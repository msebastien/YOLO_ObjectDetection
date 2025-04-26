import logging
import numpy as np
import cv2
from typing import Union, Tuple, Self
from ctypes import c_bool, c_int
from queue import Empty, Full
from multiprocessing import Queue, Process, Value
from mediaresource import MediaResource

logger = logging.getLogger(__name__)


class MediaReader(object):
    def __init__(
        self,
        resource: MediaResource,
        timeout: float = 1.0,
        buffer_duration: int = 10,
    ) -> None:
        self._resource = resource
        self._timeout = timeout

        # Synchronized values (stored in Shared Memory)
        self._can_capture = Value(c_bool, False)
        self._queue_frame_count = Value(c_int, 0)
        self._queue_frame_dropped_count = Value(c_int, 0)
        self._read_frame_count = Value(c_int, 0)

        self._p = None
        self._frames = None

        if self._resource.is_image():
            self._max_queue_size = 1
        else:
            self._max_queue_size = int(buffer_duration * self._resource.fps())

    @classmethod
    def from_location(
        cls,
        resource_location: Union[str, int],
        timeout: float = 1.0,
        buffer_duration: int = 20,
    ) -> Self:
        return cls(MediaResource.create(resource_location), timeout, buffer_duration)

    def is_running(self) -> bool:
        return self._p is not None and self._p.is_alive()

    def can_read(self) -> bool:
        return self._can_capture.value or (
            self._frames is not None and not self._frames.empty()
        )

    def frame_size(self) -> Tuple[int, int]:
        return self._resource.frame_size()

    def fps(self) -> float:
        return self._resource.fps()

    def start(self) -> Self:
        if not self.is_running():
            logger.info("Starting...")
            if self._resource.is_initialized():
                self._resource.release()

            self._resource.open()
            self._can_capture.value = (
                self._resource.is_initialized() or self._resource.is_image()
            )

            # Spawn process for reading data in parallel
            self._p = self._create_process(self._max_queue_size)
        return self

    def stop(self) -> Self:
        if self.is_running():
            logger.info("Stopping...")

            # Close loop in process
            self._can_capture.value = False
            # Close video capture and/or free allocated frame data
            self._resource.release()

            # Indicate that no more data will be put on this queue by the current
            # process. The background thread will quit once it has flushed
            # all buffered data to the pipe.
            logger.info("Closing queue...")
            self._frames.close()
            self._frames.join_thread()
            #  Allow exit without flushing the queue in some cases,
            #  but can lead to frame data loss
            # self._frames.cancel_join_thread()

            # Wait until video writer child process terminates
            self._p.join()

            self._frames = None
            self._p = None
        return self

    def read(self) -> Union[cv2.typing.MatLike, None]:
        if not self._frames:
            return None

        if not self.is_running() and not self.can_read():
            self.start()

        try:
            frame = self._frames.get(timeout=self._timeout)
        except Empty:
            logger.warning("No frame available to read in the queue.")
            return np.array([])
        else:
            if self._resource.is_image():
                # frame is actually a full MediaResource object
                # storing the image.
                self._resource.copy(frame)

            self._queue_frame_count.value -= 1
            self._read_frame_count.value += 1
            logger.info(f"Reading Frame n°{self._read_frame_count.value}...")

            return frame if not self._resource.is_image() else self._resource.read()[1]

    def _create_process(self, max_queue_size):
        if not self.is_running():
            # Create frame queue
            if not self._frames:
                self._frames = Queue(maxsize=max_queue_size)

            # Initializes the Parallel process with the `reader_thread` function
            # the arguments that the function takes is mentioned in the args var
            p = Process(
                name="ReaderProcess",
                target=self._reader_thread,
                args=(
                    self._resource,
                    self._frames,
                    self._queue_frame_count,
                    self._queue_frame_dropped_count,
                    self._read_frame_count,
                    self._can_capture,
                ),
            )
            # daemon true means, exit when main program stops
            p.daemon = True
            p.start()
            return p
        return None

    def _reader_thread(
        self,
        resource,
        queue,
        queue_count,
        dropped_count,
        read_count,
        can_capture,
    ):
        while can_capture.value:
            # Capture frame
            ret, frame = resource.read()

            if not ret:
                can_capture.value = False
                break

            if resource.is_image() and (
                read_count.value >= 1 or queue_count.value >= 1
            ):
                can_capture.value = False
                break

            # Add captured frames to queue waiting to be read and processed
            try:
                if not resource.is_image():
                    queue.put(frame, timeout=self._timeout)
                else:
                    queue.put(resource, timeout=self._timeout)
            except Full:
                logger.warning("Frame queue is full. A frame will be dequeued.")
                queue.get()
                dropped_count.value += 1
                queue_count.value -= 1
            else:
                logger.info("Frame captured added to queue")
                queue_count.value += 1

            logger.debug(
                f"Queue:{"Empty" if queue.empty() else "Not Empty" if not queue.full() else "Full"}/"
                f"Type:{resource.type().name}/"
                f"In Queue:{queue_count.value}/"
                f"Read:{read_count.value}/"
                f"Dropped:{dropped_count.value}"
            )

        logger.info("Finished")
