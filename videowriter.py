import os
import random
import string
import logging
from typing import Union, Tuple, Self
import cv2
from ctypes import c_bool, c_int
from queue import Empty, Full
from multiprocessing import Process, Queue, Value, current_process

logger = logging.getLogger(__name__)


class VideoWriter(object):
    """
    Write frames to a video file using a separate process for parallel
    execution.
    """

    def __init__(
        self,
        file_name: str,
        fps: float,
        frame_size: Tuple[int, int],
        timeout: int = 1,
        buffer_duration: int = 10,
    ) -> None:
        # Create file and get path to it
        self.file_name = file_name
        self._output_file_path = self._create_output_file(self.file_name)
        self._timeout = timeout

        # Set up codec and output video settings
        self.fps = fps
        self.frame_size = frame_size
        self._codec = cv2.VideoWriter.fourcc(*"vp80")
        self._output_video = cv2.VideoWriter()

        # Synchronized values (stored in Shared Memory)
        self._is_open = Value(c_bool, False)
        self._queue_frame_count = Value(c_int, 0)
        self._written_frame_count = Value(c_int, 0)
        self._queue_frame_dropped_count = Value(c_int, 0)

        self._p = None
        self._frames = None
        self._max_queue_size = int(buffer_duration * self.fps)

    def is_running(self) -> bool:
        """
        This method allows to know if the video writer process is active or not.
        Returns:
            bool: True if the video writer process is running
                  in the background, else False.
        """
        return self._p is not None and self._p.is_alive()

    def is_file_open(self) -> bool:
        """
        Indicates whether the video file is open (True) or closed (False).
        Returns:
            bool: True (open), else False (closed).
        """
        return self._is_open.value

    def path(self) -> str:
        """
        Path to the output video file.
        Returns:
            string: path
        """
        return self._output_file_path

    def start(self) -> Self:
        """
        Start the Video Writer process.
        Returns:
            VideoWriter: Current instance of VideoWriter.
        """
        if not self.is_running():
            logger.info("Start Video Writer")
            if self._is_open.value:
                self._output_video.release()

            # Open video file for writing data to it
            self._is_open.value = self._output_video.open(
                filename=self._output_file_path,
                fourcc=self._codec,
                fps=self.fps,
                frameSize=self.frame_size,
            )
            # Spawn process for writing data in parallel
            self._p = self._create_process(self._max_queue_size)
        return self

    def stop(self) -> Self:
        """
        Stop the Video Writer process.
        Returns:
            VideoWriter: Current instance of VideoWriter.
        """
        if self.is_running():
            # Free allocated queue data and join thread
            self._frames.close()
            self._frames.join_thread()
            #   Allow exit without flushing the queue in some cases,
            #   but can lead to frame data loss
            # self._frames.cancel_join_thread()

            # Close video file (which will stop the writer process)
            self._output_video.release()
            self._is_open.value = False

            # Wait until video writer child process terminates
            self._p.join()

            self._frames = None
            self._p = None
        return self

    def write(self, image: cv2.typing.MatLike) -> None:
        """
        Add image to the frame queue to be written to a file.
        Args:
            image (numpy.ndarray): a multi-dimensional array representing a BGR image.
        """
        if self._frames is None:
            return

        if image is None and not image.any():
            logger.warning("Image is None or an empty array.")

        if not self.is_running():
            self.start()

        try:
            self._frames.put(image, timeout=self._timeout)
        except Full:
            logger.warning("Frame Queue is full. A frame will be dequeued.")
            self._frames.get()
            self._queue_frame_count.value -= 1
            self._queue_frame_dropped_count.value += 1
            logger.info(f"Dropped Frames:{self._queue_frame_dropped_count.value}")
        else:
            self._queue_frame_count.value += 1

    def _create_process(self, max_queue_size: int) -> Union[Process, None]:
        """
        Initialize and start a separate process to write frames to a video file.
        Args:
            queue_max_size (int): Maximum size allowed for the frame queue.

        Returns:
            multiprocessing.Process: Initialized and started process.
        """
        if not self.is_running():
            # Create frame queue
            if not self._frames:
                self._frames = Queue(maxsize=max_queue_size)

            # Initializes the Parallel process with the `writer_thread` function
            # the arguments that the function takes is mentioned in the args var
            p = Process(
                name="WriterProcess",
                target=self._writer_thread,
                args=(
                    self._output_video,
                    self._frames,
                    self._queue_frame_count,
                    self._written_frame_count,
                    self._queue_frame_dropped_count,
                ),
            )
            # daemon true means, exit when main program stops
            p.daemon = True
            p.start()
            return p
        return None

    def _create_output_file(self, file_name: str) -> str:
        """
        Create a file to save the annotated video output.
        Args:
            file_name (string): Name of output video file excluding the file extension.

        Returns:
            string: Path to the output video file.
        """
        id = "".join(
            random.choices(string.ascii_uppercase + string.ascii_lowercase, k=5)
        )
        output_video_path = f"captures/{file_name}_{id}.webm"
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        open(output_video_path, "xb").close()

        return output_video_path

    def _writer_thread(
        self,
        video: cv2.VideoWriter,
        queue: Queue,
        queue_count: int,
        written_count: int,
        dropped_count: int,
    ) -> None:
        while self.is_file_open() or not queue.empty():
            # Process frames
            try:
                frame = queue.get(timeout=self._timeout)
            except Empty:
                logger.warning("No frame available in the queue.")
                continue
            else:
                logger.info(f"Writing frame n°{written_count.value+1}...")
                queue_count.value -= 1
                video.write(frame)
                written_count.value += 1

            # Print debug info
            logger.debug(
                f"File:{"Open" if self.is_file_open() else "Closed"}/"
                f"Queue:{"Empty" if queue.empty() else "Not Empty" if not queue.full() else "Full"}/"
                f"In Queue:{queue_count.value}/"
                f"Written:{written_count.value}/"
                f"Dropped:{dropped_count.value}"
            )
