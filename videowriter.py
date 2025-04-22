import random
import string
import logging
import cv2
from pathlib import Path
from ctypes import c_bool, c_int
from queue import Empty, Full
from multiprocessing import Process, Queue, Value, current_process


class VideoWriter(object):
    """
    Write frames to a video file using a separate process for parallel
    execution.
    """

    def __init__(
        self,
        file_name,
        fps,
        frame_size,
        timeout=1,
        buffer_duration=10,
        verbose=False,
    ):
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
        self._verbose = Value(c_bool, verbose)
        self._queue_frame_count = Value(c_int, 0)
        self._written_frame_count = Value(c_int, 0)
        self._queue_frame_dropped_count = Value(c_int, 0)

        self._p = None
        self._frames = None
        self._max_queue_size = int(buffer_duration * self.fps)

    def is_running(self):
        """
        This method allows to know if the video writer process is active or not
        Returns:
            bool: True if the video writer process is running
                  in the background, else False
        """
        return self._p is not None and self._p.is_alive()

    def is_file_open(self):
        """
        Indicates whether the video file is open (True) or closed (False)
        Returns:
            bool: True (open), else False (closed)
        """
        return self._is_open.value

    def path(self):
        """
        Path to the output video file
        Returns:
            string: path
        """
        return self._output_file_path

    def start(self):
        """
        Start the Video Writer process
        Returns:
            VideoWriter: Current instance of VideoWriter
        """
        if not self.is_running():
            print("Start Video Writer")
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

    def stop(self):
        """
        Stop the Video Writer process
        Returns:
            VideoWriter: Current instance of VideoWriter
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

    def write(self, image):
        """
        Add image to the frame queue to be written to a file
        Args:
            image (numpy.ndarray): a multi-dimensional array representing a BGR image
        """
        if self._frames is None:
            return

        if image is None and not image.any():
            print("[VideoWriter] Image is None or an empty array")

        if not self.is_running():
            self.start()

        try:
            self._frames.put(image, timeout=self._timeout)
        except Full:
            print("[VideoWriter] Frame Queue is full. A frame will be dequeued.")
            self._frames.get()
            self._queue_frame_count.value -= 1
            self._queue_frame_dropped_count.value += 1
            print(
                f"[VideoWriter] Dropped Frames:{self._queue_frame_dropped_count.value}"
            )
        else:
            self._queue_frame_count.value += 1

    def _create_process(self, max_queue_size):
        """
        Initialize and start a separate process to write frames to a video file
        Args:
            queue_max_size (int): Maximum size allowed for the frame queue

        Returns:
            multiprocessing.Process: Initialized and started process
        """
        if not self.is_running():
            # Create frame queue
            if not self._frames:
                self._frames = Queue(maxsize=max_queue_size)

            # Initializes the Parallel process with the `writer_thread` function
            # the arguments that the function takes is mentioned in the args var
            p = Process(
                name="Video Writer",
                target=self._writer_thread,
                args=(self._output_video, self._frames),
            )
            # daemon true means, exit when main program stops
            p.daemon = True
            p.start()
            return p

    def _create_output_file(self, file_name):
        """
        Create a file to save the annotated video output
        Args:
            file_name (string): Name of output video file excluding the file extension

        Returns:
            string: Path to the output video file
        """
        id = "".join(
            random.choices(string.ascii_uppercase + string.ascii_lowercase, k=5)
        )
        output_video_path = f"captures/{file_name}_{id}.webm"
        path = Path(output_video_path)
        path.parent.mkdir(exist_ok=True, parents=True)
        open(output_video_path, "xb").close()

        return output_video_path

    def _writer_thread(self, video, queue):
        while self.is_file_open() or not queue.empty():
            # Process frames
            try:
                frame = queue.get(timeout=self._timeout)
            except Empty:
                print("VideoWriter thread: No frame available in the queue")
                continue
            else:
                print(f"Writing frame n°{self._written_frame_count.value+1}...")
                self._queue_frame_count.value -= 1
                video.write(frame)
                self._written_frame_count.value += 1

            # Print debug info
            if self._verbose.value:
                print(
                    f"{current_process().name}/"
                    f"File:{"Open" if self.is_file_open() else "Closed"}/"
                    f"Queue:{"Empty" if queue.empty() else "Not Empty" if not queue.full() else "Full"}/"
                    f"In Queue:{self._queue_frame_count.value}/"
                    f"Written:{self._written_frame_count.value}/"
                    f"Dropped:{self._queue_frame_dropped_count.value}"
                )
