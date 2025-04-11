import random
import string
import cv2
from ctypes import c_bool
from multiprocessing import Process, Queue, Value, current_process
from pathlib import Path


class VideoWriter(object):
    """
    Write frames to a video file using a separate process for parallel
    execution.
    """

    def __init__(self, file_name, fps, frame_size, debug=False):
        # Create file and get path to it
        self.file_name = file_name
        self._output_file_path = self._create_output_file(self.file_name)

        # Set up codec and output video settings
        self.fps = fps
        self.frame_size = frame_size
        self._codec = cv2.VideoWriter.fourcc(*"vp80")
        self._output_video = cv2.VideoWriter()

        # Synchronized values (stored in Shared Memory)
        self._is_open = Value(c_bool, False)
        self._debug = Value(c_bool, debug)

        self._p = None
        self._queue_max_size = int(5 * self.fps)
        self._frames = None

    def is_running(self):
        return self._p is not None and self._p.is_alive()

    def is_file_open(self):
        """
        Indicates whether the video file is open (True) or closed (False)
        Returns:
            bool: True (open), else False (closed)
        """
        return self._is_open.value

    def get_path(self):
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
            # Open video file for writing data to it
            self._is_open.value = self._output_video.open(
                filename=self._output_file_path,
                fourcc=self._codec,
                fps=self.fps,
                frameSize=self.frame_size,
            )
            # Spawn process for writing data in parallel
            self._p = self._create_process(self._queue_max_size)
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
            self._frames.cancel_join_thread()

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
        
        if self._frames.full():
            self._frames.get()

        self._frames.put(image)

    def _create_process(self, queue_max_size):
        """
        Initialize and start a separate process to write frames to a video file
        Args:
            queue_max_size (int): Maximum size allowed for the frame queue

        Returns:
            multiprocessing.Process: Initialized and started process
        """
        if not self.is_running():
            # Create frame queue
            self._frames = Queue(maxsize=queue_max_size)

            # Initializes the Parallel process with the `writer_thread` function
            # the arguments that the function takes is mentioned in the args var
            p = Process(
                name="Video Writer Process",
                target=self._writer_thread,
                args=(self._output_video, self._frames, self._is_open, self._debug),
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

    def _writer_thread(self, video, queue, is_file_open, debug):
        while is_file_open.value or not queue.empty():
            # Process frames
            if not queue.empty():
                frame = queue.get()
                if frame is not None and frame.any():
                    video.write(frame)

            # Print debug info
            if debug.value:
                print(
                    f"{current_process().name}/"
                    f"File Open:{is_file_open.value}/"
                    f"Queue Empty:{queue.empty()}"
                )
