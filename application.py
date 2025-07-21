from typing import Union
from pathlib import Path
import logging
import cv2

from mediaresource import MediaResource
from mediareader import MediaReader
from detect import Detect
from videowriter import VideoWriter
from display import Display
import utils

logger = logging.getLogger(__name__)


class Application(object):
    def __init__(
        self,
        resource_location: Union[str, int],
        model: Union[str, Path],
        confidence: float,
    ) -> None:
        self._resource = MediaResource.create(resource_location)
        self._reader = MediaReader(self._resource)
        self.conf_threshold = confidence

        # TODO: Create Task using a classmethod
        # Task.create(model, confidence)
        self._inference = Detect(model, confidence)

        self._is_stream = not self._resource.is_image()
        self._video = VideoWriter(
            file_name="annotated_output",
            fps=self._resource.fps(),
            frame_size=self._resource.frame_size(),
        )

        self._display = Display.create(self._resource)
        self._output_file = ""

    def run(self) -> None:
        self._reader.start()

        # Retrieve output video/image file path
        if self._is_stream:
            self._video.start()
            self._output_file = self._video.path()
        else:
            self.output_file = utils._get_image_path()

        while self._reader.can_read():
            # Acquisition
            frame = self._reader.read()

            # Predict and save the newly annotated frame
            annotated = self._inference.predict(frame)

            if self._is_stream:
                # Write to video file in a separate process
                self._video.write(annotated)
            else:
                utils._write_image(annotated, self.output_file)

            # Display frame in a window
            self._display.show(annotated)

            if self._display.close_requested():
                self._reader.stop()
                self._video.stop()
                self._display.close()

        logger.info(
            f"Annotated {self._reader.resource_type().name} file saved!"
            f"({self._output_file})"
        )
