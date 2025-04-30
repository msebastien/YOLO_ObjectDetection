from typing import Union
from pathlib import Path
import logging
import cv2

from mediaresource import MediaResource
from mediareader import MediaReader
from inference import Inference
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
        self._reader = MediaReader.from_location(resource_location)
        self.conf_threshold = confidence

        self._inference = Inference(model, confidence)

        self._is_stream = not resource.is_image()
        if self._is_stream:
            self._video = VideoWriter(
                file_name="annotated_output",
                fps=resource.fps(),
                frame_size=resource.frame_size(),
            )

        self._display = Display.create(resource)
        self._output_file = None

    def run(self) -> None:
        self._reader.start()

        if self._is_stream:
            self._output_file = self._video.path()
            self._video.start()
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

        logger.info(
            f"Annotated {self._reader.resource_type()} file saved!"
            f"({self.output_file})"
        )

        cv2.destroyAllWindows()
