from pathlib import Path
from typing import Union
import logging
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

# import supervision as sv

from task import Task

logger = logging.getLogger(__name__)


class Pose(Task):
    def __init__(
        self,
        model: Union[str, Path] = "yolo11s-pose.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_path = model if model else "yolo11s-pose.pt"
        self._model = YOLO(model=self.model_path, task="pose")
        self._conf_threshold = confidence_threshold

    def predict(self, frame):
        pass

    def _annotate(
        self,
        results: Results,
        original_img: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        pass
