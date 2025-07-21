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
    """
    Pose estimation task class for YOLO models.
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11s-pose.pt",
        confidence_threshold: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self._verbose = verbose
        self._name = "pose"

        self.model_path = model if model else "yolo11s-pose.pt"
        self._model = YOLO(
            model=self.model_path,
            task=self._name,
            verbose=self._verbose,
        )

        # Sets the minimum confidence threshold for detections.
        # Lower values increase recall but may introduce more false positives.
        # Used during validation to compute precision-recall curves.
        self._conf_threshold = confidence_threshold
        # Threshold for Non-Maximum Supression
        # Controls duplicate detection elimination
        self._intersection_over_union = 0.7

    def train(
        self,
        dataset="coco-pose.yaml",
        img_size=640,
        epochs=100,
        device="cpu",
        plots=False,
    ):
        pass

    def predict(self, frame, half=False, device="cpu", visualize=False):
        pass

    def val(self, dataset="coco-pose.yaml", device="cpu"):
        pass

    def _annotate(
        self,
        results: Results,
        original_img: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        pass
