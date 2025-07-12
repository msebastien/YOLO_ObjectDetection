from pathlib import Path
from typing import Union
import datetime
import logging
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results

# import supervision as sv

from task import Task

logger = logging.getLogger(__name__)


class Segment(Task):
    def __init__(
        self,
        model: Union[str, Path] = "yolo11s-seg.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_path = model if model else "yolo11s-seg.pt"
        self._model = YOLO(model=self.model_path, task="segment")
        self._conf_threshold = confidence_threshold

    def train(
        self,
        dataset="coco.yaml",
        img_size=640,
        epochs=100,
        device="cpu",
        plots=False,
    ):
        date = datetime.datetime.now()
        name = date.strftime(f"{self._model.model_name}_segment_%Y-%m-%d_%H-%M-%S")
        return self._model.train(
            data=dataset,
            imgsz=img_size,
            multi_scale=False,
            epochs=epochs,
            patience=100,
            batch=16,
            fraction=1.0,
            save=True,
            save_period=-1,
            resume=False,
            amp=True,
            cache=True,
            workers=8,
            device=device,
            project="training",
            name=name,
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=plots,
            profile=False,
        )

    def predict(self, frame):
        pass

    def _annotate(
        self,
        results: Results,
        original_img: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        pass
