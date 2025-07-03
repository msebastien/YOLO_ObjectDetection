from pathlib import Path
from typing import Union
import datetime
import logging
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import supervision as sv

from task import Task

logger = logging.getLogger(__name__)


class Detection(Task):
    def __init__(
        self,
        model: Union[str, Path] = "yolo11n.pt",
        confidence_threshold: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self._verbose = verbose

        # self.model_path = model if model else "models/detect/yolov12s_handgestures.pt"
        self.model_path = model if model else "yolo11n.pt"
        self._model = YOLO(model=self.model_path, task="detect", verbose=self._verbose)
        self._conf_threshold = confidence_threshold

    def train(self, dataset="coco.yaml", img_size=640, epochs=100, device="cpu", plots=False):
        date = datetime.datetime.now()
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
            name=f"{date.strftime(f"{self._model.model_name}_%Y-%m-%d_%H-%M-%S")}",
            overlap_mask=True,
            mask_ratio=4,
            dropout=0.0,
            val=True,
            plots=plots,
            profile=False,
        )

    def predict(self, frame):
        if frame is not None and frame.any():
            width, height = (frame.shape[1], frame.shape[0])
            results = self._model.predict(
                source=frame,
                imgsz=(width, height),
                conf=self._conf_threshold,
            )
            return self._annotate(results, frame)
        return None

    def val(self, dataset="coco.yaml", device="cpu"):
        date = datetime.datetime.now()
        return self._model.val(
            data=dataset,
            batch=16,
            # Sets the minimum confidence threshold for detections.
            # Lower values increase recall but may introduce more false positives.
            # Used during validation to compute precision-recall curves.
            conf=self._conf_threshold,
            # Threshold for Non-Maximum Supression
            # Controls duplicate detection elimination
            iou=0.7,
            device=device,
            project="validation",
            name=f"{date.strftime(f"{self._model.model_name}_%Y-%m-%d_%H-%M-%S")}",
            plots=True,
            save_txt=True,
            save_conf=True,
            agnostic_nms=False,
            single_cls=False,
            workers=8,
            verbose=self._verbose,
        )

    def _annotate(
        self,
        results: Results,
        original_img: cv2.typing.MatLike,
    ) -> cv2.typing.MatLike:
        annotated_image = original_img

        if len(results) > 0:
            detections = sv.Detections.from_ultralytics(results[0])

            # Draw bounding box
            box_annotator = sv.BoxAnnotator()
            annotated_image = box_annotator.annotate(
                scene=annotated_image, detections=detections
            )

            # Define custom labels for bounding boxes
            labels = [
                f"{class_name} ({confidence:.2f})"
                for class_name, confidence in zip(
                    detections["class_name"], detections.confidence
                )
            ]

            # Display labels
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
            annotated_image = label_annotator.annotate(
                scene=annotated_image,
                detections=detections,
                labels=labels,
            )

        return annotated_image
