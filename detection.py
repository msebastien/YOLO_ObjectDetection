from pathlib import Path
from typing import Union
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
        model: Union[str, Path] = "yolov12s.pt",
        confidence_threshold: float = 0.25,
    ) -> None:
        self.model_path = model if model else "models/detect/yolov12s_handgestures.pt"
        self._model = YOLO(model=self.model_path, task="detect").va
        self._conf_threshold = confidence_threshold

    def train(self, dataset, epochs, device="cpu", resume=False):
        return self._model.train(
            data=dataset,
            epochs=epochs,
            workers=8,
            device=device,
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

    def val(self, dataset, device, verbose):
        pass

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
