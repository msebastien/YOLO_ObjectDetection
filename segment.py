from pathlib import Path
from typing import Union
import datetime
import logging
import cv2
from ultralytics import YOLO
from ultralytics.engine.results import Results
import supervision as sv
from supervision.annotators.utils import ColorLookup

from task import Task

logger = logging.getLogger(__name__)


class Segment(Task):
    """
    Segmentation task class for YOLO models.
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolo11s-seg.pt",
        confidence_threshold: float = 0.25,
        verbose: bool = False,
    ) -> None:
        self._verbose = verbose
        self._name = "segment"

        self.model_path = model if model else "yolo11s-seg.pt"
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
        dataset="coco.yaml",
        img_size=640,
        epochs=100,
        device="cpu",
        plots=False,
    ):
        date = datetime.datetime.now()
        dir_name = date.strftime(
            f"{self._model.model_name}_{self._name}_%Y-%m-%d_%H-%M-%S"
        )

        results = self._model.train(
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
            name=dir_name,
            val=True,
            plots=plots,
            profile=False,
            # Segmentation specific parameters
            overlap_mask=True,
            mask_ratio=4,
        )

        return results

    def predict(self, frame, half=False, device="cpu", visualize=False):
        if frame is not None and frame.any():
            date = datetime.datetime.now()
            dir_name = date.strftime(
                f"{self._model.model_name}_{self._name}_%Y-%m-%d_%H-%M-%S"
            )
            width, height = (frame.shape[1], frame.shape[0])

            results = self._model.predict(
                # Inference args
                source=frame,
                imgsz=(width, height),
                conf=self._conf_threshold,
                iou=0.7,
                half=half,
                device=device,
                max_det=300,
                visualize=visualize,
                augment=False,
                agnostic_nms=False,
                project="prediction",
                name=dir_name,
                verbose=self._verbose,
                # Segmentation specific args
                retina_masks=False,
                # Visualization args
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show_labels=True,
                show_conf=True,
                show_boxes=True,
            )

            return self._annotate(results, frame)

        return None

    def val(self, dataset="coco.yaml", device="cpu"):
        date = datetime.datetime.now()
        dir_name = date.strftime(
            f"{self._model.model_name}_{self._name}_%Y-%m-%d_%H-%M-%S"
        )

        results = self._model.val(
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
            name=dir_name,
            plots=True,
            save_txt=True,
            save_conf=True,
            agnostic_nms=False,
            single_cls=False,
            workers=8,
            verbose=self._verbose,
        )

        return results

    def _annotate(
        self,
        results: Results,
        original_img: cv2.typing.MatLike,
        use_labels: bool = True,
    ) -> cv2.typing.MatLike:
        annotated_image = original_img

        if len(results) > 0:
            detections = sv.Detections.from_ultralytics(results[0])

            # Draw segmentation masks
            mask_annotator = sv.MaskAnnotator(
                opacity=0.5, color_lookup=ColorLookup.CLASS
            )
            annotated_image = mask_annotator.annotate(
                frame=annotated_image,
                detections=detections,
                show_labels=True,
                show_conf=True,
            )

            if use_labels:
                # Define custom labels
                labels = [
                    f"{class_name} ({confidence:.2f})"
                    for class_name, confidence in zip(
                        detections["class_name"], detections.confidence
                    )
                ]

                # Display labels
                label_annotator = sv.LabelAnnotator(
                    text_position=sv.Position.TOP_CENTER,
                )
                annotated_image = label_annotator.annotate(
                    scene=annotated_image,
                    detections=detections,
                    labels=labels,
                )

        return annotated_image
