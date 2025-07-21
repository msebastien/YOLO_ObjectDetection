from abc import ABC, abstractmethod
from typing import Union, Dict
import cv2
from ultralytics.utils.metrics import DetMetrics


class Task(ABC):
    # https://www.ultralytics.com/glossary/epoch
    @abstractmethod
    def train(
        self,
        dataset: str,
        img_size: int,
        epochs: int,
        device: str,
        plots: bool,
    ) -> Union[Dict, None]:
        pass

    @abstractmethod
    def predict(
        self,
        frame: cv2.typing.MatLike,
        half: bool,
        device: str,
        visualize: bool,
    ) -> Union[cv2.typing.MatLike, None]:
        pass

    @abstractmethod
    def val(self, dataset: str, device: str) -> DetMetrics:
        pass
