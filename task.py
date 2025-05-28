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
        epochs: int,
        device: str,
        resume: bool,
    ) -> Union[Dict, None]:
        pass

    @abstractmethod
    def predict(
        self,
        frame: cv2.typing.MatLike,
    ) -> Union[cv2.typing.MatLike, None]:
        pass

    @abstractmethod
    def val(self, dataset: str, device: str, verbose: bool) -> DetMetrics:
        pass
