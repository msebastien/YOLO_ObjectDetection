from abc import ABC, abstractmethod
import cv2
import cv2.typing

from mediaresource import MediaResource, MediaResourceType
from window import Window


class Display(ABC):
    @abstractmethod
    def show(self, frame: cv2.typing.NumPyArrayNumeric) -> None:
        pass

    @abstractmethod
    def close_requested(self) -> bool:
        pass

    @classmethod
    def create(cls, resource: MediaResource, title: str = None):
        match resource.type():
            case MediaResourceType.STREAM:
                return SDLDisplay(*resource.frame_size(), title)
            case MediaResourceType.IMAGE:
                return CVImageDisplay(title)
            case _:
                return SDLDisplay(*resource.frame_size(), title)


class SDLDisplay(Display):
    def __init__(self, width, height, title="Video"):
        self._window = Window(width, height, title)

    def show(self, frame):
        self._window.paint(frame)

    def close_requested(self):
        return self._window.close_requested()


class CVImageDisplay(Display):
    def __init__(self, title="Image"):
        self._title = title

    def show(self, frame):
        cv2.imshow(self._title, frame)

    def close_requested(self):
        return cv2.waitKey(1) & 0xFF == ord("q")
