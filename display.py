from abc import ABC, abstractmethod
import cv2
import cv2.typing

from mediaresource import MediaResource, MediaResourceType
from window import Window


class Display(ABC):
    @abstractmethod
    def show(self, frame: cv2.typing.MatLike) -> None:
        pass

    @abstractmethod
    def close_requested(self) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @classmethod
    def create(cls, resource: MediaResource):
        match resource.type():
            case MediaResourceType.STREAM:
                return SDLDisplay(*resource.frame_size())
            case MediaResourceType.IMAGE:
                return CVImageDisplay()
            case _:
                return SDLDisplay(*resource.frame_size())


class SDLDisplay(Display):
    def __init__(self, width, height, title="Video"):
        self._window = Window(width, height, title)

    def show(self, frame):
        self._window.paint(frame)

    def close_requested(self):
        return self._window.close_requested()

    def close(self):
        self._window.close()


class CVImageDisplay(Display):
    def __init__(self, title="Image"):
        self._title = title
        self._close = False

    def show(self, frame):
        cv2.imshow(self._title, frame)

    def close_requested(self):
        while cv2.getWindowProperty(self._title, cv2.WND_PROP_VISIBLE) >= 1:
            keyCode = cv2.waitKey(delay=1000)
            if keyCode & 0xFF == ord("q"):
                break
        return True

    def close(self):
        cv2.destroyWindow(self._title)
