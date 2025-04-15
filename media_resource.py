from enum import Enum
from typing import Tuple, overload
import cv2


class MediaResourceType(Enum):
    IMAGE = 0
    STREAM = 1


class MediaResource(object):
    @overload
    def __init__(self, filename: str, filetype: MediaResourceType) -> None:
        self._resource_location = filename
        self._type = filetype
        self._capture_api = cv2.CAP_ANY
        self._is_camera = False

        self._res = None

        if filetype == MediaResourceType.STREAM:
            self._res = cv2.VideoCapture(self._resource_location, self._capture_api)

    @overload
    def __init__(self, camera_id: int) -> None:
        self._resource_location = camera_id
        self._type = MediaResourceType.STREAM
        self._capture_api = cv2.CAP_V4L2
        self._is_camera = True
        self._res = cv2.VideoCapture(self._resource_location, self._capture_api)

    def read(self) -> Tuple[bool, cv2.typing.MatLike]:
        ret = False
        frame = None

        if self._type == MediaResourceType.STREAM:
            ret, frame = self._res.read()
        else:
            ret = True
            frame = self._res = cv2.imread(self._resource_location, cv2.COLOR_RGB2BGR)

        return ret, frame

    def is_opened(self) -> bool:
        ret = False

        if type == MediaResourceType.STREAM:
            ret = self._res.isOpened()

        return ret

    def type(self) -> MediaResourceType:
        return self._type

    def fps(self) -> float:
        freq = 0.0

        if type == MediaResourceType.STREAM:
            freq = self._res.get(cv2.CAP_PROP_FPS)

        return freq

    def path(self) -> str:
        path = ""

        if self._is_camera:
            path = "/dev/video" + self._resource_location
        else:
            path = self._resource_location

        return path

    def frame_size(self) -> Tuple[int, int]:
        width = height = 0
        if type == MediaResourceType.STREAM:
            width = int(self._res.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._res.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            width = self._res.shape[1] if self._res and self._res.any() else 0
            height = self._res.shape[0] if self._res and self._res.any() else 0

        return width, height
