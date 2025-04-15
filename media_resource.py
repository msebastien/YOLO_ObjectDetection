from enum import Enum
from typing import Tuple, Union
import cv2


class MediaResourceType(Enum):
    IMAGE = 0
    STREAM = 1


class MediaResource(object):
    def __init__(
        self,
        location: Union[str, int],
        res_type: MediaResourceType,
    ) -> None:
        self._resource_location = location
        self._type = res_type
        self._capture_api = cv2.CAP_ANY
        self._is_camera = type(location) is int

        # If a camera ID is specified
        if self._is_camera:
            self._type = MediaResourceType.STREAM
            self._capture_api = cv2.CAP_V4L2

        if res_type == MediaResourceType.STREAM:
            self._res = cv2.VideoCapture(self._resource_location, self._capture_api)
        else:
            self._res = None

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
