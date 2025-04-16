from enum import Enum
from typing import Tuple, overload
import magic
import cv2


class MediaResourceType(Enum):
    IMAGE = 0
    STREAM = 1


class MediaResource(object):
    @overload
    def __init__(self, filename: str) -> None:
        self._resource_location = filename
        self._type = self._detect_media_type(filename)
        self._capture_api = cv2.CAP_ANY
        self._is_camera = False

        self._res = None

        if self._type == MediaResourceType.STREAM:
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
        elif self._type == MediaResourceType.IMAGE:
            ret = True
            frame = self._res = cv2.imread(self._resource_location, cv2.COLOR_RGB2BGR)

        return ret, frame

    def release(self):
        if self._res and self._res.any():
            self._res.release()
            self._res = None

    def is_opened(self) -> bool:
        ret = False

        if self._type == MediaResourceType.STREAM:
            ret = self._res.isOpened()

        return ret

    def type(self) -> MediaResourceType:
        return self._type

    def fps(self) -> float:
        freq = 0.0

        if self._type == MediaResourceType.STREAM:
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
        if self._type == MediaResourceType.STREAM:
            width = int(self._res.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._res.get(cv2.CAP_PROP_FRAME_HEIGHT))
        elif self._type == MediaResourceType.IMAGE:
            width = self._res.shape[1] if self._res and self._res.any() else 0
            height = self._res.shape[0] if self._res and self._res.any() else 0

        return width, height

    @staticmethod
    def _detect_media_type(filename: str) -> MediaResourceType:
        detected_type = None
        mimetype = magic.from_file(filename, mime=True)
        if mimetype:
            media = mimetype.split("/")[0]
            if media == "video":
                detected_type = MediaResourceType.STREAM
            elif media == "image":
                detected_type = MediaResourceType.IMAGE
        return detected_type
