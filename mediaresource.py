from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Tuple, Self
import magic
import cv2


class MediaResourceType(Enum):
    IMAGE = 0
    STREAM = 1


class MediaResource(ABC):
    @abstractmethod
    def read(self) -> Tuple[bool, cv2.typing.MatLike]:
        pass

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def release(self) -> None:
        pass

    @abstractmethod
    def is_initialized(self) -> bool:
        pass

    @abstractmethod
    def is_image(self) -> bool:
        pass

    @abstractmethod
    def type(self) -> MediaResourceType:
        pass

    @abstractmethod
    def path(self) -> str:
        pass

    @abstractmethod
    def frame_size(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def get(self, property_id: int) -> float:
        pass

    @abstractmethod
    def set(self, property_id: int, value: float) -> bool:
        pass

    @classmethod
    def create(cls, resource_location: Union[str, int]) -> Self:
        if isinstance(resource_location, str):
            type = cls._detect_media_type(resource_location)
            if type == MediaResourceType.STREAM:
                return StreamResource.create_from_file(resource_location)
            else:
                return ImageResource(resource_location)
        else:
            return StreamResource.create_from_device(resource_location)

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


class StreamResource(MediaResource):
    def __init__(
        self,
        resource_location: Union[str, int],
        capture_api: int,
        is_camera: bool,
    ):
        self._resource_location = resource_location
        self._type = MediaResourceType.STREAM
        self._capture_api = capture_api
        self._is_camera = is_camera
        self._resource = cv2.VideoCapture(
            self._resource_location,
            self._capture_api,
            (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY),
        )

    @classmethod
    def create_from_file(cls, filename: str) -> Self:
        return cls(filename, cv2.CAP_ANY, False)

    @classmethod
    def create_from_device(
        cls, camera_id: int, capture_api: int = cv2.CAP_V4L2
    ) -> Self:
        return cls(camera_id, capture_api, True)

    def read(self) -> Tuple[bool, cv2.typing.MatLike]:
        return self._resource.read()

    def open(self) -> None:
        if not self._resource:
            self._resource = cv2.VideoCapture(
                self._resource_location,
                self._capture_api,
                (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY),
            )

    def release(self) -> None:
        if self._resource:
            self._resource.release()
            self._resource = None

    def is_initialized(self) -> bool:
        return self._resource.isOpened()

    def is_image(self) -> bool:
        return False

    def type(self) -> MediaResourceType:
        return self._type

    def fps(self) -> float:
        return self._resource.get(cv2.CAP_PROP_FPS)

    def path(self) -> str:
        path = self._resource_location

        if self._is_camera:
            path = "/dev/video" + self._resource_location

        return path

    def frame_size(self) -> Tuple[int, int]:
        width = int(self._resource.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._resource.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def get(self, property_id: int) -> float:
        return self._resource.get(property_id)

    def set(self, property_id: int, value: float) -> bool:
        return self._resource.set(property_id, value)


class ImageResource(MediaResource):
    def __init__(self, filename: str):
        self._resource_location = filename
        self._type = MediaResourceType.IMAGE
        self._resource = cv2.imread(self._resource_location, cv2.COLOR_RGB2BGR)

    def read(self) -> Tuple[bool, cv2.typing.MatLike]:
        self._resource = cv2.imread(self._resource_location, cv2.COLOR_RGB2BGR)
        return self.is_initialized(), self._resource

    def open(self) -> None:
        return

    def release(self) -> None:
        if self.is_initialized():
            self._resource = None

    def is_initialized(self) -> bool:
        return self._resource is not None and self._resource.any()

    def is_image(self) -> bool:
        return True

    def type(self) -> MediaResourceType:
        return self._type

    def fps(self) -> float:
        return 0.0

    def path(self) -> str:
        return self._resource_location

    def frame_size(self) -> Tuple[int, int]:
        width = self._resource.shape[1] if self.is_initialized() else 0
        height = self._resource.shape[0] if self.is_initialized() else 0
        return width, height

    def get(self, property_id: int) -> float:
        return 0

    def set(self, property_id: int, value: float) -> bool:
        return False
