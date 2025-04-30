import os
import random
import string
import tempfile
import cv2


def copy_video_to_temp_file(file_path):
    file_ext = os.path.splitext(file_path)[1]
    video_path = tempfile.mktemp(suffix=file_ext)

    with open(video_path, "wb") as f:
        with open(file_path, "rb") as g:
            f.write(g.read())


def _get_image_path():
    id = "".join(
        random.choices(
            string.ascii_uppercase + string.ascii_lowercase,
            k=5,
        )
    )
    return f"captures/annotated_output_{id}.jpg"


def _write_image(frame, path):
    cv2.imwrite(path, frame)
