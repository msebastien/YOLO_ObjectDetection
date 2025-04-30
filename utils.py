import os
import tempfile


def copy_video_to_temp_file(file_path):
    file_ext = os.path.splitext(file_path)[1]
    video_path = tempfile.mktemp(suffix=file_ext)

    with open(video_path, "wb") as f:
        with open(file_path, "rb") as g:
            f.write(g.read())
