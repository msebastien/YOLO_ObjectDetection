import os
import random
import string
import argparse
from enum import Enum
import tempfile
import cv2
from ultralytics import YOLO
import supervision as sv

from display import Display
from videowriter import VideoWriter


class Resource(Enum):
    IMAGE = 0
    VIDEO = 1
    CAMERA = 2


def copy_video_to_temp_file(file_path):
    file_ext = os.path.splitext(file_path)[1]
    video_path = tempfile.mktemp(suffix=file_ext)

    with open(video_path, "wb") as f:
        with open(file_path, "rb") as g:
            f.write(g.read())


def annotate_frame(results, original_img):
    annotated_image = original_img

    if len(results) > 0:
        result = results[0]
        detections = sv.Detections.from_ultralytics(result)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # annotated_image = result.orig_img
        annotated_image = box_annotator.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections
        )

    return annotated_image


def yolo_inference(resource, type, custom_model, confidence):
    model = YOLO("model/yolov12s-handgestures.pt")
    if custom_model:
        model = YOLO(custom_model)

    if type == Resource.IMAGE:
        image = cv2.imread(resource, cv2.COLOR_RGB2BGR)
        height, width = image.shape[0], image.shape[1]

        # Predict
        results = model.predict(source=image, imgsz=(width, height), conf=confidence)

        # Annotated image
        annotated_image = annotate_frame(results)

        # Save the result
        id = "".join(
            random.choices(string.ascii_uppercase + string.ascii_lowercase, k=5)
        )
        output_image_path = f"captures/annotated_output_{id}.jpg"
        cv2.imwrite(output_image_path, annotated_image)

        # Display the result
        while True:
            cv2.imshow("Annotated Image", annotated_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return output_image_path, None

    else:
        # Setup acquisition
        cap = cv2.VideoCapture(resource)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)

        # Create display window
        display = Display(frame_width, frame_height)

        # Initialize VideoWriter utility and start process
        video = VideoWriter(
            file_name="annotated_output",
            fps=fps,
            frame_size=frame_size,
        ).start()

        # Acquisition
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict and save the newly annotated frame
            results = model.predict(source=frame, imgsz=frame_size, conf=confidence)
            annotated_frame = annotate_frame(results, frame)

            # Display in a window and write to a temp file
            should_quit = display.paint(annotated_frame)

            # Write to video file in a separate process
            video.write(annotated_frame)

            if should_quit or cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        video.stop()

        return None, video.get_path()


def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        prog="YOLO Hand Gesture",
        description="Hand Gestures recognition app powered by YOLOv12",
    )
    parser.add_argument(
        "-t", "--threshold", default=0.25, help="Confidence threshold for detection"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help="Path to specify the path to another custom model",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        help="Integer representing the camera id (ex: 0)",
    )
    group.add_argument(
        "-s",
        "--stream",
        default="",
        help='Path to a video stream, which can be a file or a device (ex: "/dev/video0" or "test.mp4")',
    )
    group.add_argument(
        "-i",
        "--image",
        default="",
        help='Path to an image file (ex: "/home/user/picture.jpg")',
    )

    args = parser.parse_args()

    # Video file/stream acquisition
    resource = args.camera
    type = Resource.CAMERA

    if args.stream:
        resource = args.stream
        type = Resource.VIDEO
    elif args.image:
        resource = args.image
        type = Resource.IMAGE

    annotated_image_path, annotated_video_path = yolo_inference(
        resource, type, args.model, args.threshold
    )

    if annotated_image_path:
        print(f"Annotated image file saved! ({annotated_image_path})")
    elif annotated_video_path:
        print(f"Annotated video file saved! ({annotated_video_path})")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
