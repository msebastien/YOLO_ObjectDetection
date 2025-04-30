import os
import sys
import random
import string
import logging
import logging.handlers
import argparse
import cv2
from ultralytics import YOLO

from application import Application
from mediaresource import MediaResource, MediaResourceType
from mediareader import MediaReader
from window import Window
from videowriter import VideoWriter


logger = logging.getLogger(__name__)


"""
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
"""


def yolo_inference(resource, type, custom_model, confidence):
    model = YOLO("models/yolov12s_handgestures.pt")
    if custom_model:
        model = YOLO(custom_model)

    reader = MediaReader.from_location(resource).start()

    if type == MediaResourceType.IMAGE:
        image = reader.read()
        image_size = reader.frame_size()
        logger.info(f"Image size:{image_size}")

        # Predict
        results = model.predict(source=image, imgsz=image_size, conf=confidence)

        # Annotated image
        annotated_image = annotate_frame(results, image)

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
        frame_size = reader.frame_size()
        fps = reader.fps()

        # Create display window
        display = Window(*frame_size)

        # Initialize VideoWriter utility and start process
        video = VideoWriter(
            file_name="annotated_output",
            fps=fps,
            frame_size=frame_size,
        ).start()

        # Acquisition
        should_quit = False
        while reader.can_read():
            logger.info("Reader is reading...")
            frame = reader.read()
            if frame is not None and frame.any():
                # Predict and save the newly annotated frame
                results = model.predict(source=frame, imgsz=frame_size, conf=confidence)
                annotated_frame = annotate_frame(results, frame)

                # Display in a window and write to a temp file
                should_quit = display.paint(annotated_frame)

                # Write to video file in a separate process
                video.write(annotated_frame)

            if should_quit:
                reader.stop()
                video.stop()

        return None, video.path()


def main():
    # Instantiate the parser
    parser = argparse.ArgumentParser(
        prog="YOLO Object Detection",
        description="Object Detection app powered by YOLOv12",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="",
        help="Path to specify the path to another model",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If specified, write additional information in the console "
        "about the app execution",
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
        help='Path to a video file (ex: "test.mp4")',
    )
    group.add_argument(
        "-i",
        "--image",
        default="",
        help='Path to an image file (ex: "/home/user/picture.jpg")',
    )

    args = parser.parse_args()

    # Init root logger
    # Change root logger level from WARNING (default) to NOTSET
    # in order for all messages to be delegated.
    logging.getLogger().setLevel(logging.NOTSET)

    # Add stdout handler for displaying logs in console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if not args.verbose else logging.DEBUG)
    formatter = logging.Formatter(
        "[%(levelname)s][%(name)s][%(processName)s]%(message)s"
    )
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    # Create log directory
    log_filename = "logs/object_detector_app.log"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    # Add file rotating handler, with level DEBUG
    rotatingHandler = logging.handlers.RotatingFileHandler(
        filename=log_filename,
        maxBytes=1000000,
        backupCount=5,
    )
    rotatingHandler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s\t[%(levelname)s][%(name)s][%(processName)s]%(message)s"
    )
    rotatingHandler.setFormatter(formatter)
    logging.getLogger().addHandler(rotatingHandler)

    logger.info("Started")

    # Video file/stream
    resource = args.camera
    # type = MediaResourceType.STREAM

    if args.stream:
        resource = args.stream

    if args.image:
        resource = args.image
        # type = MediaResourceType.IMAGE

    # logger.info(f"Resource Type: {type.name}")

    """
    annotated_image_path, annotated_video_path = yolo_inference(
        resource, type, args.model, args.threshold
    )

    if annotated_image_path:
        logger.info(f"Annotated image file saved! ({annotated_image_path})")
    elif annotated_video_path:
        logger.info(f"Annotated video file saved! ({annotated_video_path})")

    cv2.destroyAllWindows()
    """
    app = Application(resource, args.model, args.threshold)
    app.run()

    logger.info("Finished")


if __name__ == "__main__":
    main()
