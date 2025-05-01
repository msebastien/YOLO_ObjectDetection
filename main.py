import os
import sys
import logging
import logging.handlers
import argparse

from application import Application

logger = logging.getLogger(__name__)


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

    if args.stream:
        resource = args.stream

    if args.image:
        resource = args.image

    app = Application(resource, args.model, args.threshold)
    app.run()

    logger.info("Finished")


if __name__ == "__main__":
    main()
