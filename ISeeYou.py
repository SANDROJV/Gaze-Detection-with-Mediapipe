import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.getLogger("mediapipe").setLevel(getattr(logging, os.getenv("MEDIAPIPE_LOGGING_LEVEL", "ERROR")))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.getenv("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
from gaze_tracking_engine import GazeTrackingEngine


def main():
    gaze_engine = GazeTrackingEngine()

    # Start webcam feed
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flips the frame for a natural view
        frame = cv2.flip(frame, 1)
        gaze_engine._analyze(frame)
        annotated_frame = gaze_engine.annotated_frame()
        cv2.imshow("ISeeYou Gaze Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
