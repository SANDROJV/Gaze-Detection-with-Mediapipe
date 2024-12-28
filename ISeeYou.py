import os
import logging
import cv2
from tracking_engine import TrackingEngine


def main():
    # Suppress warnings and logs
    logging.getLogger("mediapipe").setLevel(logging.ERROR)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    engine = TrackingEngine()

    # Start webcam feed
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

    while cam.isOpened():
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Flips the frame for a natural view
        frame = cv2.flip(frame, 1)

        engine._analyze(frame)
        annotated_frame = engine.annotated_frame()
        cv2.imshow("ISeeYou Gaze Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
