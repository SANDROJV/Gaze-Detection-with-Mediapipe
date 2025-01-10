import os
import logging
import time

from dotenv import load_dotenv

load_dotenv()
logging.getLogger("mediapipe").setLevel(getattr(logging, os.getenv("MEDIAPIPE_LOGGING_LEVEL", "ERROR")))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.getenv("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
from gaze_tracking_engine import GazeTrackingEngine
from gaze_mapping_engine import GazeMappingEngine
from heartrate_tracking_engine import HeartRateTrackingEngine
from anomaly_tracking_engine import AnomalyTrackingEngine


def main():
    gaze_engine = GazeTrackingEngine()
    gaze_mapping = GazeMappingEngine()
    heart_rate_engine = HeartRateTrackingEngine()
    anomaly_tracker = AnomalyTrackingEngine()

    # Start webcam feed
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

    try:
        while cam.isOpened():
            ret, frame = cam.read()
            if not ret and frame is None:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)  # Flips the frame for a natural view
            gaze_engine._analyze(frame)
            annotated_frame = gaze_engine.annotated_frame()

            forehead_region = gaze_engine._analyze(frame)
            if forehead_region is not None:
                magnified_forehead = heart_rate_engine.process_forehead(forehead_region)
                heart_rate_engine.calculate_heart_rate(magnified_forehead)
            annotated_frame = heart_rate_engine.annotate_frame(annotated_frame)

            horizontal, vertical = gaze_engine.calculate_gaze()

            if isinstance(horizontal, (int, float)) and isinstance(vertical, (int, float)):
                anomaly_tracker.analyze_live_data(horizontal, vertical)
                anomaly_tracker.draw_status(annotated_frame)
            else:
                print(f"Invalid gaze data: horizontal={horizontal}, vertical={vertical}")

            cv2.imshow("ISeeYou Gaze Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break


    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        gaze_mapping.generate_heatmap()


if __name__ == "__main__":
    main()
