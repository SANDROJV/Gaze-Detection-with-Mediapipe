import os
import logging
import numpy as np
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), 'Engines', 'Data', '.env')
load_dotenv()
logging.getLogger("mediapipe").setLevel(getattr(logging, os.getenv("MEDIAPIPE_LOGGING_LEVEL", "ERROR")))
os.environ["TF_ENABLE_ONEDNN_OPTS"] = os.getenv("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
from Engines.gaze_tracking_engine import GazeTrackingEngine
from Engines.gaze_mapping_engine import GazeMappingEngine
from Engines.heartrate_tracking_engine import HeartRateTrackingEngine
from Engines.anomaly_tracking_engine import AnomalyTrackingEngine
from Engines.road_tracking_engine import RoadTrackingEngine


def combine_frames(driving_frame, webcam_frame, zoom_factor=3):
    """
    Combines the driving video frame with a centered, square, zoomed-in webcam frame.

    Parameters:
        driving_frame: The main driving video frame.
        webcam_frame: The webcam feed frame to overlay.
        zoom_factor: The factor by which to zoom into the webcam frame.
    """
    h, w, _ = webcam_frame.shape
    square_size = min(h, w)  # Ensure the webcam frame is square
    half_square = square_size // 2

    # Crop the webcam frame around its center
    center_x, center_y = w // 2, h // 2
    x1 = max(0, center_x - half_square)
    y1 = max(0, center_y - half_square)
    x2 = min(w, center_x + half_square)
    y2 = min(h, center_y + half_square)

    cropped_frame = webcam_frame[y1:y2, x1:x2]

    # Apply zoom
    zoomed_frame = cv2.resize(
        cropped_frame,
        None,
        fx=zoom_factor,
        fy=zoom_factor,
        interpolation=cv2.INTER_LINEAR,
    )

    # Resize for overlay
    inset_size = driving_frame.shape[0] // 3  # Adjust size of the inset frame
    resized_webcam_frame = cv2.resize(zoomed_frame, (inset_size, inset_size))

    # Overlay the webcam frame on the driving frame
    combined_frame = driving_frame.copy()
    x_offset = combined_frame.shape[1] - inset_size - 10
    y_offset = 10
    combined_frame[y_offset:y_offset + inset_size, x_offset:x_offset + inset_size] = resized_webcam_frame

    return combined_frame


def main():
    gaze_engine = GazeTrackingEngine()
    gaze_mapping = GazeMappingEngine()
    heart_rate_engine = HeartRateTrackingEngine()
    anomaly_tracker = AnomalyTrackingEngine()
    road_tracker = RoadTrackingEngine()

    # Start webcam feed and video
    cam = cv2.VideoCapture(0)
    video_path = "Engines/Data/road_video.mp4"
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    try:
        while cam.isOpened() and video.isOpened():
            ret_cam, webcam_frame = cam.read()
            ret_video, driving_frame = video.read()

            if not ret_cam and webcam_frame is None:
                print("Failed to grab webcam frame.")
                break

            if not ret_video or driving_frame is None:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            webcam_frame = cv2.flip(webcam_frame, 1)  # Flips the frame for a natural view
            gaze_engine._analyze(webcam_frame)
            driving_frame = gaze_engine.annotate_frame(driving_frame, 30, 60)

            road_direction = road_tracker.process_frame(driving_frame)
            driving_frame = road_tracker.annotate_frame(driving_frame, road_direction, 30, 30)

            horizontal, vertical = gaze_engine.calculate_gaze()
            head_direction = gaze_engine.calculate_head_orientation()
            if isinstance(horizontal, (int, float)) and isinstance(vertical, (int, float)):
                gaze_mapping.record_gaze(horizontal, vertical)
                anomaly_tracker.analyze_live_data(horizontal, vertical, road_direction, head_direction)
                anomaly_tracker.annotate_frame(driving_frame, 30, 120)
            else:
                print(f"Invalid gaze data: horizontal={horizontal}, vertical={vertical}")

            forehead_region = gaze_engine.extract_forehead_region(webcam_frame)
            if forehead_region is not None and isinstance(forehead_region, np.ndarray):
                processed_forehead = heart_rate_engine.process_forehead(forehead_region, webcam_frame)
                heart_rate_engine.calculate_heart_rate(processed_forehead)
            driving_frame = heart_rate_engine.annotate_frame(driving_frame, 30, 150)

            combined_frame = combine_frames(driving_frame, webcam_frame)
            cv2.imshow("ISeeYou", combined_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
    finally:
        cam.release()
        video.release()
        cv2.destroyAllWindows()
        gaze_mapping.generate_heatmap()


if __name__ == "__main__":
    main()
