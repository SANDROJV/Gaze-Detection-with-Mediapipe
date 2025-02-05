import cv2
import numpy as np
from collections import deque


class RoadTrackingEngine:
    def __init__(self):
        self.direction_buffer = deque(maxlen=5)  # Stores slopes over multiple frames

    def process_frame(self, frame):
        """
        Detect road direction from a single frame.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid frame input.")

        # Step 1: Edge Detection
        edges = self._canny(frame)

        # Step 2: Region of Interest
        region = self._region_of_interest(edges, frame.shape)

        # Step 3: Analyze Road Direction
        direction = self._detect_road_direction(region)

        return direction

    def _canny(self, image):
        """
        Apply Canny Edge Detection with Gaussian Blur.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 100)
        return edges

    def _region_of_interest(self, image, shape):
        """
        Apply a mask to focus on the road region.
        """
        height, width = shape[:2]
        polygons = np.array([[
            (int(width * 0.1), height),
            (int(width * 0.9), height),
            (int(width * 0.55), int(height * 0.6))
        ]], dtype=np.int32)

        mask = np.zeros_like(image)
        cv2.fillPoly(mask, polygons, 255)
        return cv2.bitwise_and(image, mask)

    def _detect_road_direction(self, region):
        """
        Analyze road direction based on detected lines in the region of interest.
        """
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(region, 2, np.pi / 180, 50, minLineLength=50, maxLineGap=100)

        if lines is None or len(lines) == 0:
            return "Forward"

        slopes = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:  # Avoid division by zero (vertical lines)
                continue

            slope = (y2 - y1) / (x2 - x1)
            slopes.append(slope)

        if not slopes:
            return "Forward"

        avg_slope = np.mean(slopes)  # Compute average slope

        # Update direction buffer with rolling average
        self.direction_buffer.append(avg_slope)

        avg_slope_buffered = np.mean(self.direction_buffer)

        # Determine turn direction based on average slope
        if avg_slope_buffered < 0.15:
            return "Left"
        elif avg_slope_buffered > 0.25:
            return "Right"
        else:
            return "Forward"

    def annotate_frame(self, frame, road_direction, x, y):
        """
        Annotate the frame with the road direction as text.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid frame input.")
        cv2.putText(
            frame,
            f"Road Direction: {road_direction}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return frame
