import cv2
import numpy as np
from collections import deque


class RoadTrackingEngine:
    def __init__(self):
        self.low_threshold = 50
        self.high_threshold = 150
        self.direction_buffer = deque(maxlen=10)

    def process_frame(self, frame):
        """
        Detect road direction from a single frame.
        """
        # Step 1: Edge Detection
        edges = self._canny(frame)

        # Step 2: Region of Interest
        region = self._region_of_interest(edges, frame.shape)

        # Step 3: Analyze Road Direction
        direction = self._detect_road_direction(region)

        # Step 4: Smooth Direction with Buffer
        self.direction_buffer.append(direction)
        stable_direction = self._get_stable_direction()

        return stable_direction

    def _canny(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.low_threshold, self.high_threshold)
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
        lines = cv2.HoughLinesP(region, 2, np.pi / 180, 100, minLineLength=50, maxLineGap=150)

        if lines is None or len(lines) == 0:
            return "Forward"

        left_slopes = []
        right_slopes = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if slope < -0.5:  # Left-sloping lines
                left_slopes.append(slope)
            elif slope > 0.5:  # Right-sloping lines
                right_slopes.append(slope)

        # Determine road direction based on the balance of left and right slopes
        if len(left_slopes) > len(right_slopes):
            return "Left"
        elif len(right_slopes) > len(left_slopes):
            return "Right"
        else:
            return "Forward"

    def _get_stable_direction(self):
        """
        Determine the most frequent direction in the buffer.
        """
        if not self.direction_buffer:
            return "Forward"
        return max(set(self.direction_buffer), key=self.direction_buffer.count)

    def annotate_frame(self, frame, road_direction):
        """
        Annotate the frame with the road direction as text.
        """
        cv2.putText(
            frame,
            f"Road Direction: {road_direction}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        return frame
