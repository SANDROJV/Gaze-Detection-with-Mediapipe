import cv2
import mediapipe as mp


class TrackingEngine:

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def _analyze(self, frame):
        """
        Detects face and initializes eye regions using MediaPipe.
        """
        self.frame = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(150, 150, 150), thickness=1, circle_radius=0
                    ),
                )
                h, w, _ = frame.shape

                # Left eye landmarks
                self.eye_left = [
                    (
                        int(face_landmarks.landmark[33].x * w),
                        int(face_landmarks.landmark[33].y * h),
                    ),  # Left corner
                    (
                        int(face_landmarks.landmark[133].x * w),
                        int(face_landmarks.landmark[133].y * h),
                    ),  # Right corner
                    (
                        int(face_landmarks.landmark[468].x * w),
                        int(face_landmarks.landmark[468].y * h),
                    ),  # Iris center
                    (
                        int(face_landmarks.landmark[159].x * w),
                        int(face_landmarks.landmark[159].y * h),
                    ),  # Top eyelid
                    (
                        int(face_landmarks.landmark[145].x * w),
                        int(face_landmarks.landmark[145].y * h),
                    ),  # Bottom eyelid
                ]

                # Right eye landmarks
                self.eye_right = [
                    (
                        int(face_landmarks.landmark[362].x * w),
                        int(face_landmarks.landmark[362].y * h),
                    ),  # Left corner
                    (
                        int(face_landmarks.landmark[263].x * w),
                        int(face_landmarks.landmark[263].y * h),
                    ),  # Right corner
                    (
                        int(face_landmarks.landmark[473].x * w),
                        int(face_landmarks.landmark[473].y * h),
                    ),  # Iris center
                    (
                        int(face_landmarks.landmark[386].x * w),
                        int(face_landmarks.landmark[386].y * h),
                    ),  # Top eyelid
                    (
                        int(face_landmarks.landmark[374].x * w),
                        int(face_landmarks.landmark[374].y * h),
                    ),  # Bottom eyelid
                ]

    def horizontal_ratio(self):
        """
        Calculates horizontal ratio based on the distance between the left and right landmarks.
        """
        if self.eye_left and self.eye_right:
            # Eye corners
            left_eye_corners = (self.eye_left[0], self.eye_left[1])
            right_eye_corners = (self.eye_right[0], self.eye_right[1])

            # Iris centers
            left_iris_center = self.eye_left[2]  # Center of the left iris
            right_iris_center = self.eye_right[2]  # Center of the right iris

            # Calculate horizontal gaze ratio for both eyes
            left_eye_range = left_eye_corners[1][0] - left_eye_corners[0][0]
            left_ratio = (left_iris_center[0] - left_eye_corners[0][0]) / left_eye_range
            right_eye_range = right_eye_corners[1][0] - right_eye_corners[0][0]
            right_ratio = (right_iris_center[0] - right_eye_corners[0][0]) / right_eye_range

            # Average ratio for both eyes
            avg_ratio = (left_ratio + right_ratio) / 2
            return avg_ratio
        return None

    def vertical_ratio(self):
        """
        Calculates vertical ratio based on the distance between the top and bottom eyelids.
        """
        if self.eye_left and self.eye_right:
            # Top eyelid
            left_top = self.eye_left[3]
            right_top = self.eye_right[3]

            # Bottom eyelid
            left_bottom = self.eye_left[4]
            right_bottom = self.eye_right[4]

            # Eye height
            left_eye_height = left_bottom[1] - left_top[1]
            right_eye_height = right_bottom[1] - right_top[1]

            # Horizontal range (for normalization)
            left_eye_range = (self.eye_left[1][0] - self.eye_left[0][0])
            left_ratio = left_eye_height / left_eye_range
            right_eye_range = (self.eye_right[1][0] - self.eye_right[0][0])
            right_ratio = right_eye_height / right_eye_range

            # Average ratio for both eyes
            avg_ratio = (left_ratio + right_ratio) / 2
            return avg_ratio
        return None

    def calculate_gaze(self):
        """
        Determines the gaze direction based on the horizontal ratio.
        :return: String indicating gaze direction.
        """

        vertical_ratio = self.vertical_ratio()
        if vertical_ratio is not None and vertical_ratio <= 0.1:
            return "Eyes Closed"

        horizontal_ratio = self.horizontal_ratio()
        if horizontal_ratio is not None:
            if horizontal_ratio <= 0.35:
                return "Looking Left"
            elif horizontal_ratio >= 0.65:
                return "Looking Right"
            else:
                return "Looking Center"

        return "No Gaze Detected"

    def annotated_frame(self):
        """
        Overlays gaze direction and other features on the frame.
        """
        frame = self.frame.copy()

        # Eye point drawing
        if self.eye_left:
            for point in self.eye_left:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)  # Green for left eye
        if self.eye_right:
            for point in self.eye_right:
                cv2.circle(frame, point, 2, (0, 255, 0), -1)  # Green for right eye

        gaze_text = self.calculate_gaze()
        cv2.putText(
            frame, gaze_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1
        )

        return frame
