import cv2
import mediapipe as mp
import numpy as np


class GazeTrackingEngine:

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.head_orientation = None

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.mp_drawing = mp.solutions.drawing_utils

    def _analyze(self, frame):
        """
        Detects face, initializes eye regions, and draws landmarks using MediaPipe.
        """
        self.frame = frame
        results = self.mp_face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            # Draw landmarks on the frame
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=landmarks,
                landmark_drawing_spec=None,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(150, 150, 150), thickness=1, circle_radius=0
                ),
            )

            self._extract_eye_landmarks(landmarks, frame.shape[:2])
            self.head_orientation = self._calculate_head_orientation(landmarks, frame.shape[1], frame.shape[0])

    def _extract_eye_landmarks(self, landmarks, frame_shape):
        """
        Extracts landmarks for left and right eyes.
        """
        h, w = frame_shape

        def get_landmark_points(indices):
            return [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in indices]

        self.eye_left = get_landmark_points([33, 133, 468, 159, 145])
        self.eye_right = get_landmark_points([362, 263, 473, 386, 374])

    def _calculate_head_orientation(self, landmarks, w, h):
        """
        Calculates the head orientation (yaw, pitch) using specific landmarks.
        """
        indices = [1, 33, 61, 199, 263, 291]  # Indices for key landmarks
        face_3d = []
        face_2d = []

        for i in indices:
            lm = landmarks.landmark[i]
            x, y = int(lm.x * w), int(lm.y * h)
            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])

        if len(face_3d) < 4 or len(face_2d) < 4:
            return {"yaw": 0, "pitch": 0}

        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        focal_length = w
        cam_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

        if not success:
            return {"yaw": 0, "pitch": 0}  # Default orientation if solvePnP fails

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        yaw = angles[1] * 3  # Scale factor
        pitch = angles[0] * 3  # Scale factor

        return {"yaw": yaw, "pitch": pitch}

    def _calculate_ratios(self, eye):
        """
        Calculate horizontal and vertical ratios for a given eye.
        """
        epsilon = 1e-6  # Small value to avoid division by zero
        horizontal_ratio = (eye[2][0] - eye[0][0]) / (eye[1][0] - eye[0][0] + epsilon)
        vertical_ratio = (eye[2][1] - eye[3][1]) / (eye[4][1] - eye[3][1] + epsilon)
        return horizontal_ratio, vertical_ratio

    def _are_eyes_closed(self):
        """
        Detects if the eyes are blinking based on the height of the eyelids.
        """
        if self.eye_left and self.eye_right:
            left_eye_height = self.eye_left[4][1] - self.eye_left[3][1]
            right_eye_height = self.eye_right[4][1] - self.eye_right[3][1]
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            return avg_eye_height < 5
        return False

    def calculate_gaze(self):
        """
        Determines the gaze direction based on eye and head orientation.
        """
        if self._are_eyes_closed():
            return "Eyes Closed"

        horizontal_ratio, vertical_ratio = self._get_gaze_details()

        if horizontal_ratio is None or vertical_ratio is None:
            return "No Gaze Detected"

        if vertical_ratio < 0.35:
            return "Looking Up"
        elif vertical_ratio > 0.65:
            return "Looking Down"

        if horizontal_ratio < 0.35:
            return "Looking Left"
        elif horizontal_ratio > 0.65:
            return "Looking Right"
        else:
            return "Looking Center"

    def _get_gaze_details(self):
        """
        Determines horizontal and vertical gaze details.
        """
        if self.eye_left and self.eye_right:
            left_horizontal, left_vertical = self._calculate_ratios(self.eye_left)
            right_horizontal, right_vertical = self._calculate_ratios(self.eye_right)

            avg_horizontal = (left_horizontal + right_horizontal) / 2
            avg_vertical = (left_vertical + right_vertical) / 2

            return avg_horizontal, avg_vertical
        return None, None

    def annotated_frame(self):
        """
        Overlays gaze direction, blinking state, and head orientation on the frame.
        """
        frame = self.frame.copy()

        for eye in (self.eye_left, self.eye_right):
            if eye:
                for point in eye:
                    cv2.circle(frame, point, 2, (0, 255, 0), -1)

        gaze_text = self.calculate_gaze()
        cv2.putText(frame, gaze_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        if self.head_orientation:
            yaw, pitch = self.head_orientation["yaw"], self.head_orientation["pitch"]
            orientation_text = f"Head: {'Left' if yaw < -0.04 else 'Right' if yaw > 0.04 else 'Center'}, "
            orientation_text += f"{'Up' if pitch > 0.04 else 'Down' if pitch < -0.04 else 'Center'}"
            cv2.putText(frame, orientation_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        return frame
