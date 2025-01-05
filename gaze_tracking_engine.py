import cv2
import mediapipe as mp
import numpy as np
import configparser


class GazeTrackingEngine:

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.head_orientation = None

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.horizontal_left_threshold = float(self.config['THRESHOLDS']['HORIZONTAL_LEFT'])
        self.horizontal_right_threshold = float(self.config['THRESHOLDS']['HORIZONTAL_RIGHT'])
        self.vertical_up_threshold = float(self.config['THRESHOLDS']['VERTICAL_UP'])
        self.vertical_down_threshold = float(self.config['THRESHOLDS']['VERTICAL_DOWN'])

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
        indices = [1, 33, 61, 199, 263, 291]
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
            return {"yaw": 0, "pitch": 0}

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        yaw = angles[1] * 3
        pitch = angles[0] * 3

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

            left_eye_closed = left_eye_height < 5
            right_eye_closed = right_eye_height < 5

            both_eyes_closed = left_eye_closed and right_eye_closed

            return both_eyes_closed

        return False, False, False

    def calculate_gaze(self):
        """
        Calculates numerical values for horizontal and vertical gaze.
        Returns:
            horizontal (float): Horizontal gaze value (-2 for no gaze, -1 for left, 0 for center, 1 for right).
            vertical (float): Vertical gaze value (-2 for no gaze, -1 for up, 0 for center, 1 for down).
        """
        if self._are_eyes_closed():
            return -2, -2

        horizontal_ratio, vertical_ratio = self._get_gaze_details()

        if horizontal_ratio is None or vertical_ratio is None:
            return None, None

        horizontal = (
            -1 if horizontal_ratio < self.horizontal_left_threshold
            else 1 if horizontal_ratio > self.horizontal_right_threshold
            else 0
        )

        vertical = (
            -1 if vertical_ratio < self.vertical_up_threshold
            else 1 if vertical_ratio > self.vertical_down_threshold
            else 0
        )

        return horizontal, vertical

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

        horizontal, vertical = self.calculate_gaze()
        if horizontal == None and vertical == None:
            gaze_text = "No Gaze Detected"
        elif horizontal == -2 and vertical == -2:
            gaze_text = "Eyes closed"
        elif horizontal == 0 and vertical == 0:
            gaze_text = "Looking Center"
        else:
            gaze_text = f"{'Left' if horizontal == -1 else 'Right' if horizontal == 1 else 'Center'} and {'Up' if vertical == -1 else 'Down' if vertical == 1 else 'Center'}"

        cv2.putText(frame, gaze_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        if self.head_orientation:
            yaw, pitch = self.head_orientation["yaw"], self.head_orientation["pitch"]
            orientation_text = f"Head: {'Left' if yaw < -0.04 else 'Right' if yaw > 0.04 else 'Center'}, "
            orientation_text += f"{'Up' if pitch > 0.04 else 'Down' if pitch < -0.04 else 'Center'}"
            cv2.putText(frame, orientation_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)

        return frame
