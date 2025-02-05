import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from collections import deque
import time


class HeartRateTrackingEngine:
    def __init__(self):
        """
        Initialize HeartRateTrackingEngine.
        - fps: Frames per second of the input video.
        - freq_band: Tuple indicating the bandpass filter range in Hz.
        - amplification: Factor to amplify color changes.
        - change_threshold: Threshold for detecting rapid heart rate changes (as a fraction of the current heart rate).
        - warning_duration: Duration (in seconds) for sustained change before triggering a warning.
        """
        self.fps = 30
        self.buffer_size = self.fps * 5  # 5 seconds of data
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.heart_rates = deque(maxlen=5)  # Store the last 5 heart rates
        self.heart_rate = 0
        self.change_threshold = 0.15
        self.warning_duration = 3
        self.previous_heart_rate = 0
        self.last_change_time = None
        self.warning_active = False
        self.frame_shape = None
        self.amplification = 15

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data, axis=0)

    def build_gaussian_pyramid(self, image, levels=3, kernel=(5, 5)):
        pyramid = [image]
        for _ in range(levels - 1):
            image = cv2.GaussianBlur(image, kernel, 0)
            image = cv2.pyrDown(image)
            pyramid.append(image)
        return pyramid

    def collapse_gaussian_pyramid(self, pyramid):
        image = pyramid[-1]
        for level in reversed(pyramid[:-1]):
            image = cv2.pyrUp(image, dstsize=(level.shape[1], level.shape[0]))
            image = cv2.add(image, level)
        return image

    def process_forehead(self, forehead_region, full_face):
        """
        Process the forehead region: smooth, resize, amplify changes.
        Fallback to full face if forehead region is not valid.
        """

        def _process_region(frame):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pyramid = self.build_gaussian_pyramid(gray, levels=3)

            if self.frame_shape is None:
                self.frame_shape = pyramid[-1].shape

            resized_frame = cv2.resize(pyramid[-1], (self.frame_shape[1], self.frame_shape[0]))
            self.frame_buffer.append(resized_frame)

            if len(self.frame_buffer) == self.buffer_size:
                stacked_frames = np.stack(self.frame_buffer, axis=0)
                filtered_signal = self._butter_bandpass_filter(
                    stacked_frames, 0.8, 3.0, self.fps
                )
                collapsed_frame = self.collapse_gaussian_pyramid([filtered_signal])

                heart_rate = self.calculate_heart_rate(collapsed_frame)
                if heart_rate is not None:
                    self.heart_rates.append(round(heart_rate))

                    if len(self.heart_rates) == 5:
                        self.heart_rate = round(np.mean(self.heart_rates))
                        self.heart_rates.clear()

                    self._check_heart_rate_stability()
                return self.heart_rate
            return None

        # Check forehead validity
        if forehead_region is not None and isinstance(forehead_region, np.ndarray) and forehead_region.shape[0] > 0:
            return _process_region(forehead_region)

        # Fallback to full face
        if full_face is not None and isinstance(full_face, np.ndarray) and full_face.shape[0] > 0:
            return _process_region(full_face)

        # No valid region to process
        print("No valid region provided for processing.")
        return None

    def calculate_heart_rate(self, signal):
        if signal is None or not isinstance(signal, np.ndarray) or signal.size == 0:
            return None
        try:
            mean_signal = np.mean(signal, axis=0)
            if mean_signal.ndim != 1:
                mean_signal = mean_signal.flatten()
            if mean_signal.size == 0:
                return None

            peaks, _ = find_peaks(mean_signal, distance=self.fps / 2)

            if len(peaks) > 1:
                peak_intervals = np.diff(peaks) / self.fps
                avg_interval = np.mean(peak_intervals)
                if avg_interval == 0:
                    return None
                bpm = 60 / avg_interval
                return bpm
        except Exception as e:
            print(f"Error calculating heart rate: {e}")
        return None

    def _check_heart_rate_stability(self):
        """
        Check for rapid changes in heart rate and activate warnings if needed.
        """
        if not self.previous_heart_rate:
            self.previous_heart_rate = self.heart_rate
            self.warning_active = False
            return

        change_ratio = abs(self.heart_rate - self.previous_heart_rate) / self.previous_heart_rate
        current_time = time.time()

        if change_ratio > self.change_threshold:
            if self.last_change_time and current_time - self.last_change_time > self.warning_duration:
                self.warning_active = True
            else:
                self.last_change_time = current_time
        else:
            self.last_change_time = None
            self.warning_active = False

        self.previous_heart_rate = self.heart_rate

    def annotate_frame(self, frame, x, y):
        cv2.putText(
            frame,
            f"Heart Rate: {self.heart_rate} BPM",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        if self.warning_active:
            cv2.putText(
                frame,
                "WARNING: Unsteady heart rate! Pull over!",
                (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2
            )

        return frame
