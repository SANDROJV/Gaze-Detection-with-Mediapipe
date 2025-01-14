import cv2
import numpy as np
from scipy.signal import butter, lfilter
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
        self.freq_band = (0.8, 3.0)
        self.amplification = 15
        self.signal_buffer = []
        self.buffer_size = self.fps * 5  # 5 seconds of data
        self.heart_rate = 0
        self.change_threshold = 0.15
        self.warning_duration = 3
        self.previous_heart_rate = 0
        self.last_change_time = None
        self.warning_active = False

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        return lfilter(b, a, data, axis=0)

    def process_forehead(self, forehead_region):
        """
        Process the forehead region: smooth, resize, amplify changes.
        """
        if forehead_region is None or len(forehead_region.shape) != 3:
            return None

        # Apply Gaussian blur to reduce noise
        blurred_forehead = cv2.GaussianBlur(forehead_region, (5, 5), 0)

        # Resize for consistency
        fixed_size = (64, 64)
        resized_forehead = cv2.resize(blurred_forehead, fixed_size, interpolation=cv2.INTER_LINEAR)

        # Magnify changes
        amplified_forehead = (resized_forehead - np.mean(resized_forehead)) * self.amplification
        magnified_forehead = np.clip(amplified_forehead + resized_forehead, 0, 255).astype(np.uint8)

        return magnified_forehead

    def calculate_heart_rate(self, magnified_forehead):
        """
        Calculate heart rate based on magnified forehead data.
        """
        if magnified_forehead is None or len(magnified_forehead.shape) != 3:
            return self.heart_rate

        # Compute mean red channel intensity over the forehead
        avg_red_channel = np.mean(magnified_forehead[:, :, 2])
        self.signal_buffer.append(avg_red_channel)

        # Maintain buffer size
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

        # Skip calculation if buffer isn't full
        if len(self.signal_buffer) < self.buffer_size:
            return self.heart_rate

        # Bandpass filter the signal
        filtered_signal = self._butter_bandpass_filter(
            np.array(self.signal_buffer), self.freq_band[0], self.freq_band[1], self.fps
        )

        # Perform FFT to extract the dominant frequency
        fft_result = np.fft.fft(filtered_signal)
        freqs = np.fft.fftfreq(len(fft_result), d=1.0 / self.fps)
        valid_idx = (freqs >= self.freq_band[0]) & (freqs <= self.freq_band[1])

        if valid_idx.any():
            peak_frequency = freqs[valid_idx][np.argmax(np.abs(fft_result[valid_idx]))]
            if self.freq_band[0] <= peak_frequency <= self.freq_band[1]:
                self.heart_rate = int(peak_frequency * 60)  # Convert Hz to BPM

        self._check_heart_rate_stability()

        return self.heart_rate

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

    def annotate_frame(self, frame):
        """
        Annotate the frame with the current heart rate and warnings if active.
        """
        cv2.putText(
            frame,
            f"Heart Rate: {self.heart_rate} BPM",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        if self.warning_active:
            cv2.putText(
                frame,
                "WARNING: Unsteady heart rate! Pull over!",
                (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                1
            )

        return frame
