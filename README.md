
# ISeeYou: Driver Monitoring System

## Overview
ISeeYou is an advanced driver monitoring system that enhances road safety by integrating multiple features, including:

- **Gaze Tracking**: Tracks driver gaze direction using Mediapipe.
- **Anomaly Detection**: Identifies deviations from normal gaze behavior.
- **Heart Rate Monitoring**: Monitors heart rate using Eulerian Video Magnification.
- **Road Direction Detection**: Detects lane directions in real-time using OpenCV.

## Features
1. **Real-Time Gaze Tracking**:
   - Tracks eye movements and head orientation.
   - Detects closed eyes and alerts the driver.

2. **Anomaly Detection**:
   - Uses a neural network trained on gaze data to detect unusual behavior.
   - Issues warnings when anomalies persist.

3. **Heart Rate Monitoring**:
   - Processes the forehead region to detect color intensity changes.
   - Calculates heart rate and issues alerts for irregularities.

4. **Road Direction Detection**:
   - Analyzes lane directions using edge detection and Hough Transform.
   - Provides directional guidance.

## Requirements
- **Programming Language**: Python 3.8+
- **Libraries**:
  - Mediapipe
  - OpenCV
  - PyTorch
  - NumPy
  - Seaborn
  - Matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/ISeeYou.git
   ```

2. Navigate to the project directory:
   ```bash
   cd ISeeYou
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main program:
   ```bash
   python ISeeYou.py
   ```

2. Ensure a webcam is connected for real-time monitoring.

## Project Structure
- **ISeeYou.py**: Main program orchestrating all components.
- **Engines/**:
  - `gaze_tracking_engine.py`: Handles gaze detection and head orientation.
  - `gaze_mapping_engine.py`: Records gaze data and generates heatmaps.
  - `heartrate_tracking_engine.py`: Monitors heart rate using EVM.
  - `road_tracking_engine.py`: Detects and annotates road directions.
  - `anomaly_tracking_engine.py`: Identifies anomalies in gaze behavior.
- **Data/**:
  - `study_data.csv`: Historical gaze data for model training.
  - `session_data.csv`: Real-time session data for heatmap generation.
  - `road_video.mp4`: Sample video for testing road tracking.

## References
- Mediapipe Face Mesh: [GitHub Documentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md)
- Gaze Tracking: [GitHub Repository](https://github.com/antoinelame/GazeTracking)
- Eulerian Video Magnification (EVM): [GitHub Repository](https://github.com/flyingzhao/PyEVM/tree/master)
- Road Lane Detection: [GeeksforGeeks Article](https://www.geeksforgeeks.org/opencv-real-time-road-lane-detection/)
