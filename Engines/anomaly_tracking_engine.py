import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import time


class GazeDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index, :2]
        y = self.data[index, 2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class AnomalyTrackingEngine:
    def __init__(self):
        self.csv_file = "./Engines/Data/study_data.csv"
        self.model_path = "./Engines/Data/gaze_model.pth"
        self.train_epochs = 20
        self.batch_size = 16
        self.status_text = "Driver's gaze is normal."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.model.to(self.device)
        self.center_threshold = 0.05
        self.drift_start_time = None
        self.alert_duration = 2
        self.gaze_data = self.load_gaze_data()
        self.train_model()

    def build_model(self):
        """
        Build a simple neural network for gaze anomaly detection.
        """
        return nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def load_gaze_data(self):
        """
        Load gaze data from the study CSV file and assign labels (normal: 0, anomaly: 1).
        """
        try:
            if os.path.exists(self.csv_file):
                df = pd.read_csv(self.csv_file)
                df["Label"] = np.where((df["X"].abs() > 0.1) | (df["Y"].abs() > 0.1), 1, 0)
                return df[["X", "Y", "Label"]].to_numpy()
            else:
                print(f"Study CSV file {self.csv_file} does not exist.")
                return np.array([])
        except Exception as e:
            print(f"Error loading study data: {e}")
            return np.array([])

    def train_model(self):
        """
        Train the anomaly detection model on the gaze data.
        """
        if len(self.gaze_data) == 0:
            return

        dataset = GazeDataset(self.gaze_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        self.model.train()
        for epoch in range(self.train_epochs):
            epoch_loss = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x).squeeze()
                y = y.view_as(y_pred)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        """
        Load the trained model from file.
        """
        try:
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        except Exception as e:
            print(f"Error loading model: {e}")

    def analyze_live_data(self, horizontal, vertical, road_direction, head_direction):
        """
        Analyze live gaze data for anomalies using the trained model, considering road direction.
        """
        with torch.no_grad():
            input_data = torch.tensor([[horizontal, vertical]], dtype=torch.float32).to(self.device)
            prediction = self.model(input_data).item()
            is_anomaly = prediction > 0.5

            # Check gaze direction against road direction
            road_match = (
                    (road_direction == "Forward" and horizontal == 0) or
                    (road_direction == "Left" and horizontal == -1) or
                    (road_direction == "Right" and horizontal == 1)
            )

            # Check head direction against road direction
            head_match = (
                    (head_direction == "Center" and road_direction == "Forward") or
                    (head_direction == "Left" and road_direction == "Left") or
                    (head_direction == "Right" and road_direction == "Right")
            )

            current_time = time.time()

            if is_anomaly and not road_match and not head_match:
                if self.drift_start_time is None:
                    self.drift_start_time = current_time
                elif current_time - self.drift_start_time > self.alert_duration:
                    self.status_text = "Warning: Driver is not paying attention!"
            else:
                self.drift_start_time = None
                self.status_text = "Driver's gaze is normal."

    def annotate_frame(self, frame, x, y):
        """
        Draw the current status text on the given frame.
        """
        color = (0, 0, 255) if "Warning" in self.status_text else (255, 255, 255)
        font_scale = 0.55 if "Warning" in self.status_text else 0.5
        thickness = 2 if "Warning" in self.status_text else 1
        cv2.putText(frame, self.status_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
