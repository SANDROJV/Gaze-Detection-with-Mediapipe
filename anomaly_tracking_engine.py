import csv
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
    def __init__(self, csv_file="study_data.csv",
                 model_path="gaze_model.pth", train_epochs=20, batch_size=16):
        self.csv_file = csv_file
        self.model_path = model_path
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.status_text = "Driver's gaze is normal."
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model()
        self.model.to(self.device)
        self.center_threshold = 0.05
        self.drift_start_time = None
        self.alert_duration = 4

        # Load and preprocess gaze data
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

        # Prepare dataset and data loader
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
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()


        # Save the trained model
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

    def analyze_live_data(self, horizontal, vertical):
        """
        Analyze live gaze data for anomalies using the trained model.
        """
        with torch.no_grad():
            input_data = torch.tensor([[horizontal, vertical]], dtype=torch.float32).to(self.device)
            prediction = self.model(input_data).item()
            is_anomaly = prediction > 0.5

            current_time = time.time()

            if is_anomaly:
                if self.drift_start_time is None:
                    self.drift_start_time = current_time
                elif current_time - self.drift_start_time > self.alert_duration:
                    self.status_text = "Warning: Driver is not paying attention!"
            else:
                self.drift_start_time = None
                self.status_text = "Driver's gaze is normal."

    def draw_status(self, frame):
        """
        Draw the current status text on the given frame.
        """
        color = (0, 0, 255) if "Warning" in self.status_text else (255, 255, 255)
        font_scale = 0.7 if "Warning" in self.status_text else 0.5
        cv2.putText(frame, self.status_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
