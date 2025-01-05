import os
import csv
import time
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import configparser


class GazeMappingEngine:
    def __init__(self, csv_file="gaze_data.csv", interval=1.5):
        self.csv_file = csv_file
        self.interval = interval
        self.last_record_time = 0

        self.config = configparser.ConfigParser()
        self.config.read('config.ini')

        self.horizontal_left_threshold = float(self.config['THRESHOLDS']['HORIZONTAL_LEFT'])
        self.horizontal_right_threshold = float(self.config['THRESHOLDS']['HORIZONTAL_RIGHT'])
        self.vertical_up_threshold = float(self.config['THRESHOLDS']['VERTICAL_UP'])
        self.vertical_down_threshold = float(self.config['THRESHOLDS']['VERTICAL_DOWN'])

        if not os.path.exists(self.csv_file):
            self._initialize_csv()

    def _initialize_csv(self):
        """
        Initializes the CSV file with headers.
        """
        if os.path.exists(self.csv_file): os.remove(self.csv_file)
        with open(self.csv_file, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Timestamp", "X", "Y"])

    def record_gaze(self, horizontal, vertical):
        """
        Records gaze data (X, Y) into the CSV file every `interval` seconds.
        """
        try:
            current_time = time.time()
            if current_time - self.last_record_time >= self.interval:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.csv_file, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, horizontal, vertical])
                self.last_record_time = current_time
        except ValueError as e:
            print(f"Error recording gaze: {e}")

    def generate_heatmap(self):
        """
        Generates a heatmap from the gaze data stored in the CSV file.
        """
        try:
            # Load the CSV data
            df = pd.read_csv(self.csv_file)

            # Create a pivot table for the heatmap
            heatmap_data = pd.crosstab(index=df["Y"], columns=df["X"])

            # Reindex to ensure all combinations are present
            heatmap_data = heatmap_data.reindex(index=[1, 0, -1], columns=[-1, 0, 1], fill_value=0)

            # Create the heatmap plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt="d",
                cmap="coolwarm",
                xticklabels=["Left", "Center", "Right"],
                yticklabels=["Up", "Center", "Down"],
                cbar=True,
                square=True
            )
            plt.title("Gaze Heatmap")
            plt.xlabel("Horizontal Gaze")
            plt.ylabel("Vertical Gaze")
            plt.show()

        except Exception as e:
            print(f"Error generating heatmap: {e}")
