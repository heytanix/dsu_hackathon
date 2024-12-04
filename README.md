# Real-Time Multi-Object Tracking System with GUI
Overview
This project integrates a real-time multi-object tracking system using YOLOv5 with a Tkinter GUI. It captures live video from the webcam, performs object detection and tracking, and displays the results along with logs in a graphical interface.

## Key Features
Start and Stop Buttons: Control the tracking process.
Live Video Display: Shows real-time video with bounding boxes around detected objects.
Logging Section: Provides real-time logs of detected objects and their appearance times.
Dependencies
Ensure you have the following Python libraries installed:

pip install torch torchvision opencv-python pillow tkinter
Code Explanation

## Imports and Setup
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import torch
import time
Tkinter: Provides GUI components.
Pillow (PIL): Handles image conversion for display in Tkinter.
OpenCV: Captures video from the webcam and processes frames.
PyTorch: Loads and runs the YOLOv5 model.
Threading: Allows video processing to run concurrently with the GUI.

## Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
YOLOv5m: Medium-sized YOLOv5 model for balanced performance and speed.

## GUI Class Definition
class TrackingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Multi-Object Tracking")
        self.root.geometry("900x700")
        ...
TrackingApp class: Defines the main GUI and functionality.
Window Setup: Sets title and size.
Video Source: Captures video from the default webcam.

## GUI Components
self.video_frame = ttk.Label(root)
self.video_frame.pack(pady=20)

self.start_button = ttk.Button(root, text="Start Tracking", command=self.start_tracking)
self.start_button.pack(side=tk.LEFT, padx=20)

self.stop_button = ttk.Button(root, text="Stop Tracking", command=self.stop_tracking)
self.stop_button.pack(side=tk.RIGHT, padx=20)

self.log_text = tk.Text(root, height=10, width=100)
self.log_text.pack(pady=10)
video_frame: Displays the live video feed.
start_button and stop_button: Control the tracking process.
log_text: Logs detected objects and their IDs.

## Start and Stop Functions
def start_tracking(self):
    if not self.is_running:
        self.is_running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

def stop_tracking(self):
    self.is_running = False
    self.cap.release()
    cv2.destroyAllWindows()
start_tracking: Initiates a separate thread for video processing.
stop_tracking: Stops the tracking process and releases resources.

## Video Processing and Detection
def process_video(self):
    while self.is_running:
        ret, frame = self.cap.read()
        if not ret:
            break
        
        results = model(frame)  # Object detection using YOLOv5
        detections = results.xyxy[0].cpu().numpy()
        ...
        
        # Convert and display the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)
        self.video_frame.img_tk = img_tk
        self.video_frame.config(image=img_tk)
        self.root.update_idletasks()
Frame Capture: Reads frames from the webcam.
Object Detection: Runs YOLOv5 on each frame and retrieves detection results.
Frame Display: Converts the frame to RGB and displays it in the GUI.

## Logging Mechanism
def log_data(self, message):
    self.log_text.insert(tk.END, message + '\n')
    self.log_text.see(tk.END)
Logs detected objects and their IDs in the Text widget.
Customization Options
Class Labels: Modify the class_labels dictionary to track different object classes.
Confidence Threshold: Adjust the conf > 0.4 threshold for detection sensitivity.
Video Source: Change self.cap = cv2.VideoCapture(0) to use a different video source or file.
Running the Application
Ensure your webcam is connected.

## Run the script:
python tracking_app.py
Use the Start and Stop buttons to control tracking.
