import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
import time

# Load the pre-trained YOLOv5 large model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # 'yolov5l' for high accuracy

# COCO class names
class_names = model.names  # Get class names from the model

class SORT:
    def _init_(self):
        self.trackers = []
        self.next_id = 0

    def update(self, detections):
        updated_tracks = []
        for det in detections:
            # Initialize Kalman filter for each new detection
            tracker = KalmanFilter(dim_x=7, dim_z=4)
            tracker.x[:4] = np.array([det[0], det[1], det[2], det[3]]).reshape((4, 1))
            tracker.F = np.eye(7)  # State transition matrix
            tracker.H = np.eye(4, 7)  # Measurement function
            tracker.P *= 10.

            # Assign unique IDs and add to trackers list
            tracker.id = self.next_id
            self.next_id += 1
            updated_tracks.append(tracker)

        return updated_tracks

# Capture video from webcam (default camera index is 0)
cap = cv2.VideoCapture(0)  # Change to a video file path if needed

# Initialize SORT tracker
tracker = SORT()

# Variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Object detection using YOLOv5
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2, conf, class)

    # Filter detections for vehicles (car, truck, bus, etc.)
    vehicle_classes = [2, 3, 5, 7]  # COCO classes: 2=car, 3=motorbike, 5=bus, 7=truck
    vehicles = [d for d in detections if int(d[5]) in vehicle_classes and d[4] > 0.4]  # Confidence threshold

    # Update SORT tracker with vehicle detections
    tracks = tracker.update(vehicles)

    # Draw Bounding Boxes and Track IDs along with object names and FPS
    for track in tracks:
        x1, y1, x2, y2 = track.x[:4].flatten()
        
        # Get the confidence score and class name from the original detection (if available)
        confidence_score = next((d[4] for d in vehicles if d[0] == track.x[0]), None)
        class_id = next((int(d[5]) for d in vehicles if d[0] == track.x[0]), None)
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track.id}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if confidence_score is not None:
            cv2.putText(frame, f'Conf: {confidence_score:.2f}', (int(x1), int(y1) - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Confidence in red

        if class_id is not None:
            object_name = class_names[class_id]
            cv2.putText(frame, f'Obj: {object_name}', (int(x1), int(y1) - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)  # Object name in yellow

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time

    # Display FPS at top left corner in blue
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)  # FPS in blue

    # Display the frame with tracking and accuracy
    cv2.imshow('Real-Time Multi-Object Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
