from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter
import time
import base64
import io
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # 'yolov5l' for high accuracy

class SORT:
    def __init__(self):
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

# Initialize SORT tracker
tracker = SORT()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Function to generate video frames and send over SocketIO
def generate_frames():
    frame_count = 0
    start_time = time.time()
    fps = 0
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
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track.id}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Calculate FPS
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            fps = frame_count / elapsed_time

        # Convert frame to base64 for sending over SocketIO
        _, buffer = cv2.imencode('.jpg', frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')

        # Emit the frame along with FPS and tracked objects to the frontend
        socketio.emit('video_frame', {
            'frame': frame_data,
            'fps': fps,
            'tracks': [{'id': track.id, 'bbox': track.x[:4].flatten().tolist()} for track in tracks]
        })

        time.sleep(0.1)  # Simulate 10fps

@app.route('/')
def index():
    return render_template('index.html')

# Start the video frame generation when SocketIO connection is made
@socketio.on('connect')
def handle_connect():
    print("Client connected.")
    socketio.start_background_task(generate_frames)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
