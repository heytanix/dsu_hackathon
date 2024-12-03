import cv2
import numpy as np
import torch
from filterpy.kalman import KalmanFilter
import time
import threading
import base64
import os
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")  # Allow all origins for SocketIO

# Ensure upload directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained YOLOv5 large model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # 'yolov5l' for high accuracy

# COCO class names
class_names = model.names  # Get class names from the model

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

# Global variables for video streaming
global_frame = None
global_fps = 0
global_tracks = []
current_video_path = None

def process_video():
    global global_frame, global_fps, global_tracks, current_video_path
    
    # Wait for a video to be uploaded
    while current_video_path is None:
        time.sleep(1)
    
    # Capture video from uploaded file
    cap = cv2.VideoCapture(current_video_path)

    # Initialize SORT tracker
    tracker = SORT()

    # Variables for FPS calculation
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

        # Draw Bounding Boxes and Track IDs
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

        # Encode frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')

        # Broadcast frame and track information via SocketIO
        socketio.emit('video_frame', {
            'frame': f'data:image/jpeg;base64,{frame_base64}',
            'fps': fps,
            'tracks': [
                {
                    'id': track.id,
                    'bbox': {
                        'x1': int(track.x[0]),
                        'y1': int(track.x[1]),
                        'x2': int(track.x[2]),
                        'y2': int(track.x[3])
                    }
                } for track in tracks
            ]
        })

        time.sleep(0.01)  # Small delay to reduce CPU usage

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_video_path
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'image' not in request.files:
            return 'No file uploaded', 400
        
        file = request.files['image']
        
        # If no file is selected
        if file.filename == '':
            return 'No selected file', 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Update current video path for processing
        current_video_path = filepath
        
        # Restart video processing thread
        video_thread = threading.Thread(target=process_video)
        video_thread.daemon = True
        video_thread.start()
        
        return render_template('index.html', filename=filename)
    
    return render_template('index.html')

if __name__ == '__main__':
    # Get local IP address
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"Server running on http://{local_ip}:5000")
    
    # Run SocketIO server on local network
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)