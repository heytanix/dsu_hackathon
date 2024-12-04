# Real-Time Multi-Object Tracking using YOLOv5 and SORT with GUI

## Overview
This Python project demonstrates real-time multi-object tracking using YOLOv5 for object detection and the Simple Online and Realtime Tracking (SORT) algorithm. It also integrates a graphical user interface (GUI) built with Tkinter to display live video feed, detected objects, and real-time tracking logs.

## Key Features
1. **Object Detection with YOLOv5m**: Utilizes the medium version of YOLOv5 for better accuracy in detecting objects.
2. **Object Tracking with SORT**: Tracks detected objects across frames using a simplified Kalman filter-based tracker.
3. **GUI Integration**: Displays live video feed and logs detected object data in a user-friendly interface.
4. **Custom COCO Class Mapping**: Tracks specific object types like cars, motorbikes, buses, trucks, bicycles, and people.
5. **Data Logging**: Logs object IDs, types, durations, and timestamps to a file and displays them in the GUI.

---

## Installation
To run this project, ensure the following Python libraries are installed:

```bash
pip install opencv-python torch pandas requests Pillow filterpy
```

---

## Code Breakdown

### 1. Object Detection and Tracking
The core functionality includes:

- **YOLOv5 Model Loading**:
  ```python
  model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
  ```

- **SORT Algorithm**:
  A simple implementation of the SORT tracker, utilizing a Kalman filter for predicting object positions.

- **Object Tracking**:
  Objects are tracked across frames using their IDs. New objects are logged with appearance timestamps, and objects leaving the frame are logged with their durations.

### 2. Real-Time Tracking GUI
The GUI is built using Tkinter and provides the following:

- **Start/Stop Buttons**:
  Controls for initiating and terminating object tracking.

- **Video Display**:
  Displays the live video feed with bounding boxes and labels for detected objects.

- **Log Text Area**:
  Shows real-time logs of detected object data.

---

## Usage

1. **Run the Script**:
   ```bash
   python tracking_script.py
   ```

2. **Start Tracking**:
   Click the "Start Tracking" button in the GUI to begin object detection and tracking.

3. **View Logs**:
   Logs will appear in the text area of the GUI and in the `tracking_log.txt` file.

4. **Stop Tracking**:
   Click the "Stop Tracking" button to terminate the tracking process.

---

## GUI Components

### Video Feed
Displays the real-time video feed from the webcam with bounding boxes and labels for detected objects.

### Controls
- **Start Tracking**: Initiates the detection and tracking process.
- **Stop Tracking**: Ends the detection and tracking process.

### Log Area
Logs details of detected objects, including:
- **ID**: Unique identifier for the object.
- **Type**: Object label (e.g., Car, Person).
- **Duration**: Time the object was in the frame.
- **Timestamps**: First and last seen times.

---

## Future Improvements
- **Enhance Tracker Logic**: Replace simplistic ID generation with a more robust tracking mechanism.
- **Custom Class Training**: Train YOLOv5 on custom datasets for specific use cases.
- **Export Logs**: Add functionality to export logs in CSV or JSON format.
- **Performance Optimization**: Improve real-time processing efficiency for higher resolution video feeds.

---

## Sample Output

### Logs
Logs are saved in `tracking_log.txt` with the following format:
```
ID: 1234, Type: Car, Duration: 5.67 seconds, First seen: 2024-12-04 15:34:21, Last seen: 2024-12-04 15:34:26
```

### GUI
The GUI displays:
1. Video feed with bounding boxes and labels.
2. Logs in the text area below the video feed.

---

## Requirements
- **Hardware**: A webcam and a system with reasonable processing power (preferably with a GPU).
- **Software**: Python 3.7+, OpenCV, PyTorch, and Tkinter.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments
- **Ultralytics** for YOLOv5.
- **SORT Algorithm** by Alex Bewley.
