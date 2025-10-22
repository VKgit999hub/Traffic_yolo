import os
import sys
import argparse
import glob
import time
import math
from collections import OrderedDict

import cv2
import numpy as np
from ultralytics import YOLO

class SpeedTracker:
    def __init__(self, max_disappeared=30, max_distance=100, pixels_per_meter=10):
        """
        Initialize the speed tracker
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for matching detections to existing tracks
            pixels_per_meter: Conversion factor from pixels to meters (calibrate based on your setup)
        """
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.pixels_per_meter = pixels_per_meter
        
        # Speed calculation parameters
        self.position_history = OrderedDict()  # Store position history for speed calculation
        self.speed_history = OrderedDict()     # Store speed history for smoothing
        self.max_history_length = 10          # Number of frames to keep in history
        
    def register(self, centroid, timestamp):
        """Register a new object with its centroid and timestamp"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.position_history[self.next_object_id] = [(centroid, timestamp)]
        self.speed_history[self.next_object_id] = []
        self.next_object_id += 1
        
    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.position_history:
            del self.position_history[object_id]
        if object_id in self.speed_history:
            del self.speed_history[object_id]
            
    def calculate_speed(self, object_id, new_centroid, timestamp):
        """Calculate speed for an object based on position history"""
        if object_id not in self.position_history:
            return 0.0
            
        history = self.position_history[object_id]
        
        # Add new position to history
        history.append((new_centroid, timestamp))
        
        # Keep only recent history
        if len(history) > self.max_history_length:
            history.pop(0)
            
        # Calculate speed if we have at least 2 points
        if len(history) < 2:
            return 0.0
            
        # Use the last few points for speed calculation
        recent_points = history[-5:] if len(history) >= 5 else history
        
        if len(recent_points) < 2:
            return 0.0
            
        # Calculate speed using linear regression or simple distance/time
        speeds = []
        for i in range(1, len(recent_points)):
            prev_pos, prev_time = recent_points[i-1]
            curr_pos, curr_time = recent_points[i]
            
            # Calculate distance in pixels
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            distance_pixels = math.sqrt(dx*dx + dy*dy)
            
            # Convert to meters
            distance_meters = distance_pixels / self.pixels_per_meter
            
            # Calculate time difference
            time_diff = curr_time - prev_time
            
            if time_diff > 0:
                speed_mps = distance_meters / time_diff  # meters per second
                speeds.append(speed_mps)
        
        if not speeds:
            return 0.0
            
        # Average the speeds and convert to km/h
        avg_speed_mps = np.mean(speeds)
        speed_kmh = avg_speed_mps * 3.6
        
        # Smooth the speed using history
        if object_id not in self.speed_history:
            self.speed_history[object_id] = []
            
        self.speed_history[object_id].append(speed_kmh)
        
        # Keep only recent speed history
        if len(self.speed_history[object_id]) > 5:
            self.speed_history[object_id].pop(0)
            
        # Return smoothed speed
        return np.mean(self.speed_history[object_id])
        
    def update(self, detections, timestamp):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection centroids [(x, y), ...]
            timestamp: Current timestamp
            
        Returns:
            Dictionary of {object_id: (centroid, speed)}
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            return {}
            
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection, timestamp)
        else:
            # Match detections to existing objects
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            # Calculate distance matrix
            distances = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - 
                                     np.array(detections), axis=2)
            
            # Find the minimum distance matches
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            # Update existing objects
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                if distances[row, col] > self.max_distance:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = detections[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
                
            # Handle unmatched detections and objects
            unused_row_indices = set(range(0, distances.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, distances.shape[1])).difference(used_col_indices)
            
            # If more objects than detections, mark objects as disappeared
            if distances.shape[0] >= distances.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
                        
            # If more detections than objects, register new objects
            else:
                for col in unused_col_indices:
                    self.register(detections[col], timestamp)
        
        # Calculate speeds and return results
        results = {}
        for object_id, centroid in self.objects.items():
            speed = self.calculate_speed(object_id, centroid, timestamp)
            results[object_id] = (centroid, speed)
            
        return results

# Define and parse user input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')
parser.add_argument('--pixels_per_meter', help='Pixels per meter conversion factor for speed calculation (default: 10)',
                    type=float, default=10.0)

args = parser.parse_args()

# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record
pixels_per_meter = args.pixels_per_meter

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Initialize speed tracker
tracker = SpeedTracker(max_disappeared=30, max_distance=100, pixels_per_meter=pixels_per_meter)

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    print(img_source)
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':
    if source_type == 'video': 
        cap_arg = img_source
    elif source_type == 'usb': 
        cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Begin inference loop
while True:
    t_start = time.perf_counter()
    current_timestamp = time.time()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder':
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb':
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Extract centroids from detections for tracking
    detection_centroids = []
    detection_info = []  # Store full detection info
    
    for i in range(len(detections)):
        # Get bounding box coordinates
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        # Get confidence and class
        conf = detections[i].conf.item()
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]

        # Only process detections above threshold
        if conf > min_thresh:
            # Calculate centroid
            centroid_x = int((xmin + xmax) / 2)
            centroid_y = int((ymin + ymax) / 2)
            centroid = (centroid_x, centroid_y)
            
            detection_centroids.append(centroid)
            detection_info.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'confidence': conf,
                'class_idx': classidx,
                'class_name': classname,
                'centroid': centroid
            })

    # Update tracker with current detections
    tracked_objects = tracker.update(detection_centroids, current_timestamp)

    # Draw tracking results
    object_count = 0
    for object_id, (centroid, speed) in tracked_objects.items():
        # Find corresponding detection info
        detection = None
        min_distance = float('inf')
        
        for det_info in detection_info:
            dist = math.sqrt((det_info['centroid'][0] - centroid[0])**2 + 
                           (det_info['centroid'][1] - centroid[1])**2)
            if dist < min_distance:
                min_distance = dist
                detection = det_info
        
        if detection:
            xmin, ymin, xmax, ymax = detection['bbox']
            classname = detection['class_name']
            conf = detection['confidence']
            classidx = detection['class_idx']
            
            # Draw bounding box
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            # Draw centroid
            cv2.circle(frame, centroid, 4, color, -1)
            
            # Create label with ID, class, confidence, and speed
            speed_text = f"{speed:.1f} km/h" if speed > 0.1 else "0.0 km/h"
            label = f'ID:{object_id} {classname}: {int(conf*100)}% | {speed_text}'
            
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            
            # Draw label background
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                         (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            
            # Draw label text
            cv2.putText(frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            object_count += 1

    # Calculate and draw framerate (if using video, USB, or Picamera source)
    if source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    
    # Display detection results
    cv2.putText(frame, f'Tracked objects: {object_count}', (10, 45), 
               cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.putText(frame, f'Conversion: {pixels_per_meter:.1f} pixels/meter', (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
    
    cv2.imshow('YOLO Detection with Speed Tracking', frame)
    if record: 
        recorder.write(frame)

    # Handle key presses
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'):
        cv2.imwrite('capture.png', frame)
    
    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS
    avg_frame_rate = np.mean(frame_rate_buffer)

# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: 
    recorder.release()
cv2.destroyAllWindows()