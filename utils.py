"""
Utility functions for Smart Home Face Recognition System
"""
import cv2
import numpy as np
import os
from datetime import datetime

def draw_fps(frame, fps, position=(10, 60)):
    """Draw FPS counter on frame"""
    cv2.putText(frame, f"FPS: {fps:.1f}", position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame

def draw_timestamp(frame, position=(10, 90)):
    """Draw timestamp on frame"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def get_face_box(frame, face_location, scale=1.0):
    """Extract face ROI from frame"""
    top, right, bottom, left = face_location
    
    if scale != 1.0:
        width = right - left
        height = bottom - top
        center_x = left + width // 2
        center_y = top + height // 2
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        left = max(0, center_x - new_width // 2)
        right = min(frame.shape[1], center_x + new_width // 2)
        top = max(0, center_y - new_height // 2)
        bottom = min(frame.shape[0], center_y + new_height // 2)
    
    return frame[top:bottom, left:right]

def save_detection_log(detection_data, log_file="detections.log"):
    """Save detection data to log file"""
    with open(log_file, "a") as f:
        timestamp = datetime.now().isoformat()
        for name, confidence in detection_data:
            f.write(f"{timestamp},{name},{confidence}\n")

def load_encodings_safe(encodings_file):
    """Safely load encodings with error handling"""
    if not os.path.exists(encodings_file):
        print(f"Error: File not found: {encodings_file}")
        return None, None
    
    try:
        import pickle
        with open(encodings_file, "rb") as f:
            data = pickle.load(f)
        return data.get("encodings", []), data.get("names", [])
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return None, None

def calculate_face_similarity(encoding1, encoding2):
    """Calculate similarity between two face encodings"""
    if encoding1 is None or encoding2 is None:
        return 0.0
    
    distance = np.linalg.norm(np.array(encoding1) - np.array(encoding2))
    similarity = max(0, 1 - distance)
    return similarity

def resize_frame_maintain_aspect(frame, target_width=None, target_height=None):
    """Resize frame maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    if target_width and not target_height:
        ratio = target_width / width
        new_dims = (target_width, int(height * ratio))
    elif target_height and not target_width:
        ratio = target_height / height
        new_dims = (int(width * ratio), target_height)
    elif target_width and target_height:
        ratio_w = target_width / width
        ratio_h = target_height / height
        ratio = min(ratio_w, ratio_h)
        new_dims = (int(width * ratio), int(height * ratio))
    else:
        return frame
    
    return cv2.resize(frame, new_dims, interpolation=cv2.INTER_AREA)

def add_overlay_text(frame, text, position=(10, 30), 
                     bg_color=(0, 0, 0), text_color=(255, 255, 255)):
    """Add text with background overlay"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness)
    
    x, y = position
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - text_height - 10), 
                  (x + text_width + 10, y + baseline), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    cv2.putText(frame, text, (x + 5, y), font, font_scale, text_color, thickness)
    return frame

def detect_motion(prev_frame, current_frame, threshold=25):
    """Detect motion between frames"""
    if prev_frame is None:
        return False
    
    diff = cv2.absdiff(prev_frame, current_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return len(contours) > 0

def get_camera_properties(video):
    """Get camera properties"""
    if not video.isOpened():
        return None
    
    return {
        'width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(video.get(cv2.CAP_PROP_FPS)),
        'brightness': int(video.get(cv2.CAP_PROP_BRIGHTNESS)),
        'contrast': int(video.get(cv2.CAP_PROP_CONTRAST)),
    }

def set_camera_properties(video, **kwargs):
    """Set camera properties"""
    property_map = {
        'width': cv2.CAP_PROP_FRAME_WIDTH,
        'height': cv2.CAP_PROP_FRAME_HEIGHT,
        'fps': cv2.CAP_PROP_FPS,
        'brightness': cv2.CAP_PROP_BRIGHTNESS,
        'contrast': cv2.CAP_PROP_CONTRAST,
        'saturation': cv2.CAP_PROP_SATURATION,
    }
    
    for name, value in kwargs.items():
        if name in property_map:
            video.set(property_map[name], value)
