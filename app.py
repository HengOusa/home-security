import cv2
import pickle
import numpy as np
import os
from datetime import datetime

# Configuration
FACE_TOLERANCE = 0.6  # Lower = more strict matching (0.4-0.7 is typical)
CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to display
ENCODINGS_FILE = "smart-home-security/encodings/encodings.pkl"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load known faces
print("Loading known faces...")
if not os.path.exists(ENCODINGS_FILE):
    print(f"Error: Encodings file not found at {ENCODINGS_FILE}")
    print("Please run encode_faces.py first to generate encodings")
    exit(1)

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print(f"Loaded {len(known_names)} known faces: {known_names}")

# Start camera
print("Starting camera...")
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Get camera properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
print(f"Camera resolution: {frame_width}x{frame_height} @ {fps}fps")

# Create window
cv2.namedWindow("Face Recognition - Smart Home", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Face Recognition - Smart Home", 1280, 720)

def detect_faces_haar(frame):
    """Detect faces using OpenCV Haar Cascade"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces

def get_face_encoding_from_image(frame, face_location):
    """Extract simple face encoding using histogram comparison"""
    x, y, w, h = face_location
    face_roi = frame[y:y+h, x:x+w]
    
    # Resize to fixed size
    face_resized = cv2.resize(face_roi, (128, 128))
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Compute histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist)
    
    return hist.flatten()

def compare_faces_simple(face_encoding, known_encodings, tolerance=FACE_TOLERANCE):
    """Compare face encoding using histogram similarity"""
    results = []
    
    for known_encoding in known_encodings:
        # Use correlation comparison
        correlation = cv2.compareHist(
            face_encoding, 
            known_encoding, 
            cv2.HISTCMP_CORREL
        )
        
        if correlation > tolerance:
            confidence = correlation
        else:
            confidence = 0.0
        
        results.append(confidence)
    
    return results

def recognize_faces(face_locations, frame):
    """Recognize faces from detected locations"""
    results = []
    
    for (x, y, w, h) in face_locations:
        # Get face encoding
        face_encoding = get_face_encoding_from_image(frame, (x, y, w, h))
        
        # Compare with known faces
        similarities = compare_faces_simple(face_encoding, known_encodings)
        
        if len(similarities) > 0:
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            
            # Convert similarity to confidence
            confidence = min(1.0, best_similarity)
            
            if confidence >= CONFIDENCE_THRESHOLD:
                name = known_names[best_match_index]
            else:
                name = "Unknown"
        else:
            name = "Unknown"
            confidence = 0.0
        
        results.append({
            'name': name,
            'confidence': confidence,
            'location': (x, y, w, h)
        })
    
    return results

def draw_results(frame, recognition_results):
    """Draw recognition results on frame"""
    for result in recognition_results:
        x, y, w, h = result['location']
        name = result['name']
        confidence = result['confidence']
        
        # Color based on recognition status
        if name == "Unknown":
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green
        
        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Draw filled rectangle for text background
        cv2.rectangle(frame, (x, y - 35), (x + w, y), color, -1)
        
        # Draw name
        if confidence > 0:
            text = f"{name} ({confidence*100:.1f}%)"
        else:
            text = name
        
        cv2.putText(frame, text, (x + 5, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

# Main loop
print("\nStarting face recognition. Press 'q' to quit.")
print("Note: Using OpenCV Haar Cascade for face detection")
frame_count = 0
start_time = datetime.now()

try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("Error: Failed to read frame")
            break
        
        frame_count += 1
        
        # Detect faces using Haar Cascade
        face_locations = detect_faces_haar(frame)
        
        # Recognize faces
        recognition_results = recognize_faces(face_locations, frame)
        
        # Draw results
        frame = draw_results(frame, recognition_results)
        
        # Add info bar
        elapsed = (datetime.now() - start_time).total_seconds()
        fps = frame_count / elapsed if elapsed > 0 else 0
        info_text = f"Faces: {len(face_locations)} | FPS: {fps:.1f} | Known: {len(known_names)} | Mode: Haar Cascade"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow("Face Recognition - Smart Home", frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    # Cleanup
    video.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")
    print("Camera released. Goodbye!")
