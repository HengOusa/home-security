"""
Face Encoding Generator for Smart Home Security System
Uses OpenCV for face detection and histogram-based encoding.
"""
import cv2
import os
import pickle
import numpy as np
from datetime import datetime

# Configuration
KNOWN_FACES_DIR = "d:/Laragon/www/all_project/smart-home-security/known_faces"
ENCODINGS_FILE = "smart-home-security/encodings/encodings.pkl"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_histogram(image_path):
    """Extract histogram features from face in an image"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"  Error: Could not load image {image_path}")
            return None, False
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print(f"  Warning: No face found in {os.path.basename(image_path)}")
            return None, False
        
        # Use the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        
        # Extract face ROI
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize to fixed size
        face_resized = cv2.resize(face_roi, (128, 128))
        
        # Compute histogram
        hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        
        return hist.flatten(), True
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None, False

def encode_all_faces():
    """Encode all faces in the known_faces directory"""
    print("Starting face encoding process...")
    print(f"Known faces directory: {KNOWN_FACES_DIR}")
    print(f"Output file: {ENCODINGS_FILE}")
    print("-" * 50)
    
    # Check if directory exists
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory not found: {KNOWN_FACES_DIR}")
        return False
    
    # Get all image files
    image_files = [f for f in os.listdir(KNOWN_FACES_DIR) 
                if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    if not image_files:
        print("No image files found in the directory")
        return False
    
    print(f"Found {len(image_files)} image(s)")
    print()
    
    known_encodings = []
    known_names = []
    success_count = 0
    
    for filename in sorted(image_files):
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"Processing: {filename}...")
        
        encoding, success = get_face_histogram(filepath)
        
        if success:
            known_encodings.append(encoding)
            # Use filename without extension as name
            name = os.path.splitext(filename)[0]
            known_names.append(name)
            print(f"  ✓ Encoded: {name}")
            success_count += 1
        else:
            print(f"  ✗ Failed to encode face")
    
    print()
    print("-" * 50)
    print(f"Successfully encoded {success_count}/{len(image_files)} faces")
    
    if success_count == 0:
        print("Error: No faces could be encoded")
        return False
    
    # Save to pickle file
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "created_at": datetime.now().isoformat(),
        "count": success_count
    }
    
    # Create directory if needed
    os.makedirs(os.path.dirname(ENCODINGS_FILE) if os.path.dirname(ENCODINGS_FILE) else ".", exist_ok=True)
    
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Encodings saved to: {ENCODINGS_FILE}")
    print("Encoding complete!")
    return True

def add_new_faces():
    """Add new faces to existing encodings"""
    if not os.path.exists(ENCODINGS_FILE):
        print("No existing encodings file. Creating new one...")
        return encode_all_faces()
    
    # Load existing encodings
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    
    existing_names = set(data["names"])
    known_encodings = data["encodings"]
    known_names = data["names"]
    
    print(f"Loaded {len(known_names)} existing faces")
    print()
    
    # Get all image files
    image_files = [f for f in os.listdir(KNOWN_FACES_DIR) 
                if f.lower().endswith(IMAGE_EXTENSIONS)]
    
    new_count = 0
    for filename in sorted(image_files):
        name = os.path.splitext(filename)[0]
        
        # Skip if already exists
        if name in existing_names:
            print(f"Skipping (already exists): {filename}")
            continue
        
        filepath = os.path.join(KNOWN_FACES_DIR, filename)
        print(f"Processing: {filename}...")
        
        encoding, success = get_face_histogram(filepath)
        
        if success:
            known_encodings.append(encoding)
            known_names.append(name)
            print(f"  ✓ Added: {name}")
            new_count += 1
    
    if new_count == 0:
        print("No new faces to add")
        return True
    
    # Save updated encodings
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "updated_at": datetime.now().isoformat(),
        "count": len(known_names)
    }
    
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Added {new_count} new face(s)")
    print(f"Total faces: {len(known_names)}")
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--add":
        add_new_faces()
    else:
        encode_all_faces()
