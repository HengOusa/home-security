"""
Face Collection Script for Smart Home Security System
This script captures faces from webcam or adds faces from images.
"""
import cv2
import os
import pickle
from datetime import datetime

# Configuration
ENCODINGS_FILE = "smart-home-security/encodings/encodings.pkl"
CAPTURE_DIR = "smart-home-security/known_faces"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_face_histogram(image_path):
    """Extract histogram features from face in an image"""
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
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
        print(f"Error: {e}")
        return None, False

def capture_face_from_webcam(name):
    """Capture a face from the webcam"""
    print("\nCapturing face from webcam...")
    print("Look at the camera and press 'c' to capture, 'q' to cancel")
    
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("Error: Could not open camera")
        return None
    
    cv2.namedWindow("Capture Face")
    
    captured_face = None
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Detect face
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        # Draw rectangle around face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.putText(frame, f"Name: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'c' to capture, 'q' to cancel", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Capture Face", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and len(faces) > 0:
            # Capture the face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (128, 128))
            hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
            cv2.normalize(hist, hist)
            captured_face = hist.flatten()
            break
        elif key == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()
    
    return captured_face

def add_face_to_encodings(name, encoding):
    """Add a new face to the encodings file"""
    # Create directory if needed
    os.makedirs(os.path.dirname(ENCODINGS_FILE) if os.path.dirname(ENCODINGS_FILE) else ".", exist_ok=True)
    
    # Load existing or create new
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    else:
        known_encodings = []
        known_names = []
    
    # Add new face
    known_encodings.append(encoding)
    known_names.append(name)
    
    # Save
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "updated_at": datetime.now().isoformat(),
        "count": len(known_names)
    }
    
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    
    print(f"✓ Added '{name}' to the system ({len(known_names)} known faces now)")

def add_face_from_image(image_path, name=None):
    """Add a face from an image file"""
    if name is None:
        name = os.path.splitext(os.path.basename(image_path))[0]
    
    encoding, success = get_face_histogram(image_path)
    
    if success:
        add_face_to_encodings(name, encoding)
        
        # Also save a copy of the image
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        dest_path = os.path.join(CAPTURE_DIR, f"{name}.jpg")
        import shutil
        shutil.copy(image_path, dest_path)
        print(f"  Saved image to: {dest_path}")
    else:
        print(f"✗ Could not detect face in {image_path}")

def list_known_faces():
    """List all known faces"""
    if not os.path.exists(ENCODINGS_FILE):
        print("No known faces yet!")
        return
    
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    
    names = data["names"]
    print(f"\nKnown faces ({len(names)}):")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")

def remove_face(name):
    """Remove a face from the system"""
    if not os.path.exists(ENCODINGS_FILE):
        print("No known faces yet!")
        return
    
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    
    known_encodings = data["encodings"]
    known_names = data["names"]
    
    if name not in known_names:
        print(f"Face '{name}' not found!")
        return
    
    index = known_names.index(name)
    known_encodings.pop(index)
    known_names.pop(index)
    
    data = {
        "encodings": known_encodings,
        "names": known_names,
        "updated_at": datetime.now().isoformat(),
        "count": len(known_names)
    }
    
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
    
    print(f"✓ Removed '{name}' from the system")

if __name__ == "__main__":
    import sys
    
    print("=" * 50)
    print("Face Collection Tool for Smart Home Security")
    print("=" * 50)
    print("\nOptions:")
    print("  1. Capture new face from webcam")
    print("  2. Add face from image file")
    print("  3. List known faces")
    print("  4. Remove a face")
    print("  5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        name = input("Enter name for this person: ").strip()
        if name:
            encoding = capture_face_from_webcam(name)
            if encoding is not None:
                add_face_to_encodings(name, encoding)
                print(f"\n✓ Face captured and added successfully!")
        else:
            print("Name cannot be empty!")
    
    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            name = input("Enter name (or press Enter for filename): ").strip() or None
            add_face_from_image(image_path, name)
            print("\n✓ Face added successfully!")
        else:
            print("Image file not found!")
    
    elif choice == "3":
        list_known_faces()
    
    elif choice == "4":
        list_known_faces()
        name = input("\nEnter name to remove: ").strip()
        if name:
            remove_face(name)
            print("\n✓ Face removed successfully!")
    
    else:
        print("Exiting...")
