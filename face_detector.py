import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
from datetime import datetime
import pickle
import time

class FaceRecognitionSystem:
    def __init__(self):
        # Initialize MongoDB connection
        try:
            self.client = MongoClient('mongodb://localhost:27017/')
            self.db = self.client['face_recognition_db']
            self.users_collection = self.db['users']
            print("MongoDB connection successful")
        except Exception as e:
            raise RuntimeError(f"MongoDB connection failed: {e}")
            
        # Initialize video capture with default camera
        self.video_capture = self._initialize_camera()
        
        # Load known faces from database
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        
    def _initialize_camera(self):
        """Initialize the camera with improved settings"""
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise RuntimeError("Unable to access the camera. Please check your camera connection.")
            
        # Set camera properties for better face detection
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
        capture.set(cv2.CAP_PROP_BRIGHTNESS, 150)  # Adjust brightness
        
        # Warm up the camera
        for _ in range(5):
            capture.read()
            time.sleep(0.1)
            
        print("Successfully opened default camera (camera 0)")
        return capture
        
    def load_known_faces(self):
        try:
            users = self.users_collection.find({})
            count = 0
            for user in users:
                face_encoding = pickle.loads(user['face_encoding'])
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(user['name'])
                count += 1
            print(f"Successfully loaded {count} known faces from database")
        except Exception as e:
            raise RuntimeError(f"Error loading known faces: {e}")

    def register_new_face(self, name, max_attempts=5):
        """Capture face and store encoding in the database with multiple attempts"""
        for attempt in range(max_attempts):
            # Capture multiple frames to ensure camera is stabilized
            for _ in range(3):
                self.video_capture.read()
                
            # Capture the actual frame for processing
            ret, frame = self.video_capture.read()
            if not ret:
                continue
                
            # Convert the image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations in the frame
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            
            if len(face_locations) == 0:
                if attempt < max_attempts - 1:
                    print(f"Attempt {attempt + 1}: No face detected, trying again...")
                    time.sleep(1)
                    continue
                return False, "No face detected after multiple attempts. Please check lighting and camera position."
                
            if len(face_locations) > 1:
                return False, "Multiple faces detected. Please ensure only one person is in frame."
                
            # Extract the face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if len(face_encodings) == 0:
                continue
                
            face_encoding = face_encodings[0]
            
            # Save to database
            try:
                # Check if name already exists
                existing_user = self.users_collection.find_one({'name': name})
                if existing_user:
                    return False, f"User {name} already exists in the database."
                    
                self.users_collection.insert_one({
                    'name': name,
                    'face_encoding': pickle.dumps(face_encoding),
                    'created_at': datetime.now()
                })
                return True, f"User {name} registered successfully."
            except Exception as e:
                return False, f"Error saving to database: {str(e)}"
                
        return False, "Failed to capture a clear face encoding after multiple attempts."

    def cleanup(self):
        """Clean up resources"""
        if self.video_capture.isOpened():
            self.video_capture.release()
            print("Camera released.")
        if self.client:
            self.client.close()
            print("MongoDB connection closed.")