import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
from datetime import datetime
import os
import pickle

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

        # Initialize video capture with enhanced error checking
        self.video_capture = self._initialize_camera()
        
        # Load known faces from database
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def _initialize_camera(self):
        """Initialize camera with multiple attempts and detailed error reporting"""
        available_cameras = self._list_available_cameras()
        if not available_cameras:
            raise RuntimeError("No cameras found on the system")
        
        print(f"Available camera indices: {available_cameras}")
        
        # Try each available camera
        for camera_index in available_cameras:
            try:
                capture = cv2.VideoCapture(camera_index)
                if capture.isOpened():
                    print(f"Successfully opened camera at index {camera_index}")
                    # Test reading a frame
                    ret, frame = capture.read()
                    if ret and frame is not None:
                        return capture
                    else:
                        print(f"Camera {camera_index} opened but cannot read frames")
                        capture.release()
                else:
                    print(f"Failed to open camera at index {camera_index}")
            except Exception as e:
                print(f"Error trying camera {camera_index}: {str(e)}")
        
        raise RuntimeError("Could not initialize any available camera")

    def _list_available_cameras(self, max_cameras=10):
        """List all available camera indices"""
        available = []
        for i in range(max_cameras):
            try:
                temp_capture = cv2.VideoCapture(i)
                if temp_capture.isOpened():
                    available.append(i)
                    temp_capture.release()
            except Exception:
                continue
        return available

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

    # ... rest of the class implementation remains the same ...