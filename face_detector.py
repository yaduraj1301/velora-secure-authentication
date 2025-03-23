import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
from datetime import datetime
import pickle
import time
import logging
import threading
import re
import base64

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FaceRecognitionSystem:
    def __init__(self):
        """Initialize MongoDB, video capture, and load known faces."""
        try:
            self.client = MongoClient('mongodb://localhost:27017/')
            self.db = self.client['face_recognition_db']
            self.users_collection = self.db['users']
            self.users_collection.create_index("name", unique=True)  # Ensure fast lookups
            logging.info("MongoDB connection successful")
        except Exception as e:
            raise RuntimeError(f"MongoDB connection failed: {e}")
            
        self.video_capture = self._initialize_camera()
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()
        self.frame = None
        self.running = True

        # Start a separate thread for video capture
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
    def _initialize_camera(self):
        """Initialize and configure the camera."""
        capture = cv2.VideoCapture(0)
        if not capture.isOpened():
            raise RuntimeError("Unable to access the camera. Please check your camera connection.")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Allow camera to adjust to lighting conditions
        time.sleep(2)
        logging.info("Camera initialized successfully")
        return capture
    
    def _capture_frames(self):
        """Continuously capture frames in a separate thread for better performance."""
        while self.running:
            ret, frame = self.video_capture.read()
            if ret:
                self.frame = frame
            time.sleep(0.05)


    def load_known_faces(self):
        """Load face encodings from the database.
        
        If the stored face_encoding is in a JS format (i.e. base64 or binary image)
        rather than a Python pickle, decode it, compute the face encoding using the
        same OpenCV logic as register_new_face, pickle it, and update the DB.
        """
        try:
            users = self.users_collection.find({})
            for user in users:
                stored_data = user['face_encoding']
                try:
                    # Try to load as a pickled face encoding (Python format)
                    face_encoding = pickle.loads(stored_data)
                except Exception as pickle_error:
                    logging.warning(f"Pickle load failed for user {user['name']} (likely stored from JS). Attempting to compute face encoding using CV logic.")
                    try:
                        # Determine if stored_data is a string (base64) or already binary
                        if isinstance(stored_data, str):
                            # Remove potential data URI header if present
                            base64_str = re.sub(r"^data:image\/\w+;base64,", "", stored_data)
                            image_data = base64.b64decode(base64_str)
                        else:
                            image_data = stored_data

                        # Convert binary data to a NumPy array and decode the image using OpenCV
                        np_arr = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is None:
                            logging.error(f"Failed to decode image for user {user['name']}.")
                            continue

                        # Convert to RGB for face_recognition
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        # Detect face locations using the same 'hog' model
                        face_locations = face_recognition.face_locations(rgb_img, model="hog")
                        if not face_locations:
                            logging.warning(f"No face detected in the image for user {user['name']}.")
                            continue

                        # Compute face encoding
                        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                        if not face_encodings:
                            logging.warning(f"Face encoding failed for user {user['name']}.")
                            continue
                        face_encoding = face_encodings[0]

                        # Pickle the computed encoding
                        pickled_encoding = pickle.dumps(face_encoding)

                        # Update the database record with the pickled encoding
                        self.users_collection.update_one(
                            {'_id': user['_id']},
                            {'$set': {'face_encoding': pickled_encoding}}
                        )
                        logging.info(f"Updated user {user['name']} with pickled face encoding from JS data.")

                    except Exception as js_error:
                        logging.error(f"Error processing JS face data for user {user['name']}: {js_error}")
                        continue

                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(user['name'])
            
            logging.info(f"Loaded {len(self.known_face_encodings)} known faces from database")
        except Exception as e:
            raise RuntimeError(f"Error loading known faces: {e}")


    def register_new_face(self, name, max_attempts=5):
        """Capture face and store encoding in the database."""
        for attempt in range(max_attempts):
            time.sleep(1)  # Allow time for the camera to stabilize
            
            frame = self.frame
            if frame is None:
                logging.warning(f"Attempt {attempt + 1}: No frame captured.")
                continue

            # Convert the image to RGB and preprocess
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")

            if len(face_locations) == 0:
                logging.warning(f"Attempt {attempt + 1}: No face detected.")
                continue
                
            if len(face_locations) > 1:
                logging.warning("Multiple faces detected. Please ensure only one person is in frame.")
                return False, "Multiple faces detected."

            # Extract the face encoding
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            if not face_encodings:
                logging.warning(f"Attempt {attempt + 1}: Face encoding failed.")
                continue
                
            face_encoding = face_encodings[0]

            # Save to database
            try:
                if self.users_collection.find_one({'name': name}):
                    return False, f"User {name} already exists."

                self.users_collection.insert_one({
                    'name': name,
                    'face_encoding': pickle.dumps(face_encoding),
                    'created_at': datetime.now()
                })
                return True, f"User {name} registered successfully."
            except Exception as e:
                logging.error(f"Error saving to database: {e}")
                return False, "Database error."

        return False, "Failed to capture a clear face encoding."

    def cleanup(self):
        """Release resources and stop video capture."""
        self.running = False
        self.capture_thread.join()
        if self.video_capture.isOpened():
            self.video_capture.release()
            logging.info("Camera released.")
        if self.client:
            self.client.close()
            logging.info("MongoDB connection closed.")
