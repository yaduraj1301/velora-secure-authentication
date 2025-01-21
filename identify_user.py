import cv2
import face_recognition
from face_detector import FaceRecognitionSystem
import time
import numpy as np
from scipy.spatial import distance as dist

def calculate_ear(eye_landmarks):
    """Calculate eye aspect ratio"""
    # Compute euclidean distances between vertical eye landmarks
    A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    # Compute euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    # Calculate eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def identify_user():
    """
    This function identifies the user by comparing the captured face with the stored face encodings.
    Added enhanced error handling, diagnostic information, and improved blink detection.
    """
    face_system = None
    try:
        print("Initializing face recognition system...")
        face_system = FaceRecognitionSystem()
        
        print("\nStarting face identification...")
        print("Press 'q' to exit or blink twice to release camera")
        
        # Initialize blink detection variables with refined parameters
        EAR_THRESHOLD = 0.25  # Increased threshold for better sensitivity
        BLINK_CONSEC_FRAMES = 3  # Increased to reduce false positives
        blink_counter = 0
        total_blinks = 0
        last_blink_time = time.time()
        BLINK_TIMEOUT = 1.5  # Increased window for double blink detection
        
        # Add EAR smoothing
        ear_history = []
        SMOOTHING_WINDOW = 3
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            # Capture a frame from the camera
            ret, frame = face_system.video_capture.read()
            if not ret:
                print("⚠️ Failed to capture frame, retrying...")
                time.sleep(1)
                continue
                
            frame_count += 1
            if frame_count % 30 == 0:  # Log FPS every 30 frames
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"Camera FPS: {fps:.2f}")
            
            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Process detected faces
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                face_landmarks = face_recognition.face_landmarks(rgb_frame, face_locations)
                
                for face_encoding, landmarks, face_location in zip(face_encodings, face_landmarks, face_locations):
                    # Calculate average EAR for both eyes
                    left_eye = landmarks['left_eye']
                    right_eye = landmarks['right_eye']
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Apply smoothing to EAR values
                    ear_history.append(avg_ear)
                    if len(ear_history) > SMOOTHING_WINDOW:
                        ear_history.pop(0)
                    smoothed_ear = sum(ear_history) / len(ear_history)
                    
                    # Detect blink with smoothed EAR
                    if smoothed_ear < EAR_THRESHOLD:
                        blink_counter += 1
                    else:
                        if blink_counter >= BLINK_CONSEC_FRAMES:
                            current_time = time.time()
                            if current_time - last_blink_time < BLINK_TIMEOUT:
                                total_blinks += 1
                                print(f"Blink detected! Total blinks: {total_blinks}")
                            else:
                                total_blinks = 1
                                print("First blink detected!")
                            last_blink_time = current_time
                        blink_counter = 0
                    
                    # Check for double blink
                    if total_blinks >= 2:
                        print("\nDouble blink confirmed - releasing camera")
                        return
                    
                    matches = face_recognition.compare_faces(
                        face_system.known_face_encodings,
                        face_encoding,
                        tolerance=0.6
                    )
                    
                    if True in matches:
                        match_index = matches.index(True)
                        name = face_system.known_face_names[match_index]
                        
                        # Draw rectangle and name
                        top, right, bottom, left = face_location
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                        
                        # Draw EAR information
                        cv2.putText(frame, f"EAR: {smoothed_ear:.3f}", (left, bottom + 20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Blinks: {total_blinks}", (left, bottom + 40),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Add visual indicator for blink detection
                        if smoothed_ear < EAR_THRESHOLD:
                            cv2.putText(frame, "BLINK!", (left, top - 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        print(f"Identified: {name}")
                    else:
                        print("Unknown face detected")
            
            # Display the frame
            cv2.imshow('Face Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nUser interrupted the process")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if face_system:
            face_system.cleanup()
        cv2.destroyAllWindows()
        print("\nFace recognition system shutdown complete")

if __name__ == "__main__":
    identify_user()