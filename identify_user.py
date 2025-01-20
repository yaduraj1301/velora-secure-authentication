import cv2
import face_recognition
from face_detector import FaceRecognitionSystem
import time

def identify_user():
    """
    This function identifies the user by comparing the captured face with the stored face encodings.
    Added enhanced error handling and diagnostic information.
    """
    face_system = None
    try:
        print("Initializing face recognition system...")
        face_system = FaceRecognitionSystem()
        
        print("\nStarting face identification...")
        print("Press 'q' to exit")
        
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
                
                for face_encoding, face_location in zip(face_encodings, face_locations):
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