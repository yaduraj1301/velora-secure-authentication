from face_detector import FaceRecognitionSystem
import sys
import time
import cv2

def register_new_user():
    """Main function to handle user registration process with improved error handling"""
    print("Starting user registration process")
    face_system = None
    
    try:
        # Initialize the face recognition system
        face_system = FaceRecognitionSystem()
        
        # Check if the camera is accessible
        if not face_system.video_capture.isOpened():
            print("Error: Unable to access the camera. Please check your camera.")
            return
            
        while True:
            # Get user name
            name = input("\nEnter the name of the person to register (or 'q' to quit): ").strip()
            
            if not name:
                print("Name cannot be empty. Please try again.")
                continue
                
            if name.lower() == 'q':
                break
                
            # Validate name
            if not name.replace(" ", "").isalpha():
                print("Name should only contain letters and spaces. Please try again.")
                continue
                
            if len(name) < 3 or len(name) > 30:
                print("Name should be between 3 and 30 characters. Please try again.")
                continue
                
            print("\nPreparing to capture face...")
            print("Guidelines for best results:")
            print("1. Ensure good lighting on your face")
            print("2. Look directly at the camera")
            print("3. Keep a neutral expression")
            print("4. Maintain a distance of about 2 feet from the camera")
            
            input("Press Enter when ready (or Ctrl+C to cancel)...")
            
            print("\nCapturing... Please stay still...")
            
            # Attempt registration with feedback
            success, message = face_system.register_new_face(name)
            
            if success:
                print("\n✅ " + message)
            else:
                print("\n❌ " + message)
                print("\nTips for successful registration:")
                print("- Ensure there is sufficient lighting")
                print("- Face the camera directly")
                print("- Keep still during capture")
                print("- Ensure no other faces are in frame")
            
            # Ask for another registration
            while True:
                retry = input("\nRegister another user? (y/n): ").lower()
                if retry in ['y', 'n']:
                    break
                print("Please enter 'y' for yes or 'n' for no.")
            
            if retry != 'y':
                break
                
    except KeyboardInterrupt:
        print("\nRegistration process cancelled by user.")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Please check the logs for more details.")
    finally:
        if face_system:
            print("\nCleaning up resources...")
            face_system.cleanup()
            print("Registration process completed.")

def main():
    try:
        register_new_user()
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()