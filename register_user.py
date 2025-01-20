from face_detector import FaceRecognitionSystem
import sys
import time

def register_new_user():
    """
    Main function to handle user registration process with face recognition.
    Includes error handling, user interaction, and cleanup.
    """
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
            if any(char.isdigit() for char in name):
                print("Name should not contain numbers. Please try again.")
                continue

            if len(name) < 3 or len(name) > 30:  # Length check
                print("Name should be between 3 and 30 characters. Please try again.")
                continue

            print("\nLooking for face... Press 'q' to cancel.")
            print("Please ensure good lighting and look directly at the camera.")

            # Allow camera to adjust
            time.sleep(2)

            # Attempt registration
            success, message = face_system.register_new_face(name)

            if success:
                print("\n✅ " + message)
            else:
                print("\n❌ " + message)

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
            print("Cleaning up resources")
            face_system.cleanup()
        print("\nRegistration process completed.")

def main():
    """
    Entry point of the script.
    Handles the top-level execution and error boundary.
    """
    try:
        register_new_user()
    except Exception as e:
        print(f"\nA critical error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
