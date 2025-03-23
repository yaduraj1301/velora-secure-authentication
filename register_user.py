from face_detector import FaceRecognitionSystem
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def register_new_user():
    """Main function to handle user registration."""
    logging.info("Starting user registration process")
    face_system = None
    
    try:
        face_system = FaceRecognitionSystem()

        if not face_system.video_capture.isOpened():
            logging.error("Unable to access the camera.")
            return
            
        while True:
            name = input("\nEnter name (or 'q' to quit): ").strip()

            if not name:
                logging.warning("Name cannot be empty.")
                continue

            if name.lower() == 'q':
                break

            if not name.replace(" ", "").isalpha():
                logging.warning("Invalid name format.")
                continue

            if len(name) < 3 or len(name) > 30:
                logging.warning("Name length must be 3-30 characters.")
                continue

            logging.info("Capturing face. Follow guidelines for best results.")

            input("Press Enter to start...")

            success, message = face_system.register_new_face(name)

            if success:
                logging.info(f"✅ {message}")
            else:
                logging.error(f"❌ {message}")

            retry = input("\nRegister another user? (y/n): ").lower()
            if retry != 'y':
                break
                
    except KeyboardInterrupt:
        logging.info("\nProcess cancelled by user.")
    except Exception as e:
        logging.error(f"\nError: {str(e)}")
    finally:
        if face_system:
            logging.info("Cleaning up resources...")
            face_system.cleanup()

if __name__ == "__main__":
    register_new_user()
