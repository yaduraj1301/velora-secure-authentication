import cv2
import face_recognition
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

image_path = "1.jpg"

# Load the image in color (just to double check in case it's not reading correctly in RGB)
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    logging.error("Failed to load image.")
    exit(1)

# Check if image is 8-bit (uint8)
if image.dtype != np.uint8:
    logging.error(f"Unexpected image dtype: {image.dtype}. Expected uint8.")
    exit(1)

# Convert image to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Check the shape and type of the image
logging.debug(f"Image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")

# Perform face detection
try:
    face_locations = face_recognition.face_locations(rgb_image)
    logging.info(f"Detected faces at locations: {face_locations}")
except Exception as e:
    logging.error(f"Face detection error: {e}")
