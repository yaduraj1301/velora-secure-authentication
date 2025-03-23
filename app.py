from flask import Flask, request, jsonify
from pymongo import MongoClient
import face_recognition
import base64
import pickle
import numpy as np
import cv2
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
users_collection = db['users']

logging.basicConfig(level=logging.INFO)

def decode_base64_image(image_base64):
    try:
        # Remove any header (e.g., "data:image/png;base64,")
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        # Add padding if necessary
        missing_padding = len(image_base64) % 4
        if missing_padding:
            image_base64 += "=" * (4 - missing_padding)
        image_data = base64.b64decode(image_base64)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logging.error(f"Error decoding base64 image: {e}")
        return None


def get_face_encoding(image):
    try:
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        if not face_locations:
            return None
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        return face_encodings[0] if face_encodings else None
    except Exception as e:
        logging.error(f"Error extracting face encoding: {e}")
        return None

@app.route('/face-recognizer', methods=['POST'])
def face_recognizer():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    image_base64 = data['image']
    image = decode_base64_image(image_base64)
    if image is None:
        return jsonify({'error': 'Invalid image data'}), 400
    
    face_encoding = get_face_encoding(image)
    if face_encoding is None:
        return jsonify({'error': 'No face detected'}), 400
    
    users = users_collection.find({})
    for user in users:
        stored_encoding = pickle.loads(user['face_encoding'])
        match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)
        if match[0]:
            return jsonify({'name': user['name']}), 200
    
    return jsonify({'name': 'Unknown'}), 200


def decode_base64(data):
    """Decodes Base64 string, adding padding if necessary"""
    data += "=" * ((4 - len(data) % 4) % 4)  # Fix missing padding
    return base64.b64decode(data)

@app.route('/verify-image', methods=['POST'])
def verify_image():
    try:
        data = request.json.get("image", "")
        image_data = data.split(",")[-1]  # Remove the data URL prefix if present
        image_bytes = decode_base64(image_data)

        with open("received_image.png", "wb") as f:
            f.write(image_bytes)

        return jsonify({"message": "Image received successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
