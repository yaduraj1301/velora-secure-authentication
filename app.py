from flask import Flask, request, jsonify
from pymongo import MongoClient
import face_recognition
import base64
import pickle
import numpy as np
import cv2
import logging
import re
import threading
import time
from flask_cors import CORS
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification


# Define the path to your saved model folder
model_path = "./models"

# Load the tokenizer (using the original model base)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load the saved model
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()  # Set model to evaluation mode



def predict_toxicity(text):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Perform inference without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and compute probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

    # Determine the predicted class (assuming label 1 = toxic)
    predicted_class = 'toxic' if probabilities[1] > probabilities[0] else 'non-toxic'
    return {
        'text': text,
        'predicted_class': predicted_class,
        'non_toxic_probability': probabilities[0].item(),
        'toxic_probability': probabilities[1].item()
    }


app = Flask(__name__)
CORS(app)

client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']
users_collection = db['users']

logging.basicConfig(level=logging.INFO)

# Global variables to store known faces in memory.
known_face_encodings = []
known_face_names = []
# A simple checksum (e.g., number of users) to detect changes.
db_checksum = None

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

def update_known_faces():
    """
    Background job that periodically:
    1. Scans the database to update any records stored in a JS (Base64) format into pickled encodings.
    2. Loads all face encodings and names into memory.
    """
    global known_face_encodings, known_face_names, db_checksum
    while True:
        try:
            users = list(users_collection.find({}))
            # Compute a simple checksum (e.g., total count)
            new_checksum = len(users)
            # Only update if the checksum has changed or force update every time
            if new_checksum != db_checksum:
                logging.info("Detected change in user database. Updating known faces...")
                db_checksum = new_checksum
            # Clear current lists
            known_face_encodings = []
            known_face_names = []
            for user in users:
                stored_data = user['face_encoding']
                try:
                    # Attempt to load as pickled data
                    face_encoding = pickle.loads(stored_data)
                except Exception as pickle_error:
                    logging.warning(f"Pickle load failed for user {user['name']} (likely stored from JS). Converting.")
                    try:
                        # If stored_data is a string, assume Base64 and remove header.
                        if isinstance(stored_data, str):
                            base64_str = re.sub(r"^data:image\/\w+;base64,", "", stored_data)
                            image_data = base64.b64decode(base64_str)
                        else:
                            image_data = stored_data

                        np_arr = np.frombuffer(image_data, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is None:
                            logging.error(f"Failed to decode image for user {user['name']}.")
                            continue

                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_img, model="hog")
                        if not face_locations:
                            logging.warning(f"No face detected for user {user['name']}.")
                            continue

                        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
                        if not face_encodings:
                            logging.warning(f"Face encoding failed for user {user['name']}.")
                            continue

                        face_encoding = face_encodings[0]
                        # Pickle the computed encoding
                        pickled_encoding = pickle.dumps(face_encoding)
                        users_collection.update_one({'_id': user['_id']}, {'$set': {'face_encoding': pickled_encoding}})
                        logging.info(f"Updated user {user['name']} with pickled encoding.")
                    except Exception as js_error:
                        logging.error(f"Error converting JS data for user {user['name']}: {js_error}")
                        continue

                known_face_encodings.append(face_encoding)
                known_face_names.append(user['name'])
            logging.info(f"Loaded {len(known_face_encodings)} known faces into memory.")
        except Exception as e:
            logging.error(f"Error in background update: {e}")
        # Sleep for 60 seconds (adjust as needed)
        time.sleep(10)

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

    # Compare against the known faces in memory.
    for stored_encoding, name in zip(known_face_encodings, known_face_names):
        match = face_recognition.compare_faces([stored_encoding], face_encoding, tolerance=0.6)
        if match[0]:
            return jsonify({'name': name}), 200

    return jsonify({'name': 'Unknown'}), 200

@app.route('/find-toxicity', methods=['POST'])
def find_toxicity():
    print(request)
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    if not text:
        return jsonify({'error': 'Empty text provided'}), 400

    result = predict_toxicity(text)
    return jsonify(result), 200


def predict_toxicity(text):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    # Perform inference without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract logits and compute probabilities
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]

    # Determine the predicted class (assuming label 1 = toxic)
    predicted_class = 'toxic' if probabilities[1] > probabilities[0] else 'non-toxic'
    return {
        'text': text,
        'predicted_class': predicted_class,
        'non_toxic_probability': probabilities[0].item(),
        'toxic_probability': probabilities[1].item()
    }


if __name__ == '__main__':
    # Start the background thread to update known faces.
    updater_thread = threading.Thread(target=update_known_faces, daemon=True)
    updater_thread.start()
    app.run(debug=True)
