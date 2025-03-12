from flask import Flask, request, jsonify
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_path = "resnet50_pretrained_model.h5"
model_url = "https://drive.google.com/file/d/1yeM4q1ae3Zli3kPCIGMm9LktGvEHCc1E/view?usp=sharing"

if not os.path.exists(model_path):
    logger.info("Downloading model...")
    try:
        response = requests.get(model_url)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(model_path, "wb") as f:
            f.write(response.content)
        logger.info("Model downloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise

# Load the model
try:
    model = load_model(model_path)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

app = Flask(__name__)

# Basic authentication
auth = HTTPBasicAuth()

# User credentials (for demonstration purposes)
users = {
    "admin": generate_password_hash("password")
}

@auth.verify_password
def verify_password(username, password):
    if username in users and check_password_hash(users.get(username), password):
        return username
    
def preprocess_image(image: Image.Image) -> np.ndarray:
    try:
        # Resize and normalize the image
        image = image.resize((32, 32))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise ValueError("Invalid image format")
    
    
def predict_class(image: np.ndarray) -> str:
    try:
        predictions = model.predict(image)
        predicted_class = class_names[np.argmax(predictions)]
        return predicted_class
    except Exception as e:
        logger.error(f"Error predicting class: {e}")
        raise ValueError("Prediction failed")
    

@app.route('/predict', methods=['POST'])
@auth.login_required

def predict():
    try:
        # Check if an image is provided
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        # Read and preprocess the image
        file = request.files['file']
        image = Image.open(io.BytesIO(file.read()))
        image = preprocess_image(image)

        # Make prediction
        predicted_class = predict_class(image)

        # Log the prediction
        logger.info(f"Prediction successful: {predicted_class}")

        return jsonify({"predicted_class": predicted_class})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({"error": "Internal server error"}), 500
    
@app.route('/')
def root():
    return jsonify({"message": "CIFAR-10 Image Classification API"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)