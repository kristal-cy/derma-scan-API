from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import UnidentifiedImageError

app = Flask(__name__)
CORS(app)

# Load pre-trained CNN model (e.g., ResNet or any model trained on skin conditions)
MODEL_PATH = "final_model_skin_new.h5"  # Path to your pre-trained model
model = load_model(MODEL_PATH)

# Labels corresponding to the model output (assumes the model has specific skin condition classes)
class_names = ['acne', 'oily', 'perioral_dermatitis', 'dry', 'normal', 'aging', 'vitiligo']

# Confidence threshold (set this based on your model's performance)
CONFIDENCE_THRESHOLD = 0.5  # If confidence is below 50%, reject the image

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "✅ Skin Detection API is ready"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "⚠️ No file uploaded"}), 400

    file = request.files['file']
    print(f"File received: {file.filename}")

    if file.filename == '':
        return jsonify({"error": "⚠️ File name is empty"}), 400

    try:
        filename = secure_filename(file.filename)
        os.makedirs("temp", exist_ok=True)
        filepath = os.path.join("temp", filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")

        # Validate image
        try:
            img = image.load_img(filepath, target_size=(224, 224))
        except UnidentifiedImageError:
            return jsonify({"error": "❌ Uploaded file is not a valid image"}), 400

        # Preprocessing the image
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction with the pre-trained CNN model
        pred = model.predict(img_array)
        if len(pred[0]) != len(class_names):
            return jsonify({"error": "❌ Model output count does not match the number of labels"}), 500

        class_idx = np.argmax(pred[0])
        label = class_names[class_idx]
        confidence = float(pred[0][class_idx])

        print(f"Prediction: {label} with confidence {confidence}")

        # Check if the confidence is below the threshold
        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({"error": "❌ The image does not appear to be a skin condition. Please upload a skin-related image."}), 400

        # Return the prediction if confidence is high enough
        result = {
            "label": label,
            "confidence": round(confidence * 100, 2)  # percentage
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"❌ Error during prediction: {str(e)}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} deleted after prediction")

if __name__ == '__main__':
    app.run(debug=True)
