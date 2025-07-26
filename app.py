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

# Load model
MODEL_PATH = "final_model_skin_new.h5"
model = load_model(MODEL_PATH)

# Labels corresponding to the model output
class_names = ['acne', 'oily', 'perioral_dermatitis', 'dry', 'normal', 'aging', 'vitiligo']

# List to store prediction history (in-memory only, not persistent)
predictions_history = []

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

        # Preprocessing
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        pred = model.predict(img_array)
        if len(pred[0]) != len(class_names):
            return jsonify({"error": "❌ Model output count does not match the number of labels"}), 500

        class_idx = np.argmax(pred[0])
        label = class_names[class_idx]
        confidence = float(pred[0][class_idx])

        print(f"Prediction: {label} with confidence {confidence}")

        result = {
            "label": label,
            "confidence": round(confidence * 100, 2)  # percentage
        }

        # Save prediction result to history
        predictions_history.append(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"❌ Error during prediction: {str(e)}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} deleted after prediction")

@app.route('/history', methods=['GET'])
def history():
    # Return all past predictions
    return jsonify(predictions_history)

if __name__ == '__main__':
    app.run(debug=True)
