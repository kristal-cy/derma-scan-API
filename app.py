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
MODEL_PATH = "model_kulit.h5"
model = load_model(MODEL_PATH)

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)  # Hentikan program jika model gagal load

# Label sesuai urutan output model
class_names = ['jerawat_parah', 'berminyak', 'jerawat_sedang', 'kering', 'jerawat_ringan']

# List untuk menyimpan history prediksi (hanya di memori, tidak persistent)
predictions_history = []

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "✅ API Deteksi Kulit Siap Digunakan"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "⚠️ Tidak ada file yang dikirim"}), 400

    file = request.files['file']
    print(f"File diterima: {file.filename}")

    if file.filename == '':
        return jsonify({"error": "⚠️ Nama file kosong"}), 400

    try:
        filename = secure_filename(file.filename)
        os.makedirs("temp", exist_ok=True)
        filepath = os.path.join("temp", filename)
        file.save(filepath)
        print(f"File disimpan di: {filepath}")

        # Validasi gambar
        try:
            img = image.load_img(filepath, target_size=(224, 224))
        except UnidentifiedImageError:
            return jsonify({"error": "❌ File bukan gambar yang valid"}), 400

        # Preprocessing
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Prediction
        pred = model.predict(img_array)
        if len(pred[0]) != len(class_names):
            return jsonify({"error": "❌ Jumlah output model tidak cocok dengan jumlah label"}), 500

        class_idx = np.argmax(pred[0])
        label = class_names[class_idx]
        confidence = float(pred[0][class_idx])

        print(f"Prediksi: {label} dengan confidence {confidence}")

        result = {
            "label": label,
            "confidence": round(confidence * 100, 2)  # dalam persen
        }

        # Simpan hasil prediksi ke history
        predictions_history.append(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"❌ Terjadi kesalahan saat prediksi: {str(e)}"}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"File {filepath} dihapus setelah prediksi")

@app.route('/history', methods=['GET'])
def history():
    # Kembalikan semua hasil prediksi yang sudah dilakukan
    return jsonify(predictions_history)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
