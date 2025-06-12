import os 
from flask import Flask ,request, jsonify
from flask_cors import CORS

from keras_model import KerasModel
from tflite_model import TFLiteModel


app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
    return "Server is running!"

@app.route("/predict/keras", methods=["POST"])
def predict_keras():
    try:
        file = request.files['file']
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        model = KerasModel(file_path)
        result = model.predict()
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/predict/tflite", methods=["POST"])
def predict_tflite():
    try:
        file = request.files['file']
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        model = TFLiteModel(file_path)
        result = model.predict()
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)