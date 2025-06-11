import os 
from flask import Flask ,request, jsonify
from flask_cors import CORS

from model import Model

app = Flask(__name__)
CORS(app)

@app.route("/")
def main():
    return "Server is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files['file']
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        model = Model(file_path)
        result = model.predict()
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)