from flask import Flask, request, jsonify
import joblib

model = joblib.load("career_predictor.pkl")

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    # Extract features
    features = [data['skills'], data['interests'], data['cgpa'], data['projects']]
    # Preprocess same as training
    pred = model.predict([features])
    return jsonify({"career_path": pred[0]})

if __name__ == "__main__":
    app.run()
