

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # allows React (different port) to call this API

model = joblib.load('model/best_model.pkl')
scaler = joblib.load('model/scaler.pkl')
feature_names = joblib.load('model/feature_names.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Expect a dict with keys matching feature_names (V1..V28, Amount)
        input_values = [float(data[feature]) for feature in feature_names]
    except KeyError as e:
        return jsonify({"error": f"Missing field: {str(e)}"}), 400

    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of fraud

    return jsonify({
        "prediction": int(prediction),
        "label": "Fraudulent" if prediction == 1 else "Genuine",
        "fraud_probability": round(float(probability), 4)
    })

@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({"features": feature_names})

if __name__ == '__main__':
    app.run(debug=True, port=5000)