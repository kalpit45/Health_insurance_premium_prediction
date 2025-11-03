from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("../app/insurance_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    features = np.array([[
        data["age"], data["sex"], data["bmi"], data["smoker"]
    ]])
    
    prediction = model.predict(features)[0]
    return jsonify({"Predicted Premium": prediction})

app.run(debug=True)
