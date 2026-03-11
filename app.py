from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load trained model
model = joblib.load("model/fraud_model.joblib")

# Encoders (MUST match training)
transaction_type_encoder = LabelEncoder()
merchant_category_encoder = LabelEncoder()

transaction_type_encoder.fit(["Online", "POS", "ATM"])
merchant_category_encoder.fit(["Food", "Shopping", "Travel", "Electronics"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        amount = float(request.form["transaction_amount"])
        transaction_type = request.form["transaction_type"]
        merchant_category = request.form["merchant_category"]
        card_present = int(request.form["card_present"])
        international = int(request.form["international"])
        transaction_hour = int(request.form["transaction_hour"])

        # Encode categorical values
        transaction_type_encoded = transaction_type_encoder.transform([transaction_type])[0]
        merchant_category_encoded = merchant_category_encoder.transform([merchant_category])[0]

        # Feature order MUST match training
        features = np.array([[
            amount,
            transaction_type_encoded,
            merchant_category_encoded,
            card_present,
            international,
            transaction_hour
        ]])

        # Prediction
        prediction = model.predict(features)[0]

        result = "Fraud Transaction ❌" if prediction == 1 else "Genuine Transaction ✅"
        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error occurred: {e}"


if __name__ == "__main__":
    app.run(debug=True)
    