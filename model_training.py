import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("fraud.csv")

# Encode categorical columns
transaction_type_encoder = LabelEncoder()
merchant_category_encoder = LabelEncoder()

data["transaction_type"] = transaction_type_encoder.fit_transform(data["transaction_type"])
data["merchant_category"] = merchant_category_encoder.fit_transform(data["merchant_category"])

# Features and target
X = data[[
    "transaction_amount",
    "transaction_type",
    "merchant_category",
    "card_present",
    "international",
    "transaction_hour"
]]

y = data["is_fraud"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/fraud_model.joblib")

print("✅ Model trained and saved successfully!")