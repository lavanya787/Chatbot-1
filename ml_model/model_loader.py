# ml_model/model_loader.py

import os
from joblib import load

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../saved_models/model.joblib")

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found.")
        return None
    try:
        model = load(MODEL_PATH)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None
