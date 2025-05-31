# app/model_loader.py
import joblib
import requests
from io import BytesIO

_model = None

def get_model():
    global _model
    if _model is None:
        model_url = (
            "https://firebasestorage.googleapis.com/v0/b/toxmap-b74f4.firebasestorage.app/o/svm_model_fix.pkl?alt=media&token=6f44d9f9-0284-4286-bc09-723030cbdc9f"
        )
        response = requests.get(model_url)
        response.raise_for_status()
        _model = joblib.load(BytesIO(response.content))
    return _model
