# app/model_loader.py
import joblib
import requests
from io import BytesIO

_model = None

def get_model():
    global _model
    if _model is None:
        model_url = (
            "https://firebasestorage.googleapis.com/v0/b/toxmap-b74f4.appspot.com/o/"
            "svm_model_final.pkl?alt=media&token=83df2ee0-e577-4d7f-aa1b-5c59362cec85"
        )
        response = requests.get(model_url)
        response.raise_for_status()
        _model = joblib.load(BytesIO(response.content))
    return _model
