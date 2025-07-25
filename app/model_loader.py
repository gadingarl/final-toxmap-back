import os
import joblib
import numpy as np
from PIL import Image
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'svm.pkl')

model = joblib.load(MODEL_PATH)

if hasattr(model, 'support_vectors_'):
    n_features = model.support_vectors_.shape[1]
else:
    n_features = 34992

mobilenet_model = MobileNetV2(weights="imagenet", include_top=False, pooling="avg", input_shape=(224, 224, 3))

label_mapping = {
    0: "Non_Toxic",
    1: "Baterai",
    2: "Kabel", 
    3: "LampuLED",
    4: "Aerosol",
    5: "PembersihLantai"
}

dropbox_color_mapping = {
    "Baterai": "Merah",
    "Kabel": "Merah", 
    "LampuLED": "Merah",
    "Aerosol": "Kuning",
    "PembersihLantai": "Kuning",
    "Non_Toxic": "Tidak Ada"
}

def extract_features(image):
    image_resized = cv2.resize(image, (224, 224))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_array = np.expand_dims(image_rgb, axis=0)
    image_array = preprocess_input(image_array)
    features = mobilenet_model.predict(image_array)
    return features.flatten()

def get_confidence(features):
    if hasattr(model, "decision_function"):
        return float(np.max(model.decision_function(features)))
    elif hasattr(model, "predict_proba"):
        return float(np.max(model.predict_proba(features)))
    else:
        return 0.5

def process_prediction(prediction, confidence):
    if confidence < 0.1:
        pred_label = "Non_Toxic"
    else:
        pred_label = label_mapping.get(prediction, "Non_Toxic")
    dropbox_color = dropbox_color_mapping.get(pred_label, "Tidak Ada")
    return pred_label, dropbox_color

def predict_image(file_bytes):
    image = Image.open(BytesIO(file_bytes)).convert('RGB')
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    features = extract_features(image_bgr).reshape(1, -1)
    prediction = model.predict(features)[0]
    confidence = get_confidence(features)
    return process_prediction(prediction, confidence)
