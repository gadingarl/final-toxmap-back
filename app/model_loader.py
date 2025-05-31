from PIL import Image
from io import BytesIO
import numpy as np
import joblib
import requests
import os
import math

MODEL_URL = os.getenv("MODEL_URL")
if not MODEL_URL:
    raise Exception("❌ MODEL_URL tidak ditemukan di environment")

LOCAL_MODEL_PATH = "svm_model_fix.pkl"

# Unduh model jika belum ada
if not os.path.exists(LOCAL_MODEL_PATH):
    print("⬇️ Mengunduh model dari Firebase Storage...")
    response = requests.get(MODEL_URL)
    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model berhasil diunduh!")

model = joblib.load(LOCAL_MODEL_PATH)
n_features = model.support_vectors_.shape[1]

def get_closest_hw(pixel_count):
    sqrt_val = int(math.sqrt(pixel_count))
    for h in range(sqrt_val, 0, -1):
        if pixel_count % h == 0:
            return h, pixel_count // h
    return 1, pixel_count

total_pixel = n_features // 3
target_h, target_w = get_closest_hw(total_pixel)

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

def predict_image(file_bytes):
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
        image = image.resize((target_w, target_h))

        img_array = np.asarray(image, dtype=np.uint8)
        if img_array.shape != (target_h, target_w, 3):
            raise ValueError(f"❌ Shape salah: {img_array.shape}, harus ({target_h}, {target_w}, 3)")

        flat = img_array.flatten().reshape(1, -1)
        if flat.shape[1] != n_features:
            raise ValueError(f"❌ Jumlah fitur salah: {flat.shape[1]} ≠ {n_features}")

        prediction = model.predict(flat)[0]
        pred_label = label_mapping.get(prediction, "Unknown")
        dropbox_color = dropbox_color_mapping.get(pred_label, "Tidak Ada")

        return pred_label, dropbox_color

    except Exception as e:
        raise ValueError(f"❌ Gagal memproses gambar: {str(e)}")
