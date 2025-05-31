from PIL import Image
from io import BytesIO
import numpy as np
import joblib
import requests
import os
import math

# ✅ Link ke model fix terbaru di Firebase Storage
MODEL_URL = "https://firebasestorage.googleapis.com/v0/b/toxmap-b74f4.firebasestorage.app/o/svm_model_fix.pkl?alt=media&token=6f44d9f9-0284-4286-bc09-723030cbdc9f"

# Nama file lokal cache-nya
LOCAL_MODEL_PATH = "svm_model_fix.pkl"

# Download model kalau belum ada
if not os.path.exists(LOCAL_MODEL_PATH):
    print("⬇️ Mengunduh model dari Firebase Storage...")
    response = requests.get(MODEL_URL)
    with open(LOCAL_MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("✅ Model berhasil diunduh!")

# Load model dari file
model = joblib.load(LOCAL_MODEL_PATH)
n_features = model.support_vectors_.shape[1]

# Hitung ukuran gambar target berdasarkan jumlah fitur
def get_closest_hw(pixel_count):
    sqrt_val = int(math.sqrt(pixel_count))
    for h in range(sqrt_val, 0, -1):
        if pixel_count % h == 0:
            w = pixel_count // h
            return h, w
    return 1, pixel_count

total_pixel = n_features // 3
target_h, target_w = get_closest_hw(total_pixel)

# Label dan mapping warna dropbox
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

# Fungsi prediksi gambar
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
