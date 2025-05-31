# app/main.py
import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.model_loader import get_model
from app.firebase_helper import save_scan_result, upload_image_to_storage
from PIL import Image
import numpy as np
from io import BytesIO

# Inisialisasi FastAPI
app = FastAPI(
    title="TOXMAP - B3 Waste Classifier",
    description="API klasifikasi sampah B3 dengan Firebase Storage dan Firestore",
    version="1.0"
)

# Konfigurasi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

@app.get("/")
async def root():
    return {"message": "TOXMAP Backend is running!"}

@app.post("/predict/")
async def predict(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        # Validasi file
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="❌ Format tidak valid. Gunakan jpg/jpeg/png.")

        file_data = await file.read()
        unique_filename = f"{uuid.uuid4()}.jpg"
        image_url = upload_image_to_storage(file_data, unique_filename)

        # Proses gambar
        image = Image.open(BytesIO(file_data)).convert("RGB")
        model = get_model()
        n_features = model.support_vectors_.shape[1]

        img_array = np.asarray(image.resize((int((n_features // 3) ** 0.5), int((n_features // 3) ** 0.5))), dtype=np.uint8)
        flat = img_array.flatten().reshape(1, -1)

        prediction = model.predict(flat)[0]

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

        result_label = label_mapping.get(prediction, "Unknown")
        dropbox_color = dropbox_color_mapping.get(result_label, "Tidak Ada")

        save_scan_result(user_id, result_label, dropbox_color, image_url)

        return {
            "result": result_label,
            "dropbox_color": dropbox_color,
            "image_url": image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")
