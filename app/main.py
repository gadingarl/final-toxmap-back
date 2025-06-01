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
import uvicorn

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
        print(">>> /predict HIT")
        
        file_extension = file.filename.split('.')[-1].lower()
        print(f">>> File received: {file.filename}, ext: {file_extension}")

        if file_extension not in ALLOWED_EXTENSIONS or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="❌ Format tidak valid. Gunakan jpg/jpeg/png.")

        file_data = await file.read()
        print(f">>> File size: {len(file_data)} bytes")

        unique_filename = f"{uuid.uuid4()}.jpg"
        print(f">>> Uploading to Firebase Storage as: {unique_filename}")

        image_url = upload_image_to_storage(file_data, unique_filename)
        print(f">>> Image URL: {image_url}")

        image = Image.open(BytesIO(file_data)).convert("RGB")
        print(f">>> Image converted to RGB")

        model = get_model()
        print(">>> Model loaded")

        n_features = model.support_vectors_.shape[1]
        print(f">>> n_features: {n_features}")

        img_array = np.asarray(image.resize((int((n_features // 3) ** 0.5), int((n_features // 3) ** 0.5))), dtype=np.uint8)
        flat = img_array.flatten().reshape(1, -1)

        prediction = model.predict(flat)[0]
        print(f">>> Prediction: {prediction}")

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
        print(">>> Scan result saved")

        return {
            "result": result_label,
            "dropbox_color": dropbox_color,
            "image_url": image_url
        }

    except Exception as e:
        print(f"❌ ERROR in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")

# Untuk dijalankan di Render
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     uvicorn.run("app.main:app", host="0.0.0.0", port=port)

