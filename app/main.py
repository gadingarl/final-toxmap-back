import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_admin import credentials, firestore, initialize_app
from app.model_loader import predict_image
from app.firebase_helper import save_scan_result, upload_image_to_storage

# Inisialisasi Firebase dari secret file
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not cred_path:
    raise Exception("❌ GOOGLE_APPLICATION_CREDENTIALS tidak ditemukan.")
cred = credentials.Certificate(cred_path)
initialize_app(cred)
db = firestore.client()

# Init FastAPI
app = FastAPI(
    title="TOXMAP - B3 Waste Classifier",
    description="API klasifikasi sampah B3 dengan Firebase Storage dan Firestore",
    version="1.0"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Sesuaikan jika mau lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

@app.get("/")
async def root():
    return {"message": "TOXMAP Backend Ready!"}

@app.post("/predict/")
async def predict(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
        file_extension = file.filename.split('.')[-1].lower()
        if file_extension not in ALLOWED_EXTENSIONS or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="❌ Format tidak valid. Gunakan jpg/jpeg/png.")

        file_data = await file.read()
        unique_filename = f"{uuid.uuid4()}.jpg"
        image_url = upload_image_to_storage(file_data, unique_filename)

        result_label, dropbox_color = predict_image(file_data)
        save_scan_result(user_id, result_label, dropbox_color, image_url)

        return {
            "result": result_label,
            "dropbox_color": dropbox_color,
            "image_url": image_url
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")

@app.get("/scan-history/{user_id}")
async def get_scan_history(user_id: str):
    try:
        scans = db.collection("scan_history").where(
            "user_id", "==", user_id).order_by(
            "timestamp", direction=firestore.Query.DESCENDING).stream()

        results = []
        for doc in scans:
            data = doc.to_dict()
            data["scan_id"] = doc.id
            results.append(data)

        return {"history": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Gagal mengambil data: {str(e)}")
