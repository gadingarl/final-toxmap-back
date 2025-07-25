from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from firebase_helper import init_firebase, save_scan_result, upload_image_to_storage, db
from model_loader import model, n_features, predict_image, label_mapping
import uuid

app = FastAPI(
    title="TOXMAP - B3 Waste Classifier",
    description="API klasifikasi sampah B3",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_firebase()

@app.get("/")
async def root():
    return {"message": "TOXMAP Backend Ready!"}

@app.get("/model-info")
async def model_info():
    return {
        "expected_features": n_features,
        "target_dimensions": "224x224",
        "model_type": type(model).__name__,
        "label_mapping": label_mapping
    }

@app.post("/predict/")
async def predict(user_id: str = Form(...), file: UploadFile = File(...)):
    try:
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
            "timestamp").stream()
            
        results = []
        for doc in scans:
            data = doc.to_dict()
            data["scan_id"] = doc.id
            results.append(data)
            
        return {"history": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Gagal mengambil data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=15017)
