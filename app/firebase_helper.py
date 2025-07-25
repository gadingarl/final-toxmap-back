import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, storage
import uuid

bucket = None
db = None

def init_firebase():
    global bucket, db

    service_key_path = os.path.join(os.path.dirname(__file__), '..', 'serviceAccountKey.json')
    with open(service_key_path, 'r') as f:
        cred_dict = json.load(f)

    if not firebase_admin._apps:
        cred = credentials.Certificate(cred_dict)
        BUCKET_NAME = "toxmap-b74f4.appspot.com"
        firebase_admin.initialize_app(cred, {
            'storageBucket': BUCKET_NAME
        })

    bucket = storage.bucket()
    db = firestore.client()

def upload_image_to_storage(file_bytes, filename):
    try:
        blob = bucket.blob(f"scan_images/{filename}")
        blob.upload_from_string(file_bytes, content_type="image/jpeg")
        blob.make_public()
        return blob.public_url
    except Exception as e:
        raise ValueError(f"Failed to upload image: {str(e)}")

def save_scan_result(user_id, result_label, dropbox_color, image_url=""):
    doc_ref = db.collection("scan_history").document(str(uuid.uuid4()))
    doc_ref.set({
        "user_id": user_id,
        "result": result_label,
        "dropbox_color": dropbox_color,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "image_url": image_url
    })
