import os
import uuid
import firebase_admin
from firebase_admin import credentials, firestore, storage

# --- Inisialisasi Firebase ---
cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not cred_path:
    raise Exception("❌ GOOGLE_APPLICATION_CREDENTIALS tidak ditemukan.")
cred = credentials.Certificate(cred_path)

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'toxmap-b74f4.firebasestorage.app'
    })

# --- Koneksi Firestore & Storage ---
db = firestore.client()
bucket = storage.bucket()


# --- Simpan hasil scan ke Firestore ---
def save_scan_result(user_id, result_label, dropbox_color, image_url=""):
    doc_ref = db.collection("scan_history").document(str(uuid.uuid4()))
    doc_ref.set({
        "user_id": user_id,
        "result": result_label,
        "dropbox_color": dropbox_color,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "image_url": image_url
    })


# --- Upload gambar ke Storage (ke folder scan_images/) ---
def upload_image_to_storage(file_bytes, filename):
    blob = bucket.blob(f"scan_images/{filename}")
    blob.upload_from_string(file_bytes, content_type="image/jpeg")
    blob.make_public()
    return blob.public_url
