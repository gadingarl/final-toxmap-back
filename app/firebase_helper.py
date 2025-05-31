import os
import json
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Ambil isi service account dari environment variable
firebase_key_str = os.environ.get("FIREBASE_SERVICE_KEY")
if not firebase_key_str:
    raise ValueError("Environment variable FIREBASE_SERVICE_KEY tidak ditemukan.")

# Ubah string JSON menjadi dict
cred_dict = json.loads(firebase_key_str)
cred = credentials.Certificate(cred_dict)

# Inisialisasi Firebase sekali saja
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'toxmap-b74f4.appspot.com'
    })

# Firebase clients
db = firestore.client()
bucket = storage.bucket()

# Fungsi untuk menyimpan hasil scan ke Firestore
def save_scan_result(user_id, result_label, dropbox_color, image_url=""):
    import uuid
    doc_ref = db.collection("scan_history").document(str(uuid.uuid4()))
    doc_ref.set({
        "user_id": user_id,
        "result": result_label,
        "dropbox_color": dropbox_color,
        "timestamp": firestore.SERVER_TIMESTAMP,
        "image_url": image_url
    })

# Fungsi untuk mengupload gambar ke Firebase Storage
def upload_image_to_storage(file_bytes, filename):
    blob = bucket.blob(f"scan_images/{filename}")
    blob.upload_from_string(file_bytes, content_type="image/jpeg")
    blob.make_public()
    return blob.public_url
