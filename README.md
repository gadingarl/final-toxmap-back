# TOXMAP Backend

TOXMAP adalah proyek tugas akhir yang mengembangkan sistem klasifikasi sampah berbahaya dan beracun (B3) berbasis citra menggunakan model Machine Learning.  
Backend ini dibangun dengan **FastAPI** dan terintegrasi dengan **Firebase** untuk penyimpanan gambar dan hasil klasifikasi.

## 🚀 Fitur

- Klasifikasi gambar sampah menggunakan model SVM (.pkl)
- Upload gambar ke Firebase Storage
- Simpan hasil klasifikasi ke Firestore (riwayat pengguna)
- Dukungan autentikasi Firebase Admin SDK
- Endpoint REST API yang terstruktur dan mudah diakses

## 🧱 Struktur Proyek

```
app/
├── main.py              # Endpoint FastAPI untuk prediksi
├── model_loader.py      # Load model SVM dari Firebase Storage
├── firebase_helper.py   # Integrasi Firestore dan Storage
├── utils.py             # (Opsional) Fungsi pendukung
└── __init__.py          # (Opsional) Penanda package
```

## ⚙️ Konfigurasi Environment

Set environment variables berikut di platform deployment seperti **Render**:

| Variable                       | Keterangan                                          |
|-------------------------------|-----------------------------------------------------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path ke file JSON Service Account Firebase        |
| `MODEL_URL`                   | Link publik ke file `.pkl` di Firebase Storage      |

## 📦 Instalasi Dependency

```bash
pip install -r requirements.txt
```

Contoh isi `requirements.txt`:

```
fastapi==0.110.0
uvicorn==0.29.0
python-multipart==0.0.9
firebase-admin==6.4.0
scikit-learn==1.4.0
joblib==1.3.2
opencv-python-headless==4.9.0.80
pillow==10.2.0
numpy==1.24.4
requests==2.31.0
```

## 🏁 Menjalankan Secara Lokal

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Pastikan file `serviceAccountKey.json` disiapkan dan `MODEL_URL` diset secara lokal.

## 🔗 Endpoint API

| Method | Endpoint       | Deskripsi                              |
|--------|----------------|----------------------------------------|
| GET    | `/`            | Cek apakah backend berjalan            |
| GET    | `/test-model`  | Tes apakah model berhasil diload       |
| POST   | `/predict`     | Kirim gambar dan terima hasil klasifikasi |

### Contoh `POST /predict` (form-data)

- `user_id`: ID pengguna
- `file`: Gambar (.jpg / .png)

Respons:
```json
{
  "result": "Aerosol",
  "dropbox_color": "Kuning",
  "image_url": "https://..."
}
```

## 🔐 Setup Firebase

1. Buat project di [Firebase Console](https://console.firebase.google.com/)
2. Aktifkan Firestore & Storage
3. Buat Service Account dan unduh file JSON
4. Upload JSON ke Render sebagai Secret File
5. Set path ke `GOOGLE_APPLICATION_CREDENTIALS`

## 👨‍💻 Developer

Dikembangkan oleh **bbb**  
Program Studi Teknik Telekomunikasi, Telkom University  
Tugas Akhir 2025 — Sistem Klasifikasi Sampah B3 Berbasis Citra
