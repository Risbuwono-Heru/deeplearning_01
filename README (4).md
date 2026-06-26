# 🩻 Klasifikasi Skoliosis X-Ray

Aplikasi web Streamlit untuk mendeteksi skoliosis pada citra X-Ray tulang belakang
menggunakan Transfer Learning (ResNet50 & DenseNet121).

---

## 📁 Struktur Proyek

```
.
├── app.py                  # Aplikasi utama Streamlit
├── requirements.txt        # Dependensi Python
├── README.md               # Dokumentasi ini
└── models/                 # (opsional) simpan file .keras di sini jika lokal
    ├── best_resnet50.keras
    └── best_densenet121.keras
```

---

## ⚙️ Setup & Menjalankan Lokal

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Konfigurasi bobot model

Buka `app.py` dan edit bagian `GDRIVE_IDS`:

```python
GDRIVE_IDS = {
    "ResNet50":     "FILE_ID_DARI_GOOGLE_DRIVE",   # ← ganti ini
    "DenseNet121":  "FILE_ID_DARI_GOOGLE_DRIVE",   # ← ganti ini
}
```

**Cara mendapatkan File ID Google Drive:**
1. Upload file `.keras` ke Google Drive
2. Klik kanan → *Get link* → ubah akses ke **Anyone with the link**
3. Salin ID dari URL: `https://drive.google.com/file/d/**<FILE_ID>**/view`

**Alternatif lokal:** Taruh file `.keras` di folder `models/` — aplikasi akan otomatis mendeteksinya.

### 3. Jalankan aplikasi

```bash
streamlit run app.py
```

---

## 🚀 Deploy ke Streamlit Community Cloud

1. Push repo ini ke GitHub (pastikan `app.py` dan `requirements.txt` ada di root)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo → set `app.py` sebagai entry point
4. Klik **Deploy**

> **Catatan:** File model tidak boleh di-push ke GitHub (ukuran terlalu besar).
> Gunakan Google Drive (`GDRIVE_IDS`) agar model diunduh otomatis saat deploy.

---

## 🧠 Arsitektur Model

| Arsitektur | Base Model | Output | Loss |
|---|---|---|---|
| ResNet50 | ResNet50 (ImageNet) + GAP + Dropout(0.3) + Dense(1, sigmoid) | Binary | Binary Crossentropy |
| DenseNet121 | DenseNet121 (ImageNet) + GAP + Dropout(0.3) + Dense(1, sigmoid) | Binary | Binary Crossentropy |

**Kelas:** `Normal` (0) vs `Scoliosis` (1) — threshold 0.5

---

## 📊 Fitur Aplikasi

- ✅ Pilih arsitektur model (ResNet50 / DenseNet121) via sidebar
- ✅ Upload citra X-Ray (JPG/PNG)
- ✅ Prediksi kelas + confidence score
- ✅ Visualisasi Grad-CAM (heatmap + overlay)
- ✅ Auto-download bobot dari Google Drive
- ✅ Penanganan error saat model gagal dimuat
