# RadarUMKMBogor API

> Flask ML API untuk prediksi daya tarik produk UMKM Bogor.

🚀 **Live:** [radarumkmbogor-api.onrender.com](https://radarumkmbogor-api.onrender.com)

---

## Deskripsi

REST API berbasis Flask yang menyajikan model Machine Learning (Random Forest) untuk memprediksi apakah suatu produk berpotensi **menarik** atau **kurang menarik** di marketplace. Model dilatih dari dataset 597 produk UMKM Kota & Kabupaten Bogor.

---

## Deployment

Dihosting di **Render** (Free tier — cold start ±30 detik jika tidak aktif).

| Endpoint | URL |
|---|---|
| Base URL | `https://radarumkmbogor-api.onrender.com` |
| Health Check | `https://radarumkmbogor-api.onrender.com/health` |
| Predict | `https://radarumkmbogor-api.onrender.com/predict` |

---

## API Endpoints

### GET /health

Cek status server dan model.

```bash
curl https://radarumkmbogor-api.onrender.com/health
```

Response:

```json
{
  "status": "ok",
  "dataset_rows": 597,
  "model": "model_umkm_bogor.joblib"
}
```

---

### POST /predict

Prediksi daya tarik produk.

```bash
curl -X POST https://radarumkmbogor-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nama_produk": "Lapis Talas Bogor Original",
    "kategori": "Makanan & Minuman",
    "sub_kategori": "Makanan Khas",
    "harga_produk": 75000,
    "rating": 4.8
  }'
```

Request Body:

| Field | Tipe | Keterangan |
|---|---|---|
| `nama_produk` | string | Nama lengkap produk |
| `kategori` | string | Kategori utama produk |
| `sub_kategori` | string (opsional) | Sub-kategori produk |
| `harga_produk` | integer | Harga dalam rupiah |
| `rating` | float | Target rating (0.0 – 5.0) |

Response:

```json
{
  "prediction_label": 1,
  "prediction_score": 85.5,
  "insight": "Produk ini memiliki potensi tinggi...",
  "competitors": [...]
}
```

---

## Stack

| Komponen | Teknologi |
|---|---|
| Framework | Flask + Flask-CORS |
| Model | Random Forest (scikit-learn) |
| Text Processing | PySastrawi (Stemmer Bahasa Indonesia) |
| Dataset | 597 produk UMKM Bogor (Tokopedia, Shopee, Lazada) |
| Model File | `model_umkm_bogor.joblib` |

---

## Struktur

```
flask_api/
├── app.py                      # Main Flask application
├── model_umkm_bogor.joblib     # Model Random Forest terlatih
├── dataset_umkm_bogor.csv      # Dataset referensi kompetitor
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Menjalankan Lokal (Opsional)

> API sudah tersedia di Render — tidak perlu dijalankan lokal untuk development.

Jika diperlukan untuk development/debugging:

```bash
pip install -r requirements.txt
python app.py
```

Server akan berjalan di `http://localhost:5001`.

---

## Lisensi

Repositori ini dibuat untuk keperluan akademik program MBKM.
