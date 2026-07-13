# RadarUMKM Bogor — API

> Flask ML API untuk prediksi daya tarik produk UMKM Bogor **+ fitur Paling Digemari**.

🚀 **Live:** [radarumkmbogor-api.onrender.com](https://radarumkmbogor-api.onrender.com)

---

## Deskripsi

REST API berbasis Flask yang menyajikan model Machine Learning (**Random Forest v3**) untuk memprediksi apakah suatu produk berpotensi **menarik** atau **kurang menarik** di marketplace, sekaligus menampilkan **produk paling digemari** dan **insight pasar keseluruhan** berdasarkan kategori dan sub-kategori yang diinputkan.

Model dilatih dari dataset **1.027 produk** UMKM Kota & Kabupaten Bogor yang dikumpulkan dari Tokopedia, Shopee, dan Lazada.

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

Cek status server, model, dan versi.

```bash
curl https://radarumkmbogor-api.onrender.com/health
```

Response:

```json
{
  "status": "ok",
  "model": "model_umkm_bogor_v3.joblib",
  "dataset_rows": 1027,
  "versi": "v3",
  "fitur_baru": ["jumlah_log", "revenue_proxy_log", "popularity_score", "produk_terpopuler"],
  "message": "Flask ML API v3 siap ✅"
}
```

---

### POST /predict

Prediksi daya tarik produk + insight pasar + produk paling digemari.

```bash
curl -X POST https://radarumkmbogor-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nama_produk"  : "Lapis Talas Bogor Original",
    "kategori"     : "Makanan",
    "sub_kategori" : "Kue & Roti",
    "harga_produk" : 55000
  }'
```

#### Request Body

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `nama_produk` | string | ✅ | Nama lengkap produk (harus mengandung identitas Bogor) |
| `kategori` | string | ✅ | Salah satu: `Makanan`, `Minuman`, `Pakaian & Fashion`, `Aksesoris & Souvenir` |
| `sub_kategori` | string | ✅ | Sub-kategori produk (lihat daftar di bawah) |
| `harga_produk` | integer | ✅ | Harga dalam rupiah (harus > 0) |

#### Sub-Kategori yang Tersedia

| Kategori | Sub-Kategori |
|---|---|
| Makanan | `Camilan & Snack`, `Kue & Roti`, `Lauk & Bahan Makanan`, `Makanan Tradisional` |
| Minuman | `Kopi`, `Teh`, `Minuman Tradisional` |
| Pakaian & Fashion | `Atasan & Pakaian Kasual`, `Pakaian Tradisional`, `Pakaian Anak` |
| Aksesoris & Souvenir | `Aksesoris & Souvenir` |

#### Response

```json
{
  "status"              : "success",
  "kesimpulan"          : "🌟 SANGAT MENARIK — Peluang laku 74.5%",
  "peluang_laku_persen" : 74.5,
  "alasan"              : [
    "Terdapat 4 produk serupa di marketplace.",
    "Harga Anda kompetitif, hanya 12% di bawah median pasar (Rp51.000).",
    "Produk sejenis terbukti laku keras (rata-rata 320 terjual).",
    "Model menilai kombinasi nama, kategori, dan harga sangat sesuai tren."
  ],

  "konteks_harga": {
    "median_pasar"   : 51000,
    "rasio_vs_pasar" : 0.88,
    "segmen"         : "Menengah",
    "selisih_persen" : -12.1
  },

  "kompetitor": [
    {
      "nama"            : "Lapis Talas Bogor Sangkuriang Blackforest",
      "harga"           : 51000,
      "rating"          : 5.0,
      "terjual"         : 425,
      "marketplace"     : "shopee",
      "url_produk"      : "https://shopee.co.id/...",
      "kemiripan_persen": 82.3
    }
  ],

  "produk_terpopuler": {
    "label"    : "Top 5 Produk Paling Digemari di 'Makanan — Kue & Roti'",
    "deskripsi": "Produk-produk yang paling diminati berdasarkan kombinasi jumlah penjualan dan rating.",
    "produk"   : [
      {
        "nama"            : "Lapis Talas Bogor Sangkuriang Blackforest",
        "kategori"        : "Makanan",
        "sub_kategori"    : "Kue & Roti",
        "harga"           : 51000,
        "jumlah_terjual"  : 425,
        "rating"          : 5.0,
        "popularity_score": 30.27,
        "marketplace"     : "shopee",
        "url_produk"      : "https://shopee.co.id/...",
        "nama_toko"       : "Anum Sari Snack"
      }
    ]
  },

  "insight_pasar": {
    "narasi"                    : "Secara keseluruhan, produk yang paling banyak diminati adalah kategori 'Makanan' dengan total 45.230 penjualan...",
    "kategori_terpopuler"       : "Makanan",
    "sub_kategori_terpopuler"   : "Kue & Roti",
    "posisi_kategori_anda"      : 1,
    "total_kategori"            : 4,
    "ranking_semua_kategori"    : [
      { "rank": 1, "kategori": "Makanan",   "total_terjual": 45230, "avg_rating": 4.85 },
      { "rank": 2, "kategori": "Minuman",   "total_terjual": 32100, "avg_rating": 4.90 },
      { "rank": 3, "kategori": "Pakaian & Fashion", "total_terjual": 28000, "avg_rating": 4.72 },
      { "rank": 4, "kategori": "Aksesoris & Souvenir", "total_terjual": 5400, "avg_rating": 4.88 }
    ],
    "top5_sub_kategori_global"  : [...],
    "sub_kategori_dalam_kategori_ini": [...]
  }
}
```

---

## Cara Kerja Model

### Pipeline Machine Learning (v3)

```
Input Pengguna
│
├─ nama_produk  → [Text Cleaning] → TF-IDF (1000 fitur, unigram+bigram)
├─ kategori     → One-Hot Encoding
├─ sub_kategori → One-Hot Encoding
└─ harga_produk → Feature Engineering
                    ├─ rasio_harga       = harga / median_kategori
                    ├─ zscore_harga      = (harga - mean) / std dalam kategori
                    ├─ log_harga         = log1p(harga)
                    ├─ segmen_harga      = 0(murah) / 1(menengah) / 2(premium)
                    ├─ jumlah_log        = log1p(jumlah_terjual estimasi kompetitor)
                    ├─ revenue_proxy_log = log1p(harga × jumlah estimasi)
                    └─ popularity_score  = rating × log1p(jumlah estimasi)
                              ↓
                    StandardScaler
                              ↓
               RandomForestClassifier (tuned GridSearchCV)
                              ↓
              Output: Probabilitas laku (0.0 – 1.0)
```

### Strategi Labeling

Dataset **tidak memiliki label eksplisit**. Label dibuat secara otomatis:

```
label = 1 (Menarik)        jika jumlah_terjual > median(jumlah_terjual)
label = 0 (Kurang Menarik) jika jumlah_terjual ≤ median(jumlah_terjual)
```

Median jumlah_terjual ≈ **34 unit** → distribusi hampir seimbang (50.1% vs 49.9%).

### Fitur Utama

| Fitur | Deskripsi | Kenapa Penting |
|---|---|---|
| `rasio_harga` | harga / median pasar kategori | Model tahu apakah harga "mahal" atau "murah" secara RELATIF |
| `zscore_harga` | deviasi harga dari mean kategori | Mendeteksi harga yang sangat ekstrem |
| `popularity_score` | rating × log1p(jumlah_terjual) | Gabungkan kualitas dan volume penjualan |
| `revenue_proxy_log` | log(harga × jumlah_estimasi) | Proxy estimasi omzet produk |
| TF-IDF nama produk | representasi teks nama | Model belajar dari kata-kata produk populer |

### Fitur Paling Digemari

Setelah prediksi, sistem otomatis menghitung **top-5 produk paling digemari** dalam kategori & sub-kategori yang sama berdasarkan:

```
popularity_score = rating × log1p(jumlah_terjual)
```

Rumus ini menggabungkan:
- **Kualitas** produk (rating pelanggan)
- **Volume** penjualan (jumlah terjual)
- **Efek diminishing returns** — produk dengan 1000 terjual tidak 10× lebih populer dari 100 terjual

---

## Stack Teknologi

| Komponen | Teknologi |
|---|---|
| Framework | Flask 3.0 + Flask-CORS |
| Model | Random Forest (scikit-learn, GridSearchCV tuned) |
| Text Processing | PySastrawi (Stemmer Bahasa Indonesia) |
| Similarity Search | TF-IDF + Cosine Similarity |
| Dataset | 1.027 produk UMKM Bogor (Tokopedia, Shopee, Lazada) |
| Model File | `models/model_umkm_bogor_v3.joblib` |
| Market Stats | `data/market_stats_v3.csv`, `data/market_stats_sub_kategori_v3.csv` |

---

## Struktur File

```
RadarUMKMBogor-API/
├── app.py                              # Flask API v3
├── requirements.txt
├── README.md
├── data/                               # Dataset & Statistik Pasar
│   ├── processed/
│   │   └── dataset_preprocessed.csv    # Dataset bersih (1027 produk)
│   ├── raw/                            # Dataset original (sebelum preprocessing)
│   ├── market_stats_v3.csv             # Statistik pasar per kategori
│   └── market_stats_sub_kategori_v3.csv# Statistik pasar per sub_kategori
├── models/                             # File Model Machine Learning
│   ├── model_umkm_bogor_v3.joblib      # Model Random Forest v3
│   └── preprocessors/                  # File pendukung transformasi (kalau ada)
├── notebooks/                          # Eksperimen, Training & Perbandingan Model
│   ├── retrain_model.ipynb             # Notebook training model v3
│   ├── compare_models.ipynb            # Komparasi Random Forest vs Logistic Regression
│   └── data_preprocessing.ipynb        # Notebook preprocessing data
└── docs/                               # Dokumentasi
    └── random_forest_rules.md          # Visualisasi logika & aturan klasifikasi
```

---

## Menjalankan Lokal

```bash
pip install -r requirements.txt

# Jalankan training dulu (jika model belum ada)
jupyter notebook notebooks/retrain_model.ipynb

# Jalankan API
python app.py
```

Server berjalan di `http://localhost:5000`.

---

## Changelog

| Versi | Perubahan |
|---|---|
| v1 | Model baseline Random Forest, fitur: harga absolut + rating |
| v2 | Fitur harga relatif pasar: rasio_harga, zscore_harga, log_harga, segmen_harga |
| v3 | Dataset 1027 produk, fitur baru: jumlah_log, revenue_proxy_log, popularity_score. Tambah **fitur Paling Digemari** di response `/predict` |

---

## Lisensi

Repositori ini dibuat untuk keperluan akademik program MBKM.
