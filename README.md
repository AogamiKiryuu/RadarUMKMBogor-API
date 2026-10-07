# RadarUMKM Bogor — API

> REST ML Microservice berbasis Flask untuk Prediksi Daya Tarik Produk UMKM Bogor, Analisis Posisi Harga Pasar, Pencarian Kompetitor, dan Pemetaan Tren Komoditas Marketplace.

🚀 **Production Live:** [radarumkmbogor-api.onrender.com](https://radarumkmbogor-api.onrender.com)  
📦 **Model Aktif:** `models/model_umkm_bogor_v4.joblib` (Random Forest Classifier v4 — Fair Sektoral & Bebas Target Leakage)

---

## 📌 Ringkasan Sistem

Layanan ini menyajikan mesin inferensi Machine Learning dan Text Mining untuk mendukung pengambilan keputusan bisnis pelaku UMKM di Kota dan Kabupaten Bogor:
1. **Prediksi Peluang Laku (Random Forest v4):** Mengestimasi probabilitas daya tarik produk berdasarkan teks nama, kategori/sub-kategori, dan profil harga relatif pasar.
2. **Pencarian Kompetitor Serupa (TF-IDF + Cosine Similarity):** Mengidentifikasi top-6 produk pesaing terdekat di pasar marketplace untuk mengatasi *cold-start problem* produk baru.
3. **Business Rules & Guardrails:** Validasi harga deterministik dan koreksi probabilitas jika rasio harga berada jauh di luar kewajaran pasar kategori.
4. **Insight Tren Pasar & Produk Terpopuler:** Menyajikan peringkat popularitas produk dan urutan minat konsumen lintas kategori.

Model dilatih dari **1.027 produk** UMKM Bogor yang dikumpulkan melalui *web scraping* dari Tokopedia, Shopee, dan Lazada (kondisi 100% bersih, tanpa duplikasi dan tanpa missing values).

---

## 🚀 Deployment

Dihosting di **Render** sebagai Python Web Service (Gunicorn / Flask):

| Parameter | Keterangan |
|---|---|
| **Base URL** | `https://radarumkmbogor-api.onrender.com` |
| **Health Check** | `GET /health` |
| **Inference Endpoint** | `POST /predict` |
| **Spesifikasi Runtime** | Python 3.12 / 3.13, Scikit-Learn 1.6+, PySastrawi |

> ℹ️ *Catatan Free Tier Render:* Jika server tidak menerima request dalam jangka waktu tertentu, server akan masuk ke mode *idle*. Request pertama membutuhkan waktu *cold start* sekitar 30 detik.

---

## 📡 API Endpoints

### 1. GET `/health`
Pemeriksaan status server, nama model yang aktif, jumlah baris dataset, dan kategori tren.

```bash
curl https://radarumkmbogor-api.onrender.com/health
```

#### Response (HTTP 200 OK):
```json
{
  "status": "ok",
  "model": "model_umkm_bogor_v4.joblib",
  "dataset_rows": 1027,
  "kategori_tren": [
    "Makanan",
    "Pakaian & Fashion",
    "Minuman",
    "Aksesoris & Souvenir"
  ],
  "top_kategori": "Makanan",
  "message": "Flask ML API v4 siap digunakan ✅ (Model Fair Per Kategori & Bebas Leakage)"
}
```

---

### 2. POST `/predict`
Endpoint utama untuk mengeksekusi inferensi daya tarik produk baru, pencarian kompetitor, dan analisis harga.

```bash
curl -X POST https://radarumkmbogor-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nama_produk"  : "Lapis Talas Bogor Sangkuriang Original Keju",
    "kategori"     : "Makanan",
    "sub_kategori" : "Kue & Roti",
    "harga_produk" : 45000
  }'
```

#### Request Payload:
| Field | Tipe | Wajib | Keterangan & Validasi |
|---|---|:---:|---|
| `nama_produk` | String | ✅ | Nama produk (harus memuat identitas Bogor / padanan lokal) |
| `kategori` | String | ✅ | Salah satu dari 4 kategori: `Makanan`, `Minuman`, `Pakaian & Fashion`, `Aksesoris & Souvenir` |
| `sub_kategori` | String | ✅ | Sub-kategori yang sesuai dengan kategori induk (lihat daftar di bawah) |
| `harga_produk` | Number | ✅ | Harga jual dalam Rupiah (harus bertanda positif $> 0$) |

#### Daftar Kategori & Sub-Kategori Valid:
- **Makanan:** `Camilan & Snack`, `Kue & Roti`, `Lauk & Bahan Makanan`, `Makanan Tradisional`
- **Minuman:** `Kopi`, `Teh`, `Minuman Tradisional`
- **Pakaian & Fashion:** `Atasan & Pakaian Kasual`, `Pakaian Tradisional`, `Pakaian Anak`
- **Aksesoris & Souvenir:** `Aksesoris & Souvenir`

#### Response Output (HTTP 200 OK):
```json
{
  "status": "success",
  "kesimpulan": "🌟 SANGAT MENARIK — Peluang laku 83.2%",
  "peluang_laku_persen": 83.2,
  "alasan": [
    "Terdapat 6 produk serupa di marketplace.",
    "Harga Anda (Rp45,000) kompetitif, sekitar 11.8% di bawah median pasar (Rp51,000).",
    "Produk sejenis di kategori ini terbukti memiliki perputaran pasar yang sehat.",
    "Model Random Forest v4 menilai teks nama dan posisi harga sangat bersaing di industri Makanan."
  ],
  "konteks_harga": {
    "median_pasar": 51000,
    "rasio_vs_pasar": 0.88,
    "segmen": "Menengah",
    "selisih_persen": -11.8
  },
  "kompetitor": [
    {
      "nama": "Lapis Talas Bogor Sangkuriang Original Keju 500gr",
      "harga": 51000,
      "rating": 4.9,
      "terjual": 450,
      "marketplace": "Shopee",
      "url_produk": "https://shopee.co.id/...",
      "kemiripan_persen": 84.5
    }
  ],
  "produk_terpopuler": {
    "label": "Top 5 Produk Paling Digemari di 'Makanan — Kue & Roti'",
    "deskripsi": "Produk-produk yang paling diminati berdasarkan volume penjualan dan rating tertinggi.",
    "produk": [...]
  },
  "insight_pasar": {
    "narasi": "Secara keseluruhan, produk yang paling banyak diminati pembeli adalah kategori 'Makanan'...",
    "kategori_terpopuler": "Makanan",
    "sub_kategori_terpopuler": "Kue & Roti",
    "posisi_kategori_anda": 1,
    "total_kategori": 4,
    "ranking_semua_kategori": [...],
    "top5_sub_kategori_global": [...],
    "sub_kategori_dalam_kategori_ini": [...]
  }
}
```

---

## 🧠 Metodologi & Arsitektur Model v4

### 1. Pembentukan Target (Stratifikasi Intra-Kategori)
Berbeda dengan Model v3 lama yang menggunakan batas median global 34 unit (yang menimbulkan bias berat terhadap produk busana), **Model v4** menggunakan persentil penjualan intra-kategori (*category-stratified ranking*):

$$\text{rank\_pct}(produk \mid kategori) = \frac{\text{peringkat penjualan produk}}{\text{total produk kategori}}$$

$$\text{label} = \begin{cases} 1 \text{ (Menarik)}, & \text{jika } \text{rank\_pct} > 0.50 \\ 0 \text{ (Kurang Menarik)}, & \text{jika } \text{rank\_pct} \le 0.50 \end{cases}$$

- **Makanan:** 208 (0) vs 208 (1) $\rightarrow$ 50% : 50%
- **Minuman:** 85 (0) vs 86 (1) $\rightarrow$ 50% : 50%
- **Pakaian & Fashion:** 207 (0) vs 208 (1) $\rightarrow$ 50% : 50%
- **Aksesoris & Souvenir:** 12 (0) vs 13 (1) $\rightarrow$ 50% : 50%
- **Total Global:** 512 (0) vs 515 (1) $\rightarrow$ Rasio keseimbangan: **0.994** (sempurna).

### 2. Eliminasi Target Leakage
Pada Model v4, fitur `jumlah_log`, `revenue_proxy_log`, dan `popularity_score_new` **dicabut sepenuhnya dari input model klasifikasi**. Input model hanya mengandalkan fitur intrinsik teks, identitas kategori, dan profil harga relatif:

```
Masukan Model (8 kolom DataFrame → 1.019 dimensi fitur):
├── Teks (1 kolom)        : 'nama_produk_clean' → TfidfVectorizer (1.000 fitur unigram & bigram)
├── Kategorikal (2 kolom) : ['kategori', 'sub_kategori'] → OneHotEncoder (14 fitur)
└── Numerik (5 kolom)     : ['rasio_harga', 'zscore_harga', 'log_harga', 'segmen_harga', 'rating']
                            → StandardScaler (5 fitur)
```

### 3. Hasil Evaluasi Ilmiah (Holdout Test 206 Sampel Independen)
- **Akurasi:** **67.48%**
- **Precision:** **68.00%**
- **Recall:** **66.02%**
- **F1-Score:** **66.99%**
- **ROC-AUC:** **0.7242**
- **5-Fold Cross Validation:** **62.73%** ($\pm 3.12\%$)
- **Top 5 Feature Importance:** `zscore_harga` (3.56%), `rasio_harga` (3.28%), `log_harga` (3.10%), `rating` (2.95%), `lapis` (1.45%).

> Model v4 menghasilkan metrik ilmiah yang realistis, berbobot, dan bebas dari tuduhan *overfitting/leakage* pada pengujian akademik.

---

## 📂 Struktur Repositori

```
RadarUMKMBogor-API/
├── app.py                                   # Entrypoint Flask API v4 & logic inference
├── train_model_v4.py                        # Script otomatisasi training Model v4
├── requirements.txt                         # Dependensi library Python
├── README.md                                # Dokumentasi resmi API & Model v4
├── tren_per_kategori.csv                    # Data agregasi tren penjualan per kategori
├── tren_per_sub_kategori.csv                # Data agregasi tren penjualan per sub-kategori
├── data/
│   ├── processed/
│   │   └── dataset_preprocessed.csv         # Dataset bersih (1.027 produk UMKM Bogor)
│   ├── market_stats_v4.csv                  # Statistik pasar median/mean/std per kategori
│   └── market_stats_sub_kategori_v4.csv     # Statistik pasar per sub-kategori
├── models/
│   └── model_umkm_bogor_v4.joblib           # Artefak biner Pipeline Random Forest v4
└── notebooks/
    └── audit_model_v4_evaluation.ipynb      # Notebook evaluasi resmi Model v4 & visualisasi
```

---

## 💻 Menjalankan Secara Lokal

```bash
# 1. Clone repositori
git clone https://github.com/AogamiKiryuu/RadarUMKMBogor-API.git
cd RadarUMKMBogor-API

# 2. Buat virtual environment & install dependensi
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# 3. Jalankan server Flask
python app.py
```

Server akan aktif secara lokal di `http://localhost:5000`.

---

## 📜 Riwayat Pembaruan (Changelog)

- **v4 (Oktober 2026 - Rilis Resmi Skripsi):**
  - Pembaruan strategi labeling berbasis persentil 50% intra-kategori (`rank_pct > 0.50`), menjamin keadilan 50:50 di seluruh sektor (menghilangkan bias pakaian).
  - Eliminasi total target leakage (`jumlah_log`, `revenue_proxy_log`, `popularity_score_new` dicabut dari input model).
  - Peningkatan ensemble pohon: `n_estimators = 200`, `class_weight = 'balanced'`.
  - Akurasi realistis: 67.48%, ROC-AUC: 0.7242, 5-Fold CV: 62.73%.
  - Pembersihan file model dan script lama.
- **v3 (Versi Transisi):** Pengenalan fitur harga relatif, threshold global 34 unit.
- **v2 (Historis):** Eksperimen skor komposit terjual dan rating.
- **v1 (Baseline):** Model Random Forest awal berbasis harga absolut.

---

## 🎓 Lisensi & Afiliasi

Dikembangkan sebagai bagian dari penelitian tugas akhir / program MBKM untuk digitalisasi dan pemberdayaan pelaku UMKM lokal di Kota & Kabupaten Bogor.
