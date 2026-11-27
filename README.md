# RadarUMKM Bogor — API (v5 · IndoBERT ONNX INT8 + LightGBM)

> REST ML Microservice berbasis Flask untuk Prediksi Daya Tarik Produk UMKM Bogor, Analisis Posisi Harga Pasar, Pencarian Kompetitor Semantik, dan Pemetaan Tren Komoditas Marketplace.

🚀 **Production Live:** [radarumkmbogor-api.onrender.com](https://radarumkmbogor-api.onrender.com)  
🌐 **Web App Live:** [radarumkm.vercel.app](https://radarumkm.vercel.app)  
📦 **Engine Aktif:** `v5_indobert_onnx_int8_lightgbm` (**IndoBERT Base-P1 ONNX INT8 + LightGBM**)  
⚡ **Efisiensi Infrastruktur:** RAM Puncak **310,30 MB** (muat di Render Free Tier 512 MB tanpa PyTorch/Transformers) · Latensi Inferensi **~50–280 ms**

---

## 📌 Ringkasan Sistem (v5)

Layanan ini menyajikan mesin inferensi *Natural Language Processing* (NLP) dan *Machine Learning* untuk mendukung pengambilan keputusan bisnis pelaku UMKM di Kota dan Kabupaten Bogor:

1. **Ekstraksi Representasi Teks Semantik (IndoBERT ONNX INT8):** Menggunakan `indobenchmark/indobert-base-p1` yang telah diekspor ke format **ONNX** dan dikuantisasi **Dynamic INT8** (`118,87 MB`, menyusut 4× lipat dari 495,64 MB FP32 dengan fidelitas vektor 99,91%). Mampu membedakan konteks semantik produk secara akurat (misalnya membedakan *"Lapis Talas"* [Makanan] dengan *"Gantungan Kunci Miniatur Lapis Talas"* [Aksesoris & Souvenir]).
2. **Klasifikasi Daya Tarik Pasar (LightGBM — PTQ v5):** Menggabungkan 768 dimensi *embedding* IndoBERT dengan 19 fitur tabular (kategori, sub-kategori, rasio harga relatif, z-score harga, log harga, segmen harga, dan proksi rating *cold-start*).
3. **Pencarian Kompetitor Semantik & Proksi Cold-Start:** Mengidentifikasi top-5 produk pesaing terdekat di pasar marketplace berbasis kemiripan vektor *embedding* IndoBERT yang sudah di-*precompute* (`indobert_embeddings_int8.npy`).
4. **Business Rules & Guardrails:** Validasi kecocokan kategori berbasis prioritas *head-noun*, validasi identitas Bogor (*regex word-boundary*), deteksi produk di luar cakupan dataset (*out-of-domain*), serta koreksi probabilitas untuk harga ekstrem (*outlier*).
5. **Insight Tren Pasar & Produk Terpopuler:** Menyajikan peringkat popularitas produk (`rating × log1p(jumlah_terjual)`) dan urutan minat konsumen lintas kategori/sub-kategori.

Model dilatih dari **1.027 produk** UMKM Bogor yang dikumpulkan melalui *web scraping* dari Tokopedia, Shopee, dan Lazada (100% bersih, tanpa duplikasi dan tanpa *missing values*).

---

## 🚀 Deployment (Render Free Tier 512 MB)

Dihosting di **Render** sebagai Python Web Service (`gunicorn --workers 1 --timeout 120 app:app`):

| Parameter | Keterangan |
|---|---|
| **Base URL** | `https://radarumkmbogor-api.onrender.com` |
| **Health Check** | `GET /health` |
| **Inference Endpoint** | `POST /predict` |
| **Start Command** | `gunicorn --workers 1 --timeout 120 app:app` |
| **Runtime Utama** | Python 3.11+, `onnxruntime`, `tokenizers`, `lightgbm`, `scikit-learn`, `Flask` |
| **Auto-Download ONNX** | `download_onnx.py` otomatis mengunduh `indobert_base_p1_int8.onnx` (118,87 MB) dari GitHub Releases (`v5.0-onnx`) saat boot, dengan *safe fallback* ke v4 jika aset tidak tersedia |
| **Konsumsi RAM Live** | **~310 MB** (margin aman ~200 MB di bawah batas 512 MB Render Free Tier) |

---

## 📡 API Endpoints

### 1. `GET /health`
Pemeriksaan status server, engine model yang aktif, jumlah baris dataset, dan kategori tren.

```bash
curl https://radarumkmbogor-api.onrender.com/health
```

#### Response (HTTP 200 OK):
```json
{
  "status": "ok",
  "engine": "v5_indobert_onnx_int8_lightgbm",
  "model": "model_umkm_bogor_v5_indobert_lgbm.joblib (ONNX INT8)",
  "dataset_rows": 1027,
  "kategori_tren": [
    "Makanan",
    "Aksesoris & Souvenir",
    "Pakaian & Fashion",
    "Minuman"
  ],
  "top_kategori": "Makanan",
  "message": "Flask ML API v5 siap digunakan ✅ (IndoBERT ONNX INT8 + LightGBM & Semantic Cosine)"
}
```

---

### 2. `POST /predict`
Endpoint utama untuk inferensi daya tarik produk baru, pencarian kompetitor semantik, dan analisis posisi harga pasar.

```bash
curl -X POST https://radarumkmbogor-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nama_produk"  : "Gantungan Kunci Miniatur Lapis Talas Bogor",
    "kategori"     : "Aksesoris & Souvenir",
    "sub_kategori" : "",
    "harga_produk" : 15000
  }'
```

#### Request Payload:
| Field | Tipe | Wajib | Keterangan & Validasi |
|---|---|:---:|---|
| `nama_produk` | String | ✅ | Nama produk (harus memuat identitas Bogor / padanan lokal) |
| `kategori` | String | ✅ | Salah satu dari 4 kategori: `Makanan`, `Minuman`, `Pakaian & Fashion`, `Aksesoris & Souvenir` |
| `sub_kategori` | String | Opsional | Sub-kategori yang sesuai dengan kategori induk |
| `harga_produk` | Number | ✅ | Harga jual dalam Rupiah (harus $> 0$) |

#### Response Output (HTTP 200 OK):
```json
{
  "status": "success",
  "model_engine": "v5_indobert_onnx_int8_lightgbm",
  "kesimpulan": "🌟 SANGAT MENARIK — Peluang laku 70.1%",
  "peluang_laku_persen": 70.1,
  "alasan": [...],
  "peringatan_dataset": null,
  "konteks_harga": {
    "median_pasar": 25000,
    "rasio_vs_pasar": 0.6,
    "segmen": "Menengah",
    "selisih_persen": -40.0
  },
  "kompetitor": [
    {
      "nama": "gantungan kunci tugu kujang bogor",
      "harga": 12000,
      "rating": 4.9,
      "terjual": 120,
      "marketplace": "Shopee",
      "url_produk": "https://shopee.co.id/...",
      "kemiripan_persen": 82.1
    }
  ],
  "produk_terpopuler": {...},
  "insight_pasar": {...}
}
```

---

## 🧠 Evaluasi Ilmiah & Benchmark Model

Seluruh evaluasi dilakukan secara **bebas *data leakage*** (split 80:20 *stratified* dilakukan sebelum perhitungan statistik harga pasar; pelabelan intra-kategori bebas *ties* dengan `rank(method='average', pct=True) > 0.5`).

| Konfigurasi Model | Akurasi | Precision | Recall | F1-Score | ROC-AUC | 5-Fold CV |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **IndoBERT (FP32 768d) + LightGBM** *(Eksperimen Terbaik)* | **74,76%** | **77,52%** | **81,30%** | **79,37%** | **0,7893** | 69,18% ± 3,97% |
| **IndoBERT (ONNX INT8 PTQ) + LightGBM** *(Produksi v5 Render)* | **72,82%** | 75,57% | 80,49% | **77,95%** | **0,7889** | 69,18% ± 3,97% |
| IndoBERT (FP32 768d) + XGBoost | 72,33% | 76,61% | 77,24% | 76,92% | 0,7779 | 67,85% ± 2,86% |
| Baseline v4 (TF-IDF Bigram + Random Forest) | 70,39% | 73,44% | 76,42% | 74,90% | 0,7611 | 68,33% ± 3,29% |

---

## 📂 Struktur Repositori

```
RadarUMKMBogor-API/
├── app.py                                       # Entrypoint Flask API v5 (IndoBERT ONNX INT8 + LightGBM)
├── download_onnx.py                             # Pengunduh otomatis indobert_base_p1_int8.onnx dari GitHub Releases
├── export_indobert_onnx.py                      # Skrip ekspor IndoBERT ke ONNX & kuantisasi INT8 + benchmark RAM
├── train_model_v5_indobert_onnx.py              # Pipeline training PTQ v5 (IndoBERT ONNX INT8 + LightGBM)
├── experiment_comparison.py                     # Benchmark 12 kombinasi (RF/XGB/LGBM × TF-IDF/W2V/IndoBERT)
├── experiment_tuning.py                         # Eksperimen optimasi 14 skenario (PCA, Hybrid Fusion, Tuning, Voting)
├── train_model_v4.py                            # Pipeline baseline historis v4
├── requirements.txt                             # Dependensi produksi ringan (tanpa PyTorch/Transformers)
├── data/
│   ├── processed/dataset_preprocessed.csv       # Dataset bersih (1.027 produk UMKM Bogor)
│   ├── indobert_embeddings_int8.npy             # Cache vektor embedding IndoBERT ONNX INT8 katalog (3,16 MB)
│   ├── market_stats_v4.csv                      # Statistik harga pasar per kategori (train-only)
│   ├── benchmark_bu_uti_results.json            # Hasil benchmark 12 kombinasi
│   ├── tuning_results_bu_uti.json               # Hasil eksperimen optimasi 14 skenario
│   └── eval_model_v5_onnx_metrics.json          # Metrik evaluasi resmi Model v5 ONNX INT8
└── models/
    ├── model_umkm_bogor_v5_indobert_lgbm.joblib # Bundle produksi LightGBM + OHE + Scaler (327,7 KB)
    ├── indobert_tokenizer.json                  # Fast Rust tokenizer IndoBERT (692,4 KB)
    └── model_umkm_bogor_v4.joblib               # Fallback model v4
```

*(File `models/indobert_base_p1_int8.onnx` berukuran 118,87 MB dihosting pada GitHub Releases tag `v5.0-onnx` dan diunduh otomatis saat startup.)*

---

## 💻 Menjalankan Secara Lokal

```bash
git clone https://github.com/AogamiKiryuu/RadarUMKMBogor-API.git
cd RadarUMKMBogor-API
python -m venv .venv
.venv\Scripts\activate   # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python app.py
```

Saat dijalankan pertama kali, `download_onnx.py` otomatis mengunduh `indobert_base_p1_int8.onnx` ke folder `models/`, lalu server aktif di `http://localhost:5000`.

---

## 📜 Riwayat Pembaruan (Changelog)

- **v5 (November 2026 — Migrasi Penuh IndoBERT ONNX INT8 + LightGBM):**
  - Migrasi penuh mesin produksi dari TF-IDF + Random Forest ke **IndoBERT (ONNX INT8) + LightGBM** dengan pencarian kompetitor semantik.
  - Kompresi model IndoBERT 4× lipat (`495,64 MB` → `118,87 MB`) via Dynamic INT8 Quantization sehingga berjalan lancar di Render Free Tier 512 MB (RAM puncak **310,30 MB**, latensi **~50–280 ms**).
  - Penambahan prioritas *head-noun* pada validasi kategori agar produk turunan (*"Gantungan Kunci Miniatur Lapis Talas"*) tidak salah diklasifikasikan sebagai Makanan.
  - Penambahan skrip benchmark 12 kombinasi & optimasi 14 skenario.
- **v4 (Oktober–November 2026):** Pelabelan intra-kategori bebas *ties*, eliminasi *target leakage* & *train-serve skew*, statistik harga pasar *train-only*.
- **v1–v3 (Historis):** Prototipe awal Random Forest dan fitur harga relatif.
