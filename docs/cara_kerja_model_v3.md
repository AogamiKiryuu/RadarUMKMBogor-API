# Dokumentasi Lengkap: Cara Kerja Model Machine Learning v3

Dokumen ini menjelaskan secara rinci seluruh alur kerja model Machine Learning v3 pada sistem **RadarUMKM Bogor**, mulai dari tahap persiapan data (training) hingga tahap saat model berjalan di server (inference).

> **Versi Dokumen:** v3 — Diperbarui 16 Juli 2026  
> **Perubahan terbaru:** Validasi sub kategori, peringatan dataset, guardrail harga abnormal.

---

## 1. Fase Pelatihan (Training Phase)

### A. Dataset dan Preprocessing
Model ini dilatih menggunakan dataset yang berisi **1.027 produk UMKM** Kota dan Kabupaten Bogor yang dikumpulkan melalui **scraping marketplace** (Tokopedia, Shopee, Lazada). Karena proses scraping tidak menjamin cakupan semua produk, dataset ini merepresentasikan sebagian besar — namun tidak semua — produk UMKM Bogor yang beredar di internet.

Data mentah melalui proses pembersihan (*preprocessing*):
1. Menghapus data duplikat dan data dengan kolom yang kosong (NaN).
2. Membatasi pencilan ekstrim (*outliers*) menggunakan metode *Interquartile Range (IQR Clipping)* agar data lebih relevan.
3. Membersihkan kolom `nama_produk` dari karakter simbol dan *stopwords* (kata umum) menggunakan algoritma **Sastrawi (Stemmer Bahasa Indonesia)**. Hasilnya disimpan pada kolom `nama_produk_clean`.

### B. Strategi Labeling (Auto-Labeling)
Karena dataset bawaan **tidak memiliki label eksplisit** (apakah suatu produk laku/tidak laku), label dibuat secara matematis berdasarkan volume penjualan historis:
- Sistem mencari **Nilai Median** dari kolom `jumlah_terjual` seluruh produk. Median pada dataset ini adalah **34 unit**.
- **Label 1 (Menarik):** Diberikan jika produk terjual `> 34` unit.
- **Label 0 (Kurang Menarik):** Diberikan jika produk terjual `<= 34` unit.
- *Hasil:* Dataset menjadi seimbang (*balanced*) dengan komposisi kelas 50.1% vs 49.9%.

### C. Rekayasa Fitur (Feature Engineering)
Model tidak hanya melihat harga mentah, melainkan melakukan ekstraksi fitur mendalam:

| # | Fitur | Penjelasan |
|---|---|---|
| 1 | **TF-IDF Vectorizer** | Memecah `nama_produk_clean` menjadi vektor 1000 dimensi (bobot kata seperti *"lapis"*, *"talas"*, *"kopi"*) |
| 2 | **One-Hot Encoder** | Mengubah `kategori` & `sub_kategori` menjadi representasi biner (0/1) |
| 3 | `rasio_harga` | Perbandingan harga produk dengan **median harga kategorinya** |
| 4 | `zscore_harga` | Deviasi harga dari rata-rata pasar dalam satuan standar deviasi |
| 5 | `segmen_harga` | Klasifikasi ke Segmen 0 (Murah), 1 (Menengah), 2 (Premium) berdasarkan kuartil |
| 6 | `popularity_score` | `rating × log1p(jumlah_terjual)` — fitur terpenting, merepresentasikan kualitas & volume |
| 7 | `revenue_proxy_log` | `log1p(harga × jumlah_terjual)` — perkiraan omzet produk |
| 8 | `jumlah_log` | `log1p(jumlah_terjual)` — transformasi log untuk meredam efek outlier penjualan |

Semua nilai numerik di atas kemudian dinormalisasi ukurannya menggunakan `StandardScaler`.

### D. Algoritma dan Evaluasi Model
- **Algoritma Utama:** **Random Forest Classifier** — dipilih karena kemampuannya mendeteksi hubungan *non-linear* yang sangat kuat (misal: relasi antara harga mahal dan kata kunci tertentu).
- **Hyperparameter Tuning:** Menggunakan `GridSearchCV` dengan pencarian parameter terbaik melalui *5-Fold Stratified Cross Validation*.
- **Hasil Evaluasi (Holdout Test 20%):**

| Metrik | Nilai |
|---|---|
| Akurasi | **98.54%** |
| AUC-ROC | **99.96%** |

---

## 2. Fase Produksi / Prediksi (Inference Phase via API)

Saat user mengirimkan request ke Endpoint API (`POST /predict`), berikut urutan kerja sistem di balik layar:

### A. Validasi Input (4 Lapis)

**Lapis 1 — Validasi Harga:**
- Menolak (Error 400) jika harga `<= 0`.

**Lapis 2 — Validasi Kategori:**
- Sistem mencocokkan kata kunci dalam nama produk terhadap kamus `PRODUK_KATEGORI_MAP`.
- Jika kata kunci mengindikasikan kategori berbeda dengan yang dipilih user → Error 400.
- Contoh: nama "Kopi Puncak Bogor" + kategori "Makanan" → **ditolak**, seharusnya "Minuman".

**Lapis 3 — Validasi Sub Kategori:**
- Sistem mencocokkan kata kunci dalam nama produk terhadap kamus `PRODUK_SUB_KATEGORI_MAP`.
- Jika sub kategori tidak sesuai → Error 400 dengan pesan koreksi.
- Contoh: nama "Bolu Talas Bogor" + sub kategori "Lauk & Bahan Makanan" → **ditolak**, seharusnya "Kue & Roti".

**Sub Kategori Valid per Kategori:**

| Kategori | Sub Kategori yang Valid |
|---|---|
| Makanan | Kue & Roti · Camilan & Snack · Lauk & Bahan Makanan · Makanan Tradisional |
| Minuman | Kopi · Teh · Minuman Tradisional |
| Pakaian & Fashion | Atasan & Pakaian Kasual · Pakaian Tradisional |
| Aksesoris & Souvenir | Aksesoris & Souvenir |

**Lapis 4 — Validasi Identitas Bogor:**
- Sistem mengecek apakah nama produk mengandung kata kunci wilayah/produk khas Bogor.
- Jika tidak ada kompetitor serupa DAN tidak ada identitas Bogor → Error 400.
- Jika ada kompetitor tapi tidak ada identitas Bogor → Warning 400 (saran menambahkan identitas).

### B. Pencarian Kompetitor (Cosine Similarity)
Sebelum memprediksi peluang, sistem mencari produk serupa di database untuk digunakan sebagai acuan:
1. Nama produk input dibersihkan dengan stemmer (`clean_text`), lalu diubah menjadi vektor TF-IDF.
2. Dilakukan *Cosine Similarity* antara input user vs **1.027 produk** di database training (pre-cached saat startup).
3. Diambil **Top 6 produk** dengan skor similarity > 0.05 sebagai kompetitor.
4. Dari kompetitor ini, sistem mengambil:
   - `rating_est` = rata-rata rating kompetitor
   - `jumlah_est` = median jumlah terjual kompetitor

Jika tidak ada kompetitor, sistem menggunakan nilai default: `rating_est = 3.5`, `jumlah_est = 30`.

### C. Deteksi Cakupan Dataset (Peringatan Dataset)
Karena data bersumber dari scraping, tidak semua produk UMKM Bogor terwakili. Sistem mengevaluasi `max_sim_score` (skor kemiripan tertinggi) untuk menentukan level peringatan:

| Level | Kondisi | Keterangan |
|---|---|---|
| `tidak_ditemukan` | Tidak ada kompetitor (sim = 0) | Prediksi sepenuhnya berbasis estimasi kategori & harga |
| `kemiripan_rendah` | max_sim < 15% | Data sangat terbatas, hasil sebagai gambaran umum |
| `kemiripan_sedang` | max_sim 15–35% | Prediksi berbasis produk serupa (bukan persis sama) |
| `null` | max_sim ≥ 35% | Produk terwakili dengan baik — tidak ada peringatan |

Peringatan ini **tidak memblokir prediksi**, hanya ditambahkan di field `peringatan_dataset` pada response.

### D. Kalkulasi Fitur dan Prediksi Random Forest
1. Sistem menghitung fitur harga relatif (`rasio_harga`, `zscore_harga`, `segmen_harga`, `log_harga`) dari `market_stats_v3.csv` yang di-cache di memori saat startup.
2. Fitur numerik digabung dengan teks dan kategori, lalu dimasukkan ke Pipeline Random Forest.
3. Model mengeluarkan **probabilitas (0.00–1.00)**:
   - `>= 0.70` → 🌟 SANGAT MENARIK
   - `0.50–0.69` → ✅ CUKUP MENARIK
   - `< 0.50` → ⚠️ KURANG MENARIK

### E. Guardrail Harga Abnormal
Setelah prediksi Random Forest keluar, sistem menerapkan **batas atas probabilitas** berdasarkan rasio harga terhadap median pasar — untuk mengoreksi prediksi yang tidak realistis secara bisnis:

| Kondisi Harga | Guardrail |
|---|---|
| Harga > 5× median pasar (`rasio >= 5.0`) | Probabilitas **max 2%** |
| Harga > 3× median pasar (`rasio >= 3.0`) | Probabilitas **max 15%** |
| Harga > 2× median pasar (`rasio >= 2.0`) | Probabilitas **max 35%** |
| Harga > 1.3× median pasar (`rasio >= 1.3`) | Probabilitas **max 45%** |
| Harga < 30% median pasar (`rasio < 0.3`) | Probabilitas **max 35%** *(terlalu murah — mencurigakan)* |

### F. Pembuatan Alasan Bisnis (Reasoning)
Sistem merangkai kalimat alasan secara dinamis berdasarkan kondisi aktual:
- Jumlah kompetitor yang ditemukan
- Selisih harga vs median pasar
- Rata-rata penjualan kompetitor
- Level probabilitas akhir

### G. Insight Pasar & Produk Terpopuler
Di akhir proses, sistem mengambil data tambahan (di-cache saat startup):
1. **Top 5 produk terpopuler** di kategori/sub-kategori yang sama, diurutkan berdasarkan `popularity_score`.
2. **Ranking semua kategori** berdasarkan total popularitas.
3. **Top 5 sub-kategori terpopuler** dalam kategori yang sama.

---

## 3. Diagram Alur Singkat

```
Input User (nama, kategori, sub_kategori, harga)
    │
    ▼
[Validasi 4 Lapis] ──── Gagal ──→ Error 400
    │ Lolos
    ▼
[Cosine Similarity → Cari Kompetitor]
    │
    ▼
[Deteksi Coverage Dataset → peringatan_dataset]
    │
    ▼
[Hitung Fitur Harga Relatif]
    │
    ▼
[Random Forest Predict → Probabilitas]
    │
    ▼
[Guardrail Harga Abnormal]
    │
    ▼
[Bangun Alasan + Insight Pasar]
    │
    ▼
Response JSON (success)
```

---

## 4. Output Akhir JSON

Response sukses mengandung:

| Field | Isi |
|---|---|
| `peluang_laku_persen` | Persentase peluang produk laku (0–100) |
| `kesimpulan` | Label dan status prediksi |
| `alasan` | Array kalimat alasan bisnis |
| `peringatan_dataset` | Peringatan jika produk tidak/kurang terwakili di dataset (`null` jika aman) |
| `konteks_harga` | Median pasar, segmen, selisih harga |
| `kompetitor` | Daftar produk serupa yang ditemukan |
| `produk_terpopuler` | Top 5 produk paling digemari di kategori/sub-kategori |
| `insight_pasar` | Ranking kategori, sub-kategori terpopuler, narasi pasar |
