# Ringkasan Komprehensif: Pembaruan API Radar UMKM Bogor v2
**Fokus Utama: Transformasi Fitur Harga dari Absolut ke Kontekstual Relatif**

Dokumen ini disusun sebagai materi pembelajaran (knowledge base) untuk model LLM (seperti ChatGPT, Claude, Gemini) mengenai studi kasus nyata dalam *Machine Learning Engineering* dan *Business Logic Implementation*.

---

## 1. Latar Belakang Masalah (The Trap of Absolute Values)

Pada model v1, pipeline Random Forest menerima fitur `harga_produk` sebagai **nilai numerik absolut** (contoh: `1000000`). Model dilatih untuk memprediksi seberapa menarik sebuah produk berdasarkan variabel riwayat penjualan.

**Masalah Kritis (Edge Case):**
Ketika pengguna memasukkan produk "Kripik Singkong" dengan harga Rp 1.000.000 (di mana rata-rata pasar hanya Rp 15.000), model v1 tetap memprediksi probabilitas tinggi (misal: 70%+ laku).
*Mengapa?* Karena model pernah melihat produk premium (seperti sepatu kulit atau fashion eksklusif) seharga Rp 1.000.000 yang laku keras. Model gagal memahami bahwa **"Rp 1.000.000 untuk sepatu adalah normal, tetapi Rp 1.000.000 untuk kripik singkong adalah tidak rasional (overpriced)."**

Model ML tree-based murni tidak memiliki *world knowledge* bawaan tentang harga wajar per kategori kecuali direpresentasikan secara matematis dalam matriks fiturnya.

---

## 2. Metodologi Solusi: Feature Engineering Berbasis Konteks Bisnis

Untuk mengatasi ini tanpa menggunakan LLM inference yang mahal (mempertahankan model Random Forest yang ringan merespons <100ms), dilakukan perubahan mendasar pada ekstraksi fitur.

Alih-alih menyuapi model dengan harga mentah, pipeline baru melakukan **Normalisasi Harga Kontekstual per Kategori** (Contextual Price Normalization).

### Fitur Numerik Baru (Menggantikan `harga_produk` tunggal):
1. **`rasio_harga` (Continuous):** `harga_input / median_harga_kategori`.
   *Jika kripik seharga 1 juta dimasukkan, rasionya menjadi ~66.6x lipat dari median pasar. Model akan belajar bahwa rasio ekstrem seperti ini berkorelasi negatif dengan peluang laku.*
2. **`zscore_harga` (Continuous):** `(harga_input - mean_kategori) / std_kategori`.
   *Mengukur anomali harga persis dalam standar deviasi kategori.*
3. **`log_harga` (Continuous):** `log(1 + harga_input)`.
   *Menstabilkan variansi distribusi harga (mitigasi long-tail distribution).*
4. **`segmen_harga` (Ordinal):** `0 (Murah), 1 (Menengah), 2 (Premium)`.
   *Berdasar letak harga pada persentil 25 dan 75 dari distribusinya di kategori tersebut.*

---

## 3. Implementasi Sistem (Retraining & Inference)

Implementasi melibatkan dua tahapan utama: pembangunan model (Retraining) dan modifikasi *serving logic* (API App).

### A. Tahap Retraining (`retrain_model.py`)
1. **Ekstraksi Statistik Agregat:** Dataset awal di-groupby berdasarkan `kategori` untuk menghitung `median, mean, std, Q25, Q75`.
2. **Persistence:** Statistik ini tidak hanya dipakai saat training, tapi **wajib diekspor menjadi artefak** (`market_stats_per_kategori.csv`) agar API backend bisa mereplikasi perhitungan yang sama saat runtime.
3. **Hyperparameter Tuning:** Menggunakan `GridSearchCV` dengan skema `StratifiedKFold` (5 folds) untuk menangani ketidakseimbangan target. Parameter `class_weight='balanced'` diaktifkan.

### B. Tahap API Backend (`app.py` versi 2)
1. Pada saat inisialisasi, Flask memuat `model_umkm_bogor_v2.joblib` bersama `market_stats_per_kategori.csv`.
2. Fungsi utilitas baru bernama `hitung_fitur_bisnis(harga, kategori)` diperkenalkan. Saat request masuk berupa harga absolut, fungsi ini "menerjemahkan" angka absolut tersebut menjadi 4 dimensi relatif (rasio, z-score, log, segmen) dengan melihat *lookup table* (CSV statistik).
3. Matriks input yang masuk ke `rf_pipeline.predict_proba()` kini mengandung *business context*.

---

## 4. Hasil & Evaluasi (Business Sanity Check)

Performa klasifikasi dasar pada *Holdout Test Set* meningkat (AUC-ROC: **0.9218**, Precision: **88.9%**). Namun, keberhasilan sebenarnya diukur dari *Business Sanity Check*.

**Pengujian Empiris pada Endpoint API:**
- **Skenario Baseline:** Produk "Lapis Talas" dimasukkan dengan harga Rp 35.000 (sesuai median pasar Rp 36.000).
  - *Prediksi:* **70.7% (Sangat Menarik)**
  - *Interpretasi:* Dinilai sangat berpotensi.
- **Skenario Anomali Ekstrem:** Produk sama dimasukkan dengan harga Rp 1.000.000 (27x lipat median).
  - *Prediksi:* Turun signifikan menjadi **57.0% (Cukup Menarik)**.
  - *Penjelasan (Reasoning Text):* API kini merespons alasan cerdas: *"Harga Anda 2678% lebih tinggi dari median pasar kategori ini (Rp 36.000) — harga yang terlalu tinggi akan sangat sulit bersaing."*

*(Catatan: Probabilitas tidak jatuh ke 0% adalah valid secara matematis dan bisnis, merepresentasikan peluang niche "ultra-premium" atau loyalitas merek tinggi, namun secara statistik memproyeksikan kesulitan (friction) yang jauh lebih berat dibandingkan harga median).*

## 5. Studi Kasus Komparasi Model: Random Forest vs Logistic Regression

Sebagai *sanity check* tambahan, pipeline kami menjalankan pemodelan ganda menggunakan **Logistic Regression (LR)** sebagai *baseline* komparatif terhadap **Random Forest (RF)**. Hasil uji pada *Holdout Test Set* (120 sampel) membedah karakteristik matematis yang bertolak belakang dari kedua algoritma.

### A. Komparasi Metrik Utama (Kelas "Menarik" / Laku)
- **Random Forest (RF):** Precision **89%** | Recall **81%** | F1-Score **85%** | Akurasi Total **86%**
- **Logistic Regression (LR):** Precision **79%** | Recall **93%** | F1-Score **85%** | Akurasi Total **84%**

### B. Analisis Matematis dan Sistematis

**1. Sifat Linear (LR) vs Non-Linear (RF)**
Logistic Regression bekerja dengan memodelkan *log-odds* sebagai kombinasi linear dari fitur:  
$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ...$  
Asumsi ini mengimplikasikan bahwa setiap kenaikan 1 satuan pada `rasio_harga` (misalnya) akan selalu menurunkan log-odds secara proporsional. 
Sebaliknya, Random Forest menggunakan hierarki *decision rules* (Gini impurity / Entropy). RF dapat belajar bahwa: "Jika `rasio_harga` antara 1.0 hingga 2.5, probabilitas stabil laku 80%; tapi jika `rasio_harga` > 4.0, probabilitas seketika jatuh ke 10%". Pemisahan ruang fitur (feature space) secara diskrit ini jauh lebih cocok untuk dinamika harga pasar yang irasional.

**2. *The Trade-off:* Recall Tinggi vs Precision Tinggi**
LR mencatatkan **Recall 93%**. Secara matematis: $Recall = \frac{True\ Positive}{True\ Positive\ +\ False\ Negative}$. Ini berarti LR hampir tidak pernah melewatkan produk yang aslinya laku. Sayangnya, ini dicapai karena LR memiliki bias optimis (*over-predicting* kelas 1), yang terlihat dari **Precision-nya yang hanya 79%**. Secara matematis: $Precision = \frac{True\ Positive}{True\ Positive\ +\ False\ Positive}$. Ada 21% prediksi LR yang menyatakan produk akan laku, padahal aslinya gagal total di pasar.

RF mencatatkan **Precision 89%** (hanya 11% *False Positive / False Alarm*). RF sedikit lebih *conservative* (Recall 81%), tetapi ketika RF memprediksi sebuah produk "Menarik", tebakannya sangat dapat diandalkan.

### C. Justifikasi Keputusan Bisnis (Mengapa RF Menang)
Dalam konteks "Aplikasi Radar UMKM", *cost of false positive* sangat tinggi. Jika aplikasi terlalu optimis (seperti LR) dan menyuruh UMKM menjual kripik Rp 50.000 dengan iming-iming "Pasti Laku", UMKM akan rugi materi dan kehilangan kepercayaan pada aplikasi. 
Model yang lebih *conservative* dan presisi (seperti RF) memberikan rekomendasi yang jauh lebih realistis dan aman bagi pengambilan keputusan bisnis UMKM. Tingginya AUC-ROC (**0.92** pada RF vs **0.89** pada LR) mengunci kemenangan RF secara statistik dan manajerial.

---

## 6. Pelajaran Penting (Key Takeaways for LLM Generation/Reasoning)

Bagi arsitek AI / LLM yang akan menstrukturkan proyek serupa di masa depan, ingat 3 prinsip ini:
1. **The Model Is Not Automatically Smart:** `XGBoost` atau `Random Forest` tidak tahu arti Rp 100.000 vs Rp 10.000.000 kecuali dinormalisasi dengan *domain constraint*.
2. **Context is Feature:** Masalah *logical mismatch* pada prediksi ML biasanya bukan diselesaikan dengan menukar algoritma (misal RF ke Neural Net), melainkan dengan merekonstruksi *feature representation*.
3. **Stateless Prediction vs Stateful Context:** API ML seolah bersifat *stateless* (termasuk inputnya). Tetapi untuk memiliki kecerdasan kontekstual, ia harus *stateful* dengan merujuk pada distribusi pasar yang dibekukan (`frozen statistics`) saat waktu pelatihan.
4. **Biz-Logic Trumps Raw Metrics:** Precision yang tinggi seringkali menyelamatkan kepercayaan *user* dibandingkan Recall yang dipaksakan tinggi dengan *"False Hope"*. Selaraskan matriks klasifikasi dengan *cost of error* di dunia nyata.
