# Dokumentasi Lengkap: Cara Kerja Model Machine Learning v3

Dokumen ini menjelaskan secara rinci seluruh alur kerja model Machine Learning v3 pada sistem **RadarUMKM Bogor**, mulai dari tahap persiapan data (training) hingga tahap saat model berjalan di server (inference).

---

## 1. Fase Pelatihan (Training Phase)

### A. Dataset dan Preprocessing
Model ini dilatih menggunakan dataset yang berisi **1.027 produk UMKM** Kota dan Kabupaten Bogor. Data mentah ini melalui proses pembersihan (*preprocessing*):
1. Menghapus data duplikat dan data dengan kolom yang kosong (NaN).
2. Membatasi pencilan ekstrim (*outliers*) menggunakan metode *Interquartile Range (IQR Clipping)* agar data lebih relevan.
3. Membersihkan kolom `nama_produk` dari karakter simbol dan *stopwords* (kata umum) menggunakan algoritma **Sastrawi (Stemmer Bahasa Indonesia)**. Hasilnya disimpan pada kolom `nama_produk_clean`.

### B. Strategi Labeling (Auto-Labeling)
Karena dataset bawaan **tidak memiliki label eksplisit** (apakah suatu produk laku/tidak laku), label dibuat secara matematis berdasarkan volume penjualan historis:
*   Sistem mencari **Nilai Median** dari kolom `jumlah_terjual` seluruh produk. Median pada dataset ini adalah **34 unit**.
*   **Label 1 (Menarik):** Diberikan jika produk terjual `> 34` unit.
*   **Label 0 (Kurang Menarik):** Diberikan jika produk terjual `<= 34` unit.
*   *Hasil:* Dataset menjadi seimbang (*balanced*) dengan komposisi kelas 50.1% vs 49.9%.

### C. Rekayasa Fitur (Feature Engineering)
Model tidak hanya melihat harga mentah, melainkan melakukan ekstraksi fitur mendalam:
1.  **Fitur Teks (TF-IDF Vectorizer):** Memecah `nama_produk_clean` menjadi vektor numerik 1000 dimensi untuk melihat bobot pentingnya kata-kata (seperti *"lapis"*, *"talas"*, *"kopi"*).
2.  **Fitur Kategorikal (One-Hot Encoder):** Mengubah `kategori` dan `sub_kategori` menjadi biner (0/1).
3.  **Fitur Harga Relatif:** 
    *   `rasio_harga`: Membandingkan harga produk dengan median harga di kategorinya (mengukur seberapa "mahal" dibanding pasar).
    *   `zscore_harga`: Mengukur deviasi (penyimpangan) harga dari rata-rata pasar.
    *   `segmen_harga`: Mengklasifikasikan harga ke Segmen 0 (Murah), 1 (Menengah), atau 2 (Premium) berdasarkan kuartil.
4.  **Fitur Bisnis Proksi:**
    *   `popularity_score`: Diperoleh dari rumus `rating * log1p(jumlah_terjual)`. Fitur terpenting yang merepresentasikan kepuasan *dan* volume penjualan sekaligus.
    *   `revenue_proxy_log`: Perkiraan omzet yang didapat dari `log1p(harga * jumlah_terjual)`.

Semua nilai numerik di atas kemudian dinormalisasi ukurannya menggunakan `StandardScaler`.

### D. Algoritma dan Evaluasi Model
*   **Algoritma Utama:** **Random Forest Classifier**. Karena kemampuannya mendeteksi hubungan *non-linear* yang sangat kuat (misalnya hubungan antara harga yang mahal dengan kata kunci tertentu pada nama).
*   **Hyperparameter Tuning:** Menggunakan `GridSearchCV` yang melakukan pencarian parameter terbaik (kedalaman pohon, jumlah pohon) melalui *5-Fold Stratified Cross Validation*.
*   **Hasil Evaluasi (Holdout Test 20%):** Model mencapai akurasi **98.54%** dan nilai **AUC-ROC 99.96%**, menjadikannya sangat handal untuk membedakan peluang sukses suatu produk.

---

## 2. Fase Produksi / Prediksi (Inference Phase via API)

Saat user mengirimkan request ke Endpoint API (`POST /predict`), berikut adalah urutan mesin bekerja di balik layar:

### A. Validasi Identitas Lokal & Input
1. Sistem menolak (Error 400) jika harga diinput <= 0.
2. Sistem mengecek kamus internal `KATA_KUNCI_WILAYAH_BOGOR`. Jika user menginput produk tanpa unsur Bogor (misal: "Jaket Polos Hitam"), sistem mencari tahu apakah produk tersebut relevan. Jika tidak ada kompetitor terkait, proses akan digagalkan. Jika ada, sistem hanya memberikan `warning` agar user melengkapi nama wilayahnya.

### B. Pencarian Kompetitor (Cosine Similarity)
Sebelum memprediksi peluang, sistem harus tahu kondisi kompetisi untuk produk tersebut:
1. Mengubah nama produk inputan menjadi vektor TF-IDF.
2. Melakukan *Cosine Similarity* antara input user melawan **1.027 produk** di *database* training.
3. Mengambil **Top 5 Kompetitor Paling Mirip**.
4. Dari kelima kompetitor ini, sistem mengambil rata-rata `rating` dan median `jumlah_terjual` sebagai nilai estimasi atau target proyeksi untuk produk user (`rating_est` dan `jumlah_est`).

### C. Kalkulasi Fitur dan Prediksi Random Forest
1. Sistem menghitung fitur-fitur harga (`rasio_harga`, `segmen_harga`, dll) berdasarkan harga input user dibandingkan dengan data statistik kategori yang di-cache di memori (`market_stats_v3.csv`).
2. Input ini dimasukkan ke dalam `Pipeline` Random Forest.
3. Model mengeluarkan *Output* berupa **Probabilitas (0.00 hingga 1.00)**.
    *   `>= 0.7`: SANGAT MENARIK
    *   `0.5 - 0.69`: CUKUP MENARIK
    *   `< 0.5`: KURANG MENARIK

### D. Pembuatan Keputusan Bisnis (Reasoning Generation)
Sistem tidak hanya memberikan angka persentase, tetapi secara dinamis merangkai kalimat alasan (teks) berdasarkan logika IF-ELSE di dalam `app.py`:
*   *Seberapa mahal produk user dibanding pasar?*
*   *Ada berapa banyak pesaing (kompetitor) dari pencarian Cosine Similarity?*
*   *Apakah rata-rata penjualan kompetitor tinggi atau rendah?*

### E. Perhitungan "Produk Paling Digemari" (Insight Pasar)
Di akhir proses prediksi, sistem membaca file `market_stats_sub_kategori_v3.csv` untuk:
1. Mem-filter database hanya pada kategori dan sub-kategori produk inputan user.
2. Mengurutkan (*sorting*) produk secara menurun berdasarkan `popularity_score`.
3. Mengembalikan 5 produk teratas sebagai fitur tambahan: **Top 5 Produk Paling Digemari**.
4. Mengembalikan statistik *ranking* keseluruhan kategori untuk memberikan konteks pasar *(Market Insight)* kepada user (contoh: "Kategori Makanan mendominasi dengan 96rb penjualan...").

---

**Output Akhir JSON:**
Gabungan dari persentase peluang laku, status menarik/tidak, alasan bisnis dinamis, kompetitor terdekat, top produk, dan wawasan pasar keseluruhan!
