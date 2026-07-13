# Aturan Pengambilan Keputusan Model (Random Forest v3)

Berikut adalah representasi alur kerja dan logika pengambilan keputusan (*rules*) dari **Random Forest Pipeline** yang kita gunakan dalam projek RadarUMKM Bogor v3.

## 1. Arsitektur Pipeline Pemrosesan

Sebelum masuk ke pohon keputusan Random Forest, data input diubah terlebih dahulu oleh `ColumnTransformer` menjadi bentuk angka terstandarisasi.

```mermaid
graph TD
    A[Input Produk Baru] --> B{Fitur Teks: nama_produk}
    A --> C{Fitur Kategori & Sub-Kategori}
    A --> D{Fitur Harga & Statistik Pasar}

    B -->|TF-IDF Vectorizer| B1[Vektor Teks 1000 Dimensi]
    C -->|One-Hot Encoder| C1[Representasi Biner Kategori]
    D -->|StandardScaler| D1[Rasio Harga, Z-Score, Popularity Score Terstandar]

    B1 --> E[Concatenate Features]
    C1 --> E
    D1 --> E

    E --> F[Random Forest Classifier v3]
    F --> G{Voting Ensemble 100 Pohon}
    G --> H[Output: Probabilitas Peluang Laku %]
```

## 2. Logika Aturan Keputusan (Representasi Decision Tree)

Karena Random Forest terdiri dari 100 pohon keputusan (*n_estimators=100*), masing-masing pohon mempelajari kombinasi aturan yang berbeda secara acak. Berikut adalah representasi pohon keputusan tipikal berdasarkan bobot pentingnya fitur (*Feature Importance*) dari model kita:

```mermaid
graph TD
    %% Styling Nodes
    classDef root fill:#4f46e5,stroke:#312e81,stroke-width:2px,color:#fff;
    classDef split fill:#0891b2,stroke:#155e75,stroke-width:2px,color:#fff;
    classDef leaf1 fill:#16a34a,stroke:#14532d,stroke-width:2px,color:#fff;
    classDef leaf0 fill:#dc2626,stroke:#7f1d1d,stroke-width:2px,color:#fff;

    RootNode["Apakah popularity_score (Kompetitor)<br> >= 12.5? (Terjual & Rating Tinggi)"]:::root
    
    %% Root Level Split
    RootNode -->|Ya| LeftChild["Apakah rasio_harga <br> <= 1.3? (Harga kompetitif)"]:::split
    RootNode -->|Tidak| RightChild["Apakah nama_produk mengandung<br>kata kunci khas/ikonik Bogor?"]:::split

    %% Left Branch (High Popularity)
    LeftChild -->|Ya| Leaf_A["🌟 Menarik<br>(Peluang Laku Tinggi)"]:::leaf1
    LeftChild -->|Tidak| Leaf_B["⚠️ Kurang Menarik<br>(Harga Terlalu Mahal)"]:::leaf0

    %% Right Branch (Low Popularity)
    RightChild -->|Ya| LeftSubChild["Apakah rasio_harga <br> <= 0.8? (Harga Murah)"]:::split
    RightChild -->|Tidak| Leaf_C["⚠️ Kurang Menarik<br>(Kurang Relevan/Pasar Sepi)"]:::leaf0

    %% Deep Sub Branch
    LeftSubChild -->|Ya| Leaf_D["✅ Menarik<br>(Peluang Terbuka - Murah)"]:::leaf1
    LeftSubChild -->|Tidak| Leaf_E["⚠️ Kurang Menarik<br>(Kompetitor Sedikit & Mahal)"]:::leaf0
```

## 3. Penjelasan Aturan Utama (*Core Rules*)

Model Random Forest v3 menentukan status keaktifan produk berdasarkan hierarki aturan berikut:

1. **Aturan Popularitas Utama (`popularity_score`):**
   - Jika produk sejenis (kompetitor) memiliki performa penjualan historis yang kuat (`jumlah_terjual` tinggi) dan `rating` rata-rata di atas median, model langsung mengklasifikasikan produk ini ke zona potensial laku.
2. **Aturan Elastisitas Harga (`rasio_harga`):**
   - Sekalipun produknya terpopuler, jika harga yang dimasukkan user melebihi **1.3 kali (30% lebih mahal)** dari median pasar kategori tersebut (`rasio_harga > 1.3`), model akan otomatis menurunkan probabilitas kelayakan lakunya menjadi **Kurang Menarik** karena dinilai tidak kompetitif.
3. **Aturan Relevansi Teks (`nama_produk_clean` melalui TF-IDF):**
   - Kata kunci identitas Bogor (seperti *"Lapis Talas"*, *"Roti Unyil"*, *"Kopi Puncak"*) yang memiliki bobot TF-IDF tinggi akan menaikkan skor klasifikasi secara instan karena secara historis produk dengan kata kunci tersebut terbukti sangat diminati di dataset.
