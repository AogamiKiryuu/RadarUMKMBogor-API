# Visualisasi Random Forest v3

Berikut adalah dua diagram interaktif yang merepresentasikan bagaimana model di *backend* bekerja:

## 1. Arsitektur Pipeline Pemrosesan
Bagaimana data dari input user ditransformasikan sebelum diprediksi:

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

## 2. Decision Tree Rules (Logika Keputusan)
Ini adalah simulasi 1 dari 100 pohon yang ada di dalam Random Forest. Algoritma mencari kombinasi fitur yang paling berpengaruh secara berurutan:

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
