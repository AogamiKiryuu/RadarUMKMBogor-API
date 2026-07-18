# Penjelasan Diagram Rules Random Forest v3

Dokumen ini adalah penjelasan detail dari visualisasi *Decision Tree* (Pohon Keputusan) pada file `diagram_rules.mermaid`. Saat di-*deploy*, Model Random Forest membuat ratusan pohon, dan visualisasi ini adalah rangkuman dari keputusan yang paling sering diambil oleh sistem berdasarkan **Feature Importance**.

---

## 1. Node Akar (Root Node)
**Kondisi:** `Apakah popularity_score >= 12.5?`

*   **Apa itu `popularity_score`?** Ini adalah skor yang dihitung dari perkalian antara rating kompetitor terdekat dan logaritma jumlah produk yang terjual. Rumusnya: `Rating × Log(Terjual + 1)`.
*   **Kenapa angkanya 12.5?** Angka ini adalah titik potong (*threshold*) yang paling memisahkan produk laku dan tidak laku di dataset UMKM Bogor kita. 
*   **Maknanya:** Jika skor `>= 12.5`, artinya kompetitor untuk produk ini memiliki penjualan historis dan rating yang *sangat bagus* (pasar memang suka produk jenis ini). Jika `< 12.5`, berarti produk ini berada di *niche* (pasar sempit) atau kurang populer.

---

## 2. Percabangan Kiri (Pasar Populer)
Jika produk tersebut berada di pasar yang populer (kiri), tantangan selanjutnya adalah **harga**.

**Kondisi:** `Apakah rasio_harga <= 1.3?`

*   **Apa itu `rasio_harga`?** Ini adalah perbandingan harga inputan dengan **Median Harga di Kategori tersebut**.
*   **Kenapa angkanya 1.3?** Angka `1.3` berarti **30% lebih mahal** daripada median harga pasar (1.0 = sama dengan pasar).
*   **Maknanya:** Meskipun pasarnya ramai (misal jualan *Lapis Talas*), jika pengguna mematok harga yang ekstrim mahalnya (> 30% dari harga tengah kompetitor di aplikasi), maka peluang laku menurun drastis.
    *   Jika `<= 1.3`: **🌟 SANGAT MENARIK** (Harga logis, pasarnya ada).
    *   Jika `> 1.3`: **⚠️ KURANG MENARIK** (Pasarnya ada, tapi terlalu mahal/kalah bersaing).

---

## 3. Percabangan Kanan (Pasar Sepi / Baru)
Jika produk tersebut tidak memiliki historis penjualan yang meyakinkan (`popularity_score < 12.5`), model akan mencari keunggulan lain.

**Kondisi:** `Apakah nama_produk mengandung kata kunci khas/ikonik Bogor?`

*   **Penjelasan TF-IDF:** Model mengekstrak kata kunci (seperti *"Bogor"*, *"Talas"*, *"Puncak"*, *"Unyil"*, dll) menggunakan vektor teks.
*   **Maknanya:** Jika produk bukan barang pasaran yang sering dibeli, peluang menangnya ada di keunikan identitas sebagai oleh-oleh khas.
    *   **Jika TIDAK mengandung kata kunci:** **⚠️ KURANG MENARIK** (Pasar tidak jelas, bukan barang populer, dan tidak punya identitas lokal).
    *   **Jika YA mengandung kata kunci:** Masuk ke evaluasi harga level selanjutnya (Node di bawahnya).

---

## 4. Evaluasi Harga Pada Pasar Niche (Cabang Paling Kanan Bawah)
Jika produk memiliki unsur khas Bogor tetapi `popularity_score`-nya rendah.

**Kondisi:** `Apakah rasio_harga <= 0.8?`

*   **Kenapa angkanya 0.8?** Angka `0.8` berarti harga dipatok **20% lebih murah** dari harga median pasar (berani *banting harga* atau masuk segmen bawah).
*   **Maknanya:** Karena produk ini unik (khas Bogor) tapi secara historis kompetitor sejenis belum banyak mencetak angka terjual (popularitas rendah), cara paling aman agar laku adalah dengan memberikan harga miring (murah).
    *   Jika `<= 0.8`: **✅ MENARIK** (Produk unik, khas lokal, dan berani harga murah).
    *   Jika `> 0.8`: **⚠️ KURANG MENARIK** (Produk belum terkenal tapi dijual standar atau mahal).
