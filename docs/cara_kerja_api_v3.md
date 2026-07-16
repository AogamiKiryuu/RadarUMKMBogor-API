# Dokumentasi API: RadarUMKM Bogor — Flask ML API v3

Dokumentasi lengkap endpoint HTTP yang tersedia pada server **RadarUMKM Bogor API**.

> **Base URL (Development):** `http://localhost:5000`  
> **Framework:** Flask (Python)  
> **Model:** Random Forest v3 (`model_umkm_bogor_v3.joblib`)  
> **Versi Dokumen:** v3 — Diperbarui 16 Juli 2026

---

## Daftar Endpoint

| Method | Path | Keterangan |
|---|---|---|
| GET | `/health` | Cek status server dan model |
| POST | `/predict` | Prediksi peluang laku produk UMKM |

---

## GET `/health`

Mengecek apakah server dan model berjalan dengan benar.

**Request:** Tidak perlu body.

**Response Sukses (200):**
```json
{
  "status": "ok",
  "model": "model_umkm_bogor_v3.joblib",
  "dataset_rows": 1027,
  "versi": "v3",
  "fitur_baru": ["jumlah_log", "revenue_proxy_log", "popularity_score", "produk_terpopuler"],
  "message": "Flask ML API v3 siap ✅ (fitur Paling Digemari aktif)"
}
```

---

## POST `/predict`

Endpoint utama untuk memprediksi peluang laku suatu produk UMKM Bogor berdasarkan nama, kategori, sub kategori, dan harga.

### Request Body (JSON)

| Field | Tipe | Wajib | Keterangan |
|---|---|---|---|
| `nama_produk` | string | ✅ | Nama produk, sebaiknya mengandung identitas Bogor |
| `kategori` | string | ✅ | Kategori utama produk |
| `sub_kategori` | string | ✅ | Sub kategori produk |
| `harga_produk` | number | ✅ | Harga produk dalam Rupiah (harus > 0) |

**Kategori dan Sub Kategori yang Valid:**

| Kategori | Sub Kategori |
|---|---|
| `Makanan` | `Kue & Roti` · `Camilan & Snack` · `Lauk & Bahan Makanan` · `Makanan Tradisional` |
| `Minuman` | `Kopi` · `Teh` · `Minuman Tradisional` |
| `Pakaian & Fashion` | `Atasan & Pakaian Kasual` · `Pakaian Tradisional` |
| `Aksesoris & Souvenir` | `Aksesoris & Souvenir` |

**Contoh Request:**
```json
{
  "nama_produk": "Bolu Talas Bogor",
  "kategori": "Makanan",
  "sub_kategori": "Kue & Roti",
  "harga_produk": 38000
}
```

---

### Alur Validasi (Sebelum Prediksi)

Request akan ditolak (`400`) jika salah satu kondisi berikut terpenuhi:

| # | Kondisi | Pesan Error |
|---|---|---|
| 1 | `harga_produk <= 0` | "Harga produk tidak boleh 0 atau minus." |
| 2 | Kata kunci produk tidak sesuai kategori yang dipilih | "Kategori tidak sesuai untuk produk '...'..." |
| 3 | Kata kunci produk tidak sesuai sub kategori yang dipilih | "Sub Kategori tidak sesuai untuk produk '...'..." |
| 4 | Tidak ada identitas Bogor DAN tidak ada kompetitor di dataset | "Produk '...' tidak terdeteksi sebagai produk khas Bogor." |
| 5 | Ada kompetitor tapi tidak ada identitas Bogor | Warning: "...tidak mencantumkan identitas Bogor secara eksplisit." |

---

### Response Sukses (200)

```json
{
  "status": "success",
  "kesimpulan": "🌟 SANGAT MENARIK — Peluang laku 94.0%",
  "peluang_laku_persen": 94.0,
  "alasan": [
    "Produk serupa sudah banyak dijual (6 kompetitor ditemukan).",
    "Harga anda 47% di bawah median pasar (Rp38.000) — sangat kompetitif, berpotensi menarik banyak pembeli.",
    "Produk sejenis terbukti laku keras (rata-rata 169 terjual).",
    "Model menilai kombinasi nama, kategori, dan posisi harga sangat sesuai tren pasar."
  ],

  "peringatan_dataset": null,

  "konteks_harga": {
    "median_pasar": 38000,
    "rasio_vs_pasar": 0.53,
    "segmen": "Murah",
    "selisih_persen": -47.4
  },

  "kompetitor": [
    {
      "nama": "Bolu Talas Bogor dan Bolu Pandan Bogor by Rakuni Bakery",
      "harga": 68000,
      "rating": 4.9,
      "terjual": 36,
      "marketplace": "tokopedia",
      "url_produk": "https://...",
      "kemiripan_persen": 79.4
    }
  ],

  "produk_terpopuler": {
    "label": "Top 5 Produk Paling Digemari di 'Makanan — Kue & Roti'",
    "deskripsi": "Produk-produk di bawah ini adalah yang paling diminati pembeli...",
    "produk": [
      {
        "nama": "Lapis Talas Bogor Original",
        "kategori": "Makanan",
        "sub_kategori": "Kue & Roti",
        "harga": 45000,
        "jumlah_terjual": 544,
        "rating": 4.9,
        "popularity_score": 33.87,
        "marketplace": "lazada",
        "url_produk": "https://...",
        "nama_toko": "Kabayan Official"
      }
    ]
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

### Field `peringatan_dataset`

Field ini memberitahu frontend bahwa produk belum/kurang terwakili di dataset scraping kami. **Prediksi tetap berjalan**, namun akurasi mungkin lebih rendah.

| Level | Kondisi | `akurasi_prediksi` |
|---|---|---|
| `tidak_ditemukan` | Tidak ada produk serupa di database | `rendah` |
| `kemiripan_rendah` | Kemiripan tertinggi < 15% | `sedang` |
| `kemiripan_sedang` | Kemiripan 15–35% | `cukup_baik` |
| `null` | Kemiripan ≥ 35% — produk terwakili baik | *(tidak ada peringatan)* |

**Contoh `peringatan_dataset` tidak null:**
```json
"peringatan_dataset": {
  "level": "tidak_ditemukan",
  "judul": "⚠️ Produk Belum Ada di Dataset Kami",
  "pesan": "Mohon maaf, produk 'Kerajinan Bambu Khas Bogor' belum tersedia dalam database referensi kami yang dikumpulkan dari hasil scraping marketplace...",
  "saran": "Coba periksa kembali nama produk, atau tambahkan kata kunci yang lebih spesifik...",
  "akurasi_prediksi": "rendah"
}
```

---

### Response Error (400 / 500)

**Format umum error:**
```json
{
  "status": "error",
  "message": "Pesan penjelasan error di sini."
}
```

**Contoh error validasi sub kategori:**
```json
{
  "status": "error",
  "message": "Sub Kategori tidak sesuai untuk produk 'Bolu Talas Bogor'. Kata kunci 'bolu' mengindikasikan sub kategori 'Kue & Roti', bukan 'Lauk & Bahan Makanan'. Silakan pilih sub kategori yang tepat."
}
```

**Error 500** — Exception tidak terduga dari server:
```json
{
  "status": "error",
  "message": "<detail exception>"
}
```

---

## Catatan Teknis

### Pre-caching saat Startup
Untuk mempercepat respons API, beberapa operasi berat dilakukan **sekali saja saat server pertama kali dinyalakan**:
- Memuat model Random Forest dari file `.joblib`
- Memuat dataset dan market stats ke memori
- Pre-komputasi matriks TF-IDF seluruh dataset (untuk Cosine Similarity)
- Pre-komputasi ranking kategori dan sub-kategori

### Guardrail Harga Abnormal
Random Forest kadang menghasilkan probabilitas yang tidak realistis secara bisnis untuk harga ekstrem. Sistem menerapkan **batas atas (cap)** probabilitas otomatis:

| Kondisi | Probabilitas Maksimum |
|---|---|
| Harga > 5× median pasar | 2% |
| Harga > 3× median pasar | 15% |
| Harga > 2× median pasar | 35% |
| Harga > 1.3× median pasar | 45% |
| Harga < 30% median pasar | 35% |

### Keterbatasan Dataset
Data dikumpulkan melalui scraping marketplace sehingga:
- Tidak semua produk UMKM Bogor terwakili
- Produk baru atau produk yang tidak dijual online mungkin tidak terdeteksi
- Hasil prediksi untuk produk yang tidak ada di dataset menggunakan estimasi berbasis kategori dan harga

Frontend disarankan untuk menampilkan `peringatan_dataset` sebagai banner/card peringatan agar user memahami keterbatasan ini.
