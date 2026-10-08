# DOKUMENTASI RESMI: KONTRAK & CARA KERJA FLASK API V4

> **Service:** RadarUMKM Bogor — Backend Machine Learning Microservice  
> **Host Production:** `https://radarumkmbogor-api.onrender.com`  
> **Versi API:** v4 (Oktober 2026)

---

## 1. Daftar Endpoint Aktif

| Method | Endpoint | Fungsi | Status Kode Utama |
|---|---|---|---|
| `GET` | `/health` | Healthcheck, versi model aktif, dan status kesiapan dataset. | `200 OK` |
| `POST` | `/predict` | Inferensi prediksi daya tarik produk UMKM dan pencarian kompetitor. | `200 OK`, `400 Bad Request` |
| `GET` | `/tren` | Daftar kategori dan ringkasan tren pasar. | `200 OK` |

---

## 2. Spesifikasi Endpoint

### A. GET `/health`
Digunakan oleh Nuxt Gateway dan monitoring Render untuk memastikan service aktif.

#### Contoh Request:
```bash
curl -X GET https://radarumkmbogor-api.onrender.com/health
```

#### Contoh Response:
```json
{
  "status": "ok",
  "model": "model_umkm_bogor_v4.joblib",
  "dataset_rows": 1027,
  "tren_loaded": true,
  "kategori_tren": [
    "Makanan",
    "Minuman",
    "Pakaian & Fashion",
    "Aksesoris & Souvenir"
  ]
}
```

---

### B. POST `/predict`
Endpoint utama untuk menghitung daya tarik produk berbasis model Machine Learning v4.

#### Spesifikasi Request Body:
```json
{
  "nama_produk": "Brownies Kukus Talas Keju Khas Bogor",
  "kategori": "Makanan",
  "sub_kategori": "Kue & Roti",
  "harga_produk": 38000
}
```

#### Aturan Validasi Input:
1. `nama_produk` (string, wajib): Harus memuat minimal satu kata dan relevan dengan kategori.
2. `kategori` (string, wajib): Salah satu dari `Makanan`, `Minuman`, `Pakaian & Fashion`, `Aksesoris & Souvenir`.
3. `sub_kategori` (string, wajib): Harus merupakan sub-kategori valid dari kategori yang dipilih.
4. `harga_produk` (number, wajib): Nilai numerik positif ($> 0$).

#### Contoh Response Berhasil (`200 OK`):
```json
{
  "peluang": 48.0,
  "label": "CUKUP MENARIK",
  "persentase": 48.0,
  "status": "success",
  "alasan": "Peluang pasar cukup baik (48.0%). Produk ini memiliki harga kompetitif dan karakteristik yang mendekati tren pasar.",
  "konteks_harga": {
    "harga_input": 38000,
    "median_pasar": 38000,
    "rata_rata_pasar": 41250,
    "selisih_persen": 0.0,
    "segmen": "Ekonomis",
    "kategori": "Makanan"
  },
  "similar_products": [
    {
      "nama_produk": "Lapis Bogor Sangkuriang Talas Keju Lembut",
      "harga": 54000,
      "jumlah_terjual": 544,
      "rating": 4.94,
      "nama_toko": "Sangkuriang Official",
      "marketplace": "Shopee",
      "kemiripan_persen": 62.4
    }
  ]
}
```

#### Contoh Response Error Deterministik (`400 Bad Request`):
Jika pengguna memasukkan sub-kategori yang tidak sesuai dengan kata kunci produk:
```json
{
  "error": "Sub Kategori tidak sesuai. Kata kunci 'sambal' mengindikasikan sub kategori 'Lauk & Bahan Makanan', bukan 'Makanan Tradisional'."
}
```

Jika produk tidak memiliki pembanding di marketplace (mencegah halusinasi AI):
```json
{
  "error": "Produk ini tidak ditemukan di database referensi 1.027 produk UMKM Bogor kami (kemiripan < 15%). Sistem menolak prediksi untuk mencegah estimasi yang tidak akurat. Silakan sesuaikan nama produk atau pastikan kata kunci relevan."
}
```
