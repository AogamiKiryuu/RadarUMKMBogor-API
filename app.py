from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app) # Mengizinkan Nuxt 3 untuk mengambil data dari Flask

print("Memuat Model dan Dataset... Mohon tunggu ⏳")

# 1. Load Model & Dataset
rf_pipeline = joblib.load('model_umkm_bogor.joblib')
df = pd.read_csv('dataset_umkm_bogor.csv')

# 2. Setup Sastrawi Stemmer
stemmer = StemmerFactory().create_stemmer()
list_stopwords = {'murah', 'promo', 'cod', 'terlaris', 'original', 'ori', 'asli', 'oleh', 'pcs', 'gr', 'gram', 'kg', 'dan', 'di', 'ke', 'dari', 'yang'}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in list_stopwords]
    return stemmer.stem(' '.join(words))

# 3. Pre-Cache TF-IDF untuk mempercepat pencarian
print("Mengoptimasi pencarian kompetitor (Pre-caching TF-IDF)...")
tfidf_vectorizer = rf_pipeline.named_steps['preprocessor'].transformers_[0][1]
X_train_text_db = tfidf_vectorizer.transform(df['nama_produk_clean'].fillna(''))
print("✅ Server Flask SIAP DIGUNAKAN!")

# ─────────────────────────────────────────────────────────────────────────────
# 4. DATA REFERENSI IDENTITAS BOGOR (Kota & Kabupaten)
# ─────────────────────────────────────────────────────────────────────────────

# Kata kunci wilayah & produk ikonik Bogor (untuk deteksi identitas)
KATA_KUNCI_WILAYAH_BOGOR = [
    # Umum
    'bogor',
    # Kecamatan / Daerah Kota Bogor
    'tanah sareal', 'bogor barat', 'bogor timur', 'bogor selatan', 'bogor utara',
    'bogor tengah', 'sempur', 'ciwaringin', 'gudang', 'paledang', 'babakan',
    # Kecamatan / Daerah Kabupaten Bogor
    'cibinong', 'citeureup', 'gunung putri', 'jonggol', 'cariu', 'tanjungsari',
    'sukamakmur', 'babakan madang', 'sentul', 'sukaraja', 'ciawi', 'cigombong',
    'caringin', 'cisarua', 'puncak', 'megamendung', 'cijeruk', 'kemang',
    'rancabungur', 'parung', 'ciseeng', 'gunung sindur', 'rumpin', 'cigudeg',
    'sukajaya', 'nanggung', 'leuwiliang', 'leuwisadeng', 'pamijahan',
    'cibungbulang', 'ciampea', 'tenjolaya', 'dramaga', 'ciomas', 'tamansari',
    'jasinga', 'tenjo', 'parungpanjang',
    # Produk & ikon khas Kota Bogor
    'lapis talas', 'talas bogor', 'roti unyil', 'asinan bogor', 'toge goreng',
    'laksa bogor', 'soto mie bogor', 'batagor bogor', 'dodol bogor',
    'manisan bogor', 'kujang', 'uncal', 'sangkuriang', 'liong',
    'noga', 'teng teng bogor', 'enting enting bogor', 'renginang bogor',
    'pie bogor', 'gepuk bogor', 'tauco bogor', 'ali agrem', 'cungkring',
    # Produk & ikon khas Kabupaten Bogor
    'kopi puncak', 'teh puncak', 'strawberry puncak', 'stroberi cisarua',
    'stroberi puncak', 'emping ciawi', 'tauco cibinong', 'kopi bogor',
    'teh bogor', 'susu bogor', 'madu bogor', 'jamur bogor',
]

# Peta produk ikonik Bogor → kategori yang BENAR
# Digunakan untuk mendeteksi mismatch kategori yang jelas salah
PRODUK_KATEGORI_MAP = {
    # ── MAKANAN ──────────────────────────────────────────────────────────────
    'lapis talas'       : 'Makanan',
    'talas bogor'       : 'Makanan',
    'talas'             : 'Makanan',
    'roti unyil'        : 'Makanan',
    'asinan'            : 'Makanan',
    'toge goreng'       : 'Makanan',
    'laksa'             : 'Makanan',
    'soto mie'          : 'Makanan',
    'batagor'           : 'Makanan',
    'dodol'             : 'Makanan',
    'manisan'           : 'Makanan',
    'noga'              : 'Makanan',
    'teng teng'         : 'Makanan',
    'enting enting'     : 'Makanan',
    'renginang'         : 'Makanan',
    'emping'            : 'Makanan',
    'pie bogor'         : 'Makanan',
    'pie talas'         : 'Makanan',
    'gepuk'             : 'Makanan',
    'tauco'             : 'Makanan',
    'ali agrem'         : 'Makanan',
    'cungkring'         : 'Makanan',
    'keripik'           : 'Makanan',
    'camilan'           : 'Makanan',
    'snack'             : 'Makanan',
    'kue'               : 'Makanan',
    'roti'              : 'Makanan',
    'lapis'             : 'Makanan',
    'abon'              : 'Makanan',
    'dendeng'           : 'Makanan',
    'sambal'            : 'Makanan',
    'sambel'            : 'Makanan',
    'tempe'             : 'Makanan',
    'tahu'              : 'Makanan',
    'madu'              : 'Makanan',
    'jamur'             : 'Makanan',
    'stroberi'          : 'Makanan',
    'strawberry'        : 'Makanan',
    'susu'              : 'Makanan',
    # ── MINUMAN ──────────────────────────────────────────────────────────────
    'kopi'              : 'Minuman',
    'teh'               : 'Minuman',
    'bandrek'           : 'Minuman',
    'minuman'           : 'Minuman',
    'jus'               : 'Minuman',
    'sirup'             : 'Minuman',
    'wedang'            : 'Minuman',
    # ── PAKAIAN & FASHION ────────────────────────────────────────────────────
    'batik'             : 'Pakaian & Fashion',
    'kebaya'            : 'Pakaian & Fashion',
    'baju'              : 'Pakaian & Fashion',
    'kaos'              : 'Pakaian & Fashion',
    'jaket'             : 'Pakaian & Fashion',
    'celana'            : 'Pakaian & Fashion',
    'kemeja'            : 'Pakaian & Fashion',
    'dress'             : 'Pakaian & Fashion',
    # ── AKSESORIS & SOUVENIR ─────────────────────────────────────────────────
    'kujang'            : 'Aksesoris & Souvenir',
    'uncal'             : 'Aksesoris & Souvenir',
    'souvenir'          : 'Aksesoris & Souvenir',
    'gantungan kunci'   : 'Aksesoris & Souvenir',
    'magnet kulkas'     : 'Aksesoris & Souvenir',
    'topi'              : 'Aksesoris & Souvenir',
    'tas'               : 'Aksesoris & Souvenir',
    'dompet'            : 'Aksesoris & Souvenir',
    'gelang'            : 'Aksesoris & Souvenir',
    'bros'              : 'Aksesoris & Souvenir',
    'miniatur'          : 'Aksesoris & Souvenir',
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": "model_umkm_bogor.joblib",
        "dataset_rows": len(df),
        "message": "Flask ML API siap digunakan ✅"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        nama_produk_input = data.get('nama_produk', '')
        harga_input = float(data.get('harga_produk', 0))
        # Rating bersifat opsional — default 0.0 untuk pedagang yang belum punya toko online
        rating_input = float(data.get('rating', 0.0))
        kategori_input = data.get('kategori', '')
        sub_kategori_input = data.get('sub_kategori', '')

        # Validasi Harga
        if harga_input <= 0:
            return jsonify({"status": "error", "message": "Harga produk tidak boleh 0 atau minus."}), 400

        nama_lower = nama_produk_input.lower()

        # ── VALIDASI 1: Cek mismatch kategori berdasarkan produk terdeteksi ──
        # Sort by key length descending agar multi-word key cocok lebih dulu
        produk_terdeteksi = None
        kategori_seharusnya = None
        for kata_kunci in sorted(PRODUK_KATEGORI_MAP.keys(), key=len, reverse=True):
            if kata_kunci in nama_lower:
                produk_terdeteksi = kata_kunci
                kategori_seharusnya = PRODUK_KATEGORI_MAP[kata_kunci]
                break

        if kategori_seharusnya and kategori_input and kategori_input != kategori_seharusnya:
            return jsonify({
                "status": "error",
                "message": (
                    f"Kategori tidak sesuai untuk produk '{nama_produk_input}'. "
                    f"Kata kunci '{produk_terdeteksi}' mengindikasikan produk ini termasuk "
                    f"kategori '{kategori_seharusnya}', bukan '{kategori_input}'. "
                    f"Silakan pilih kategori '{kategori_seharusnya}'."
                )
            }), 400

        # ── VALIDASI 2: Deteksi Identitas Bogor (Kota & Kabupaten) ──────────
        mengandung_identitas_bogor = any(kata in nama_lower for kata in KATA_KUNCI_WILAYAH_BOGOR)

        # ── Cari Kompetitor menggunakan Cosine Similarity ────────────────────
        query_clean = clean_text(nama_produk_input)
        query_vec = tfidf_vectorizer.transform([query_clean])
        sim_scores = cosine_similarity(query_vec, X_train_text_db).flatten()
        top_indices = sim_scores.argsort()[-5:][::-1]

        kompetitor_df = df.iloc[top_indices].copy()
        top_sim_scores = sim_scores[top_indices]
        kompetitor_mask = top_sim_scores > 0.05
        kompetitor_df = kompetitor_df[kompetitor_mask]
        filtered_sim_scores = top_sim_scores[kompetitor_mask]

        # Jika tidak ada identitas Bogor sama sekali → error
        if not mengandung_identitas_bogor and len(kompetitor_df) == 0:
            return jsonify({
                "status": "error",
                "message": (
                    f"Produk '{nama_produk_input}' tidak terdeteksi sebagai produk khas Bogor "
                    f"(Kota maupun Kabupaten). Pastikan nama produk mengandung identitas lokal Bogor, "
                    f"seperti 'Khas Bogor', nama kawasan (Puncak, Cisarua, Cibinong, Dramaga, dll), "
                    f"atau produk ikonik (Lapis Talas, Roti Unyil, Kopi Puncak, Renginang, dll)."
                )
            }), 400

        # Jika produk mirip ditemukan tapi tidak ada identitas Bogor di nama → peringatan
        if not mengandung_identitas_bogor:
            return jsonify({
                "status": "warning",
                "message": (
                    f"Produk '{nama_produk_input}' tidak secara eksplisit mencantumkan identitas Bogor. "
                    f"Untuk memperkuat positioning sebagai produk UMKM Bogor, tambahkan kata kunci "
                    f"seperti 'Khas Bogor', nama kawasan (Puncak, Cisarua, Cibinong, Dramaga, dll), "
                    f"atau produk ikonik pada nama produk Anda."
                )
            }), 400

        # ── PREDIKSI Machine Learning ────────────────────────────────────────
        input_df = pd.DataFrame([{
            'nama_produk_clean': query_clean, 'kategori': kategori_input,
            'sub_kategori': sub_kategori_input, 'harga_produk': harga_input, 'rating': rating_input
        }])

        probabilitas = rf_pipeline.predict_proba(input_df)[0][1]
        status_prediksi = "🌟 SANGAT MENARIK" if probabilitas >= 0.7 else ("✅ CUKUP MENARIK" if probabilitas >= 0.5 else "⚠️ KURANG MENARIK")

        # ── BANGUN ALASAN PREDIKSI ────────────────────────────────────────────
        alasan_parts = []

        if len(kompetitor_df) > 0:
            avg_harga_kompetitor = kompetitor_df['harga_produk'].mean()
            avg_terjual_kompetitor = kompetitor_df['jumlah_terjual'].mean()
            avg_rating_kompetitor = kompetitor_df['rating'].mean()
            jumlah_kompetitor = len(kompetitor_df)

            # Konteks persaingan
            if jumlah_kompetitor >= 4:
                alasan_parts.append(f"produk serupa sudah banyak dijual di marketplace ({jumlah_kompetitor} kompetitor ditemukan)")
            elif jumlah_kompetitor >= 2:
                alasan_parts.append(f"terdapat {jumlah_kompetitor} produk serupa di marketplace")
            else:
                alasan_parts.append("produk ini masih sangat jarang ditemukan di marketplace (potensi pasar terbuka lebar)")

            # Konteks harga
            selisih_persen = ((harga_input - avg_harga_kompetitor) / avg_harga_kompetitor) * 100
            if selisih_persen > 20:
                alasan_parts.append(f"harga Anda {abs(selisih_persen):.0f}% lebih tinggi dari rata-rata kompetitor (Rp{avg_harga_kompetitor:,.0f})")
            elif selisih_persen < -20:
                alasan_parts.append(f"harga Anda {abs(selisih_persen):.0f}% lebih murah dari rata-rata kompetitor (Rp{avg_harga_kompetitor:,.0f})")
            else:
                alasan_parts.append(f"harga Anda sudah kompetitif dibanding rata-rata pasar (Rp{avg_harga_kompetitor:,.0f})")

            # Konteks penjualan kompetitor
            if avg_terjual_kompetitor >= 100:
                alasan_parts.append(f"produk sejenis terbukti laku keras dengan rata-rata {avg_terjual_kompetitor:.0f} terjual")
            elif avg_terjual_kompetitor >= 20:
                alasan_parts.append(f"produk sejenis memiliki permintaan sedang dengan rata-rata {avg_terjual_kompetitor:.0f} terjual")
            else:
                alasan_parts.append(f"penjualan produk sejenis di pasar masih rendah (rata-rata {avg_terjual_kompetitor:.0f} terjual)")
        else:
            alasan_parts.append("belum ada produk serupa yang terdeteksi di marketplace, peluang untuk menjadi yang pertama sangat besar")

        # Konteks probabilitas akhir
        if probabilitas >= 0.7:
            alasan_parts.append("model menilai kombinasi nama, kategori, dan harga Anda sangat sesuai dengan tren pasar saat ini")
        elif probabilitas >= 0.5:
            alasan_parts.append("model menilai produk Anda cukup berpotensi, namun masih ada ruang untuk optimasi harga atau penamaan")
        else:
            alasan_parts.append("model menilai produk ini belum cukup kompetitif — pertimbangkan menyesuaikan harga atau memperkuat identitas produk")

        alasan = "; ".join(alasan_parts).capitalize() + "."

        # ── FORMAT KOMPETITOR (termasuk URL & marketplace) ───────────────────
        kompetitor_list = []
        for idx, (_, row) in enumerate(kompetitor_df.iterrows()):
            kompetitor_list.append({
                "nama": row['nama_produk'],
                "harga": float(row['harga_produk']),
                "rating": float(row['rating']),
                "terjual": float(row['jumlah_terjual']),
                "marketplace": str(row.get('marketplace', '')),
                "url_produk": str(row.get('url_produk', '')),
                "kemiripan_persen": round(float(filtered_sim_scores[idx]) * 100, 1),
            })

        # ── KIRIM RESPONSE ────────────────────────────────────────────────────
        return jsonify({
            "status": "success",
            "kesimpulan": status_prediksi,
            "peluang_laku_persen": round(probabilitas * 100, 1),
            "alasan": alasan,
            "kompetitor": kompetitor_list
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)