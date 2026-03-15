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
rf_pipeline = joblib.load('model_umkm_bogor_v2.joblib')
df = pd.read_csv('dataset_umkm_bogor.csv')

# 2. Load statistik pasar per kategori (dihasilkan saat retrain)
#    Dipakai untuk menghitung rasio_harga, zscore_harga, segmen_harga
market_stats = pd.read_csv('market_stats_per_kategori.csv').set_index('kategori')

# 3. Setup Sastrawi Stemmer
stemmer = StemmerFactory().create_stemmer()
list_stopwords = {'murah', 'promo', 'cod', 'terlaris', 'original', 'ori', 'asli', 'oleh', 'pcs', 'gr', 'gram', 'kg', 'dan', 'di', 'ke', 'dari', 'yang'}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in list_stopwords]
    return stemmer.stem(' '.join(words))

# 4. Pre-Cache TF-IDF untuk mempercepat pencarian
print("Mengoptimasi pencarian kompetitor (Pre-caching TF-IDF)...")
tfidf_vectorizer = rf_pipeline.named_steps['preprocessor'].transformers_[0][1]
X_train_text_db = tfidf_vectorizer.transform(df['nama_produk_clean'].fillna(''))
print("✅ Server Flask SIAP DIGUNAKAN!")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER: Hitung fitur bisnis relatif terhadap pasar
# ─────────────────────────────────────────────────────────────────────────────

def hitung_fitur_bisnis(harga: float, kategori: str) -> dict:
    """
    Menghitung fitur harga yang RELATIF terhadap pasar per kategori.
    Ini yang membuat model 'mengerti' apakah harga mahal atau murah.

    Returns dict dengan: rasio_harga, zscore_harga, log_harga, segmen_harga
    """
    if kategori in market_stats.index:
        stat = market_stats.loc[kategori]
        median_kat = stat['median_harga_kategori']
        mean_kat   = stat['mean_harga_kategori']
        std_kat    = stat['std_harga_kategori']
        q25_kat    = stat['q25_harga_kategori']
        q75_kat    = stat['q75_harga_kategori']
    else:
        # Fallback: pakai statistik global jika kategori tidak dikenal
        median_kat = df['harga_produk'].median()
        mean_kat   = df['harga_produk'].mean()
        std_kat    = df['harga_produk'].std()
        q25_kat    = df['harga_produk'].quantile(0.25)
        q75_kat    = df['harga_produk'].quantile(0.75)

    # Rasio harga vs median pasar (fitur paling penting secara bisnis)
    rasio = harga / max(median_kat, 1)
    rasio = min(rasio, 50)  # cap

    # Z-score dalam kategori
    zscore = (harga - mean_kat) / max(std_kat, 1)
    zscore = max(-5, min(5, zscore))  # clip

    # Log transform
    log_h = np.log1p(harga)

    # Segmen: 0=murah, 1=menengah, 2=premium
    if harga <= q25_kat:
        segmen = 0
    elif harga <= q75_kat:
        segmen = 1
    else:
        segmen = 2

    return {
        'rasio_harga': rasio,
        'zscore_harga': zscore,
        'log_harga': log_h,
        'segmen_harga': segmen,
        'median_pasar': median_kat,
        'selisih_persen': ((harga - median_kat) / max(median_kat, 1)) * 100,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. DATA REFERENSI IDENTITAS BOGOR (Kota & Kabupaten)
# ─────────────────────────────────────────────────────────────────────────────

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
        "model": "model_umkm_bogor_v2.joblib",
        "dataset_rows": len(df),
        "message": "Flask ML API v2 siap digunakan ✅ (fitur harga relatif pasar aktif)"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        nama_produk_input   = data.get('nama_produk', '')
        harga_input         = float(data.get('harga_produk', 0))
        kategori_input      = data.get('kategori', '')
        sub_kategori_input  = data.get('sub_kategori', '')

        # Validasi Harga
        if harga_input <= 0:
            return jsonify({"status": "error", "message": "Harga produk tidak boleh 0 atau minus."}), 400

        nama_lower = nama_produk_input.lower()

        # ── VALIDASI 1: Cek mismatch kategori ──────────────────────────────
        produk_terdeteksi  = None
        kategori_seharusnya = None
        for kata_kunci in sorted(PRODUK_KATEGORI_MAP.keys(), key=len, reverse=True):
            if kata_kunci in nama_lower:
                produk_terdeteksi   = kata_kunci
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

        # ── VALIDASI 2: Deteksi Identitas Bogor ────────────────────────────
        mengandung_identitas_bogor = any(kata in nama_lower for kata in KATA_KUNCI_WILAYAH_BOGOR)

        # ── Cari Kompetitor menggunakan Cosine Similarity ──────────────────
        query_clean = clean_text(nama_produk_input)
        query_vec   = tfidf_vectorizer.transform([query_clean])
        sim_scores  = cosine_similarity(query_vec, X_train_text_db).flatten()
        top_indices = sim_scores.argsort()[-5:][::-1]

        kompetitor_df       = df.iloc[top_indices].copy()
        top_sim_scores      = sim_scores[top_indices]
        kompetitor_mask     = top_sim_scores > 0.05
        kompetitor_df       = kompetitor_df[kompetitor_mask]
        filtered_sim_scores = top_sim_scores[kompetitor_mask]

        # ── Rating proxy dari kompetitor ───────────────────────────────────
        if len(kompetitor_df) > 0:
            rating_input = float(kompetitor_df['rating'].mean())
        else:
            rating_input = 3.5

        # Validasi identitas Bogor
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

        # ── HITUNG FITUR BISNIS RELATIF PASAR ─────────────────────────────
        # Inilah inti perbaikan v2: model sekarang menerima harga yang
        # sudah dinormalisasi terhadap konteks pasar per kategori.
        fitur_bisnis = hitung_fitur_bisnis(harga_input, kategori_input)

        # ── PREDIKSI Machine Learning ──────────────────────────────────────
        input_df = pd.DataFrame([{
            'nama_produk_clean' : query_clean,
            'kategori'          : kategori_input,
            'sub_kategori'      : sub_kategori_input,
            'rasio_harga'       : fitur_bisnis['rasio_harga'],
            'zscore_harga'      : fitur_bisnis['zscore_harga'],
            'log_harga'         : fitur_bisnis['log_harga'],
            'segmen_harga'      : fitur_bisnis['segmen_harga'],
            'rating'            : rating_input,
        }])

        probabilitas    = rf_pipeline.predict_proba(input_df)[0][1]
        peluang_persen  = round(probabilitas * 100, 1)

        if probabilitas >= 0.7:
            status_prediksi = f"🌟 SANGAT MENARIK — Model memprediksi peluang laku {peluang_persen}%"
        elif probabilitas >= 0.5:
            status_prediksi = f"✅ CUKUP MENARIK — Model memprediksi peluang laku {peluang_persen}%"
        else:
            status_prediksi = f"⚠️ KURANG MENARIK — Model memprediksi peluang laku {peluang_persen}%"

        # ── BANGUN ALASAN PREDIKSI ─────────────────────────────────────────
        alasan_parts = []
        median_pasar    = fitur_bisnis['median_pasar']
        selisih_persen  = fitur_bisnis['selisih_persen']

        if len(kompetitor_df) > 0:
            avg_harga_kompetitor    = kompetitor_df['harga_produk'].mean()
            avg_terjual_kompetitor  = kompetitor_df['jumlah_terjual'].mean()
            avg_rating_kompetitor   = kompetitor_df['rating'].mean()
            jumlah_kompetitor       = len(kompetitor_df)

            # Konteks persaingan
            if jumlah_kompetitor >= 4:
                alasan_parts.append(f"produk serupa sudah banyak dijual di marketplace ({jumlah_kompetitor} kompetitor ditemukan)")
            elif jumlah_kompetitor >= 2:
                alasan_parts.append(f"terdapat {jumlah_kompetitor} produk serupa di marketplace")
            else:
                alasan_parts.append("produk ini masih sangat jarang ditemukan di marketplace (potensi pasar terbuka lebar)")

            # Konteks harga — kini dijelaskan vs MEDIAN PASAR kategori
            if selisih_persen > 100:
                alasan_parts.append(
                    f"harga Anda {selisih_persen:.0f}% lebih tinggi dari median pasar kategori ini "
                    f"(Rp{median_pasar:,.0f}) — harga yang terlalu tinggi akan sangat sulit bersaing"
                )
            elif selisih_persen > 30:
                alasan_parts.append(
                    f"harga Anda {selisih_persen:.0f}% di atas median pasar (Rp{median_pasar:,.0f}) — "
                    f"pertimbangkan menurunkan harga atau menambah nilai tambah produk"
                )
            elif selisih_persen < -30:
                alasan_parts.append(
                    f"harga Anda {abs(selisih_persen):.0f}% di bawah median pasar (Rp{median_pasar:,.0f}) — "
                    f"sangat kompetitif, berpotensi menarik banyak pembeli"
                )
            else:
                alasan_parts.append(
                    f"harga Anda sudah kompetitif, hanya {abs(selisih_persen):.0f}% "
                    f"{'di atas' if selisih_persen > 0 else 'di bawah'} median pasar (Rp{median_pasar:,.0f})"
                )

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
            alasan_parts.append("model menilai kombinasi nama, kategori, dan posisi harga Anda sangat sesuai dengan tren pasar saat ini")
        elif probabilitas >= 0.5:
            alasan_parts.append("model menilai produk Anda cukup berpotensi, namun masih ada ruang untuk optimasi harga atau penamaan")
        else:
            alasan_parts.append("model menilai produk ini belum cukup kompetitif — pertimbangkan menyesuaikan harga mendekati median pasar atau memperkuat identitas produk")

        alasan = [part.capitalize() + "." for part in alasan_parts]

        # ── FORMAT KOMPETITOR ──────────────────────────────────────────────
        kompetitor_list = []
        for idx, (_, row) in enumerate(kompetitor_df.iterrows()):
            kompetitor_list.append({
                "nama"              : row['nama_produk'],
                "harga"             : float(row['harga_produk']),
                "rating"            : float(row['rating']),
                "terjual"           : float(row['jumlah_terjual']),
                "marketplace"       : str(row.get('marketplace', '')),
                "url_produk"        : str(row.get('url_produk', '')),
                "kemiripan_persen"  : round(float(filtered_sim_scores[idx]) * 100, 1),
            })

        # ── KIRIM RESPONSE ─────────────────────────────────────────────────
        return jsonify({
            "status"                : "success",
            "kesimpulan"            : status_prediksi,
            "peluang_laku_persen"   : peluang_persen,
            "alasan"                : alasan,
            "konteks_harga"         : {
                "median_pasar"      : round(median_pasar, 0),
                "rasio_vs_pasar"    : round(fitur_bisnis['rasio_harga'], 2),
                "segmen"            : ["Murah", "Menengah", "Premium"][fitur_bisnis['segmen_harga']],
                "selisih_persen"    : round(selisih_persen, 1),
            },
            "kompetitor"            : kompetitor_list,
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
