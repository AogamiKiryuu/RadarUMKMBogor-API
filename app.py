from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)

print("Memuat Model dan Dataset... Mohon tunggu ⏳")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load Model, Dataset & Market Stats
# ─────────────────────────────────────────────────────────────────────────────
rf_pipeline  = joblib.load('models/model_umkm_bogor_v3.joblib')
df           = pd.read_csv('data/processed/dataset_preprocessed.csv')
market_stats = pd.read_csv('data/market_stats_v3.csv').set_index('kategori')

# Market stats per sub_kategori (untuk fitur Paling Digemari)
try:
    market_stats_sub = pd.read_csv('data/market_stats_sub_kategori_v3.csv')
except FileNotFoundError:
    market_stats_sub = None

# Pastikan kolom popularity_score ada
if 'popularity_score' not in df.columns:
    df['popularity_score'] = df['rating'] * np.log1p(df['jumlah_terjual'])

# ─────────────────────────────────────────────────────────────────────────────
# 2. Setup Sastrawi Stemmer
# ─────────────────────────────────────────────────────────────────────────────
stemmer = StemmerFactory().create_stemmer()
list_stopwords = {
    'murah', 'promo', 'cod', 'terlaris', 'original', 'ori', 'asli',
    'oleh', 'pcs', 'gr', 'gram', 'kg', 'dan', 'di', 'ke', 'dari', 'yang',
    'khas', 'bogor', 'untuk', 'dengan', 'yang', 'ini', 'itu'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [w for w in text.split() if w not in list_stopwords]
    return stemmer.stem(' '.join(words))

# ─────────────────────────────────────────────────────────────────────────────
# 3. Pre-Cache TF-IDF untuk pencarian kompetitor
# ─────────────────────────────────────────────────────────────────────────────
print("Mengoptimasi pencarian kompetitor (Pre-caching TF-IDF)...")
tfidf_vectorizer  = rf_pipeline.named_steps['preprocessor'].transformers_[0][1]
X_train_text_db   = tfidf_vectorizer.transform(df['nama_produk_clean'].fillna(''))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-compute Insight Keseluruhan (di-cache saat startup)
# ─────────────────────────────────────────────────────────────────────────────
def _build_kategori_ranking():
    """Hitung ranking popularitas semua kategori dari dataset referensi."""
    stats = df.groupby('kategori').agg(
        total_popularity  = ('popularity_score', 'sum'),
        avg_popularity    = ('popularity_score', 'mean'),
        total_terjual     = ('jumlah_terjual', 'sum'),
        avg_rating        = ('rating', 'mean'),
        jumlah_produk     = ('nama_produk', 'count'),
        median_harga      = ('harga_produk', 'median'),
    ).reset_index().sort_values('total_popularity', ascending=False)
    return stats

def _build_sub_kategori_ranking():
    """Hitung ranking popularitas semua sub_kategori dari dataset referensi."""
    stats = df.groupby(['kategori', 'sub_kategori']).agg(
        total_popularity  = ('popularity_score', 'sum'),
        avg_popularity    = ('popularity_score', 'mean'),
        total_terjual     = ('jumlah_terjual', 'sum'),
        avg_rating        = ('rating', 'mean'),
        jumlah_produk     = ('nama_produk', 'count'),
        median_harga      = ('harga_produk', 'median'),
    ).reset_index().sort_values('total_popularity', ascending=False)
    return stats

KATEGORI_RANKING     = _build_kategori_ranking()
SUB_KATEGORI_RANKING = _build_sub_kategori_ranking()

print("✅ Server Flask SIAP DIGUNAKAN! (Model v3)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER: Fitur Bisnis Relatif Pasar
# ─────────────────────────────────────────────────────────────────────────────
def hitung_fitur_bisnis(harga: float, kategori: str) -> dict:
    """
    Menghitung fitur harga RELATIF terhadap pasar per kategori.
    Dipakai untuk inference model v3.
    """
    if kategori in market_stats.index:
        stat       = market_stats.loc[kategori]
        median_kat = stat['median_harga_kategori']
        mean_kat   = stat['mean_harga_kategori']
        std_kat    = stat['std_harga_kategori']
        q25_kat    = stat['q25_harga_kategori']
        q75_kat    = stat['q75_harga_kategori']
    else:
        median_kat = df['harga_produk'].median()
        mean_kat   = df['harga_produk'].mean()
        std_kat    = df['harga_produk'].std()
        q25_kat    = df['harga_produk'].quantile(0.25)
        q75_kat    = df['harga_produk'].quantile(0.75)

    rasio  = min(harga / max(median_kat, 1), 50)
    zscore = float(np.clip((harga - mean_kat) / max(std_kat, 1), -5, 5))
    log_h  = float(np.log1p(harga))

    if harga <= q25_kat:    segmen = 0
    elif harga <= q75_kat:  segmen = 1
    else:                   segmen = 2

    return {
        'rasio_harga'    : rasio,
        'zscore_harga'   : zscore,
        'log_harga'      : log_h,
        'segmen_harga'   : segmen,
        'median_pasar'   : float(median_kat),
        'selisih_persen' : ((harga - median_kat) / max(median_kat, 1)) * 100,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPER: Produk Paling Digemari
# ─────────────────────────────────────────────────────────────────────────────
def get_top_produk(kategori: str, sub_kategori: str = None, top_n: int = 5) -> list:
    """
    Mengembalikan daftar produk paling digemari di suatu kategori/sub_kategori.

    Skor popularitas = rating × log1p(jumlah_terjual)
    Menggabungkan kualitas produk (rating) dan volume penjualan (jumlah terjual).
    """
    mask = df['kategori'] == kategori
    if sub_kategori and sub_kategori.strip():
        mask_sub = df['sub_kategori'] == sub_kategori
        subset = df[mask & mask_sub].copy()
        # Fallback ke seluruh kategori jika sub_kategori tidak ada datanya
        if len(subset) == 0:
            subset = df[mask].copy()
    else:
        subset = df[mask].copy()

    if len(subset) == 0:
        return []

    subset = subset.sort_values('popularity_score', ascending=False).head(top_n)

    result = []
    for _, row in subset.iterrows():
        result.append({
            "nama"             : str(row['nama_produk']),
            "kategori"         : str(row['kategori']),
            "sub_kategori"     : str(row['sub_kategori']),
            "harga"            : float(row['harga_produk']),
            "jumlah_terjual"   : float(row['jumlah_terjual']),
            "rating"           : float(row['rating']),
            "popularity_score" : round(float(row['popularity_score']), 2),
            "marketplace"      : str(row.get('marketplace', '')),
            "url_produk"       : str(row.get('url_produk', '')),
            "nama_toko"        : str(row.get('nama_toko', '')),
        })
    return result


def get_insight_keseluruhan(kategori_input: str = None) -> dict:
    """
    Mengembalikan insight pasar keseluruhan:
    - Kategori & sub_kategori mana yang paling digemari
    - Ranking lengkap semua kategori
    - Posisi kategori yang sedang dilihat user
    """
    # Ranking semua kategori
    ranking_list = []
    for i, row in KATEGORI_RANKING.iterrows():
        ranking_list.append({
            "rank"              : int(KATEGORI_RANKING.index.get_loc(i)) + 1,
            "kategori"          : str(row['kategori']),
            "total_popularity"  : round(float(row['total_popularity']), 1),
            "avg_rating"        : round(float(row['avg_rating']), 2),
            "total_terjual"     : int(row['total_terjual']),
            "jumlah_produk"     : int(row['jumlah_produk']),
            "median_harga"      : int(row['median_harga']),
        })

    # Kategori terpopuler
    top_kategori = KATEGORI_RANKING.iloc[0]

    # Sub_kategori terpopuler
    top_sub = SUB_KATEGORI_RANKING.iloc[0]

    # Posisi kategori user dalam ranking
    posisi_kategori = None
    if kategori_input:
        mask = KATEGORI_RANKING['kategori'] == kategori_input
        if mask.any():
            posisi_kategori = int(KATEGORI_RANKING[mask].index[0]) + 1

    # Top 5 sub_kategori dari semua kategori
    top_sub_list = []
    for _, row in SUB_KATEGORI_RANKING.head(5).iterrows():
        top_sub_list.append({
            "kategori"     : str(row['kategori']),
            "sub_kategori" : str(row['sub_kategori']),
            "total_terjual": int(row['total_terjual']),
            "avg_rating"   : round(float(row['avg_rating']), 2),
        })

    return {
        "kategori_terpopuler"    : str(top_kategori['kategori']),
        "sub_kategori_terpopuler": str(top_sub['sub_kategori']),
        "posisi_kategori_anda"   : posisi_kategori,
        "total_kategori"         : len(ranking_list),
        "ranking_kategori"       : ranking_list,
        "top5_sub_kategori"      : top_sub_list,
        "narasi": (
            f"Secara keseluruhan, produk yang paling banyak diminati pembeli adalah kategori "
            f"'{top_kategori['kategori']}' dengan total estimasi {int(top_kategori['total_terjual']):,} penjualan "
            f"dan rata-rata rating {top_kategori['avg_rating']:.2f}. "
            f"Sub-kategori paling populer adalah '{top_sub['sub_kategori']}' "
            f"({top_sub['kategori']}) dengan {int(top_sub['total_terjual']):,} total terjual."
        )
    }

# ─────────────────────────────────────────────────────────────────────────────
# 7. DATA REFERENSI IDENTITAS BOGOR
# ─────────────────────────────────────────────────────────────────────────────
KATA_KUNCI_WILAYAH_BOGOR = [
    'bogor',
    'tanah sareal', 'bogor barat', 'bogor timur', 'bogor selatan', 'bogor utara',
    'bogor tengah', 'sempur', 'ciwaringin', 'gudang', 'paledang', 'babakan',
    'cibinong', 'citeureup', 'gunung putri', 'jonggol', 'cariu', 'tanjungsari',
    'sukamakmur', 'babakan madang', 'sentul', 'sukaraja', 'ciawi', 'cigombong',
    'caringin', 'cisarua', 'puncak', 'megamendung', 'cijeruk', 'kemang',
    'rancabungur', 'parung', 'ciseeng', 'gunung sindur', 'rumpin', 'cigudeg',
    'sukajaya', 'nanggung', 'leuwiliang', 'leuwisadeng', 'pamijahan',
    'cibungbulang', 'ciampea', 'tenjolaya', 'dramaga', 'ciomas', 'tamansari',
    'jasinga', 'tenjo', 'parungpanjang',
    'lapis talas', 'talas bogor', 'roti unyil', 'asinan bogor', 'toge goreng',
    'laksa bogor', 'soto mie bogor', 'batagor bogor', 'dodol bogor',
    'manisan bogor', 'kujang', 'uncal', 'sangkuriang', 'liong',
    'noga', 'teng teng bogor', 'enting enting bogor', 'renginang bogor',
    'pie bogor', 'gepuk bogor', 'tauco bogor', 'ali agrem', 'cungkring',
    'kopi puncak', 'teh puncak', 'strawberry puncak', 'stroberi cisarua',
    'stroberi puncak', 'emping ciawi', 'tauco cibinong', 'kopi bogor',
    'teh bogor', 'susu bogor', 'madu bogor', 'jamur bogor',
]

PRODUK_KATEGORI_MAP = {
    'lapis talas': 'Makanan', 'talas bogor': 'Makanan', 'talas': 'Makanan',
    'roti unyil': 'Makanan', 'asinan': 'Makanan', 'toge goreng': 'Makanan',
    'laksa': 'Makanan', 'soto mie': 'Makanan', 'batagor': 'Makanan',
    'dodol': 'Makanan', 'manisan': 'Makanan', 'noga': 'Makanan',
    'teng teng': 'Makanan', 'enting enting': 'Makanan', 'renginang': 'Makanan',
    'emping': 'Makanan', 'pie bogor': 'Makanan', 'pie talas': 'Makanan',
    'gepuk': 'Makanan', 'tauco': 'Makanan', 'ali agrem': 'Makanan',
    'cungkring': 'Makanan', 'keripik': 'Makanan', 'camilan': 'Makanan',
    'snack': 'Makanan', 'kue': 'Makanan', 'roti': 'Makanan',
    'lapis': 'Makanan', 'abon': 'Makanan', 'dendeng': 'Makanan',
    'sambal': 'Makanan', 'sambel': 'Makanan', 'tempe': 'Makanan',
    'tahu': 'Makanan', 'madu': 'Makanan', 'jamur': 'Makanan',
    'stroberi': 'Makanan', 'strawberry': 'Makanan', 'susu': 'Makanan',
    'kopi': 'Minuman', 'teh': 'Minuman', 'bandrek': 'Minuman',
    'minuman': 'Minuman', 'jus': 'Minuman', 'sirup': 'Minuman', 'wedang': 'Minuman',
    'batik': 'Pakaian & Fashion', 'kebaya': 'Pakaian & Fashion',
    'baju': 'Pakaian & Fashion', 'kaos': 'Pakaian & Fashion',
    'jaket': 'Pakaian & Fashion', 'celana': 'Pakaian & Fashion',
    'kemeja': 'Pakaian & Fashion', 'dress': 'Pakaian & Fashion',
    'kujang': 'Aksesoris & Souvenir', 'uncal': 'Aksesoris & Souvenir',
    'souvenir': 'Aksesoris & Souvenir', 'gantungan kunci': 'Aksesoris & Souvenir',
    'magnet kulkas': 'Aksesoris & Souvenir', 'topi': 'Aksesoris & Souvenir',
    'tas': 'Aksesoris & Souvenir', 'dompet': 'Aksesoris & Souvenir',
    'gelang': 'Aksesoris & Souvenir', 'bros': 'Aksesoris & Souvenir',
    'miniatur': 'Aksesoris & Souvenir',
}

# ─────────────────────────────────────────────────────────────────────────────
# Mapping kata kunci produk → sub kategori yang SEHARUSNYA
# Sub kategori valid berdasarkan dataset:
#   Makanan      : Camilan & Snack | Kue & Roti | Lauk & Bahan Makanan | Makanan Tradisional
#   Minuman      : Kopi | Teh | Minuman Tradisional
#   Pakaian & Fashion : Atasan & Pakaian Kasual | Pakaian Tradisional
#   Aksesoris & Souvenir : Aksesoris & Souvenir
# ─────────────────────────────────────────────────────────────────────────────
PRODUK_SUB_KATEGORI_MAP = {
    # ── Camilan & Snack ──────────────────────────────────────────────────────
    'keripik'      : 'Camilan & Snack',
    'camilan'      : 'Camilan & Snack',
    'snack'        : 'Camilan & Snack',
    'teng teng'    : 'Camilan & Snack',
    'enting enting': 'Camilan & Snack',
    'renginang'    : 'Camilan & Snack',
    'emping'       : 'Camilan & Snack',
    'noga'         : 'Camilan & Snack',
    'kacang'       : 'Camilan & Snack',
    'biji'         : 'Camilan & Snack',
    'crackers'     : 'Camilan & Snack',
    'biskuit'      : 'Camilan & Snack',
    'wafer'        : 'Camilan & Snack',
    'ciki'         : 'Camilan & Snack',
    'kerupuk'      : 'Camilan & Snack',

    # ── Kue & Roti ───────────────────────────────────────────────────────────
    'lapis talas'  : 'Kue & Roti',
    'pie talas'    : 'Kue & Roti',
    'pie bogor'    : 'Kue & Roti',
    'roti unyil'   : 'Kue & Roti',
    'lapis'        : 'Kue & Roti',
    'kue'          : 'Kue & Roti',
    'roti'         : 'Kue & Roti',
    'bolu'         : 'Kue & Roti',
    'brownies'     : 'Kue & Roti',
    'donat'        : 'Kue & Roti',
    'croissant'    : 'Kue & Roti',
    'mochi'        : 'Kue & Roti',
    'nastar'       : 'Kue & Roti',
    'kastengel'    : 'Kue & Roti',
    'putri salju'  : 'Kue & Roti',
    'pancake'      : 'Kue & Roti',
    'pastry'       : 'Kue & Roti',
    'cake'         : 'Kue & Roti',
    'tart'         : 'Kue & Roti',

    # ── Lauk & Bahan Makanan ─────────────────────────────────────────────────
    'abon'         : 'Lauk & Bahan Makanan',
    'dendeng'      : 'Lauk & Bahan Makanan',
    'gepuk'        : 'Lauk & Bahan Makanan',
    'tauco'        : 'Lauk & Bahan Makanan',
    'sambal'       : 'Lauk & Bahan Makanan',
    'sambel'       : 'Lauk & Bahan Makanan',
    'tempe'        : 'Lauk & Bahan Makanan',
    'tahu'         : 'Lauk & Bahan Makanan',
    'madu'         : 'Lauk & Bahan Makanan',
    'jamur'        : 'Lauk & Bahan Makanan',
    'susu'         : 'Lauk & Bahan Makanan',
    'stroberi'     : 'Lauk & Bahan Makanan',
    'strawberry'   : 'Lauk & Bahan Makanan',
    'telur'        : 'Lauk & Bahan Makanan',
    'ayam'         : 'Lauk & Bahan Makanan',
    'daging'       : 'Lauk & Bahan Makanan',
    'ikan'         : 'Lauk & Bahan Makanan',
    'bumbu'        : 'Lauk & Bahan Makanan',
    'saus'         : 'Lauk & Bahan Makanan',
    'saos'         : 'Lauk & Bahan Makanan',
    'kecap'        : 'Lauk & Bahan Makanan',
    'pete'         : 'Lauk & Bahan Makanan',
    'jengkol'      : 'Lauk & Bahan Makanan',

    # ── Makanan Tradisional ──────────────────────────────────────────────────
    'asinan'       : 'Makanan Tradisional',
    'toge goreng'  : 'Makanan Tradisional',
    'laksa'        : 'Makanan Tradisional',
    'soto mie'     : 'Makanan Tradisional',
    'batagor'      : 'Makanan Tradisional',
    'dodol'        : 'Makanan Tradisional',
    'manisan'      : 'Makanan Tradisional',
    'ali agrem'    : 'Makanan Tradisional',
    'cungkring'    : 'Makanan Tradisional',
    'nasi'         : 'Makanan Tradisional',
    'ketupat'      : 'Makanan Tradisional',
    'opak'         : 'Makanan Tradisional',
    'ongol'        : 'Makanan Tradisional',
    'geplak'       : 'Makanan Tradisional',
    'wajik'        : 'Makanan Tradisional',

    # ── Kopi ─────────────────────────────────────────────────────────────────
    'kopi'         : 'Kopi',
    'espresso'     : 'Kopi',
    'arabika'      : 'Kopi',
    'robusta'      : 'Kopi',
    'cold brew'    : 'Kopi',
    'cappuccino'   : 'Kopi',
    'latte'        : 'Kopi',

    # ── Teh ──────────────────────────────────────────────────────────────────
    'teh'          : 'Teh',
    'green tea'    : 'Teh',
    'matcha'       : 'Teh',
    'chamomile'    : 'Teh',
    'jasmine'      : 'Teh',

    # ── Minuman Tradisional ──────────────────────────────────────────────────
    'bandrek'      : 'Minuman Tradisional',
    'bajigur'      : 'Minuman Tradisional',
    'wedang'       : 'Minuman Tradisional',
    'jus'          : 'Minuman Tradisional',
    'sirup'        : 'Minuman Tradisional',
    'minuman'      : 'Minuman Tradisional',
    'es'           : 'Minuman Tradisional',
    'sari'         : 'Minuman Tradisional',

    # ── Atasan & Pakaian Kasual ──────────────────────────────────────────────
    'baju'         : 'Atasan & Pakaian Kasual',
    'kaos'         : 'Atasan & Pakaian Kasual',
    'kemeja'       : 'Atasan & Pakaian Kasual',
    'dress'        : 'Atasan & Pakaian Kasual',
    'jaket'        : 'Atasan & Pakaian Kasual',
    'blouse'       : 'Atasan & Pakaian Kasual',
    'sweater'      : 'Atasan & Pakaian Kasual',
    'hoodie'       : 'Atasan & Pakaian Kasual',
    'celana'       : 'Atasan & Pakaian Kasual',
    'rok'          : 'Atasan & Pakaian Kasual',
    'legging'      : 'Atasan & Pakaian Kasual',
    'kaus'         : 'Atasan & Pakaian Kasual',
    'polo'         : 'Atasan & Pakaian Kasual',

    # ── Pakaian Tradisional ──────────────────────────────────────────────────
    'batik'        : 'Pakaian Tradisional',
    'kebaya'       : 'Pakaian Tradisional',
    'sarung'       : 'Pakaian Tradisional',
    'kain'         : 'Pakaian Tradisional',
    'songket'      : 'Pakaian Tradisional',
    'tenun'        : 'Pakaian Tradisional',
    'lurik'        : 'Pakaian Tradisional',
    'beskap'       : 'Pakaian Tradisional',

    # ── Aksesoris & Souvenir ─────────────────────────────────────────────────
    'kujang'       : 'Aksesoris & Souvenir',
    'uncal'        : 'Aksesoris & Souvenir',
    'souvenir'     : 'Aksesoris & Souvenir',
    'gantungan kunci': 'Aksesoris & Souvenir',
    'magnet kulkas': 'Aksesoris & Souvenir',
    'topi'         : 'Aksesoris & Souvenir',
    'tas'          : 'Aksesoris & Souvenir',
    'dompet'       : 'Aksesoris & Souvenir',
    'gelang'       : 'Aksesoris & Souvenir',
    'bros'         : 'Aksesoris & Souvenir',
    'miniatur'     : 'Aksesoris & Souvenir',
    'pin'          : 'Aksesoris & Souvenir',
    'kalung'       : 'Aksesoris & Souvenir',
    'cincin'       : 'Aksesoris & Souvenir',
    'anting'       : 'Aksesoris & Souvenir',
    'hiasan'       : 'Aksesoris & Souvenir',
    'boneka'       : 'Aksesoris & Souvenir',
    'pigura'       : 'Aksesoris & Souvenir',
    'payung'       : 'Aksesoris & Souvenir',
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status"       : "ok",
        "model"        : "model_umkm_bogor_v3.joblib",
        "dataset_rows" : len(df),
        "versi"        : "v3",
        "fitur_baru"   : ["jumlah_log", "revenue_proxy_log", "popularity_score", "produk_terpopuler"],
        "message"      : "Flask ML API v3 siap ✅ (fitur Paling Digemari aktif)"
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        nama_produk_input  = data.get('nama_produk', '')
        harga_input        = float(data.get('harga_produk', 0))
        kategori_input     = data.get('kategori', '')
        sub_kategori_input = data.get('sub_kategori', '')

        # Validasi Harga
        if harga_input <= 0:
            return jsonify({"status": "error", "message": "Harga produk tidak boleh 0 atau minus."}), 400

        nama_lower = nama_produk_input.lower()

        # ── VALIDASI 1: Cek mismatch kategori ─────────────────────────────────
        produk_terdeteksi   = None
        kategori_seharusnya = None
        for kata_kunci in sorted(PRODUK_KATEGORI_MAP.keys(), key=len, reverse=True):
            if kata_kunci in nama_lower:
                produk_terdeteksi   = kata_kunci
                kategori_seharusnya = PRODUK_KATEGORI_MAP[kata_kunci]
                break

        if kategori_seharusnya and kategori_input and kategori_input != kategori_seharusnya:
            return jsonify({
                "status" : "error",
                "message": (
                    f"Kategori tidak sesuai untuk produk '{nama_produk_input}'. "
                    f"Kata kunci '{produk_terdeteksi}' mengindikasikan kategori "
                    f"'{kategori_seharusnya}', bukan '{kategori_input}'."
                )
            }), 400

        # ── VALIDASI 1b: Cek mismatch sub kategori ────────────────────────────
        sub_terdeteksi        = None
        sub_kategori_seharusnya = None
        for kata_kunci in sorted(PRODUK_SUB_KATEGORI_MAP.keys(), key=len, reverse=True):
            if kata_kunci in nama_lower:
                sub_terdeteksi          = kata_kunci
                sub_kategori_seharusnya = PRODUK_SUB_KATEGORI_MAP[kata_kunci]
                break

        if sub_kategori_seharusnya and sub_kategori_input and sub_kategori_input != sub_kategori_seharusnya:
            return jsonify({
                "status" : "error",
                "message": (
                    f"Sub Kategori tidak sesuai untuk produk '{nama_produk_input}'. "
                    f"Kata kunci '{sub_terdeteksi}' mengindikasikan sub kategori "
                    f"'{sub_kategori_seharusnya}', bukan '{sub_kategori_input}'. "
                    f"Silakan pilih sub kategori yang tepat."
                )
            }), 400

        # ── VALIDASI 2: Identitas Bogor ────────────────────────────────────────
        mengandung_identitas_bogor = any(kata in nama_lower for kata in KATA_KUNCI_WILAYAH_BOGOR)

        # ── Cari Kompetitor — Cosine Similarity ───────────────────────────────
        query_clean  = clean_text(nama_produk_input)
        query_vec    = tfidf_vectorizer.transform([query_clean])
        sim_scores   = cosine_similarity(query_vec, X_train_text_db).flatten()
        top_indices  = sim_scores.argsort()[-6:][::-1]

        kompetitor_df      = df.iloc[top_indices].copy()
        top_sim_scores     = sim_scores[top_indices]
        kompetitor_mask    = top_sim_scores > 0.05
        kompetitor_df      = kompetitor_df[kompetitor_mask]
        filtered_sim       = top_sim_scores[kompetitor_mask]

        # ── Hitung skor kemiripan tertinggi untuk deteksi coverage dataset ───
        max_sim_score = float(top_sim_scores[0]) if len(top_sim_scores) > 0 else 0.0

        # ── Estimasi Rating & Jumlah dari Kompetitor ─────────────────────────
        if len(kompetitor_df) > 0:
            rating_est = float(kompetitor_df['rating'].mean())
            jumlah_est = float(kompetitor_df['jumlah_terjual'].median())
        else:
            rating_est = 3.5
            jumlah_est = 30.0

        # ── Validasi identitas Bogor ───────────────────────────────────────────
        if not mengandung_identitas_bogor and len(kompetitor_df) == 0:
            return jsonify({
                "status" : "error",
                "message": (
                    f"Produk '{nama_produk_input}' tidak terdeteksi sebagai produk khas Bogor. "
                    f"Pastikan nama produk mengandung identitas lokal seperti 'Khas Bogor', "
                    f"nama kawasan (Puncak, Cisarua, Cibinong, Dramaga, dll), "
                    f"atau produk ikonik (Lapis Talas, Roti Unyil, Kopi Puncak, Renginang, dll)."
                )
            }), 400

        if not mengandung_identitas_bogor:
            return jsonify({
                "status" : "warning",
                "message": (
                    f"Produk '{nama_produk_input}' tidak mencantumkan identitas Bogor secara eksplisit. "
                    f"Tambahkan 'Khas Bogor', nama kawasan, atau produk ikonik pada nama produk Anda."
                )
            }), 400

        # ── Hitung Fitur Bisnis ────────────────────────────────────────────────
        fitur = hitung_fitur_bisnis(harga_input, kategori_input)

        jumlah_log        = float(np.log1p(jumlah_est))
        revenue_proxy_log = float(np.log1p(harga_input * jumlah_est))
        popularity_score  = float(rating_est * np.log1p(jumlah_est))

        # ── Prediksi ML ───────────────────────────────────────────────────────
        input_df = pd.DataFrame([{
            'nama_produk_clean' : query_clean,
            'kategori'          : kategori_input,
            'sub_kategori'      : sub_kategori_input,
            'rasio_harga'       : fitur['rasio_harga'],
            'zscore_harga'      : fitur['zscore_harga'],
            'log_harga'         : fitur['log_harga'],
            'segmen_harga'      : fitur['segmen_harga'],
            'rating'            : rating_est,
            'jumlah_log'        : jumlah_log,
            'revenue_proxy_log' : revenue_proxy_log,
            'popularity_score_new': popularity_score,
        }])

        probabilitas   = rf_pipeline.predict_proba(input_df)[0][1]
        
        # ── GUARDRAIL: Koreksi Probabilitas untuk Harga Abnormal (Outliers) ──
        # Berdasarkan bisnis rules: Jika harga terlalu jauh di atas rata-rata pasar,
        # mustahil produk akan laku, terlepas dari apa prediksi murni Random Forest.
        rasio = fitur['rasio_harga']
        
        if rasio > 1.3:
            # Harga > 30% dari median pasar (seharusnya Kurang Menarik / < 0.5)
            if rasio >= 5.0:
                probabilitas = min(probabilitas, 0.02) # Harga gila (>5x lipat pasar) -> Max 2%
            elif rasio >= 3.0:
                probabilitas = min(probabilitas, 0.15) # Sangat sulit bersaing -> Max 15%
            elif rasio >= 2.0:
                probabilitas = min(probabilitas, 0.35) # Sulit laku -> Max 35%
            else:
                probabilitas = min(probabilitas, 0.45) # Kurang menarik -> Max 45%
        elif rasio < 0.3:
            # Harga terlalu murah (< 30% dari median pasar, selisih < -70%)
            probabilitas = min(probabilitas, 0.35) # Mencurigakan -> Max 35%

        peluang_persen = round(probabilitas * 100, 1)

        if probabilitas >= 0.7:
            status_prediksi = f"🌟 SANGAT MENARIK — Peluang laku {peluang_persen}%"
        elif probabilitas >= 0.5:
            status_prediksi = f"✅ CUKUP MENARIK — Peluang laku {peluang_persen}%"
        else:
            status_prediksi = f"⚠️ KURANG MENARIK — Peluang laku {peluang_persen}%"

        # ── Bangun Alasan Prediksi ─────────────────────────────────────────────
        alasan_parts    = []
        median_pasar    = fitur['median_pasar']
        selisih_persen  = fitur['selisih_persen']

        if len(kompetitor_df) > 0:
            avg_harga_k   = kompetitor_df['harga_produk'].mean()
            avg_terjual_k = kompetitor_df['jumlah_terjual'].mean()
            n_k           = len(kompetitor_df)

            if n_k >= 4:
                alasan_parts.append(f"produk serupa sudah banyak dijual ({n_k} kompetitor ditemukan)")
            elif n_k >= 2:
                alasan_parts.append(f"terdapat {n_k} produk serupa di marketplace")
            else:
                alasan_parts.append("produk ini masih sangat jarang di marketplace (peluang terbuka lebar)")

            if selisih_persen > 100:
                alasan_parts.append(
                    f"harga Anda {selisih_persen:.0f}% di atas median pasar (Rp{median_pasar:,.0f}) — "
                    f"sangat sulit bersaing"
                )
            elif selisih_persen > 30:
                alasan_parts.append(
                    f"harga Anda {selisih_persen:.0f}% di atas median pasar (Rp{median_pasar:,.0f}) — "
                    f"pertimbangkan menyesuaikan harga atau menambah nilai tambah"
                )
            elif selisih_persen < -70:
                alasan_parts.append(
                    f"harga Anda {abs(selisih_persen):.0f}% di bawah median pasar (Rp{median_pasar:,.0f}) — "
                    f"terlalu murah, berisiko dianggap mencurigakan atau merusak harga pasar"
                )
            elif selisih_persen < -30:
                alasan_parts.append(
                    f"harga Anda {abs(selisih_persen):.0f}% di bawah median pasar (Rp{median_pasar:,.0f}) — "
                    f"sangat kompetitif, berpotensi menarik banyak pembeli"
                )
            else:
                alasan_parts.append(
                    f"harga Anda kompetitif, hanya {abs(selisih_persen):.0f}% "
                    f"{'di atas' if selisih_persen > 0 else 'di bawah'} median pasar (Rp{median_pasar:,.0f})"
                )

            if avg_terjual_k >= 100:
                alasan_parts.append(f"produk sejenis terbukti laku keras (rata-rata {avg_terjual_k:.0f} terjual)")
            elif avg_terjual_k >= 20:
                alasan_parts.append(f"produk sejenis memiliki permintaan sedang (rata-rata {avg_terjual_k:.0f} terjual)")
            else:
                alasan_parts.append(f"penjualan produk sejenis masih rendah (rata-rata {avg_terjual_k:.0f} terjual)")
        else:
            alasan_parts.append("belum ada produk serupa yang terdeteksi, peluang menjadi yang pertama sangat besar")

        if probabilitas >= 0.7:
            alasan_parts.append("model menilai kombinasi nama, kategori, dan posisi harga sangat sesuai tren pasar")
        elif probabilitas >= 0.5:
            alasan_parts.append("model menilai produk cukup berpotensi, masih ada ruang untuk optimasi")
        else:
            alasan_parts.append("model menilai produk belum cukup kompetitif — sesuaikan harga atau perkuat identitas")

        alasan = [p.capitalize() + "." for p in alasan_parts]

        # ── Format Kompetitor ─────────────────────────────────────────────────
        kompetitor_list = []
        for idx, (_, row) in enumerate(kompetitor_df.iterrows()):
            kompetitor_list.append({
                "nama"             : row['nama_produk'],
                "harga"            : float(row['harga_produk']),
                "rating"           : float(row['rating']),
                "terjual"          : float(row['jumlah_terjual']),
                "marketplace"      : str(row.get('marketplace', '')),
                "url_produk"       : str(row.get('url_produk', '')),
                "kemiripan_persen" : round(float(filtered_sim[idx]) * 100, 1),
            })

        # ── Produk Paling Digemari di Kategori/Sub-Kategori ini ──────────────
        top_produk = get_top_produk(kategori_input, sub_kategori_input, top_n=5)

        # ── Insight Pasar Keseluruhan ─────────────────────────────────────────
        insight_pasar = get_insight_keseluruhan(kategori_input)

        # ── Sub-kategori terpopuler dalam kategori yang sama ──────────────────
        sub_ranking_in_kat = SUB_KATEGORI_RANKING[
            SUB_KATEGORI_RANKING['kategori'] == kategori_input
        ].head(5)
        sub_ranking_list = []
        for _, row in sub_ranking_in_kat.iterrows():
            sub_ranking_list.append({
                "sub_kategori"  : str(row['sub_kategori']),
                "total_terjual" : int(row['total_terjual']),
                "avg_rating"    : round(float(row['avg_rating']), 2),
                "jumlah_produk" : int(row['jumlah_produk']),
            })

        # ── Bangun Peringatan Dataset (jika produk tidak/kurang terwakili) ──
        peringatan_dataset = None
        if len(kompetitor_df) == 0:
            # Produk tidak ditemukan sama sekali di dataset
            peringatan_dataset = {
                "level"   : "tidak_ditemukan",
                "judul"   : "⚠️ Produk Belum Ada di Dataset Kami",
                "pesan"   : (
                    f"Mohon maaf, produk '{nama_produk_input}' belum tersedia dalam "
                    f"database referensi kami yang dikumpulkan dari hasil scraping marketplace. "
                    f"Karena tidak ada data pembanding yang ditemukan, hasil prediksi ini "
                    f"sepenuhnya didasarkan pada estimasi kategori '{kategori_input}' "
                    f"dan posisi harga relatif terhadap pasar — bukan pada data penjualan "
                    f"produk serupa secara langsung. Gunakan hasilnya sebagai gambaran umum, "
                    f"bukan sebagai acuan pasti."
                ),
                "saran"   : (
                    "Coba periksa kembali nama produk, atau tambahkan kata kunci yang lebih "
                    "spesifik agar sistem dapat menemukan produk serupa di database."
                ),
                "akurasi_prediksi": "rendah",
            }
        elif max_sim_score < 0.15:
            # Produk ditemukan tapi kemiripannya sangat rendah
            peringatan_dataset = {
                "level"   : "kemiripan_rendah",
                "judul"   : "ℹ️ Data Referensi Produk Terbatas",
                "pesan"   : (
                    f"Produk '{nama_produk_input}' belum banyak terwakili dalam database "
                    f"referensi kami (kemiripan produk serupa: {max_sim_score*100:.0f}%). "
                    f"Data kami dikumpulkan dari scraping marketplace, sehingga tidak semua "
                    f"produk UMKM Bogor tercakup. Hasil prediksi menggunakan data produk "
                    f"terdekat yang tersedia sebagai estimasi."
                ),
                "saran"   : (
                    "Hasil prediksi tetap dapat dijadikan referensi, namun disarankan "
                    "untuk membandingkan dengan kondisi pasar aktual secara langsung."
                ),
                "akurasi_prediksi": "sedang",
            }
        elif max_sim_score < 0.35:
            # Kemiripan sedang — perlu notifikasi ringan
            peringatan_dataset = {
                "level"   : "kemiripan_sedang",
                "judul"   : "📊 Prediksi Berbasis Data Produk Serupa",
                "pesan"   : (
                    f"Produk '{nama_produk_input}' tidak ditemukan secara persis di database kami, "
                    f"namun terdapat {len(kompetitor_df)} produk serupa dengan kemiripan "
                    f"{max_sim_score*100:.0f}% yang digunakan sebagai acuan prediksi."
                ),
                "saran"   : None,
                "akurasi_prediksi": "cukup_baik",
            }

        # ── RESPONSE ──────────────────────────────────────────────────────────
        return jsonify({
            "status"              : "success",
            "kesimpulan"          : status_prediksi,
            "peluang_laku_persen" : peluang_persen,
            "alasan"              : alasan,
            "peringatan_dataset"  : peringatan_dataset,

            "konteks_harga": {
                "median_pasar"   : round(median_pasar, 0),
                "rasio_vs_pasar" : round(fitur['rasio_harga'], 2),
                "segmen"         : ["Murah", "Menengah", "Premium"][fitur['segmen_harga']],
                "selisih_persen" : round(selisih_persen, 1),
            },

            "kompetitor": kompetitor_list,

            # ── FITUR BARU v3 ──────────────────────────────────────────────────
            "produk_terpopuler": {
                "label"   : f"Top 5 Produk Paling Digemari di '{kategori_input}"
                            + (f" — {sub_kategori_input}'" if sub_kategori_input else "'"),
                "deskripsi": (
                    f"Produk-produk di bawah ini adalah yang paling diminati pembeli "
                    f"berdasarkan kombinasi jumlah penjualan dan rating tertinggi "
                    f"dalam kategori {kategori_input}"
                    + (f" sub-kategori {sub_kategori_input}" if sub_kategori_input else "")
                    + "."
                ),
                "produk": top_produk,
            },

            "insight_pasar": {
                "narasi"                 : insight_pasar['narasi'],
                "kategori_terpopuler"    : insight_pasar['kategori_terpopuler'],
                "sub_kategori_terpopuler": insight_pasar['sub_kategori_terpopuler'],
                "posisi_kategori_anda"   : insight_pasar['posisi_kategori_anda'],
                "total_kategori"         : insight_pasar['total_kategori'],
                "ranking_semua_kategori" : insight_pasar['ranking_kategori'],
                "top5_sub_kategori_global": insight_pasar['top5_sub_kategori'],
                "sub_kategori_dalam_kategori_ini": sub_ranking_list,
            },
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
