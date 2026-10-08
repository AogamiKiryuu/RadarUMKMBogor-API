import os
import re
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
CORS(app)  # Mengizinkan Nuxt 3 untuk mengambil data dari Flask

# 1. Load Model, Dataset & Market Stats
# ─────────────────────────────────────────────────────────────────────────────
rf_pipeline  = joblib.load('models/model_umkm_bogor_v4.joblib')
df           = pd.read_csv('data/processed/dataset_preprocessed.csv')
market_stats = pd.read_csv('data/market_stats_v4.csv').set_index('kategori')
tren_per_kategori = pd.read_csv('tren_per_kategori.csv')
tren_per_sub = pd.read_csv('tren_per_sub_kategori.csv')

# Market stats per sub_kategori (untuk fitur Paling Digemari)
try:
    market_stats_sub = pd.read_csv('data/market_stats_sub_kategori_v4.csv')
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
    "murah",
    "promo",
    "cod",
    "terlaris",
    "original",
    "ori",
    "asli",
    "oleh",
    "pcs",
    "gr",
    "gram",
    "kg",
    "dan",
    "di",
    "ke",
    "dari",
    "yang",
}

list_stopwords = {
    'murah', 'promo', 'cod', 'terlaris', 'original', 'ori', 'asli',
    'oleh', 'pcs', 'gr', 'gram', 'kg', 'dan', 'di', 'ke', 'dari', 'yang',
    'khas', 'bogor', 'untuk', 'dengan', 'yang', 'ini', 'itu'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in list_stopwords]
    return stemmer.stem(" ".join(words))

# ─────────────────────────────────────────────────────────────────────────────
# 4. Pre-Cache TF-IDF untuk pencarian kompetitor
# ─────────────────────────────────────────────────────────────────────────────
print("Mengoptimasi pencarian kompetitor (Pre-caching TF-IDF)...")
tfidf_vectorizer = rf_pipeline.named_steps["preprocessor"].transformers_[0][1]
X_train_text_db = tfidf_vectorizer.transform(df["nama_produk_clean"].fillna(""))

# ─────────────────────────────────────────────────────────────────────────────
# 5. Konstanta validasi harga
# ─────────────────────────────────────────────────────────────────────────────
# Harga dianggap TIDAK WAJAR jika rasionya terhadap median pasar kategori
# melebihi batas berikut. Artinya harga Anda BATAS_RASIO_TIDAK_WAJAR× lebih
# mahal dari median — sudah di luar batas akal sehat.
BATAS_RASIO_TIDAK_WAJAR = 20.0

# Untuk kategori Makanan & Minuman, batas lebih ketat karena harga komoditas
# cenderung tidak bisa terlalu jauh dari pasar.
BATAS_RASIO_MAKANAN_MINUMAN = 10.0
KATEGORI_KONSUMSI = {"Makanan", "Minuman"}

# ─────────────────────────────────────────────────────────────────────────────
# 6. HELPER: Hitung fitur bisnis relatif terhadap pasar
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

print("✅ Server Flask SIAP DIGUNAKAN! (Model v4 Hybrid Ensemble)")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HELPER: Fitur Bisnis Relatif Pasar
# ─────────────────────────────────────────────────────────────────────────────
def hitung_fitur_bisnis(harga: float, kategori: str) -> dict:
    """
    Menghitung fitur harga RELATIF terhadap pasar per kategori.
    Returns dict dengan: rasio_harga, zscore_harga, log_harga, segmen_harga,
                         median_pasar, selisih_persen
    Menghitung fitur harga RELATIF terhadap pasar per kategori.
    Dipakai untuk inference model v3.
    """
    if kategori in market_stats.index:
        stat = market_stats.loc[kategori]
        median_kat = stat["median_harga_kategori"]
        mean_kat = stat["mean_harga_kategori"]
        std_kat = stat["std_harga_kategori"]
        q25_kat = stat["q25_harga_kategori"]
        q75_kat = stat["q75_harga_kategori"]
        stat       = market_stats.loc[kategori]
        median_kat = stat['median_harga_kategori']
        mean_kat   = stat['mean_harga_kategori']
        std_kat    = stat['std_harga_kategori']
        q25_kat    = stat['q25_harga_kategori']
        q75_kat    = stat['q75_harga_kategori']
    else:
        # Fallback: pakai statistik global jika kategori tidak dikenal
        median_kat = df["harga_produk"].median()
        mean_kat = df["harga_produk"].mean()
        std_kat = df["harga_produk"].std()
        q25_kat = df["harga_produk"].quantile(0.25)
        q75_kat = df["harga_produk"].quantile(0.75)
        median_kat = df['harga_produk'].median()
        mean_kat   = df['harga_produk'].mean()
        std_kat    = df['harga_produk'].std()
        q25_kat    = df['harga_produk'].quantile(0.25)
        q75_kat    = df['harga_produk'].quantile(0.75)

    rasio = min(harga / max(median_kat, 1), 50)
    zscore = float(np.clip((harga - mean_kat) / max(std_kat, 1), -5, 5))
    log_h = float(np.log1p(harga))

    # 0 = Murah, 1 = Menengah, 2 = Mahal  (sebelumnya "premium" → kini "mahal")
    if harga <= q25_kat:
        segmen = 0
    elif harga <= q75_kat:
        segmen = 1
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
        "narasi": (
            f"Secara keseluruhan, kategori paling diminati saat ini adalah '{top_kategori['kategori']}' "
            f"dengan total popularitas {top_kategori['total_popularity']:.1f}. "
            f"Kategori '{kategori_input}' berada di posisi {posisi_kategori if posisi_kategori is not None else '-'} "
            f"dari {len(ranking_list)} kategori."
        ),
        "kategori_terpopuler": str(top_kategori['kategori']),
        "sub_kategori_terpopuler": str(top_sub['sub_kategori']),
        "posisi_kategori_anda": posisi_kategori,
        "total_kategori": len(ranking_list),
        "ranking_kategori": ranking_list,
        "top5_sub_kategori": top_sub_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. HELPER: Bangun data tren keseluruhan untuk output perbandingan
# ─────────────────────────────────────────────────────────────────────────────


def bangun_tren_output(kategori_input: str, sub_kategori_input: str) -> dict:
    """
    Menghasilkan blok tren_keseluruhan yang dikirim di response prediksi.
    Berisi:
      - peringkat semua kategori berdasarkan skor tren
      - posisi kategori & sub-kategori yang diinput user
      - kategori paling diminati secara keseluruhan
      - perbandingan sub-kategori di dalam kategori yang sama
    """
    # --- Semua kategori diurutkan ---
    # Gunakan to_dict("records") agar nilai sudah Python native (bukan pandas Series)
    semua_kategori = []
    for record in tren_per_kategori.to_dict("records"):
        semua_kategori.append(
            {
                "peringkat": int(record["peringkat"]),
                "kategori": str(record["kategori"]),
                "rata_rata_terjual": round(float(record["rata_rata_terjual"]), 1),
                "rata_rata_rating": round(float(record["rata_rata_rating"]), 2),
                "jumlah_produk": int(record["jumlah_produk"]),
                "skor_tren": round(float(record["skor_tren"]), 4),
                "adalah_kategori_anda": str(record["kategori"]) == kategori_input,
            }
        )

    # --- Posisi kategori user ---
    # Cari di semua_kategori (sudah Python dict) agar tidak ada type error pandas
    baris_kat_match = [r for r in semua_kategori if r["kategori"] == kategori_input]
    if baris_kat_match:
        b = baris_kat_match[0]
        total_kat = len(semua_kategori)
        peringkat_k = int(b["peringkat"])
        rata_terjual_kat = float(b["rata_rata_terjual"])
        rata_rating_kat = float(b["rata_rata_rating"])

        if peringkat_k == 1:
            keterangan_kategori = (
                f"Kategori '{kategori_input}' adalah yang PALING DIMINATI "
                f"dari {total_kat} kategori yang ada, dengan rata-rata "
                f"{rata_terjual_kat:.1f} produk terjual dan rating {rata_rating_kat:.2f}."
            )
        else:
            top_kat_rec = semua_kategori[0]  # indeks 0 = peringkat tertinggi
            keterangan_kategori = (
                f"Kategori '{kategori_input}' berada di peringkat {peringkat_k} "
                f"dari {total_kat} kategori. Kategori paling diminati saat ini "
                f"adalah '{top_kat_rec['kategori']}' dengan rata-rata "
                f"{top_kat_rec['rata_rata_terjual']:.1f} produk terjual "
                f"(vs {rata_terjual_kat:.1f} di kategori Anda)."
            )
    else:
        peringkat_k = None
        keterangan_kategori = (
            f"Kategori '{kategori_input}' belum ada dalam data tren historis."
        )

    # --- Kategori paling diminati secara keseluruhan ---
    # semua_kategori[0] sudah peringkat #1 (sudah diurutkan saat retrain)
    paling_diminati = {
        "kategori": semua_kategori[0]["kategori"],
        "rata_rata_terjual": semua_kategori[0]["rata_rata_terjual"],
        "rata_rata_rating": semua_kategori[0]["rata_rata_rating"],
        "skor_tren": semua_kategori[0]["skor_tren"],
    }

    # --- Sub-kategori dalam kategori yang sama ---
    sub_df = tren_per_sub[tren_per_sub["kategori"] == kategori_input].copy()
    semua_sub = []
    if not sub_df.empty:
        # Konversi ke list of dict terlebih dahulu, lalu sort dengan Python native
        sub_records = sorted(
            sub_df.to_dict(orient="records"),  # type: ignore
            key=lambda x: x["skor_tren"],
            reverse=True,
        )
        for rank_i, srecord in enumerate(sub_records, start=1):
            semua_sub.append(
                {
                    "peringkat": rank_i,
                    "sub_kategori": str(srecord["sub_kategori"]),
                    "rata_rata_terjual": round(float(srecord["rata_rata_terjual"]), 1),
                    "rata_rata_rating": round(float(srecord["rata_rata_rating"]), 2),
                    "jumlah_produk": int(srecord["jumlah_produk"]),
                    "skor_tren": round(float(srecord["skor_tren"]), 4),
                    "adalah_sub_anda": str(srecord["sub_kategori"])
                    == sub_kategori_input,
                }
            )

    # --- Keterangan posisi sub-kategori user ---
    # Cari langsung di semua_sub (Python list of dict) — tidak perlu pandas lagi
    sub_match = [s for s in semua_sub if s["sub_kategori"] == sub_kategori_input]
    if sub_match:
        rank_sub = int(sub_match[0]["peringkat"])
        total_sub = len(semua_sub)
        top_sub = semua_sub[0]  # indeks 0 = sub-kategori paling diminati
        if rank_sub == 1:
            keterangan_sub = (
                f"Sub-kategori '{sub_kategori_input}' adalah yang paling diminati "
                f"dalam kategori '{kategori_input}' ({total_sub} sub-kategori)."
            )
        else:
            keterangan_sub = (
                f"Sub-kategori '{sub_kategori_input}' berada di peringkat {rank_sub} "
                f"dari {total_sub} sub-kategori dalam '{kategori_input}'. "
                f"Sub-kategori paling diminati di kategori ini adalah "
                f"'{top_sub['sub_kategori']}' "
                f"(avg terjual {top_sub['rata_rata_terjual']:.1f})."
            )
    else:
        keterangan_sub = (
            f"Sub-kategori '{sub_kategori_input}' belum ada dalam data tren "
            f"untuk kategori '{kategori_input}'."
        )

    return {
        "paling_diminati_keseluruhan": paling_diminati,
        "keterangan_kategori": keterangan_kategori,
        "peringkat_kategori_anda": peringkat_k,
        "total_kategori": len(tren_per_kategori),
        "semua_kategori": semua_kategori,
        "keterangan_sub_kategori": keterangan_sub,
        "semua_sub_kategori": semua_sub,
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
# 8. DATA REFERENSI IDENTITAS BOGOR (Kota & Kabupaten)
# 7. DATA REFERENSI IDENTITAS BOGOR
# ─────────────────────────────────────────────────────────────────────────────
KATA_KUNCI_WILAYAH_BOGOR = [
    # ── Identitas Umum ──────────────────────────────────────────────────────
    "bogor",
    "khas bogor",
    "oleh oleh bogor",
    "produk bogor",
    "umkm bogor",
    "wisata bogor",
    # ── 6 Kecamatan Kota Bogor ──────────────────────────────────────────────
    "bogor barat",
    "bogor selatan",
    "bogor tengah",
    "bogor timur",
    "bogor utara",
    "tanah sareal",
    # ── Kelurahan penting Kota Bogor ────────────────────────────────────────
    # Bogor Tengah
    "babakan pasar",
    "cibogor",
    "ciwaringin",
    "gudang",
    "kebon kelapa",
    "pabaton",
    "paledang",
    "panaragan",
    "sempur",
    "tegal lega",
    "babakan",
    # Bogor Timur
    "baranangsiang",
    "katulampa",
    "tajur",
    "sindangrasa",
    "sindangsari",
    "sukasari",
    # Bogor Utara
    "bantarjati",
    "cibuluh",
    "ciluar",
    "cimahpar",
    "ciparigi",
    "kedunghalang",
    "tanah baru",
    "tegal gundil",
    # Bogor Selatan
    "batutulis",
    "bondongan",
    "cikaret",
    "cipaku",
    "empang",
    "harjasari",
    "muarasari",
    "pakuan",
    "pamoyanan",
    "rancamaya",
    "ranggamekar",
    # Bogor Barat
    "bubulak",
    "cilendek",
    "curug mekar",
    "gunung batu",
    "loji",
    "margajaya",
    "menteng",
    "pasirjaya",
    "semplak",
    "sindangbarang",
    "situgede",
    # Tanah Sareal
    "cibadak",
    "kayumanis",
    "kebonpedes",
    "kedungbadak",
    "kedungjaya",
    "kedungwaringin",
    "kencana",
    "mekarwangi",
    "sukadamai",
    "sukaresmi",
    # Jalan / kawasan ikonik Kota Bogor
    "suryakencana",
    "pajajaran",
    "jl pajajaran",
    "jl suryakencana",
    "pasar bogor",
    "kebun raya bogor",
    "kebun raya",
    "istana bogor",
    # ── 40 Kecamatan Kabupaten Bogor ────────────────────────────────────────
    "babakan madang",
    "bojong gede",
    "bojonggede",
    "caringin",
    "cariu",
    "ciampea",
    "ciawi",
    "cibinong",
    "cibungbulang",
    "cigombong",
    "cigudeg",
    "cijeruk",
    "cileungsi",
    "ciomas",
    "cisarua",
    "ciseeng",
    "citeureup",
    "dramaga",
    "gunung putri",
    "gunung sindur",
    "jasinga",
    "jonggol",
    "kemang",
    "klapanunggal",
    "leuwiliang",
    "leuwisadeng",
    "megamendung",
    "nanggung",
    "pamijahan",
    "parung",
    "parung panjang",
    "parungpanjang",
    "ranca bungur",
    "rancabungur",
    "rumpin",
    "sukajaya",
    "sukamakmur",
    "sukaraja",
    "tajurhalang",
    "tamansari",
    "tanjungsari",
    "tenjo",
    "tenjolaya",
    # ── Kawasan / Tempat Ikonik Bogor ───────────────────────────────────────
    "sentul",
    "sentul city",
    "puncak",
    "kawasan puncak",
    "gunung mas",
    "perkebunan teh gunung mas",
    "taman safari",
    "taman safari indonesia",
    "sirkuit sentul",
    "kebun buah mekarsari",
    "cilember",
    "curug nangka",
    "curug cipendawa",
    "curug cipamingkis",
    "sindanglaya",
    "cipanas",
    "ipb",
    "ipb dramaga",
    "kampus ipb",
    "ipb university",
    "pongkor",
    "antam pongkor",
    # ── Produk & Kuliner Ikonik Kota Bogor ──────────────────────────────────
    # Olahan talas
    "lapis bogor sangkuriang",
    "lapis bogor",
    "lapis talas bogor",
    "lapis talas",
    "keripik talas bogor",
    "keripik talas",
    "talas bogor",
    # Roti & pastri
    "roti unyil venus",
    "roti unyil",
    "pia apple pie bogor",
    "pie bogor",
    "strudel bogor",
    "makaroni panggang bogor",
    "makaroni panggang",
    "bika bogor talubi",
    "bika talubi",
    # Asinan & manisan
    "asinan gedung dalam",
    "asinan bogor gedung dalam",
    "asinan buah bogor",
    "asinan sayur bogor",
    "asinan bogor",
    "manisan pala bogor",
    "manisan bogor",
    # Kuliner tradisional Kota Bogor
    "toge goreng bogor",
    "toge goreng",
    "tauge goreng",
    "laksa bogor",
    "soto mie bogor",
    "soto kuning bogor",
    "soto bogor",
    "gepuk karuhun",
    "gepuk bogor",
    "cungkring",
    "doclang",
    "bapatong",
    "dodongkal",
    "lumpia basah suryakencana",
    "lumpia basah",
    "nasi liwet bogor",
    "nasi liwet",
    "sate maranggi bogor",
    "batagor bogor",
    "siomay bogor",
    "pepes sagu",
    "dodol bogor",
    # Minuman khas Kota Bogor
    "kopi liong bulan",
    "liong bulan",
    "kopi nikmat bogor",
    "es pala bogor",
    "es pala",
    "bir pletok bogor",
    "bir pletok",
    "bajigur",
    "bandrek",
    # Penanda brand lokal
    "sangkuriang",
    "liong",
    "gepuk karuhun",
    "asinan sedap",
    "ngo hiang bogor",
    "ali agrem",
    "cungkring",
    "teng teng bogor",
    "enting enting bogor",
    "renginang bogor",
    "ranginang bogor",
    "noga bogor",
    "kujang bogor",
    "kujang",
    "uncal",
    # ── Produk & Kuliner Ikonik Kabupaten Bogor ─────────────────────────────
    # Kawasan Puncak / Cisarua
    "teh puncak gunung mas",
    "teh gunung mas",
    "teh puncak",
    "kopi puncak",
    "kopi gunung salak",
    "kopi leuwiliang",
    "strawberry puncak",
    "stroberi puncak",
    "stroberi cisarua",
    "strawberry cisarua",
    "selai stroberi puncak",
    "ubi cilembu puncak",
    "ubi bakar cilembu",
    "ubi cilembu",
    "keripik talas puncak",
    "pisang sale puncak",
    "getuk lindri puncak",
    "getuk lindri",
    "ranginang puncak",
    "manisan pala cisarua",
    "sirup pala puncak",
    "sirup pala",
    "cimory puncak",
    "susu cimory",
    "moo moo roll cimory",
    "priangan sari puncak",
    # Kawasan Jasinga & Barat Bogor
    "sangkolun jasinga",
    "sangkolun",
    "dodol tenjo boga rasa",
    "dodol tenjo",
    "tape singkong leuwiliang",
    "tape singkong",
    # Kabupaten umum
    "emping melinjo",
    "emping ciawi",
    "opak cigombong",
    "tempe mendoan cibinong",
    "nasi tutug oncom",
    "sale pisang kabupaten",
    "sale pisang",
    "keripik singkong bogor",
    "susu bogor",
    "madu bogor",
    "jamur bogor",
    "teh bogor",
    "kopi bogor",
    # ── Brand UMKM Terkenal Bogor ────────────────────────────────────────────
    "batik bogor tradisi",
    "kerajinan anyaman bambu bogor",
    "anyaman bambu bogor",
    "kerajinan kayu ciomas",
    "dapur sambel ibu kribo",
    "galeri puncak cisarua",
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

# CATATAN: kunci diurutkan dari LEBIH SPESIFIK ke LEBIH GENERIK.
# Di runtime, sorted(keys, key=len, reverse=True) memastikan kunci terpanjang
# (paling spesifik) dicek lebih dulu, sehingga "lapis bogor sangkuriang"
# menang atas "lapis bogor" yang menang atas "lapis".
PRODUK_KATEGORI_MAP = {
    # ══════════════════════════════════════════════════════════════════════════
    # MAKANAN — Khas Kota Bogor (spesifik → generik)
    # ══════════════════════════════════════════════════════════════════════════
    # Olahan talas — ikon utama UMKM Bogor
    "lapis bogor sangkuriang": "Makanan",  # brand terkemuka
    "lapis talas bogor": "Makanan",
    "keripik talas bogor": "Makanan",
    "lapis bogor": "Makanan",
    "lapis talas": "Makanan",
    "keripik talas": "Makanan",
    "talas bogor": "Makanan",
    "talas": "Makanan",
    # Roti, pastri & kue khas Bogor
    "roti unyil venus": "Makanan",  # brand legendaris
    "pia apple pie bogor": "Makanan",
    "makaroni panggang bogor": "Makanan",
    "bika bogor talubi": "Makanan",
    "strudel bogor": "Makanan",
    "pie talas bogor": "Makanan",
    "roti unyil": "Makanan",
    "pie bogor": "Makanan",
    "pie talas": "Makanan",
    "makaroni panggang": "Makanan",
    "makaroni bogor": "Makanan",
    "bika talubi": "Makanan",
    # Asinan & manisan — oleh-oleh legendaris
    "asinan gedung dalam": "Makanan",  # brand legendaris
    "asinan bogor gedung dalam": "Makanan",
    "asinan buah bogor": "Makanan",
    "asinan sayur bogor": "Makanan",
    "manisan pala bogor": "Makanan",
    "asinan bogor": "Makanan",
    "manisan bogor": "Makanan",
    "asinan buah": "Makanan",
    "asinan sayur": "Makanan",
    "manisan pala": "Makanan",
    "asinan": "Makanan",
    "manisan": "Makanan",
    # Kuliner tradisional Kota Bogor
    "gepuk karuhun": "Makanan",  # brand gepuk terkenal
    "toge goreng bogor": "Makanan",
    "soto kuning bogor": "Makanan",
    "soto mie bogor": "Makanan",
    "laksa bogor": "Makanan",
    "lumpia basah suryakencana": "Makanan",
    "batagor bogor": "Makanan",
    "siomay bogor": "Makanan",
    "nasi liwet bogor": "Makanan",
    "sate maranggi bogor": "Makanan",
    "toge goreng": "Makanan",
    "tauge goreng": "Makanan",
    "gepuk bogor": "Makanan",
    "soto mie": "Makanan",
    "soto kuning": "Makanan",
    "soto bogor": "Makanan",
    "laksa": "Makanan",
    "lumpia basah": "Makanan",
    "batagor": "Makanan",
    "siomay": "Makanan",
    "nasi liwet": "Makanan",
    "sate maranggi": "Makanan",
    "gepuk": "Makanan",
    "cungkring": "Makanan",
    "doclang": "Makanan",
    "bapatong": "Makanan",  # Bakso Kupat Lontong
    "dodongkal": "Makanan",
    "pepes sagu": "Makanan",
    "ngo hiang": "Makanan",
    "ali agrem": "Makanan",
    # Dodol & olahan gula merah
    "dodol bogor": "Makanan",
    "dodol tenjo boga rasa": "Makanan",
    "dodol tenjo": "Makanan",
    "dodol": "Makanan",
    "tauco bogor": "Makanan",
    "tauco": "Makanan",
    # ══════════════════════════════════════════════════════════════════════════
    # MAKANAN — Khas Kabupaten Bogor / Puncak
    # ══════════════════════════════════════════════════════════════════════════
    # Ubi & singkong
    "ubi bakar cilembu": "Makanan",
    "ubi cilembu puncak": "Makanan",
    "ubi cilembu": "Makanan",
    "keripik singkong bogor": "Makanan",
    "tape singkong leuwiliang": "Makanan",
    "tape singkong": "Makanan",
    "singkong": "Makanan",
    "ubi": "Makanan",
    # Puncak & Cisarua
    "keripik talas puncak": "Makanan",
    "pisang sale puncak": "Makanan",
    "getuk lindri puncak": "Makanan",
    "ranginang puncak": "Makanan",
    "manisan pala cisarua": "Makanan",
    "selai stroberi puncak": "Makanan",
    "sale pisang puncak": "Makanan",
    "getuk lindri": "Makanan",
    "ranginang": "Makanan",
    "renginang": "Makanan",
    "sale pisang": "Makanan",
    "pisang sale": "Makanan",
    # Jasinga & Barat Bogor
    "sangkolun jasinga": "Makanan",
    "sangkolun": "Makanan",
    # Kabupaten umum
    "tempe mendoan cibinong": "Makanan",
    "emping melinjo ciawi": "Makanan",
    "opak cigombong": "Makanan",
    "nasi tutug oncom": "Makanan",
    "emping melinjo": "Makanan",
    # Buah segar / produk kebun
    "strawberry puncak": "Makanan",
    "stroberi puncak": "Makanan",
    "stroberi cisarua": "Makanan",
    "strawberry cisarua": "Makanan",
    "stroberi": "Makanan",
    "strawberry": "Makanan",
    # ══════════════════════════════════════════════════════════════════════════
    # MAKANAN — kata kunci generik (selalu cek setelah yang spesifik)
    # ══════════════════════════════════════════════════════════════════════════
    "noga bogor": "Makanan",
    "teng teng bogor": "Makanan",
    "enting enting bogor": "Makanan",
    "renginang bogor": "Makanan",
    "emping bogor": "Makanan",
    "noga": "Makanan",
    "teng teng": "Makanan",
    "enting enting": "Makanan",
    "emping": "Makanan",
    "opak": "Makanan",
    "keripik": "Makanan",
    "camilan": "Makanan",
    "snack": "Makanan",
    "kue": "Makanan",
    "roti": "Makanan",
    "lapis": "Makanan",
    "abon": "Makanan",
    "dendeng": "Makanan",
    "sambal": "Makanan",
    "sambel": "Makanan",
    "tempe": "Makanan",
    "tahu": "Makanan",
    "madu bogor": "Makanan",
    "jamur bogor": "Makanan",
    "madu": "Makanan",
    "jamur": "Makanan",
    "bakso": "Makanan",
    "sosis": "Makanan",
    "nugget": "Makanan",
    "cireng": "Makanan",
    "cilok": "Makanan",
    # ══════════════════════════════════════════════════════════════════════════
    # MINUMAN — Khas Bogor (spesifik → generik)
    # ══════════════════════════════════════════════════════════════════════════
    # Kopi khas Bogor
    "kopi liong bulan": "Minuman",  # legendaris sejak 1945
    "kopi nikmat bogor": "Minuman",  # legendaris sejak 1951
    "kopi gunung salak": "Minuman",  # single origin arabika
    "kopi leuwiliang": "Minuman",
    "kopi puncak": "Minuman",
    "kopi bogor": "Minuman",
    "liong bulan": "Minuman",
    # Teh khas Puncak/Bogor
    "teh gunung mas puncak": "Minuman",
    "teh puncak gunung mas": "Minuman",
    "teh gunung mas": "Minuman",
    "teh puncak": "Minuman",
    "teh cisarua": "Minuman",
    "teh bogor": "Minuman",
    # Susu & dairy
    "susu cimory puncak": "Minuman",
    "moo moo roll cimory": "Minuman",
    "susu cimory": "Minuman",
    "susu bogor": "Minuman",
    "susu": "Minuman",
    # Minuman tradisional & herbal
    "bir pletok bogor": "Minuman",
    "es pala bogor": "Minuman",
    "sirup pala puncak": "Minuman",
    "wedang jahe bogor": "Minuman",
    "bir pletok": "Minuman",
    "es pala": "Minuman",
    "sirup pala": "Minuman",
    "bajigur": "Minuman",
    "bandrek": "Minuman",
    "wedang jahe": "Minuman",
    "cendol": "Minuman",
    # Generik
    "kopi": "Minuman",
    "teh": "Minuman",
    "minuman": "Minuman",
    "jus": "Minuman",
    "sirup": "Minuman",
    "wedang": "Minuman",
    "es jeruk": "Minuman",
    # ══════════════════════════════════════════════════════════════════════════
    # PAKAIAN & FASHION
    # ══════════════════════════════════════════════════════════════════════════
    "batik bogor tradisi": "Pakaian & Fashion",  # brand resmi Kota Bogor
    "batik bogor": "Pakaian & Fashion",
    "batik": "Pakaian & Fashion",
    "kebaya": "Pakaian & Fashion",
    "baju muslim": "Pakaian & Fashion",
    "baju": "Pakaian & Fashion",
    "kaos bogor": "Pakaian & Fashion",
    "kaos": "Pakaian & Fashion",
    "jaket": "Pakaian & Fashion",
    "celana": "Pakaian & Fashion",
    "kemeja": "Pakaian & Fashion",
    "dress": "Pakaian & Fashion",
    "gamis": "Pakaian & Fashion",
    "hijab": "Pakaian & Fashion",
    "kerudung": "Pakaian & Fashion",
    "sandal": "Pakaian & Fashion",
    "sepatu": "Pakaian & Fashion",
    "rok": "Pakaian & Fashion",
    "blus": "Pakaian & Fashion",
    # ══════════════════════════════════════════════════════════════════════════
    # AKSESORIS & SOUVENIR
    # ══════════════════════════════════════════════════════════════════════════
    # Kerajinan khas Bogor
    "kerajinan anyaman bambu bogor": "Aksesoris & Souvenir",
    "anyaman bambu bogor": "Aksesoris & Souvenir",
    "kerajinan bambu bogor": "Aksesoris & Souvenir",
    "kerajinan kayu ciomas": "Aksesoris & Souvenir",
    "miniatur kujang bogor": "Aksesoris & Souvenir",
    "kujang bogor": "Aksesoris & Souvenir",
    "gantungan kunci bogor": "Aksesoris & Souvenir",
    "magnet kulkas bogor": "Aksesoris & Souvenir",
    "souvenir bogor": "Aksesoris & Souvenir",
    "kerajinan anyaman bambu": "Aksesoris & Souvenir",
    "anyaman bambu": "Aksesoris & Souvenir",
    "kerajinan bambu": "Aksesoris & Souvenir",
    "kerajinan kayu": "Aksesoris & Souvenir",
    "miniatur kujang": "Aksesoris & Souvenir",
    "kujang": "Aksesoris & Souvenir",
    "uncal": "Aksesoris & Souvenir",
    # Souvenir generik
    "souvenir": "Aksesoris & Souvenir",
    "gantungan kunci": "Aksesoris & Souvenir",
    "magnet kulkas": "Aksesoris & Souvenir",
    "topi": "Aksesoris & Souvenir",
    "tas": "Aksesoris & Souvenir",
    "dompet": "Aksesoris & Souvenir",
    "gelang": "Aksesoris & Souvenir",
    "kalung": "Aksesoris & Souvenir",
    "cincin": "Aksesoris & Souvenir",
    "bros": "Aksesoris & Souvenir",
    "miniatur": "Aksesoris & Souvenir",
    "aksesoris": "Aksesoris & Souvenir",
    "pernak pernik": "Aksesoris & Souvenir",
    "figura": "Aksesoris & Souvenir",
    "pigura": "Aksesoris & Souvenir",
    "lukisan": "Aksesoris & Souvenir",
    "keramik": "Aksesoris & Souvenir",
    "anyaman": "Aksesoris & Souvenir",
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
# 9. ROUTES
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# 8. ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    top_kat = tren_per_kategori.iloc[0] if not tren_per_kategori.empty else {}
    return jsonify(
        {
            "status": "ok",
            "model": "model_umkm_bogor_v4.joblib",
            "dataset_rows": len(df),
            "kategori_tren": [str(k) for k in tren_per_kategori["kategori"].tolist()],
            "top_kategori": str(top_kat.get("kategori", "-")),
            "message": "Flask ML API v4 siap digunakan ✅ (Model Fair Per Kategori & Bebas Leakage)",
        }
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        nama_produk_input = data.get("nama_produk", "")
        harga_input = float(data.get("harga_produk", 0))
        kategori_input = data.get("kategori", "")
        sub_kategori_input = data.get("sub_kategori", "")
        nama_produk_input  = data.get('nama_produk', '')
        harga_input        = float(data.get('harga_produk', 0))
        kategori_input     = data.get('kategori', '')
        sub_kategori_input = data.get('sub_kategori', '')

        # ── Validasi harga dasar ───────────────────────────────────────────
        if harga_input <= 0:
            return jsonify(
                {
                    "status": "error",
                    "message": "Harga produk tidak boleh 0 atau minus.",
                }
            ), 400

        nama_lower = nama_produk_input.lower()

        # ── VALIDASI 1: Cek mismatch kategori ─────────────────────────────
        produk_terdeteksi = None
        # ── VALIDASI 1: Cek mismatch kategori ─────────────────────────────────
        produk_terdeteksi   = None
        kategori_seharusnya = None
        for kata_kunci in sorted(PRODUK_KATEGORI_MAP.keys(), key=len, reverse=True):
            if kata_kunci in nama_lower:
                produk_terdeteksi = kata_kunci
                kategori_seharusnya = PRODUK_KATEGORI_MAP[kata_kunci]
                break

        if (
            kategori_seharusnya
            and kategori_input
            and kategori_input != kategori_seharusnya
        ):
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        f"Kategori tidak sesuai untuk produk '{nama_produk_input}'. "
                        f"Kata kunci '{produk_terdeteksi}' mengindikasikan produk ini termasuk "
                        f"kategori '{kategori_seharusnya}', bukan '{kategori_input}'. "
                        f"Silakan pilih kategori '{kategori_seharusnya}'."
                    ),
                }
            ), 400
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
        query_clean = clean_text(nama_produk_input)
        query_vec = tfidf_vectorizer.transform([query_clean])
        sim_scores = cosine_similarity(query_vec, X_train_text_db).flatten()
        top_indices = sim_scores.argsort()[-5:][::-1]

        kompetitor_df = df.iloc[top_indices].copy()
        top_sim_scores = sim_scores[top_indices]
        kompetitor_mask = top_sim_scores > 0.05
        kompetitor_df = kompetitor_df[kompetitor_mask]
        filtered_sim_scores = top_sim_scores[kompetitor_mask]

        # ── Hitung skor kemiripan tertinggi untuk deteksi coverage dataset ───
        max_sim_score = float(top_sim_scores[0]) if len(top_sim_scores) > 0 else 0.0

        # ── Estimasi Rating & Jumlah dari Kompetitor ─────────────────────────
        if len(kompetitor_df) > 0:
            rating_est = float(kompetitor_df['rating'].mean())
            jumlah_est = float(kompetitor_df['jumlah_terjual'].median())
        else:
            rating_est = 3.5
            jumlah_est = 30.0

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

        # ── Blokir prediksi jika tidak ada produk serupa di dataset ───────────
        if len(kompetitor_df) == 0:
            return jsonify({
                "status": "error",
                "message": (
                    f"Mohon maaf, produk '{nama_produk_input}' belum tersedia dalam "
                    f"database referensi kami yang dikumpulkan dari hasil scraping marketplace. "
                    f"Tanpa data pembanding, sistem tidak dapat memberikan prediksi yang akurat dan andal."
                ),
                "saran": [
                    "Coba gunakan nama produk yang lebih umum atau lebih spesifik.",
                    f"Tambahkan kata kunci identitas Bogor yang dikenal, misalnya: "
                    f"'Lapis Talas Bogor', 'Kopi Puncak', 'Keripik Bogor', 'Asinan Bogor'.",
                    "Pastikan produk Anda memang dijual atau dikenal di marketplace Bogor.",
                ],
                "catatan": (
                    "Database kami dikumpulkan dari hasil scraping marketplace. "
                    "Kami terus memperbarui data secara berkala — produk Anda mungkin "
                    "akan terdaftar di pembaruan berikutnya."
                ),
            }), 400

        # ── Hitung Fitur Bisnis ────────────────────────────────────────────────
        fitur = hitung_fitur_bisnis(harga_input, kategori_input)
        fitur_bisnis = fitur

        # ── VALIDASI 3: Harga Tidak Wajar ─────────────────────────────────
        batas_rasio = (
            BATAS_RASIO_MAKANAN_MINUMAN
            if kategori_input in KATEGORI_KONSUMSI
            else BATAS_RASIO_TIDAK_WAJAR
        )
        if fitur_bisnis["rasio_harga"] >= batas_rasio:
            median_pasar_fmt = f"Rp{fitur_bisnis['median_pasar']:,.0f}"
            harga_fmt = f"Rp{harga_input:,.0f}"
            pesan_kategori = (
                f" Untuk kategori '{kategori_input}', harga yang masuk akal "
                f"umumnya tidak melebihi {batas_rasio:.0f}× median pasar."
            )
            return jsonify({
                "status": "error",
                "message": (
                    f"Harga {harga_fmt} tidak wajar untuk kategori '{kategori_input}'. "
                    f"Harga ini {fitur_bisnis['rasio_harga']:.1f}× lebih tinggi dari median pasar "
                    f"({median_pasar_fmt}).{pesan_kategori} "
                    f"Prediksi tidak dapat dilakukan karena data pelatihan tidak mencakup "
                    f"rentang harga sejauh ini."
                ),
                "rasio_terhadap_pasar": round(fitur_bisnis["rasio_harga"], 2),
                "median_pasar": round(fitur_bisnis["median_pasar"], 0),
            }), 400

        # ── PREDIKSI Machine Learning v4 (Bebas Target Leakage) ────────────
        input_df = pd.DataFrame([{
            'nama_produk_clean': query_clean,
            'kategori': kategori_input,
            'sub_kategori': sub_kategori_input,
            'rasio_harga': fitur['rasio_harga'],
            'zscore_harga': fitur['zscore_harga'],
            'log_harga': fitur['log_harga'],
            'segmen_harga': fitur['segmen_harga'],
            'rating': rating_est,
        }])

        probabilitas_ml = float(rf_pipeline.predict_proba(input_df)[0][1])

        # ── HYBRID ENSEMBLE: Model ML v4 + Bukti Traksi Pasar (Cosine Similarity) ──
        # Model ML v4 objektif dan bebas target leakage.
        # Namun, produk UMKM legendaris / niche yang di dataset hanya memiliki sedikit sampel
        # (seperti Roti Unyil) tetap memiliki bukti traksi nyata jika ditemukan kompetitor
        # relevan dengan rating tinggi di marketplace.
        if len(kompetitor_df) > 0 and max_sim_score >= 0.15:
            # Normalisasi rating kompetitor (3.0 -> 0.0, 5.0 -> 1.0)
            rating_factor = min(1.0, max(0.0, (rating_est - 3.0) / 2.0))
            # Skor bukti pasar memadukan tingkat kemiripan tekstual dan kepuasan pembeli
            market_proof = (max_sim_score * 0.5) + (rating_factor * 0.5)
            # Bobot ensemble: 75% Model ML + 25% Traksi Bukti Pasar
            probabilitas = (0.75 * probabilitas_ml) + (0.25 * market_proof)
        else:
            probabilitas = probabilitas_ml

        # ── GUARDRAIL: Koreksi Probabilitas untuk Harga Abnormal (Outliers) ──
        # Jika harga terlalu jauh di atas rata-rata pasar, daya saing produk akan tertekan.
        rasio = fitur['rasio_harga']
        if rasio > 1.3:
            if rasio >= 5.0:
                probabilitas = min(probabilitas, 0.02)  # Harga ekstrem (>5x lipat pasar) -> Max 2%
            elif rasio >= 3.0:
                probabilitas = min(probabilitas, 0.15)  # Sangat sulit bersaing -> Max 15%
            elif rasio >= 2.0:
                probabilitas = min(probabilitas, 0.35)  # Sulit laku -> Max 35%
            else:
                probabilitas = min(probabilitas, 0.45)  # Terlalu mahal (>30% pasar) -> Max 45%
        elif rasio < 0.3:
            probabilitas = min(probabilitas, 0.35)  # Terlalu murah (mencurigakan) -> Max 35%

        peluang_persen = round(probabilitas * 100, 1)

        # ── KLASIFIKASI TIERING REALISTIS ──────────────────────────────────
        if probabilitas >= 0.65:
            status_prediksi = f"🌟 SANGAT MENARIK — Peluang laku {peluang_persen}%"
        elif probabilitas >= 0.40:
            status_prediksi = f"✅ CUKUP MENARIK — Peluang laku {peluang_persen}%"
        else:
            status_prediksi = f"⚠️ KURANG MENARIK — Peluang laku {peluang_persen}%"

        # ── BANGUN ALASAN PREDIKSI ─────────────────────────────────────────
        alasan_parts = []
        median_pasar = fitur['median_pasar']
        selisih_persen = fitur['selisih_persen']

        if len(kompetitor_df) > 0:
            avg_terjual_k = float(kompetitor_df['jumlah_terjual'].mean())
            avg_rating_k = float(kompetitor_df['rating'].mean())
            n_k = len(kompetitor_df)

            # 1. Konteks persaingan
            if n_k >= 4:
                alasan_parts.append(
                    f"produk serupa sudah banyak dijual di marketplace ({n_k} kompetitor ditemukan)"
                )
            elif n_k >= 2:
                alasan_parts.append(
                    f"terdapat {n_k} produk serupa di marketplace"
                )
            else:
                alasan_parts.append(
                    "produk ini masih sangat jarang ditemukan di marketplace (potensi pasar terbuka lebar)"
                )

            # 2. Konteks harga vs median pasar kategori
            if selisih_persen > 100:
                alasan_parts.append(
                    f"harga Anda {selisih_persen:.0f}% lebih tinggi dari median pasar kategori ini (Rp{median_pasar:,.0f}) — "
                    f"harga yang terlalu tinggi akan sangat sulit bersaing"
                )
            elif selisih_persen > 30:
                alasan_parts.append(
                    f"harga Anda {selisih_persen:.0f}% di atas median pasar (Rp{median_pasar:,.0f}) — "
                    f"pertimbangkan menyesuaikan harga atau menambah nilai tambah produk"
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
                arah = "di atas" if selisih_persen > 0 else "di bawah"
                alasan_parts.append(
                    f"harga Anda kompetitif, hanya {abs(selisih_persen):.0f}% {arah} median pasar (Rp{median_pasar:,.0f})"
                )

            # 3. Konteks penjualan kompetitor
            if avg_terjual_k >= 100:
                alasan_parts.append(
                    f"produk sejenis terbukti laku keras (rata-rata {avg_terjual_k:.0f} terjual)"
                )
            elif avg_terjual_k >= 20:
                alasan_parts.append(
                    f"produk sejenis memiliki permintaan sedang (rata-rata {avg_terjual_k:.0f} terjual)"
                )
            else:
                alasan_parts.append(
                    f"penjualan produk sejenis di pasar masih berkembang (rata-rata {avg_terjual_k:.0f} terjual)"
                )

            # 4. Konteks rating kompetitor
            if avg_rating_k >= 4.5:
                alasan_parts.append(
                    f"produk serupa memiliki rating sangat tinggi ({avg_rating_k:.2f}/5.0) — standar kualitas di segmen ini tinggi"
                )
            elif avg_rating_k >= 4.0:
                alasan_parts.append(
                    f"rating produk serupa cukup baik ({avg_rating_k:.2f}/5.0)"
                )
        else:
            alasan_parts.append(
                "belum ada produk serupa yang terdeteksi di marketplace, peluang menjadi pionir sangat terbuka"
            )

        # 5. Konteks evaluasi model
        if probabilitas >= 0.65:
            alasan_parts.append(
                "model menilai kombinasi nama produk, kategori, dan posisi harga sangat sesuai tren pasar saat ini"
            )
        elif probabilitas >= 0.40:
            alasan_parts.append(
                "model menilai produk cukup berpotensi di pasar, namun masih ada ruang untuk optimasi harga atau diferensiasi"
            )
        else:
            alasan_parts.append(
                "model menilai produk ini belum cukup kompetitif — pertimbangkan menyesuaikan harga mendekati median pasar atau memperkuat identitas produk"
            )

        # 6. Konteks tren kategori ke alasan
        baris_kat_tren = [
            r for r in tren_per_kategori.to_dict("records")
            if r["kategori"] == kategori_input
        ]
        if baris_kat_tren:
            rank_k = int(baris_kat_tren[0]["peringkat"])
            total_k = len(tren_per_kategori)
            top_kat_nm = str(tren_per_kategori.to_dict("records")[0]["kategori"])
            if rank_k == 1:
                alasan_parts.append(
                    f"kategori '{kategori_input}' adalah kategori yang paling diminati pembeli dari {total_k} kategori yang ada — momentum pasar mendukung"
                )
            else:
                alasan_parts.append(
                    f"kategori '{kategori_input}' berada di peringkat {rank_k}/{total_k} secara keseluruhan; kategori '{top_kat_nm}' saat ini lebih banyak diminati"
                )

        alasan = [p.capitalize() + "." for p in alasan_parts]

        # ── Format Kompetitor ─────────────────────────────────────────────────
        kompetitor_list = []
        for idx, (_, row) in enumerate(kompetitor_df.iterrows()):
            kompetitor_list.append(
                {
                    "nama": str(row["nama_produk"]),
                    "harga": float(row["harga_produk"]),
                    "rating": float(row["rating"]),
                    "terjual": float(row["jumlah_terjual"]),
                    "marketplace": str(row.get("marketplace", "")),
                    "url_produk": str(row.get("url_produk", "")),
                    "kemiripan_persen": round(float(filtered_sim_scores[idx]) * 100, 1),
                }
            )

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

        # ── Bangun Peringatan Dataset (jika produk kurang terwakili) ──────────
        # Catatan: kasus kompetitor=0 sudah diblokir sebelum prediksi,
        # sehingga di sini kompetitor_df dijamin > 0.
        peringatan_dataset = None
        if max_sim_score < 0.15:
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


if __name__ == "__main__":
    app.run(debug=True)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
