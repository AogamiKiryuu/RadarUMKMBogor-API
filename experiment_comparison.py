"""
Eksperimen Perbandingan Model & Ekstraksi Fitur Teks (Revisi Bu Uti — 26 Nov 2026)
==================================================================================
Membandingkan secara adil (apple-to-apple, bebas data leakage & bebas ties):
- 3 Algoritma Klasifikasi: Random Forest (Bagging) vs XGBoost vs LightGBM (Boosting)
- 4 Metode Ekstraksi Fitur Teks:
  1. TF-IDF Unigram (1,1)
  2. TF-IDF Unigram + Bigram (1,2) [Baseline v4]
  3. Word2Vec (Skip-Gram 100d, jika gensim tersedia)
  4. IndoBERT (indobenchmark/indobert-base-p1, Mean Pooling 768d)
- Studi Kasus Konteks Kata: "Lapis Talas" (Makanan) vs "Gantungan Kunci Miniatur Lapis Talas" (Souvenir)
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

try:
    from gensim.models import Word2Vec
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_INDOBERT = True
except ImportError:
    HAS_INDOBERT = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_preprocessed.csv")
OUT_JSON = os.path.join(BASE_DIR, "data", "benchmark_bu_uti_results.json")
OUT_CSV = os.path.join(BASE_DIR, "data", "benchmark_bu_uti_results.csv")
EMB_CACHE = os.path.join(BASE_DIR, "data", "indobert_embeddings_base_p1.npy")


def prepare_data():
    df = pd.read_csv(DATA_PATH)
    # Labeling intra-kategori bebas ties (rank method='average')
    df["rank_terjual_kat"] = df.groupby("kategori")["jumlah_terjual"].rank(method="average", pct=True)
    df["label"] = (df["rank_terjual_kat"] > 0.5).astype(int)

    # Split 80:20 Stratified SEBELUM menghitung statistik harga pasar
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Statistik harga HANYA dari train_df
    stat_kat = train_df.groupby("kategori")["harga_produk"].agg(
        median_harga_kategori="median",
        mean_harga_kategori="mean",
        std_harga_kategori="std",
        q25_harga_kategori=lambda x: x.quantile(0.25),
        q75_harga_kategori=lambda x: x.quantile(0.75),
    ).reset_index()

    def apply_market(d):
        d = d.merge(stat_kat, on="kategori", how="left")
        d["rasio_harga"] = (d["harga_produk"] / d["median_harga_kategori"].replace(0, np.nan)).fillna(1.0).clip(upper=50)
        d["zscore_harga"] = ((d["harga_produk"] - d["mean_harga_kategori"]) / d["std_harga_kategori"].replace(0, 1)).clip(-5, 5)
        d["log_harga"] = np.log1p(d["harga_produk"])
        d["segmen_harga"] = np.where(
            d["harga_produk"] <= d["q25_harga_kategori"], 0,
            np.where(d["harga_produk"] <= d["q75_harga_kategori"], 1, 2)
        )
        return d

    train_idx = train_df.index.to_numpy()
    test_idx = test_df.index.to_numpy()
    train_df = apply_market(train_df)
    test_df = apply_market(test_df)

    # Tabular features (OHE + StandardScaler fit hanya pada train_df)
    cat_cols = ["kategori", "sub_kategori"]
    num_cols = ["rasio_harga", "zscore_harga", "log_harga", "segmen_harga", "rating"]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    X_tr_tab = np.hstack([ohe.fit_transform(train_df[cat_cols]), scaler.fit_transform(train_df[num_cols])])
    X_te_tab = np.hstack([ohe.transform(test_df[cat_cols]), scaler.transform(test_df[num_cols])])

    return df, train_df, test_df, train_idx, test_idx, X_tr_tab, X_te_tab


def get_indobert_embeddings(texts, cache_path=None, model_name="indobenchmark/indobert-base-p1"):
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path)
        if len(arr) == len(texts):
            print(f"✅ Memuat cache IndoBERT embeddings: {arr.shape}")
            return arr, None, None

    print(f"⏳ Memuat model {model_name} dan mengekstrak embedding ({len(texts)} teks)...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embs = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt")
            out = model(**enc)
            # Mean pooling dengan attention mask
            mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            summed = torch.sum(out.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (summed / counts).numpy()
            all_embs.append(mean_pooled)

    embs = np.vstack(all_embs)
    if cache_path:
        np.save(cache_path, embs)
    print(f"✅ Ekstraksi IndoBERT selesai dalam {time.time() - t0:.1f}s -> shape {embs.shape}")
    return embs, tokenizer, model


def encode_indobert_single(texts, tokenizer, model):
    with torch.no_grad():
        enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
        out = model(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        mean_pooled = (torch.sum(out.last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)).numpy()
    return mean_pooled


def get_w2v_features(train_texts, test_texts, vector_size=100):
    tr_tokens = [str(t).split() for t in train_texts]
    te_tokens = [str(t).split() for t in test_texts]
    w2v = Word2Vec(sentences=tr_tokens, vector_size=vector_size, window=5, min_count=1, sg=1, epochs=60, seed=42, workers=1)

    def doc_vec(tokens):
        vecs = [w2v.wv[w] for w in tokens if w in w2v.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(vector_size)

    return np.vstack([doc_vec(t) for t in tr_tokens]), np.vstack([doc_vec(t) for t in te_tokens]), w2v


def build_classifiers(y_train):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / max(pos, 1)
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_split=3, min_samples_leaf=1,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
            random_state=42, eval_metric="logloss", n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=42, verbose=-1, n_jobs=-1
        ),
    }


def evaluate_combo(feat_name, clf_name, clf_builder, X_tr, y_tr, X_te, y_te):
    clf = clf_builder()
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred)
    rec = recall_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    cm = confusion_matrix(y_te, y_pred)

    # 5-Fold Stratified CV pada Train Set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs, cv_aucs = [], []
    for tr_i, va_i in skf.split(X_tr, y_tr):
        m = clf_builder()
        m.fit(X_tr[tr_i], y_tr[tr_i])
        vp = m.predict(X_tr[va_i])
        vpr = m.predict_proba(X_tr[va_i])[:, 1]
        cv_accs.append(accuracy_score(y_tr[va_i], vp))
        cv_aucs.append(roc_auc_score(y_tr[va_i], vpr))

    return {
        "feature_extraction": feat_name,
        "model": clf_name,
        "text_dim": int(X_tr.shape[1] - 19),
        "total_dim": int(X_tr.shape[1]),
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(prec) * 100, 2),
        "recall": round(float(rec) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "roc_auc": round(float(auc), 4),
        "cv_acc_mean": round(float(np.mean(cv_accs)) * 100, 2),
        "cv_acc_std": round(float(np.std(cv_accs)) * 100, 2),
        "cv_auc_mean": round(float(np.mean(cv_aucs)), 4),
        "cm": cm.tolist(),
    }


def main():
    print("=" * 72)
    print("🚀 BENCHMARK PERBANDINGAN MODEL & EKSTRAKSI FITUR TEKS (REVISI BU UTI)")
    print("=" * 72)

    df, train_df, test_df, train_idx, test_idx, X_tr_tab, X_te_tab = prepare_data()
    y_tr = train_df["label"].to_numpy()
    y_te = test_df["label"].to_numpy()

    tr_clean = train_df["nama_produk_clean"].fillna("").tolist()
    te_clean = test_df["nama_produk_clean"].fillna("").tolist()

    feature_sets = {}

    # 1. TF-IDF Unigram (1,1)
    tf1 = TfidfVectorizer(max_features=1000, ngram_range=(1, 1), sublinear_tf=True)
    X_tr_tf1 = tf1.fit_transform(tr_clean).toarray()
    X_te_tf1 = tf1.transform(te_clean).toarray()
    feature_sets["TF-IDF Unigram (1,1)"] = (np.hstack([X_tr_tf1, X_tr_tab]), np.hstack([X_te_tf1, X_te_tab]))

    # 2. TF-IDF Unigram + Bigram (1,2) [Baseline v4]
    tf2 = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_tf2 = tf2.fit_transform(tr_clean).toarray()
    X_te_tf2 = tf2.transform(te_clean).toarray()
    feature_sets["TF-IDF Unigram+Bigram (1,2)"] = (np.hstack([X_tr_tf2, X_tr_tab]), np.hstack([X_te_tf2, X_te_tab]))

    # 3. Word2Vec (Skip-gram 100d)
    if HAS_GENSIM:
        print("⏳ Melatih Word2Vec Skip-Gram (100d) pada data latih...")
        X_tr_w2v, X_te_w2v, w2v_model = get_w2v_features(tr_clean, te_clean, vector_size=100)
        feature_sets["Word2Vec (Skip-Gram 100d)"] = (np.hstack([X_tr_w2v, X_tr_tab]), np.hstack([X_te_w2v, X_te_tab]))
    else:
        print("⚠️ gensim tidak tersedia, melewati Word2Vec.")

    # 4. IndoBERT (indobenchmark/indobert-base-p1)
    indobert_tok, indobert_mod = None, None
    if HAS_INDOBERT:
        all_raw = df["nama_produk"].fillna("").tolist()
        all_embs, indobert_tok, indobert_mod = get_indobert_embeddings(all_raw, cache_path=EMB_CACHE)
        X_tr_bert = all_embs[train_idx]
        X_te_bert = all_embs[test_idx]
        feature_sets["IndoBERT (Base-P1 768d)"] = (np.hstack([X_tr_bert, X_tr_tab]), np.hstack([X_te_bert, X_te_tab]))
    else:
        print("⚠️ transformers/torch tidak tersedia, melewati IndoBERT.")

    # Evaluasi seluruh kombinasi
    results = []
    for feat_name, (X_tr, X_te) in feature_sets.items():
        for clf_name in ["Random Forest", "XGBoost", "LightGBM"]:
            builder = lambda cn=clf_name: build_classifiers(y_tr)[cn]
            res = evaluate_combo(feat_name, clf_name, builder, X_tr, y_tr, X_te, y_te)
            results.append(res)
            print(
                f"  [{feat_name:28s}] × [{clf_name:13s}] -> "
                f"Acc={res['accuracy']:5.2f}% | F1={res['f1_score']:5.2f}% | "
                f"AUC={res['roc_auc']:.4f} | CV={res['cv_acc_mean']:5.2f}%±{res['cv_acc_std']:.2f}%"
            )

    res_df = pd.DataFrame(results).sort_values(by=["roc_auc", "accuracy"], ascending=False)
    print("\n" + "=" * 72)
    print("📊 RINGKASAN PERINGKAT KOMBINASI (DIURUTKAN BERDASARKAN ROC-AUC & AKURASI)")
    print("=" * 72)
    print(res_df[["feature_extraction", "model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_acc_mean"]].to_string(index=False))

    # Studi Kasus Konteks Kata (Bu Uti): "Lapis Talas" (Makanan) vs "Gantungan Kunci Miniatur Lapis Talas" (Souvenir)
    case_study = {}
    samples = [
        "Lapis Talas Khas Bogor",
        "Gantungan Kunci Miniatur Lapis Talas Khas Bogor",
        "Lapis Bogor Sangkuriang Full Talas Keju",
        "Gantungan Kunci Bogor Oleh-Oleh Khas Bogor Dua Sisi",
    ]
    tf_vecs = tf2.transform([s.lower().replace("khas", "").replace("bogor", "") for s in samples]).toarray()
    sim_tf = cosine_similarity(tf_vecs)
    case_study["samples"] = samples
    case_study["tfidf_sim_q1_vs_makanan"] = round(float(sim_tf[0, 2]), 4)
    case_study["tfidf_sim_q1_vs_souvenir_lapis"] = round(float(sim_tf[0, 1]), 4)
    case_study["tfidf_sim_souvenir_lapis_vs_makanan"] = round(float(sim_tf[1, 2]), 4)
    case_study["tfidf_sim_souvenir_lapis_vs_souvenir_ganci"] = round(float(sim_tf[1, 3]), 4)

    if HAS_INDOBERT:
        if indobert_tok is None:
            indobert_tok = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
            indobert_mod = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
            indobert_mod.eval()
        bert_vecs = encode_indobert_single(samples, indobert_tok, indobert_mod)
        sim_bt = cosine_similarity(bert_vecs)
        case_study["indobert_sim_q1_vs_makanan"] = round(float(sim_bt[0, 2]), 4)
        case_study["indobert_sim_q1_vs_souvenir_lapis"] = round(float(sim_bt[0, 1]), 4)
        case_study["indobert_sim_souvenir_lapis_vs_makanan"] = round(float(sim_bt[1, 2]), 4)
        case_study["indobert_sim_souvenir_lapis_vs_souvenir_ganci"] = round(float(sim_bt[1, 3]), 4)

    print("\n🔍 STUDI KASUS KONTEKS KATA (Cosine Similarity):")
    print(json.dumps(case_study, indent=2))

    res_df.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"results": results, "case_study": case_study}, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Hasil tersimpan ke:\n  - {OUT_CSV}\n  - {OUT_JSON}")


if __name__ == "__main__":
    main()
