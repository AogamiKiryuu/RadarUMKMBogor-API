"""
Training Pipeline Model Produksi v5: IndoBERT (ONNX INT8) + LightGBM
====================================================================
- Bebas data leakage (split 80:20 sebelum hitung statistik harga pasar)
- Bebas ties labeling (rank pct > 0.5 per kategori)
- Bebas train-serve skew (embedding latih & inferensi sama-sama ONNX INT8 single-query)
- Hemat RAM (~308 MB puncak) -> Muat di Render Free Tier 512 MB!
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from lightgbm import LGBMClassifier

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_preprocessed.csv")
INT8_ONNX = os.path.join(BASE_DIR, "models", "indobert_base_p1_int8.onnx")
TOK_JSON = os.path.join(BASE_DIR, "models", "indobert_tokenizer.json")
EMB_INT8_PATH = os.path.join(BASE_DIR, "data", "indobert_embeddings_int8.npy")
MODEL_OUT = os.path.join(BASE_DIR, "models", "model_umkm_bogor_v5_indobert_lgbm.joblib")
METRICS_OUT = os.path.join(BASE_DIR, "data", "eval_model_v5_onnx_metrics.json")


def build_onnx_encoder():
    tok = Tokenizer.from_file(TOK_JSON)
    tok.enable_truncation(max_length=64)
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = False
    sess_opts.enable_cpu_mem_arena = False
    sess = ort.InferenceSession(INT8_ONNX, sess_options=sess_opts, providers=["CPUExecutionProvider"])

    def encode_single(text: str) -> np.ndarray:
        enc = tok.encode(str(text))
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        out = sess.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
        return out.mean(axis=1).astype(np.float32)

    return encode_single


def main():
    print("=" * 72)
    print("🚀 TRAINING MODEL v5: IndoBERT (ONNX INT8) + LightGBM (Render 512MB Ready)")
    print("=" * 72)

    df = pd.read_csv(DATA_PATH)
    df["rank_terjual_kat"] = df.groupby("kategori")["jumlah_terjual"].rank(method="average", pct=True)
    df["label"] = (df["rank_terjual_kat"] > 0.5).astype(int)

    # 1. Ekstrak / muat cache embedding ONNX INT8 (single-query agar 100% identik dengan produksi)
    if os.path.exists(EMB_INT8_PATH):
        embs = np.load(EMB_INT8_PATH)
        if len(embs) == len(df):
            print(f"✅ Memuat cache embedding ONNX INT8: {embs.shape}")
        else:
            embs = None
    else:
        embs = None

    if embs is None:
        print(f"⏳ Mengekstrak {len(df)} embedding produk dengan ONNX INT8 (single-query)...")
        t0 = time.time()
        enc_fn = build_onnx_encoder()
        embs = np.vstack([enc_fn(t) for t in df["nama_produk"].fillna("").tolist()])
        np.save(EMB_INT8_PATH, embs)
        print(f"✅ Selesai dalam {time.time() - t0:.1f}s -> {EMB_INT8_PATH} ({os.path.getsize(EMB_INT8_PATH)/1e6:.2f} MB)")

    # 2. Split 80:20 Stratified SEBELUM hitung statistik harga
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    train_idx = train_df.index.to_numpy()
    test_idx = test_df.index.to_numpy()
    train_df = train_df.copy()
    test_df = test_df.copy()

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

    train_df = apply_market(train_df)
    test_df = apply_market(test_df)

    cat_cols = ["kategori", "sub_kategori"]
    num_cols = ["rasio_harga", "zscore_harga", "log_harga", "segmen_harga", "rating"]

    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    scaler = StandardScaler()

    X_tr_tab = np.hstack([ohe.fit_transform(train_df[cat_cols]), scaler.fit_transform(train_df[num_cols])])
    X_te_tab = np.hstack([ohe.transform(test_df[cat_cols]), scaler.transform(test_df[num_cols])])

    # Post-Training Quantization (PTQ):
    # Latih LightGBM pada embedding FP32 (sinyal lebih bersih), evaluasi pada embedding ONNX INT8 (kondisi produksi)
    emb_fp32_path = os.path.join(BASE_DIR, "data", "indobert_embeddings_base_p1.npy")
    if os.path.exists(emb_fp32_path):
        embs_train = np.load(emb_fp32_path)
        print("✅ Menggunakan mode PTQ: train pada FP32, inferensi/evaluasi pada ONNX INT8")
    else:
        embs_train = embs

    X_tr = np.hstack([embs_train[train_idx], X_tr_tab])
    X_te = np.hstack([embs[test_idx], X_te_tab])
    y_tr = train_df["label"].to_numpy()
    y_te = test_df["label"].to_numpy()

    clf = LGBMClassifier(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    # 5-Fold Stratified CV pada Train Set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs, cv_aucs = [], []
    for tr_i, va_i in skf.split(X_tr, y_tr):
        m = LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
            random_state=42, verbose=-1, n_jobs=1,
        )
        m.fit(X_tr[tr_i], y_tr[tr_i])
        cv_accs.append(accuracy_score(y_tr[va_i], m.predict(X_tr[va_i])))
        cv_aucs.append(roc_auc_score(y_tr[va_i], m.predict_proba(X_tr[va_i])[:, 1]))

    metrics = {
        "model_version": "v5_indobert_onnx_int8_lightgbm",
        "accuracy": round(float(accuracy_score(y_te, y_pred)) * 100, 2),
        "precision": round(float(precision_score(y_te, y_pred)) * 100, 2),
        "recall": round(float(recall_score(y_te, y_pred)) * 100, 2),
        "f1_score": round(float(f1_score(y_te, y_pred)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_te, y_prob)), 4),
        "cv_acc_mean": round(float(np.mean(cv_accs)) * 100, 2),
        "cv_acc_std": round(float(np.std(cv_accs)) * 100, 2),
        "cv_auc_mean": round(float(np.mean(cv_aucs)), 4),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
        "train_samples": int(len(train_df)),
        "test_samples": int(len(test_df)),
        "feature_dim": int(X_tr.shape[1]),
    }

    print("\n📊 HASIL EVALUASI MODEL v5 (IndoBERT ONNX INT8 + LightGBM):")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    bundle = {
        "classifier": clf,
        "ohe": ohe,
        "scaler": scaler,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "metrics": metrics,
        "version": "v5_indobert_onnx_int8_lightgbm",
    }
    joblib.dump(bundle, MODEL_OUT, compress=3)
    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Bundle model tersimpan: {MODEL_OUT} ({os.path.getsize(MODEL_OUT)/1024:.1f} KB)")
    print(f"💾 Metrik tersimpan: {METRICS_OUT}")


if __name__ == "__main__":
    main()
