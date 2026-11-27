"""
Eksperimen Optimasi & Hyperparameter Tuning (Bebas Data Leakage)
================================================================
Menggunakan cache indobert_embeddings_base_p1.npy (tidak perlu ekstrak ulang).
Seluruh PCA, TF-IDF, OHE, Scaler, dan pemilihan hyperparameter dilakukan
HANYA pada Data Latih (5-Fold Stratified CV), lalu dievaluasi sekali pada
Holdout Test Set (206 data).
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "dataset_preprocessed.csv")
EMB_CACHE = os.path.join(BASE_DIR, "data", "indobert_embeddings_base_p1.npy")
OUT_JSON = os.path.join(BASE_DIR, "data", "tuning_results_bu_uti.json")


def load_prepared():
    df = pd.read_csv(DATA_PATH)
    df["rank_terjual_kat"] = df.groupby("kategori")["jumlah_terjual"].rank(method="average", pct=True)
    df["label"] = (df["rank_terjual_kat"] > 0.5).astype(int)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train_idx = train_df.index.to_numpy()
    test_idx = test_df.index.to_numpy()
    train_df, test_df = train_df.copy(), test_df.copy()

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

    all_embs = np.load(EMB_CACHE)
    X_tr_bert = all_embs[train_idx]
    X_te_bert = all_embs[test_idx]

    tr_clean = train_df["nama_produk_clean"].fillna("").tolist()
    te_clean = test_df["nama_produk_clean"].fillna("").tolist()
    y_tr = train_df["label"].to_numpy()
    y_te = test_df["label"].to_numpy()

    return tr_clean, te_clean, X_tr_tab, X_te_tab, X_tr_bert, X_te_bert, y_tr, y_te


def eval_cv_and_test(name, clf_fn, X_tr, y_tr, X_te, y_te):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_accs, cv_aucs = [], []
    for ti, vi in skf.split(X_tr, y_tr):
        m = clf_fn()
        m.fit(X_tr[ti], y_tr[ti])
        cv_accs.append(accuracy_score(y_tr[vi], m.predict(X_tr[vi])))
        cv_aucs.append(roc_auc_score(y_tr[vi], m.predict_proba(X_tr[vi])[:, 1]))

    clf = clf_fn()
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    ypr = clf.predict_proba(X_te)[:, 1]
    cm = confusion_matrix(y_te, yp)
    return {
        "strategy": name,
        "dim": int(X_tr.shape[1]),
        "cv_acc_mean": round(float(np.mean(cv_accs)) * 100, 2),
        "cv_acc_std": round(float(np.std(cv_accs)) * 100, 2),
        "cv_auc_mean": round(float(np.mean(cv_aucs)), 4),
        "accuracy": round(float(accuracy_score(y_te, yp)) * 100, 2),
        "precision": round(float(precision_score(y_te, yp)) * 100, 2),
        "recall": round(float(recall_score(y_te, yp)) * 100, 2),
        "f1_score": round(float(f1_score(y_te, yp)) * 100, 2),
        "roc_auc": round(float(roc_auc_score(y_te, ypr)), 4),
        "cm": cm.tolist(),
    }


def main():
    tr_clean, te_clean, X_tr_tab, X_te_tab, X_tr_bert, X_te_bert, y_tr, y_te = load_prepared()
    neg, pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    spw = neg / max(pos, 1)

    tf2 = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr_tf2 = tf2.fit_transform(tr_clean).toarray()
    X_te_tf2 = tf2.transform(te_clean).toarray()

    results = []

    # 1. PCA compression of IndoBERT (32d, 64d, 128d) + LightGBM & XGBoost
    for n_comp in [32, 64, 128]:
        pca = PCA(n_components=n_comp, random_state=42)
        b_tr = pca.fit_transform(X_tr_bert)
        b_te = pca.transform(X_te_bert)
        Xtr = np.hstack([b_tr, X_tr_tab])
        Xte = np.hstack([b_te, X_te_tab])

        results.append(eval_cv_and_test(
            f"IndoBERT PCA-{n_comp} + LightGBM (Tuned)",
            lambda: LGBMClassifier(
                n_estimators=250, num_leaves=15, max_depth=5, learning_rate=0.03,
                min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=1.0, class_weight="balanced",
                random_state=42, verbose=-1, n_jobs=-1
            ), Xtr, y_tr, Xte, y_te
        ))
        results.append(eval_cv_and_test(
            f"IndoBERT PCA-{n_comp} + XGBoost (Tuned)",
            lambda: XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.03,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=1.5, scale_pos_weight=spw,
                random_state=42, eval_metric="logloss", n_jobs=-1
            ), Xtr, y_tr, Xte, y_te
        ))

    # 2. Hybrid Multi-View: TF-IDF Bigram + IndoBERT PCA-64
    pca64 = PCA(n_components=64, random_state=42)
    b64_tr = pca64.fit_transform(X_tr_bert)
    b64_te = pca64.transform(X_te_bert)

    svd64 = TruncatedSVD(n_components=64, random_state=42)
    t64_tr = svd64.fit_transform(X_tr_tf2)
    t64_te = svd64.transform(X_te_tf2)

    Xtr_hyb_full = np.hstack([X_tr_tf2, b64_tr, X_tr_tab])
    Xte_hyb_full = np.hstack([X_te_tf2, b64_te, X_te_tab])

    Xtr_hyb_compact = np.hstack([t64_tr, b64_tr, X_tr_tab])
    Xte_hyb_compact = np.hstack([t64_te, b64_te, X_te_tab])

    for label_hyb, Xtr_h, Xte_h in [
        ("Hybrid (TF-IDF 1000d + IndoBERT PCA-64)", Xtr_hyb_full, Xte_hyb_full),
        ("Hybrid Compact (TF-IDF SVD-64 + IndoBERT PCA-64)", Xtr_hyb_compact, Xte_hyb_compact),
    ]:
        results.append(eval_cv_and_test(
            f"{label_hyb} + LightGBM (Tuned)",
            lambda: LGBMClassifier(
                n_estimators=250, num_leaves=15, max_depth=5, learning_rate=0.03,
                min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=1.0, class_weight="balanced",
                random_state=42, verbose=-1, n_jobs=-1
            ), Xtr_h, y_tr, Xte_h, y_te
        ))
        results.append(eval_cv_and_test(
            f"{label_hyb} + XGBoost (Tuned)",
            lambda: XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.03,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.5, reg_lambda=1.5, scale_pos_weight=spw,
                random_state=42, eval_metric="logloss", n_jobs=-1
            ), Xtr_h, y_tr, Xte_h, y_te
        ))
        results.append(eval_cv_and_test(
            f"{label_hyb} + Soft Voting (RF+XGB+LGBM)",
            lambda: VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=200, min_samples_split=3, class_weight="balanced", random_state=42, n_jobs=-1)),
                    ("xgb", XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, min_child_weight=3, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.5, scale_pos_weight=spw, random_state=42, eval_metric="logloss", n_jobs=-1)),
                    ("lgbm", LGBMClassifier(n_estimators=250, num_leaves=15, max_depth=5, learning_rate=0.03, min_child_samples=15, subsample=0.8, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=1.0, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)),
                ],
                voting="soft", n_jobs=-1
            ), Xtr_h, y_tr, Xte_h, y_te
        ))

    # 3. Full IndoBERT 768d + Soft Voting (RF+XGB+LGBM) & Tuned LGBM
    Xtr_b768 = np.hstack([X_tr_bert, X_tr_tab])
    Xte_b768 = np.hstack([X_te_bert, X_te_tab])
    results.append(eval_cv_and_test(
        "IndoBERT 768d + LightGBM (Regularized Tuned)",
        lambda: LGBMClassifier(
            n_estimators=300, num_leaves=15, max_depth=5, learning_rate=0.03,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.5,
            reg_alpha=0.5, reg_lambda=1.5, class_weight="balanced",
            random_state=42, verbose=-1, n_jobs=-1
        ), Xtr_b768, y_tr, Xte_b768, y_te
    ))
    results.append(eval_cv_and_test(
        "IndoBERT 768d + Soft Voting (RF+XGB+LGBM)",
        lambda: VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=200, min_samples_split=3, class_weight="balanced", random_state=42, n_jobs=-1)),
                ("xgb", XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, min_child_weight=3, subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.5, scale_pos_weight=spw, random_state=42, eval_metric="logloss", n_jobs=-1)),
                ("lgbm", LGBMClassifier(n_estimators=250, num_leaves=15, max_depth=5, learning_rate=0.03, min_child_samples=15, subsample=0.8, colsample_bytree=0.5, reg_alpha=0.5, reg_lambda=1.5, class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1)),
            ],
            voting="soft", n_jobs=-1
        ), Xtr_b768, y_tr, Xte_b768, y_te
    ))

    df_res = pd.DataFrame(results).sort_values(by=["roc_auc", "accuracy"], ascending=False)
    print("=" * 95)
    print("🔥 HASIL EKSPERIMEN OPTIMASI & HYPERPARAMETER TUNING (BEBAS LEAKAGE)")
    print("=" * 95)
    print(df_res[["strategy", "dim", "accuracy", "precision", "recall", "f1_score", "roc_auc", "cv_acc_mean", "cv_auc_mean"]].to_string(index=False))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Disimpan ke {OUT_JSON}")


if __name__ == "__main__":
    main()
