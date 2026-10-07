"""
Script Pelatihan Model v4 — RadarUMKM Bogor
============================================
Karakteristik Model v4:
1. Labeling Biner Per Kategori (Stratified Category Median / Rank 50:50).
   - Menghilangkan bias sektoral global (makanan vs fashion).
2. Bebas Target Leakage:
   - Fitur jumlah_log, revenue_proxy_log, dan popularity_score_new DIHAPUS dari input model.
   - Fitur hanya mengandalkan teks nama produk (TF-IDF), kategori/sub-kategori (OHE),
     dan metrik harga pasar relatif (rasio_harga, zscore_harga, log_harga, segmen_harga, rating).
3. Pipeline Scikit-Learn terintegrasi (TF-IDF + OHE + Scaler + RandomForestClassifier).
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score


def train_v4():
    print("=" * 60)
    print("🚀 MEMULAI PELATIHAN MODEL v4 (LABELING PER KATEGORI & BEBAS LEAKAGE)")
    print("=" * 60)

    # 1. Load Dataset
    data_path = 'data/processed/dataset_preprocessed.csv'
    if not os.path.exists(data_path):
        data_path = 'flask_api/data/processed/dataset_preprocessed.csv'
    
    df = pd.read_csv(data_path)
    print(f"✅ Dataset berhasil dimuat: {len(df)} baris, {len(df.columns)} kolom")

    # 2. Market Statistics Per Kategori
    stat_kat = df.groupby('kategori')['harga_produk'].agg(
        median_harga_kategori='median',
        mean_harga_kategori='mean',
        std_harga_kategori='std',
        q25_harga_kategori=lambda x: x.quantile(0.25),
        q75_harga_kategori=lambda x: x.quantile(0.75),
    ).reset_index()

    pop_per_kategori = df.groupby('kategori').agg(
        total_terjual=('jumlah_terjual', 'sum'),
        avg_rating=('rating', 'mean'),
        jumlah_produk=('nama_produk', 'count'),
    ).reset_index()

    pop_per_sub = df.groupby(['kategori', 'sub_kategori']).agg(
        total_terjual=('jumlah_terjual', 'sum'),
        avg_rating=('rating', 'mean'),
        jumlah_produk=('nama_produk', 'count'),
    ).reset_index()

    market_stats_v4 = stat_kat.merge(pop_per_kategori, on='kategori', how='left')
    os.makedirs('data', exist_ok=True)
    market_stats_v4.to_csv('data/market_stats_v4.csv', index=False)
    pop_per_sub.to_csv('data/market_stats_sub_kategori_v4.csv', index=False)
    print("✅ market_stats_v4.csv dan market_stats_sub_kategori_v4.csv disimpan.")

    # 3. Feature Engineering Pasar
    df = df.merge(stat_kat, on='kategori', how='left')

    df['rasio_harga'] = (
        df['harga_produk'] / df['median_harga_kategori'].replace(0, np.nan)
    ).fillna(1.0).clip(upper=50)

    df['zscore_harga'] = (
        (df['harga_produk'] - df['mean_harga_kategori']) /
        df['std_harga_kategori'].replace(0, 1)
    ).clip(-5, 5)

    df['log_harga'] = np.log1p(df['harga_produk'])

    def segmen(r):
        if r['harga_produk'] <= r['q25_harga_kategori']:
            return 0
        elif r['harga_produk'] <= r['q75_harga_kategori']:
            return 1
        else:
            return 2

    df['segmen_harga'] = df.apply(segmen, axis=1)

    # 4. LABELING PER KATEGORI (Fair & Balanced 50% : 50%)
    # Menggunakan persentil penjualan intra-kategori untuk mengatasi perbedaan volume antar industri
    df['rank_terjual_kat'] = df.groupby('kategori')['jumlah_terjual'].rank(method='first', pct=True)
    df['label'] = (df['rank_terjual_kat'] > 0.5).astype(int)

    print("\n📊 Distribusi Frekuensi Label Per Kategori:")
    crosstab_res = pd.crosstab(df['kategori'], df['label'], margins=True)
    print(crosstab_res)

    # 5. Menyiapkan Feature Matrix (BEBAS LEAKAGE)
    text_feature = 'nama_produk_clean'
    cat_features = ['kategori', 'sub_kategori']
    num_features = [
        'rasio_harga',   # harga relatif median pasar kategori
        'zscore_harga',  # jarak deviasi standar harga
        'log_harga',     # normalisasi log harga
        'segmen_harga',  # 0=Murah, 1=Menengah, 2=Premium
        'rating',        # reputasi/rating kepuasan produk
    ]

    all_features = [text_feature] + cat_features + num_features
    X = df[all_features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n🔢 Split Data: Train={len(X_train)} sampel, Test={len(X_test)} sampel")

    # 6. Pipeline Preprocessing & Model
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), sublinear_tf=True), text_feature),
            ('cat',  OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features),
            ('num',  StandardScaler(), num_features),
        ]
    )

    rf_clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', rf_clf)
    ])

    print("\n⏳ Melatih Model Random Forest v4...")
    model_pipeline.fit(X_train, y_train)

    # 7. Evaluasi Model pada Data Uji
    y_pred = model_pipeline.predict(X_test)
    y_prob = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("📈 HASIL EVALUASI MODEL v4 (TEST SET 206 DATA)")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-Score : {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC  : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Kurang Menarik (0)', 'Menarik (1)']))

    cv = cross_val_score(model_pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv.mean():.4f} (+/- {cv.std():.4f})")

    # 8. Simpan Model Biner
    os.makedirs('models', exist_ok=True)
    model_save_path = 'models/model_umkm_bogor_v4.joblib'
    joblib.dump(model_pipeline, model_save_path, compress=3)
    print(f"\n💾 Model biner berhasil disimpan ke: {model_save_path}")

    # Simpan report JSON
    report_dict = {
        "model_version": "v4",
        "description": "Random Forest dengan Labeling Per Kategori & Bebas Target Leakage",
        "metrics": {
            "accuracy": round(float(acc), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(auc), 4),
            "cv_accuracy_mean": round(float(cv.mean()), 4),
            "cv_accuracy_std": round(float(cv.std()), 4)
        },
        "features": {
            "text": text_feature,
            "categorical": cat_features,
            "numeric": num_features,
            "total_input_columns": len(all_features)
        },
        "confusion_matrix": cm.tolist()
    }
    with open('data/eval_model_v4_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)
    print("✅ File data/eval_model_v4_metrics.json berhasil disimpan.")

    return model_pipeline, report_dict


if __name__ == '__main__':
    train_v4()
