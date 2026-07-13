"""
retrain_model.py
================
Retrain model prediksi pasar UMKM Bogor dengan fitur-fitur yang
LOGIS SECARA BISNIS — bukan sekadar harga absolut.

Fitur baru yang ditambahkan:
  1. rasio_harga        — harga produk / median harga kategori
                          (model belajar: "murah" atau "mahal" itu RELATIF terhadap pasar)
  2. zscore_harga       — z-score harga dalam kategori
                          (seberapa jauh harga dari rata-rata, dalam satuan standar deviasi)
  3. log_harga          — log10(harga) untuk menstabilkan distribusi harga yang skewed
  4. bin_harga          — segmen harga dalam kategori (murah/menengah/premium)
                          berdasarkan kuartil per kategori

Kenapa ini penting secara bisnis:
  - Harga Rp 1.000.000 untuk produk yang rata-rata pasarnya Rp 50.000
    akan menghasilkan rasio_harga = 20x → model tahu ini terlalu mahal
  - Harga Rp 50.000 untuk produk premium yang rata-rata Rp 500.000
    akan menghasilkan rasio_harga = 0.1x → model tahu ini sangat kompetitif
  - Harga absolut Rp 200.000 sendiri tidak bermakna tanpa konteks!
"""

import pandas as pd
import numpy as np
import joblib
import re
import warnings

warnings.filterwarnings('ignore')

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, f1_score, precision_score, recall_score
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  RETRAIN MODEL UMKM BOGOR — WITH BUSINESS-LOGIC FEATURES")
print("=" * 65)

df = pd.read_csv('dataset_umkm_bogor.csv')
print(f"\n✅ Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")
print(f"   Kolom: {list(df.columns)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] Preprocessing teks (Sastrawi)...")
stemmer = StemmerFactory().create_stemmer()
stopwords = {
    'murah', 'promo', 'cod', 'terlaris', 'original', 'ori', 'asli',
    'oleh', 'pcs', 'gr', 'gram', 'kg', 'dan', 'di', 'ke', 'dari', 'yang'
}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stopwords]
    return stemmer.stem(' '.join(words))

df['nama_produk_clean'] = df['nama_produk'].apply(clean_text)
print("   ✅ Selesai!")

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE ENGINEERING BISNIS — KUNCI UTAMA PERBAIKAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] Feature Engineering Bisnis...")

# --- 3a. Statistik harga per KATEGORI (konteks pasar) ---
stat_per_kategori = df.groupby('kategori')['harga_produk'].agg(
    median_harga_kategori='median',
    mean_harga_kategori='mean',
    std_harga_kategori='std',
    q25_harga_kategori=lambda x: x.quantile(0.25),
    q75_harga_kategori=lambda x: x.quantile(0.75),
).reset_index()

# Simpan statistik ini — nantinya dibutuhkan oleh app.py saat inference
stat_per_kategori.to_csv('market_stats_per_kategori.csv', index=False)
print(f"   ✅ Statistik pasar per kategori disimpan ke market_stats_per_kategori.csv")

# Merge ke df
df = df.merge(stat_per_kategori, on='kategori', how='left')

# --- 3b. Rasio harga terhadap median kategori ---
# Fitur ini adalah yang paling penting:
# rasio = 1.0 → tepat di median pasar
# rasio = 2.0 → 2x lebih mahal dari median
# rasio = 0.5 → 50% lebih murah dari median
df['rasio_harga'] = df['harga_produk'] / df['median_harga_kategori'].replace(0, np.nan)
df['rasio_harga'] = df['rasio_harga'].fillna(1.0).clip(upper=50)  # cap outlier ekstrem

# --- 3c. Z-score harga dalam kategori ---
# Seberapa jauh dari rata-rata dalam satuan standar deviasi
df['zscore_harga'] = (
    (df['harga_produk'] - df['mean_harga_kategori']) /
    df['std_harga_kategori'].replace(0, 1)
).clip(-5, 5)  # clip agar tidak ada outlier ekstrem

# --- 3d. Log harga (untuk stabilkan distribusi) ---
df['log_harga'] = np.log1p(df['harga_produk'])

# --- 3e. Segmen harga per kategori (label ordinal) ---
# 0 = murah (< Q25), 1 = menengah (Q25-Q75), 2 = premium (> Q75)
def segment_harga(row):
    if row['harga_produk'] <= row['q25_harga_kategori']:
        return 0  # murah
    elif row['harga_produk'] <= row['q75_harga_kategori']:
        return 1  # menengah
    else:
        return 2  # premium

df['segmen_harga'] = df.apply(segment_harga, axis=1)

# --- Tampilkan statistik fitur baru ---
print(f"\n   Statistik rasio_harga:")
print(f"   - Min   : {df['rasio_harga'].min():.2f}x")
print(f"   - Median: {df['rasio_harga'].median():.2f}x")
print(f"   - Max   : {df['rasio_harga'].max():.2f}x")
print(f"   - >5x   : {(df['rasio_harga'] > 5).sum()} produk (harga jauh di atas pasar)")
print(f"\n   Distribusi segmen harga:")
print(f"   {df['segmen_harga'].value_counts().rename({0:'Murah',1:'Menengah',2:'Premium'})}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. LABELING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] Membuat label...")
median_terjual = df['jumlah_terjual'].median()
df['label'] = (df['jumlah_terjual'] > median_terjual).astype(int)
print(f"   Median jumlah terjual: {median_terjual}")
print(f"   Distribusi label:\n{df['label'].value_counts().to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 5] Menyiapkan feature matrix...")

text_feature  = 'nama_produk_clean'
cat_features  = ['kategori', 'sub_kategori']
num_features  = [
    'rasio_harga',    # ← BARU: harga relatif terhadap pasar per kategori
    'zscore_harga',   # ← BARU: seberapa jauh dari rata-rata pasar
    'log_harga',      # ← BARU: stabilkan distribusi harga
    'segmen_harga',   # ← BARU: murah / menengah / premium
    'rating',         # tetap ada
    # 'harga_produk', ← DIHAPUS: diganti oleh fitur-fitur relatif di atas
]

X = df[[text_feature] + cat_features + num_features]
y = df['label']

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Total data     : {len(X)}")
print(f"   Train+Val set  : {len(X_train_val)}")
print(f"   Test set       : {len(X_test)}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. PIPELINE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), text_feature),
        ('cat',  OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num',  StandardScaler(), num_features),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. CROSS VALIDATION (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 6] Cross Validation baseline (5-Fold Stratified)...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
cv_results = cross_validate(
    rf_baseline, X_train_val, y_train_val,
    cv=skf, scoring=scoring, return_train_score=False
)

print(f"\n   {'Metrik':<15} {'Mean':>8} {'±Std':>8}")
print("   " + "-" * 35)
for metric in scoring:
    scores = cv_results[f'test_{metric}']
    print(f"   {metric:<15} {scores.mean():.4f}   ±{scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7B. CV BASELINE LOGISTIC REGRESSION (PERBANDINGAN)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 6B] Cross Validation Logistic Regression (Perbandingan)...")

lr_baseline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        max_iter=1000, 
        class_weight='balanced', 
        random_state=42, 
        solver='lbfgs'
    ))
])

lr_cv_results = cross_validate(
    lr_baseline, X_train_val, y_train_val,
    cv=skf, scoring=scoring, return_train_score=False
)

print(f"\n   [LR] {'Metrik':<15} {'Mean':>8} {'±Std':>8}")
print("   " + "-" * 35)
for metric in scoring:
    scores = lr_cv_results[f'test_{metric}']
    print(f"   {metric:<15} {scores.mean():.4f}   ±{scores.std():.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. HYPERPARAMETER TUNING (RANDOM FOREST)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 7] Hyperparameter Tuning (GridSearchCV)...")
print("   Ini mungkin butuh beberapa menit...")

param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2],
    'classifier__class_weight': ['balanced', None],  # ← tambahan: handle imbalance
}

rf_for_grid = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    rf_for_grid, param_grid, cv=skf,
    scoring='roc_auc', n_jobs=1, verbose=1  # n_jobs=1: kompatibel Python 3.14
)
grid_search.fit(X_train_val, y_train_val)

print(f"\n   ✅ Selesai!")
print(f"   Best Parameters: {grid_search.best_params_}")
print(f"   Best AUC-ROC (CV): {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUASI DI HOLDOUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 8] Evaluasi di holdout test set...")

best_rf = grid_search.best_estimator_
y_pred  = best_rf.predict(X_test)
y_proba = best_rf.predict_proba(X_test)[:, 1]

print(f"\n   {'Metrik':<15} {'Score':>8}")
print("   " + "-" * 25)
print(f"   {'Accuracy':<15} {accuracy_score(y_test, y_pred):.4f}")
print(f"   {'Precision':<15} {precision_score(y_test, y_pred):.4f}")
print(f"   {'Recall':<15} {recall_score(y_test, y_pred):.4f}")
print(f"   {'F1 Score':<15} {f1_score(y_test, y_pred):.4f}")
print(f"   {'AUC-ROC':<15} {roc_auc_score(y_test, y_proba):.4f}")
print()
print("   --- Klasifikasi RF ---")
print(classification_report(y_test, y_pred, target_names=['Kurang Menarik', 'Menarik']))

print("\n[Step 8B] Evaluasi LR di holdout test set...")
lr_baseline.fit(X_train_val, y_train_val)
lr_pred = lr_baseline.predict(X_test)
print("   --- Klasifikasi LR ---")
print(classification_report(y_test, lr_pred, target_names=['Kurang Menarik', 'Menarik']))

# ─────────────────────────────────────────────────────────────────────────────
# 10. VERIFIKASI LOGIS — SANITY CHECK BISNIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 9] Sanity check bisnis (harga jauh di atas pasar)...")

# Ambil 3 produk nyata dari dataset sebagai referensi
contoh_df = df[df['label'] == 1].head(3)
for _, row in contoh_df.iterrows():
    kategori    = row['kategori']
    stat        = stat_per_kategori[stat_per_kategori['kategori'] == kategori].iloc[0]
    median_kat  = stat['median_harga_kategori']

    # Skenario A: harga normal (1x median)
    harga_normal = median_kat
    # Skenario B: harga gila (20x median → seperti case user)
    harga_gila   = median_kat * 20

    def make_row(harga):
        rasio   = harga / median_kat if median_kat > 0 else 1.0
        zscore  = (harga - stat['mean_harga_kategori']) / max(stat['std_harga_kategori'], 1)
        log_h   = np.log1p(harga)
        q25     = stat['q25_harga_kategori']
        q75     = stat['q75_harga_kategori']
        seg     = 0 if harga <= q25 else (1 if harga <= q75 else 2)
        return pd.DataFrame([{
            'nama_produk_clean': row['nama_produk_clean'],
            'kategori': kategori,
            'sub_kategori': row['sub_kategori'],
            'rasio_harga': min(rasio, 50),
            'zscore_harga': np.clip(zscore, -5, 5),
            'log_harga': log_h,
            'segmen_harga': seg,
            'rating': row['rating'],
        }])

    prob_normal = best_rf.predict_proba(make_row(harga_normal))[0][1]
    prob_gila   = best_rf.predict_proba(make_row(harga_gila))[0][1]

    print(f"\n   Produk : {row['nama_produk'][:50]}...")
    print(f"   Kategori: {kategori} | Median pasar: Rp{median_kat:,.0f}")
    print(f"   Harga normal (1x median = Rp{harga_normal:,.0f}): peluang laku {prob_normal*100:.1f}%")
    print(f"   Harga gila   (20x median= Rp{harga_gila:,.0f}): peluang laku {prob_gila*100:.1f}%")
    print(f"   → Penurunan probabilitas: {(prob_normal - prob_gila)*100:.1f} poin ✅" 
          if prob_normal > prob_gila else f"   → ⚠️ TIDAK TURUN — perlu investigasi lebih lanjut")

# ─────────────────────────────────────────────────────────────────────────────
# 11. RETRAIN DENGAN SELURUH DATA → SIMPAN MODEL FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 10] Retrain dengan SELURUH data → simpan model final...")

best_params = grid_search.best_params_

final_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=best_params['classifier__n_estimators'],
        max_depth=best_params['classifier__max_depth'],
        min_samples_split=best_params['classifier__min_samples_split'],
        min_samples_leaf=best_params['classifier__min_samples_leaf'],
        class_weight=best_params['classifier__class_weight'],
        random_state=42
    ))
])

final_model.fit(X, y)

# Simpan model baru
joblib.dump(final_model, 'model_umkm_bogor_v2.joblib')
print(f"   ✅ Model berhasil disimpan ke: model_umkm_bogor_v2.joblib")

print("\n" + "=" * 65)
print("  SELESAI! Ringkasan perubahan:")
print("=" * 65)
print("""
  SEBELUM (v1):
    Fitur numerik: [harga_produk, rating]
    Masalah      : Model tidak tahu apakah harga "mahal" atau "murah"
                   karena tidak ada konteks pasar

  SESUDAH (v2):
    Fitur numerik: [rasio_harga, zscore_harga, log_harga, segmen_harga, rating]
    Improvement  :
      - rasio_harga: harga relatif terhadap MEDIAN PER KATEGORI
        → Rp1.000.000 untuk produk Rp50k = rasio 20x → model tahu ini mahal
      - zscore_harga: seberapa ekstrem harga vs rata-rata kategori
      - log_harga: stabilkan distribusi skewed
      - segmen_harga: murah / menengah / premium per kategori
      - class_weight: handle imbalance data
      - ngram (1,2): tangkap frasa lebih baik (e.g. "lapis talas")
""")
print("  File yang dihasilkan:")
print("  - model_umkm_bogor_v2.joblib       ← model baru")
print("  - market_stats_per_kategori.csv    ← statistik pasar (dipakai app.py)")
print("=" * 65)
