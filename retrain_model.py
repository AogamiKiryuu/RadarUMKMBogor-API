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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use("Agg")  # backend non-interaktif, aman di Colab & server
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, accuracy_score,
    roc_auc_score, f1_score, precision_score, recall_score
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  RETRAIN MODEL UMKM BOGOR v3 — BUSINESS-LOGIC + FAIR LABELING")
print("=" * 65)

df = pd.read_csv('dataset_umkm_bogor.csv')
print(f"\n✅ Dataset dimuat: {len(df)} baris, {len(df.columns)} kolom")
print(f"   Kolom: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. BERSIHKAN NILAI 0 PADA jumlah_terjual DAN rating
#    Nilai 0 menyebabkan data condong ke nol sehingga membiaskan model.
#    Solusi: ganti nilai 0 dengan rata-rata dari nilai non-zero per kolom.
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 2] Membersihkan nilai 0 pada jumlah_terjual & rating...")

zero_terjual = (df["jumlah_terjual"] == 0).sum()
zero_rating  = (df["rating"] == 0).sum()

mean_terjual_nonzero = df.loc[df["jumlah_terjual"] > 0, "jumlah_terjual"].mean()
mean_rating_nonzero  = df.loc[df["rating"] > 0, "rating"].mean()

df["jumlah_terjual"] = df["jumlah_terjual"].replace(0, round(mean_terjual_nonzero, 4))
df["rating"]         = df["rating"].replace(0, round(mean_rating_nonzero, 4))

print(f"   jumlah_terjual — {zero_terjual} nilai 0 diganti rata-rata non-zero: {mean_terjual_nonzero:.2f}")
print(f"   rating         — {zero_rating}  nilai 0 diganti rata-rata non-zero: {mean_rating_nonzero:.2f}")
print(f"   Range rating setelah perbaikan: {df['rating'].min():.2f} – {df['rating'].max():.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] Preprocessing teks (Sastrawi)...")
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
# 4. FEATURE ENGINEERING BISNIS — KUNCI UTAMA PERBAIKAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] Feature Engineering Bisnis...")

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
        return 2  # mahal  ← sebelumnya "premium", diubah menjadi "mahal"

df['segmen_harga'] = df.apply(segment_harga, axis=1)

# --- Tampilkan statistik fitur baru ---
print(f"\n   Statistik rasio_harga:")
print(f"   - Min   : {df['rasio_harga'].min():.2f}x")
print(f"   - Median: {df['rasio_harga'].median():.2f}x")
print(f"   - Max   : {df['rasio_harga'].max():.2f}x")
print(f"   - >5x   : {(df['rasio_harga'] > 5).sum()} produk (harga jauh di atas pasar)")
print(f"\n   Distribusi segmen harga:")
print(f"   {df['segmen_harga'].value_counts().rename({0:'Murah',1:'Menengah',2:'Mahal'})}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. LABELING — SKOR GABUNGAN (jumlah_terjual + rating)
#
#    SEBELUMNYA: label = jumlah_terjual > 10   ← rule angka tetap, tidak adil
#    SEKARANG  : label = skor_daya_tarik > median(skor)
#
#    skor_daya_tarik = 0.7 × norm(jumlah_terjual) + 0.3 × norm(rating)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 5] Labeling berbasis skor daya tarik gabungan...")

min_t = df["jumlah_terjual"].min()
max_t = df["jumlah_terjual"].max()
df["norm_terjual"] = (df["jumlah_terjual"] - min_t) / max(max_t - min_t, 1)
df["norm_rating"]  = df["rating"] / 5.0
df["skor_daya_tarik"] = 0.7 * df["norm_terjual"] + 0.3 * df["norm_rating"]

threshold_skor = df["skor_daya_tarik"].median()
df["label"] = (df["skor_daya_tarik"] > threshold_skor).astype(int)

print(f"   Skor daya tarik — Min   : {df['skor_daya_tarik'].min():.4f}")
print(f"   Skor daya tarik — Median: {threshold_skor:.4f}  ← threshold label")
print(f"   Skor daya tarik — Max   : {df['skor_daya_tarik'].max():.4f}")
print(f"\n   Distribusi label:")
print(f"   {df['label'].value_counts().rename({0:'Kurang Menarik',1:'Menarik'}).to_string()}")

# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 6. EKSPOR STATISTIK TREN PER KATEGORI & SUB-KATEGORI
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 6] Menghitung tren per kategori & sub-kategori...")

tren_per_kategori = (
    df.groupby("kategori")
    .agg(
        rata_rata_terjual=("jumlah_terjual", "mean"),
        median_terjual=("jumlah_terjual", "median"),
        rata_rata_rating=("rating", "mean"),
        jumlah_produk=("nama_produk", "count"),
        rata_rata_skor=("skor_daya_tarik", "mean"),
    )
    .reset_index()
)

max_avg_terjual = tren_per_kategori["rata_rata_terjual"].max()
tren_per_kategori["skor_tren"] = 0.7 * (
    tren_per_kategori["rata_rata_terjual"] / max(max_avg_terjual, 1)
) + 0.3 * (tren_per_kategori["rata_rata_rating"] / 5.0)
tren_per_kategori = tren_per_kategori.sort_values(
    by="skor_tren", ascending=False
).reset_index(drop=True)
tren_per_kategori["peringkat"] = range(1, len(tren_per_kategori) + 1)

tren_per_kategori.to_csv("tren_per_kategori.csv", index=False)
print("   ✅ Tren per kategori disimpan ke tren_per_kategori.csv")
print("\n   Peringkat kategori berdasarkan skor tren:")
for rec in tren_per_kategori.to_dict("records"):
    rank_num = int(rec["peringkat"])
    kat_name = str(rec["kategori"])
    avg_t    = float(rec["rata_rata_terjual"])
    avg_r    = float(rec["rata_rata_rating"])
    skor_v   = float(rec["skor_tren"])
    print(f"   #{rank_num} {kat_name:<30} avg_terjual={avg_t:.1f}  avg_rating={avg_r:.2f}  skor={skor_v:.4f}")

# --- Tren per SUB-KATEGORI (per kategori induk) ---
tren_per_sub = (
    df.groupby(["kategori", "sub_kategori"])
    .agg(
        rata_rata_terjual=("jumlah_terjual", "mean"),
        median_terjual=("jumlah_terjual", "median"),
        rata_rata_rating=("rating", "mean"),
        jumlah_produk=("nama_produk", "count"),
    )
    .reset_index()
)

# Gunakan transform() agar kolom 'kategori' tidak hilang ke index (pandas 2.x)
max_terjual_per_kat = (
    tren_per_sub.groupby("kategori")["rata_rata_terjual"]
    .transform("max")
    .clip(lower=1)
)
tren_per_sub["skor_tren"] = (
    0.7 * (tren_per_sub["rata_rata_terjual"] / max_terjual_per_kat)
    + 0.3 * (tren_per_sub["rata_rata_rating"] / 5.0)
)
tren_per_sub = tren_per_sub.sort_values(
    ["kategori", "skor_tren"], ascending=[True, False]
)
tren_per_sub.to_csv("tren_per_sub_kategori.csv", index=False)
print("   ✅ Tren per sub-kategori disimpan ke tren_per_sub_kategori.csv")

# 7. FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 7] Menyiapkan feature matrix...")

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
# 8. PIPELINE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=1000, ngram_range=(1, 2)), text_feature),
        ('cat',  OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num',  StandardScaler(), num_features),
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# 9. CROSS VALIDATION (BASELINE)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 8] Cross Validation baseline (5-Fold Stratified)...")

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
# 9b. CROSS VALIDATION — LOGISTIC REGRESSION (PEMBANDING)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 8b] Cross Validation LOGISTIC REGRESSION sebagai pembanding...")

lr_baseline = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

cv_results_lr = cross_validate(
    lr_baseline,
    X_train_val,
    y_train_val,
    cv=skf,
    scoring=scoring,
    return_train_score=False,
)

print(f"\n   {'Metrik':<15} {'RF Mean':>10} {'RF Std':>8}  {'LR Mean':>10} {'LR Std':>8}  Unggul")
print("   " + "-" * 70)
for metric in scoring:
    rf_s = cv_results[f"test_{metric}"]
    lr_s = cv_results_lr[f"test_{metric}"]
    winner = "RF" if rf_s.mean() >= lr_s.mean() else "LR"
    mark   = "✅" if winner == "RF" else "  "
    print(
        f"   {metric:<15} {rf_s.mean():>10.4f} {rf_s.std():>8.4f}"
        f"  {lr_s.mean():>10.4f} {lr_s.std():>8.4f}  {winner} {mark}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# 10. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 9] Hyperparameter Tuning (GridSearchCV)...")
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
# 11. EVALUASI DI HOLDOUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 10] Evaluasi di holdout test set...")

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
print(classification_report(y_test, y_pred, target_names=['Kurang Menarik', 'Menarik']))


# ─────────────────────────────────────────────────────────────────────────────
# 11b. EVALUASI LOGISTIC REGRESSION DI HOLDOUT TEST SET
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 10b] Evaluasi LOGISTIC REGRESSION di test set...")

lr_baseline.fit(X_train_val, y_train_val)
y_pred_lr  = lr_baseline.predict(X_test)
y_proba_lr = lr_baseline.predict_proba(X_test)[:, 1]

print(f"\n   {'Metrik':<15} {'Score':>8}")
print("   " + "-" * 25)
print(f"   {'Accuracy':<15} {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"   {'Precision':<15} {precision_score(y_test, y_pred_lr):.4f}")
print(f"   {'Recall':<15} {recall_score(y_test, y_pred_lr):.4f}")
print(f"   {'F1 Score':<15} {f1_score(y_test, y_pred_lr):.4f}")
print(f"   {'AUC-ROC':<15} {roc_auc_score(y_test, y_proba_lr):.4f}")
print()
print(classification_report(y_test, y_pred_lr, target_names=["Kurang Menarik", "Menarik"]))

# ─────────────────────────────────────────────────────────────────────────────
# 11c. TABEL PERBANDINGAN AKHIR: Random Forest (Tuned) vs Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PERBANDINGAN: Random Forest (Tuned) vs Logistic Regression")
print("=" * 65)

metrics_compare = {
    "Accuracy" : (accuracy_score(y_test, y_pred),    accuracy_score(y_test, y_pred_lr)),
    "Precision": (precision_score(y_test, y_pred),   precision_score(y_test, y_pred_lr)),
    "Recall"   : (recall_score(y_test, y_pred),      recall_score(y_test, y_pred_lr)),
    "F1 Score" : (f1_score(y_test, y_pred),          f1_score(y_test, y_pred_lr)),
    "AUC-ROC"  : (roc_auc_score(y_test, y_proba),    roc_auc_score(y_test, y_proba_lr)),
}

print(f"\n   {'Metrik':<12} {'Random Forest':>15} {'Logistic Reg':>14} {'Unggul':>8}")
print("   " + "-" * 55)
rf_wins = lr_wins = 0
for name, (rf_val, lr_val) in metrics_compare.items():
    if rf_val >= lr_val:
        winner = "RF ✅"
        rf_wins += 1
    else:
        winner = "LR ✅"
        lr_wins += 1
    print(f"   {name:<12} {rf_val:>15.4f} {lr_val:>14.4f} {winner:>8}")
print("   " + "-" * 55)
print(f"   {'Total Wins':<12} {'RF: ' + str(rf_wins):>15} {'LR: ' + str(lr_wins):>14}")
print()
if rf_wins > lr_wins:
    print("   → Random Forest lebih unggul: dipilih sebagai model produksi ✅")
else:
    print("   → Logistic Regression lebih unggul atau seimbang — pertimbangkan penggunaan LR")

# ─────────────────────────────────────────────────────────────────────────────
# 11d. CONFUSION MATRIX SIDE-BY-SIDE
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 10c] Membuat Confusion Matrix perbandingan...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Confusion Matrix: Random Forest vs Logistic Regression",
    fontsize=14, fontweight="bold", y=1.02,
)

models_info = [
    (y_pred,    y_proba,    "Random Forest (Tuned)",          "Blues"),
    (y_pred_lr, y_proba_lr, "Logistic Regression (Baseline)", "Oranges"),
]

for ax, (y_p, y_pb, title, cmap) in zip(axes, models_info):
    cm = confusion_matrix(y_test, y_p)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Kurang Menarik", "Menarik"],
    )
    disp.plot(ax=ax, colorbar=True, cmap=cmap)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Label Prediksi", fontsize=10)
    ax.set_ylabel("Label Aktual",   fontsize=10)
    acc = accuracy_score(y_test, y_p)
    f1v = f1_score(y_test, y_p)
    ax.text(
        0.5, -0.22,
        f"Accuracy: {acc:.4f}  |  F1: {f1v:.4f}",
        ha="center", va="bottom", fontsize=9,
        transform=ax.transAxes, color="#333333",
    )

plt.tight_layout()
plt.savefig("confusion_matrix_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✅ Confusion matrix disimpan ke: confusion_matrix_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# 12. VERIFIKASI LOGIS — SANITY CHECK BISNIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 11] Sanity check bisnis (harga jauh di atas pasar)...")

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
# 13. RETRAIN DENGAN SELURUH DATA → SIMPAN MODEL FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Step 12] Retrain dengan SELURUH data → simpan model final...")

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
print("  - market_stats_per_kategori.csv     ← statistik harga pasar")
print("  - tren_per_kategori.csv             ← peringkat tren per kategori")
print("  - tren_per_sub_kategori.csv         ← peringkat tren per sub-kategori")
print("  - confusion_matrix_comparison.png   ← visualisasi perbandingan model")
print("=" * 65)
