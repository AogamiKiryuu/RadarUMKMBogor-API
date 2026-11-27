"""
Ekspor & Kuantisasi INT8 ONNX untuk IndoBERT (indobenchmark/indobert-base-p1)
+ Benchmark Akurasi, Latensi, dan Konsumsi RAM (Target: Render Free Tier 512 MB)
"""
import os
import sys
import json
import time
import subprocess
import numpy as np
import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODELS_DIR, exist_ok=True)

FP32_ONNX = os.path.join(MODELS_DIR, "_tmp_indobert_fp32.onnx")
INT8_ONNX = os.path.join(MODELS_DIR, "indobert_base_p1_int8.onnx")
TOKENIZER_JSON = os.path.join(MODELS_DIR, "indobert_tokenizer.json")
EMB_INT8_PATH = os.path.join(DATA_DIR, "indobert_embeddings_int8.npy")
EMB_FP32_PATH = os.path.join(DATA_DIR, "indobert_embeddings_base_p1.npy")
MODEL_NAME = "indobenchmark/indobert-base-p1"


def export_and_quantize():
    import torch
    from transformers import AutoTokenizer, AutoModel
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print(f"1. Memuat {MODEL_NAME} dan menyimpan tokenizer fast...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Simpan tokenizer.json murni (Rust tokenizers, tanpa butuh library transformers di produksi)
    tokenizer.backend_tokenizer.save(TOKENIZER_JSON)
    print(f"   -> Tokenizer disimpan: {TOKENIZER_JSON} ({os.path.getsize(TOKENIZER_JSON)/1024:.1f} KB)")

    if os.path.exists(INT8_ONNX):
        print(f"   -> ONNX INT8 sudah ada: {INT8_ONNX} ({os.path.getsize(INT8_ONNX)/1e6:.2f} MB)")
        return

    print("2. Mengekspor IndoBERT ke ONNX FP32 sementara...")
    t0 = time.time()
    base_model = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    base_model.eval()

    class BertOnnxWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            out = self.m(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
            return out[0]

    wrapper = BertOnnxWrapper(base_model)
    wrapper.eval()

    dummy = tokenizer(
        ["lapis talas khas bogor"],
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy["input_ids"], dummy["attention_mask"]),
            FP32_ONNX,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence"},
                "attention_mask": {0: "batch_size", 1: "sequence"},
                "last_hidden_state": {0: "batch_size", 1: "sequence"},
            },
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"   -> Ekspor FP32 selesai ({time.time() - t0:.1f}s), ukuran: {os.path.getsize(FP32_ONNX)/1e6:.2f} MB")

    print("3. Melakukan Dynamic INT8 Quantization...")
    t1 = time.time()
    quantize_dynamic(
        model_input=FP32_ONNX,
        model_output=INT8_ONNX,
        weight_type=QuantType.QInt8,
    )
    print(f"   -> Kuantisasi INT8 selesai ({time.time() - t1:.1f}s), ukuran: {os.path.getsize(INT8_ONNX)/1e6:.2f} MB")

    # Hapus file FP32 sementara
    if os.path.exists(FP32_ONNX):
        os.remove(FP32_ONNX)
        # Hapus external data file jika ada
        ext = FP32_ONNX + ".data"
        if os.path.exists(ext):
            os.remove(ext)
    print("   -> File FP32 sementara dibersihkan.")


def benchmark_clean_process():
    """
    Jalankan proses Python BERSIH (tanpa import torch/transformers) untuk mengukur:
    - RAM RSS aktual (Flask + onnxruntime + tokenizers + LightGBM + dataset)
    - Latensi inferensi per 1 query
    - Akurasi & F1 LightGBM pada test set menggunakan embedding INT8
    """
    probe_code = r'''
import os, sys, time, json, psutil
import numpy as np
import pandas as pd
import onnxruntime as ort
from tokenizers import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from lightgbm import LGBMClassifier

proc = psutil.Process(os.getpid())
def rss_mb():
    return round(proc.memory_info().rss / (1024 * 1024), 2)

base = sys.argv[1]
int8_onnx = os.path.join(base, "models", "indobert_base_p1_int8.onnx")
tok_json = os.path.join(base, "models", "indobert_tokenizer.json")
data_csv = os.path.join(base, "data", "processed", "dataset_preprocessed.csv")
emb_fp32_path = os.path.join(base, "data", "indobert_embeddings_base_p1.npy")

ram_start = rss_mb()

# Load tokenizer & ONNX Runtime (single-threaded / memory-optimized untuk Render 512MB)
tok = Tokenizer.from_file(tok_json)
tok.enable_truncation(max_length=64)
tok.enable_padding(pad_id=0, pad_token="[PAD]")

sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = 1
sess_opts.inter_op_num_threads = 1
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.enable_mem_pattern = False
sess_opts.enable_cpu_mem_arena = False

t_load0 = time.time()
session = ort.InferenceSession(int8_onnx, sess_options=sess_opts, providers=["CPUExecutionProvider"])
load_time = time.time() - t_load0
ram_after_onnx = rss_mb()

def encode_onnx(texts):
    encs = tok.encode_batch([str(t) for t in texts])
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    out = session.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
    m = mask[:, :, None].astype(np.float32)
    pooled = (out * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
    return pooled

# Warmup + ukur latensi 1 query
_ = encode_onnx(["lapis talas bogor"])
latencies = []
sample_queries = [
    "lapis talas bogor original",
    "gantungan kunci miniatur lapis talas bogor",
    "kopi arabika puncak bogor 250gr",
    "asinan buah khas bogor segar",
    "roti unyil venus khas bogor",
]
for q in sample_queries:
    t0 = time.time()
    _ = encode_onnx([q])
    latencies.append((time.time() - t0) * 1000)

ram_after_infer = rss_mb()

# Evaluasi akurasi pada 206 test samples menggunakan ONNX INT8 (train menggunakan FP32 cache)
df = pd.read_csv(data_csv)
df["rank_terjual_kat"] = df.groupby("kategori")["jumlah_terjual"].rank(method="average", pct=True)
df["label"] = (df["rank_terjual_kat"] > 0.5).astype(int)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_idx = train_df.index.to_numpy()
test_idx = test_df.index.to_numpy()

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
    d["segmen_harga"] = np.where(d["harga_produk"] <= d["q25_harga_kategori"], 0, np.where(d["harga_produk"] <= d["q75_harga_kategori"], 1, 2))
    return d

train_df = apply_market(train_df.copy())
test_df = apply_market(test_df.copy())

cat_cols = ["kategori", "sub_kategori"]
num_cols = ["rasio_harga", "zscore_harga", "log_harga", "segmen_harga", "rating"]
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
scaler = StandardScaler()
X_tr_tab = np.hstack([ohe.fit_transform(train_df[cat_cols]), scaler.fit_transform(train_df[num_cols])])
X_te_tab = np.hstack([ohe.transform(test_df[cat_cols]), scaler.transform(test_df[num_cols])])

emb_fp32 = np.load(emb_fp32_path)
X_tr = np.hstack([emb_fp32[train_idx], X_tr_tab])

# Encode 206 test samples dengan ONNX INT8 (simulasi real-time inference di produksi!)
te_texts = test_df["nama_produk"].fillna("").tolist()
te_embs_int8 = []
for i in range(0, len(te_texts), 32):
    te_embs_int8.append(encode_onnx(te_texts[i:i+32]))
te_embs_int8 = np.vstack(te_embs_int8)

# Cek cosine similarity antara embedding FP32 vs INT8 pada test set
cos_fidelity = float(np.mean([
    cosine_similarity(emb_fp32[test_idx][i:i+1], te_embs_int8[i:i+1])[0, 0]
    for i in range(len(test_idx))
]))

clf = LGBMClassifier(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, class_weight="balanced",
    random_state=42, verbose=-1, n_jobs=1
)
clf.fit(X_tr, train_df["label"].to_numpy())

X_te_int8 = np.hstack([te_embs_int8, X_te_tab])
y_te = test_df["label"].to_numpy()
y_pred = clf.predict(X_te_int8)
y_prob = clf.predict_proba(X_te_int8)[:, 1]

ram_peak = rss_mb()

# Cek studi kasus Lapis Talas dengan ONNX INT8
case_texts = [
    "Lapis Talas Bogor Sangkuriang Original",
    "Bolu Lapis Talas Khas Bogor Keju",
    "Gantungan Kunci Miniatur Lapis Talas Bogor",
    "Souvenir Gantungan Kunci Tugu Kujang Bogor",
]
ce = encode_onnx(case_texts)
sim_food_food = float(cosine_similarity(ce[0:1], ce[1:2])[0, 0])
sim_food_souv = float(cosine_similarity(ce[0:1], ce[2:3])[0, 0])
sim_souv_souv = float(cosine_similarity(ce[2:3], ce[3:4])[0, 0])

res = {
    "onnx_size_mb": round(os.path.getsize(int8_onnx) / (1024 * 1024), 2),
    "load_time_sec": round(load_time, 2),
    "avg_latency_ms": round(float(np.mean(latencies)), 1),
    "ram_start_mb": ram_start,
    "ram_after_onnx_mb": ram_after_onnx,
    "ram_after_infer_mb": ram_after_infer,
    "ram_peak_with_lgbm_mb": ram_peak,
    "fp32_vs_int8_cosine_fidelity": round(cos_fidelity, 5),
    "accuracy": round(float(accuracy_score(y_te, y_pred)) * 100, 2),
    "precision": round(float(precision_score(y_te, y_pred)) * 100, 2),
    "recall": round(float(recall_score(y_te, y_pred)) * 100, 2),
    "f1_score": round(float(f1_score(y_te, y_pred)) * 100, 2),
    "roc_auc": round(float(roc_auc_score(y_te, y_prob)), 4),
    "case_study": {
        "food_vs_food": round(sim_food_food, 4),
        "food_vs_souvenir": round(sim_food_souv, 4),
        "souvenir_vs_souvenir": round(sim_souv_souv, 4),
    }
}
print("BENCHMARK_JSON:" + json.dumps(res, ensure_ascii=False))
'''
    print("\n4. Menjalankan benchmark proses bersih (TANPA PyTorch/Transformers)...")
    out = subprocess.check_output([sys.executable, "-c", probe_code, BASE_DIR], text=True)
    for line in out.splitlines():
        if line.startswith("BENCHMARK_JSON:"):
            data = json.loads(line.replace("BENCHMARK_JSON:", ""))
            out_json = os.path.join(DATA_DIR, "onnx_int8_benchmark.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(json.dumps(data, indent=2, ensure_ascii=False))
            return data
    print(out)


if __name__ == "__main__":
    export_and_quantize()
    benchmark_clean_process()
