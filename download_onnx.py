"""
Helper ringan untuk mengunduh model IndoBERT ONNX INT8 (~119 MB) saat build/startup Render.
Menggunakan streaming chunk (hemat RAM) dengan fallback aman jika belum diunggah.
"""
import os
import sys
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ONNX_PATH = os.path.join(BASE_DIR, "models", "indobert_base_p1_int8.onnx")
DEFAULT_ONNX_URL = os.environ.get(
    "ONNX_MODEL_URL",
    "https://github.com/AogamiKiryuu/RadarUMKMBogor-API/releases/download/v5.0-onnx/indobert_base_p1_int8.onnx",
)


def ensure_onnx_model(target_path: str = DEFAULT_ONNX_PATH, url: str = DEFAULT_ONNX_URL) -> bool:
    """Pastikan file ONNX INT8 tersedia di disk. Mengembalikan True jika siap dipakai."""
    if os.path.exists(target_path) and os.path.getsize(target_path) > 50_000_000:
        return True

    if not url:
        print("⚠️ ONNX_MODEL_URL kosong; melewati unduhan model IndoBERT ONNX.")
        return False

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = target_path + ".downloading"
    try:
        print(f"⏳ Mengunduh IndoBERT ONNX INT8 dari: {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "RadarUMKMBogor/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(tmp_path, "wb") as out:
            total = 0
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
        if total < 50_000_000:
            raise ValueError(f"Ukuran file terunduh terlalu kecil ({total} bytes)")
        os.replace(tmp_path, target_path)
        print(f"✅ Model IndoBERT ONNX INT8 berhasil diunduh ({total / (1024 * 1024):.2f} MB).")
        return True
    except Exception as exc:
        print(f"⚠️ Gagal mengunduh IndoBERT ONNX INT8 ({exc}). Sistem akan fallback ke Model v4.")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        return False


if __name__ == "__main__":
    ok = ensure_onnx_model()
    sys.exit(0 if ok else 0)  # Exit 0 agar build Render tidak gagal meski release belum diunggah
