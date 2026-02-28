# Flask ML API Server

API server untuk serve machine learning model prediksi tren pasar.

## 📁 Struktur Folder

```
flask_api/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── models/            # Folder untuk model files
│   ├── random_forest_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── scaler.pkl
│   ├── label_encoder_kategori.pkl (optional)
│   └── label_encoder_subkategori.pkl (optional)
└── README.md
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Model Files

Copy semua file model (.pkl) hasil training ke folder `models/`:

- `random_forest_model.pkl` - Model Random Forest utama
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer untuk text features
- `scaler.pkl` - Standard Scaler untuk numeric features
- `label_encoder_kategori.pkl` - (Optional) Label encoder untuk kategori
- `label_encoder_subkategori.pkl` - (Optional) Label encoder untuk sub-kategori

### 3. Run Server

```bash
python app.py
```

Server akan berjalan di: `http://localhost:5000`

## 📡 API Endpoints

### 1. Health Check

```bash
GET /
```

Response:
```json
{
  "status": "running",
  "message": "Flask ML API untuk Prediksi Tren Pasar",
  "model_loaded": true
}
```

### 2. Predict Product Attractiveness

```bash
POST /predict
Content-Type: application/json
```

Request Body:
```json
{
  "nama_produk": "Lapis Talas Bogor Original",
  "kategori": "Makanan & Minuman",
  "sub_kategori": "Makanan Khas",
  "harga_produk": 75000,
  "rating": 4.8
}
```

Response:
```json
{
  "prediction_label": 1,
  "prediction_score": 85.5,
  "confidence": 0.855,
  "message": "Produk berpotensi menarik",
  "probabilities": {
    "kurang_menarik": 0.145,
    "menarik": 0.855
  }
}
```

### 3. Test Endpoint

```bash
GET /test
```

Test endpoint dengan sample data.

## 🧪 Testing dengan curl

```bash
# Test health check
curl http://localhost:5000/

# Test prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "nama_produk": "Lapis Talas Bogor Original",
    "kategori": "Makanan & Minuman",
    "sub_kategori": "Makanan Khas",
    "harga_produk": 75000,
    "rating": 4.8
  }'
```

## 🐛 Troubleshooting

### Model files not found
Pastikan semua file .pkl ada di folder `models/` dan nama file sesuai.

### NLTK data error
Run sekali untuk download NLTK data:
```python
import nltk
nltk.download('stopwords')
```

### Import error Sastrawi
```bash
pip install --upgrade Sastrawi
```

## 📝 Notes

- Text preprocessing harus sama dengan saat training model
- Pastikan CORS enabled untuk accept request dari Nuxt app
- Untuk production, gunakan production server seperti Gunicorn
