# 🎮 Video Games Sentiment Analysis

Repositori ini berisi implementasi pipeline **Natural Language Processing (NLP)** dan **Machine Learning** untuk menganalisis dan mengklasifikasikan sentimen ulasan video game (Positif dan Negatif). Proyek ini dirancang untuk menangani dataset ulasan lokal maupun dataset skala besar (seperti Amazon Video Games Reviews atau data ulasan game lainnya via streaming/arsip gzip).

---

## 📁 Struktur Proyek

```text
Task/
│
├── data/
│   └── video_games_reviews.csv              # Dataset ulasan video games
│
├── model/
│   └── model_sentimen.joblib                # Model tersimpan (Pipeline TF-IDF + Logistic Regression)
│
├── notebook.ipynb                           # Jupyter Notebook utama (alur lengkap ML & visualisasi)
├── video_games_sentiment_analysis.ipynb     # Salinan deskriptif untuk notebook analisis sentimen
├── requirements.txt                         # Daftar pustaka & dependensi
├── .gitignore                               # Aturan pengabaian berkas Git
└── README.md                                # Dokumentasi proyek
```

---

## 🛠️ Pustaka & Dependensi

Pustaka utama yang digunakan dalam proyek ini meliputi:

- **Pandas** (`pandas`): Manipulasi, eksplorasi, dan penataan data tabular ulasan game.
- **Scikit-Learn** (`scikit-learn`): Ekstraksi fitur teks (`TfidfVectorizer`), pemodelan klasifikasi (`LogisticRegression`), dan kalkulasi metrik evaluasi (`accuracy_score`, `classification_report`, `confusion_matrix`).
- **NumPy** (`numpy`): Operasi komputasi numerik dan manipulasi array.
- **Joblib** (`joblib`): Serialisasi model machine learning (`.joblib`) untuk proses deployment dan inferensi cepat.
- **Datasets** (`datasets`): Pemuatan dan integrasi dataset NLP skala besar.
- **Requests & Gzip** (`requests`, `gzip`): Pengambilan data via HTTP dan pembacaan berkas arsip terkompresi secara streaming tanpa membebani RAM.
- **Matplotlib & Seaborn** (`matplotlib`, `seaborn`): Visualisasi distribusi sentimen, statistik panjang kata, dan confusion matrix heatmap.
- **NLTK** (`nltk`): Toolkit pemrosesan bahasa alami pendukung pra-pemrosesan teks.

---

## 🚀 Panduan Instalasi & Persiapan Lingkungan

### 1. Klon Repositori GitHub
```bash
git clone https://github.com/Raihan220106/video-games-sentiment-analysis.git
cd video-games-sentiment-analysis
```

### 2. (Opsional) Buat & Aktifkan Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

---

## 💡 Cara Menjalankan

### Menjalankan Jupyter Notebook
Buka dan jalankan `notebook.ipynb` atau `video_games_sentiment_analysis.ipynb` di VS Code, Jupyter Lab, atau Jupyter Notebook:
```bash
jupyter notebook notebook.ipynb
```

Alur di dalam notebook tersusun secara terstruktur:
1. **Pemuatan Pustaka (Library Loading)**
2. **Pemuatan Dataset** (Fungsi streaming file arsip `.json.gz` dan pemuatan data lokal)
3. **Exploratory Data Analysis (EDA)** (Distribusi label sentimen dan panjang ulasan)
4. **Pra-pemrosesan Teks (Preprocessing)** (Case folding, regex cleaning, normalisasi)
5. **Pembagian Data (Train & Test Split)** (80:20 terstratifikasi)
6. **Feature Extraction & Modeling** (Pipeline TF-IDF Vectorizer + Logistic Regression)
7. **Evaluasi Model** (Akurasi, Precision, Recall, F1-Score, Confusion Matrix Heatmap)
8. **Penyimpanan Model** (`model/model_sentimen.joblib`)
9. **Uji Inferensi Real-time** dengan ulasan baru

### Menggunakan Model Terlatih Secara Langsung (Inference Script)
Model yang tersimpan di `model/model_sentimen.joblib` dapat langsung digunakan dalam skrip Python tanpa perlu melatih ulang:

```python
import joblib

# Muat model pipeline
model = joblib.load('model/model_sentimen.joblib')

# Contoh ulasan game baru
ulasan_baru = [
    "The combat mechanics and open world exploration are absolutely incredible!",
    "Constant crashes, heavy frame drops, and greedy pay-to-win microtransactions."
]

# Prediksi label (1 = Positif, 0 = Negatif)
prediksi = model.predict(ulasan_baru)
probabilitas = model.predict_proba(ulasan_baru)

for teks, label, prob in zip(ulasan_baru, prediksi, probabilitas):
    status = "POSITIF" if label == 1 else "NEGATIF"
    print(f"Ulasan   : \"{teks}\"")
    print(f"Sentimen : {status} (Keyakinan: {max(prob)*100:.2f}%)")
    print("-" * 60)
```

---

## 📊 Hasil Evaluasi

Model mencapai akurasi tinggi pada data evaluasi dengan metrik:
- **Accuracy**: 100% pada data uji terstratifikasi
- **Precision (Positif / Negatif)**: 1.00 / 1.00
- **Recall (Positif / Negatif)**: 1.00 / 1.00
- **F1-Score**: 1.00

---

## 👤 Penulis
- **Raihan Muhammad Riswandi** ([@Raihan220106](https://github.com/Raihan220106))
