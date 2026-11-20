# 🎮 Video Games Sentiment Analysis

Repositori ini berisi implementasi pipeline **Natural Language Processing (NLP)** dan **Machine Learning** untuk menganalisis dan mengklasifikasikan sentimen ulasan video game (Positif dan Negatif) menggunakan dataset **Amazon Reviews 2023** kategori **Video_Games**.

Proyek ini dirancang secara khusus untuk menangani dataset skala besar menggunakan metode **Streaming / Chunking** baris demi baris (*line-by-line*) sehingga memori RAM tetap efisien dan tidak *out of memory*.

---

## 📁 Struktur Proyek

```text
Task/
│
├── data/
│   ├── video_games_reviews_2019.csv         # Dataset ulasan video games tersaring (<= 31 Des 2019)
│   └── video_games_reviews.csv              # Dataset ulasan sampel awal
│
├── model/
│   └── model_sentimen.joblib                # Model Pipeline terlatih (TF-IDF + Logistic Regression)
│
├── model_sentimen.joblib                    # Salinan model di root direktori
├── load_data.py                             # Modul streaming & pemuatan data bertahap
├── notebook.ipynb                           # Jupyter Notebook utama (alur lengkap ML & visualisasi)
├── video_games_sentiment_analysis.ipynb     # Salinan deskriptif untuk notebook analisis sentimen
├── requirements.txt                         # Daftar pustaka & dependensi
├── .gitignore                               # Aturan pengabaian berkas Git
└── README.md                                # Dokumentasi proyek lengkap
```

---

## 🛠️ Pustaka & Dependensi

Pustaka utama yang digunakan dalam proyek ini meliputi:

- **Pandas** (`pandas`): Manipulasi, eksplorasi, dan penataan data tabular ulasan game.
- **Scikit-Learn** (`scikit-learn`): Ekstraksi fitur teks (`TfidfVectorizer`), pemodelan klasifikasi (`LogisticRegression`), dan kalkulasi metrik evaluasi (`accuracy_score`, `classification_report`, `confusion_matrix`).
- **NumPy** (`numpy`): Operasi komputasi numerik dan manipulasi array.
- **Joblib** (`joblib`): Serialisasi model machine learning (`.joblib`) untuk proses deployment dan inferensi cepat.
- **Datasets** (`datasets`): Pemuatan dan integrasi dataset NLP skala besar via Hugging Face.
- **Requests & Zlib / Gzip** (`requests`, `zlib`, `gzip`): Pengambilan data via HTTP stream dan pembacaan berkas arsip terkompresi `.jsonl.gz` secara streaming tanpa membebani RAM.
- **Matplotlib & Seaborn** (`matplotlib`, `seaborn`): Visualisasi distribusi sentimen, tren ulasan per tahun, dan confusion matrix heatmap.
- **NLTK** (`nltk`): Toolkit pemrosesan bahasa alami pendukung pra-pemrosesan teks.

---

## 📥 Pemuatan Data Streaming & Chunking (Tahap 2)

### Karakteristik Dataset
- **Sumber**: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io) / [Hugging Face McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Kategori**: `Video_Games.jsonl.gz` (~814 MB terkompresi, gigabytes saat diekstrak)
- **Batas Waktu (Cutoff)**: `timestamp <= 1577836799999` (Hingga 31 Desember 2019, 23:59:59.999 UTC)

### Mekanisme Streaming & RAM Safety
Untuk menjaga penggunaan RAM tetap rendah:
1. Data di-stream per *chunk* (64 KB) dari URL atau berkas `.jsonl.gz`.
2. Dekompresi dilakukan secara *on-the-fly* menggunakan `zlib.decompressobj`.
3. Baris didekode dan diparsing menggunakan `json.loads(line)` satu per satu (*line-by-line*).
4. Filter langsung dievaluasi pada tiap baris: jika `timestamp <= 1577836799999`, baris disimpan; jika tidak, baris segera dibuang dari memori.

### Menjalankan Skrip Pemuatan Data Mandiri
```bash
python load_data.py
```
Hasil pemuatan akan tersimpan di `data/video_games_reviews_2019.csv`.

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

Alur di dalam notebook:
1. **Persiapan Lingkungan & Pustaka**
2. **Pengunduhan & Pemuatan Data Streaming (Filter Timestamp <= 31 Des 2019)**
3. **Exploratory Data Analysis (EDA)** (Distribusi rating 1-5, sentimen biner, tren per tahun)
4. **Pra-pemrosesan Teks (Preprocessing)** (Case folding, regex cleaning, normalisasi)
5. **Pembagian Data (Train & Test Split)** (80:20 terstratifikasi)
6. **Feature Extraction & Modeling** (Pipeline TF-IDF Vectorizer + Logistic Regression)
7. **Evaluasi Model** (Akurasi, Precision, Recall, F1-Score, Confusion Matrix Heatmap)
8. **Penyimpanan Model** (`model/model_sentimen.joblib`)
9. **Uji Inferensi Real-time** dengan contoh ulasan baru

### Menggunakan Model Terlatih Secara Langsung (Inference Script)
Model yang tersimpan di `model/model_sentimen.joblib` dapat langsung digunakan dalam skrip Python:

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

Model dievaluasi pada data uji ulasan Amazon Video Games riil (rentang tahun 1999 s/d 2019):
- **Accuracy**: ~89% pada data uji terstratifikasi
- **Precision (Positif)**: 0.95
- **Recall (Positif)**: 0.92
- **F1-Score**: 0.93 (Positif), 0.76 (Macro Avg)

---

## 👤 Penulis
- **Raihan Muhammad Riswandi** ([@Raihan220106](https://github.com/Raihan220106))
