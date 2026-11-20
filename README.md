# 🎮 Video Games Sentiment Analysis

Repositori ini berisi implementasi pipeline **Natural Language Processing (NLP)** dan **Machine Learning** untuk menganalisis dan mengklasifikasikan sentimen ulasan video game (Positif dan Negatif) menggunakan dataset **Amazon Reviews 2023** kategori **Video_Games**.

Proyek ini mencakup alur lengkap:
1. **Persiapan Lingkungan & Repositori**
2. **Pemuatan Data Bertahap (Streaming / Chunking)** dengan filter timestamp $\le 31$ Desember 2019
3. **Pembersihan Data & Pelabelan Biner (Preprocessing)**
4. **Pembagian Data (Train-Validation Split)** (*Time-based Split* & *Stratified Split*)
5. **Ekstraksi Fitur & Pemodelan Machine Learning (Pipeline TF-IDF + Class Imbalance Handling)**
6. **Evaluasi dengan Metrik Macro-F1 & Hyperparameter Tuning**
7. **Penyimpanan Model & Pengujian Inferensi Real-time**

---

## 📁 Struktur Proyek

```text
Task/
│
├── data/
│   ├── video_games_preprocessed.csv         # Dataset bersih siap latih (hasil Tahap 3)
│   ├── video_games_reviews_2019.csv         # Dataset ulasan tersaring (<= 31 Des 2019)
│   └── video_games_reviews.csv              # Dataset ulasan sampel awal
│
├── model/
│   ├── model_sentimen.joblib                # Model Pipeline TERBAIK (Tuned C=10.0, Macro-F1 0.8726)
│   ├── model_config.json                    # Metadata konfigurasi & threshold optimal
│   ├── model_linearsvc.joblib               # Pipeline LinearSVC (Calibrated)
│   ├── model_logistic_regression.joblib     # Pipeline Logistic Regression (balanced)
│   ├── model_multinomial_nb.joblib          # Pipeline Multinomial Naive Bayes
│   ├── model_lightgbm.joblib                # Pipeline LightGBM (balanced)
│   └── model_sentimen_time_based.joblib     # Pipeline evaluasi Time-based Split
│
├── model_sentimen.joblib                    # Salinan model terbaik di root direktori
├── load_data.py                             # Modul streaming, pra-pemrosesan, split & pipeline
├── notebook.ipynb                           # Jupyter Notebook utama (alur lengkap ML & visualisasi)
├── video_games_sentiment_analysis.ipynb     # Salinan deskriptif untuk notebook analisis sentimen
├── requirements.txt                         # Daftar pustaka & dependensi
├── .gitignore                               # Aturan pengabaian berkas Git
└── README.md                                # Dokumentasi proyek lengkap
```

---

## 🎯 Tahap 6: Evaluasi dengan Metrik Macro-F1 & Hyperparameter Tuning

### 1. Mengapa Metrik Macro-F1 Digunakan?
Dataset ulasan game memiliki ketidakseimbangan kelas (*class imbalance*) alami:
- **Positif (1)**: ~87%
- **Negatif (0)**: ~13%

Jika hanya mengandalkan metrik **Akurasi**, model naif yang selalu menebak "Positif" untuk setiap ulasan akan memperoleh akurasi semu sebesar 87%, meskipun gagal total mendeteksi komplain pemain.
Metrik **Macro-F1** memberikan bobot setara (*unweighted mean*) untuk kedua kelas:
$$\text{Macro-F1} = \frac{\text{F1}_{\text{Negatif}} + \text{F1}_{\text{Positif}}}{2}$$
Hal ini menjamin model dinilai secara objektif dan **tidak berat sebelah**, memastikan kelas minoritas (ulasan negatif) terdeteksi dengan presisi dan recall yang seimbang.

---

### 2. Hyperparameter Tuning

#### A. Tuning Nilai Regularisasi $C$ pada Logistic Regression
Nilai $C$ mengontrol kekuatan regularisasi L2 (semakin besar $C$, semakin rendah penalti regularisasi):

| Nilai $C$ | Akurasi | F1 (Negatif) | F1 (Positif) | Macro-F1 |
| :---: | :---: | :---: | :---: | :---: |
| 0.05 | 86.73% | 0.6270 | 0.9193 | 0.7731 |
| 0.10 | 87.40% | 0.6390 | 0.9237 | 0.7813 |
| 0.50 | 90.30% | 0.6813 | 0.9428 | 0.8121 |
| 1.00 *(Default)* | 91.97% | 0.7188 | 0.9532 | 0.8360 |
| 2.00 | 93.31% | 0.7521 | 0.9613 | 0.8567 |
| 5.00 | 94.09% | 0.7644 | 0.9662 | 0.8653 |
| **10.00 *(Optimal)*** | **94.31%** | **0.7692** | **0.9676** | **0.8684** |

> Nilai **$C = 10.0$** memberikan kapasitas optimal bagi model untuk mempelajari bobot kata kunci negatif spesifik tanpa mengalami overfitting.

#### B. Tuning Ambang Batas Probabilitas (*Probability Threshold Tuning*)
Secara default, model mengklasifikasikan kelas 1 jika $P(Y=1) \ge 0.50$. Karena proporsi data tidak seimbang, ambang batas dipindai dari $0.30$ hingga $0.80$:
- **Ambang Batas Optimal**: $P(Y=1) \ge \mathbf{0.49}$
- **Hasil Metrik Optimal**:
  - **Akurasi**: **94.54%**
  - **Macro-F1**: **0.8726**
  - **F1 (Negatif)**: **0.7763** (Precision: 0.81, Recall: 0.75)
  - **F1 (Positif)**: **0.9689** (Precision: 0.96, Recall: 0.97)

---

### 📊 Perbandingan Sebelum vs Sesudah Tuning

```text
=== SEBELUM TUNING (C=1.0, Threshold=0.50) ===
              precision    recall  f1-score   support
 Negatif (0)       0.66      0.79      0.72       116
 Positif (1)       0.97      0.94      0.95       781
    accuracy                           0.92       897
   macro avg       0.81      0.87      0.84       897

=== SESUDAH TUNING (C=10.0, Threshold=0.49) ===
              precision    recall  f1-score   support
 Negatif (0)       0.81      0.75      0.78       116
 Positif (1)       0.96      0.97      0.97       781
    accuracy                           0.95       897
   macro avg       0.89      0.86      0.87       897
```

---

## 💾 Tahap 7: Penyimpanan Model & Uji Muat Ulang (joblib.load)

Pipeline model terbaik yang telah dituning disimpan ke dalam berkas serialisasi menggunakan `joblib.dump`:
- `model_sentimen.joblib` (root direktori proyek)
- `model/model_sentimen.joblib` (folder `model/` terstruktur)
- `model/model_config.json` (metadata performa & parameter model)

### 🧪 Contoh Pemuatan Ulang & Pengujian Ulasan Baru

```python
import joblib

# 1. Muat ulang pipeline model terbaik
pipeline = joblib.load('model_sentimen.joblib')

# 2. Kalimat ulasan baru
ulasan_baru = [
    "Masterpiece of RPG! The combat is smooth, story is deep, and visuals are breathtaking.",
    "Terrible experience. The game constantly crashes every 10 minutes and save files get corrupted.",
    "Not good at all. I thought it would be fun, but the combat is clunky and gameplay loops are boring.",
    "Great co-op fun with friends! Excellent soundtrack, responsive controls, and challenging boss fights.",
    "Complete waste of money! Full of game-breaking bugs, laggy servers, and aggressive microtransactions.",
    "The story is surprisingly emotional and voice acting is top notch. Definitely worth playing."
]

# 3. Prediksi label dan probabilitas
prediksi = pipeline.predict(ulasan_baru)
probabilitas = pipeline.predict_proba(ulasan_baru)

# 4. Tampilkan hasil
for teks, pred, prob in zip(ulasan_baru, prediksi, probabilitas):
    label = "POSITIF" if pred == 1 else "NEGATIF"
    confidence = prob[pred] * 100
    print(f"[{label} ({confidence:.2f}%)] -> {teks}")
```

#### Hasil Pengujian:
```text
[POSITIF (86.65%)] -> Masterpiece of RPG! The combat is smooth, story is deep, and visuals are breathtaking.
[NEGATIF (67.63%)] -> Terrible experience. The game constantly crashes every 10 minutes and save files get corrupted.
[NEGATIF (93.63%)] -> Not good at all. I thought it would be fun, but the combat is clunky and gameplay loops are boring.
[POSITIF (99.48%)] -> Great co-op fun with friends! Excellent soundtrack, responsive controls, and challenging boss fights.
[NEGATIF (88.40%)] -> Complete waste of money! Full of game-breaking bugs, laggy servers, and aggressive microtransactions.
[POSITIF (95.97%)] -> The story is surprisingly emotional and voice acting is top notch. Definitely worth playing.
```

---

## 👤 Penulis
- **Raihan Muhammad Riswandi** ([@Raihan220106](https://github.com/Raihan220106))
