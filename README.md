# 🎮 Video Games Sentiment Analysis

Repositori ini berisi implementasi pipeline **Natural Language Processing (NLP)** dan **Machine Learning** untuk menganalisis dan mengklasifikasikan sentimen ulasan video game (Positif dan Negatif) menggunakan dataset **Amazon Reviews 2023** kategori **Video_Games**.

Proyek ini mencakup alur lengkap:
1. **Persiapan Lingkungan & Repositori**
2. **Pemuatan Data Bertahap (Streaming / Chunking)** dengan filter timestamp $\le 31$ Desember 2019
3. **Pembersihan Data & Pelabelan Biner (Preprocessing)**
4. **Pembagian Data (Train-Validation Split)** dengan pendekatan *Time-based Split* & *Stratified Split*
5. **Ekstraksi Fitur & Pemodelan Machine Learning (Pipeline TF-IDF + Class Imbalance Handling)**
6. **Evaluasi Komparatif & Pengujian Inferensi Real-time**

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
│   ├── model_sentimen.joblib                # Model Pipeline TERBAIK (LinearSVC balanced)
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

## 🧠 Tahap 5: Ekstraksi Fitur & Pemodelan

### 1. Ekstraksi Fitur: TF-IDF Vectorizer
- **`ngram_range=(1, 2)`**: Menangkap unigram (kata tunggal) dan bigram (kombinasi 2 kata). Bigram sangat krusial dalam klasifikasi sentimen game untuk menangkap **frasa negasi** seperti *"not good"*, *"no fun"*, *"never buy"*, *"waste money"*, atau *"bad optimization"*.
- **Preservasi Kata Negasi**: Stop words bawaan bahasa Inggris sering kali membuang kata negasi seperti *not*, *no*, dan *never*. Pada pipeline ini, kata-kata negasi secara eksplisit **dikecualikan dari stop words** sehingga informasi polaritas negatif tetap utuh.
- **`max_features=20000`**: Membatasi kosakata paling informatif untuk menyeimbangkan ketelitian dan efisiensi memori.
- **`sublinear_tf=True`**: Menggunakan pembobotan sublinear $1 + \log(\text{tf})$ untuk meredam frekuensi kata yang sangat sering muncul.

---

### 2. Model Klasifikasi & Penanganan Ketidakseimbangan Kelas (*Class Imbalance*)
Dataset ulasan game memiliki ketidakseimbangan kelas alami (Positif ~87%, Negatif ~13%). Untuk mengatasinya, digunakan model yang mendukung pembobotan kelas:

1. **Logistic Regression (`class_weight='balanced'`)**:
   Linear classifier yang memberikan penalti bobot berbanding terbalik dengan frekuensi kelas sehingga kelas minoritas negatif mendapat bobot perhatian lebih besar.
2. **LinearSVC (`class_weight='balanced'`)**:
   Support Vector Classifier berbasis *Maximum Margin Hyperplane* yang sangat efektif pada representasi teks berdimensi tinggi (>10.000 fitur). Model dibungkus dengan `CalibratedClassifierCV` agar mampu menghasilkan probabilitas keyakinan (`predict_proba`).
3. **Multinomial Naive Bayes (`alpha=0.5`)**:
   Model pembanding probabilistik berbasis teorema Bayes. Model ini tidak mendukung `class_weight`, sehingga berfungsi sebagai bukti empiris mengapa penanganan ketidakseimbangan kelas sangat penting.
4. **LightGBM (`class_weight='balanced'`)**:
   Model pembanding berbasis *Gradient Boosted Decision Trees* yang sangat cepat dan efisien.

---

### 3. Scikit-learn Pipeline
Semua model dibungkus dalam arsitektur `Pipeline([('tfidf', vectorizer), ('classifier', model)])`. Keunggulannya:
- Mencegah kebocoran data (*data leakage*) antara train dan validation.
- Satu objek tunggal dapat langsung disimpan via `joblib.dump()` dan dimuat untuk inferensi teks mentah secara langsung.

---

## 📊 Hasil Benchmarking Model Komparatif

Evaluasi pada **Stratified Validation Set (897 sampel)**:

| Model | Akurasi | Precision (Neg) | Recall (Neg) | F1 (Neg) | Precision (Pos) | Recall (Pos) | F1 (Pos) | **Macro F1** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 **LinearSVC (balanced)** | **94.20%** | **0.8478** | **0.6724** | **0.7500** | **0.9528** | **0.9821** | **0.9672** | **0.8586** |
| 🥈 **Logistic Regression (balanced)** | **91.97%** | 0.6571 | **0.7931** | 0.7188 | 0.9683 | 0.9385 | 0.9532 | 0.8360 |
| 🥉 **LightGBM (balanced)** | **92.31%** | 0.6975 | 0.7155 | 0.7064 | 0.9576 | 0.9539 | 0.9557 | 0.8311 |
| 4. **Multinomial Naive Bayes** | 87.63% | 1.0000 | 0.0431 | 0.0826 | 0.8756 | 1.0000 | 0.9337 | 0.5081 |

Evaluasi pada **Time-based Validation Set (Tahun 2019, 945 sampel)**:

| Model | Akurasi | F1 (Neg) | F1 (Pos) | **Macro F1** |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **LinearSVC (balanced)** | **92.70%** | **0.6145** | **0.9597** | **0.7871** |
| 🥈 **LightGBM (balanced)** | **91.22%** | 0.6066 | 0.9506 | 0.7786 |
| 🥉 **Logistic Regression (balanced)** | **90.69%** | 0.5926 | 0.9474 | 0.7700 |
| 4. **Multinomial Naive Bayes** | 90.69% | 0.0638 | 0.9510 | 0.5074 |

### 💡 Analisis Temuan:
1. **LinearSVC** meraih performa tertinggi (**Akurasi 94.20%**, **Macro F1 0.8586**). Margin hyperplane SVM sangat efektif memisahkan ruang fitur sparse TF-IDF.
2. **Logistic Regression** unggul dalam tingkat **Recall Negatif (79.31%)**, sangat baik jika fokus sistem adalah meminimalkan ulasan negatif yang lolos (*false positive*).
3. **Multinomial Naive Bayes** menunjukkan bahwa tanpa penanganan ketidakseimbangan kelas (`class_weight`), model cenderung memprediksi hampir semua ulasan sebagai Positif (Recall Negatif hanya 4.31%), membuktikan krusialnya parameter `class_weight='balanced'`.

---

## 💡 Cara Menjalankan Inferensi

```python
import joblib

# Muat model pipeline terbaik (LinearSVC)
model = joblib.load('model/model_sentimen.joblib')

# Contoh ulasan baru (menguji kata-kata negasi)
ulasan_baru = [
    "Masterpiece of RPG! The combat is smooth, story is deep, and visuals are breathtaking.",
    "Not good at all. The combat is clunky and it has no fun gameplay loops.",
    "Broken launch, heavy frame drops, constant crashing and game breaking bugs.",
    "Great co-op fun with friends. Excellent soundtrack and challenging boss fights."
]

prediksi = model.predict(ulasan_baru)
probabilitas = model.predict_proba(ulasan_baru)

for teks, label, prob in zip(ulasan_baru, prediksi, probabilitas):
    status = "POSITIF" if label == 1 else "NEGATIF"
    print(f"Ulasan   : \"{teks}\"")
    print(f"Sentimen : {status} (Keyakinan: {max(prob)*100:.2f}%)")
    print("-" * 60)
```

---

## 👤 Penulis
- **Raihan Muhammad Riswandi** ([@Raihan220106](https://github.com/Raihan220106))
