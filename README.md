# 🎮 Video Games Sentiment Analysis (Amazon Reviews 2023)

Repositori ini memuat implementasi *end-to-end* **Natural Language Processing (NLP)** dan **Machine Learning** untuk klasifikasi sentimen biner (Positif vs Negatif) pada ulasan video game dari dataset **Amazon Reviews 2023** (*cutoff* $\le 31$ Desember 2019).

---

## 🛠️ 1. Metode yang Digunakan

Pipeline pemodelan dibangun menggunakan pustaka `scikit-learn` dalam satu objek `Pipeline`:
1. **Ekstraksi Fitur Teks:** `TfidfVectorizer` dengan rentang n-gram `(1, 2)` (unigram & bigram), pembobotan `sublinear_tf=True`, batas fitur `max_features=20.000`, serta daftar stop words kustom yang **mempertahankan kata negasi** (*not, no, never, without, nothing*).
2. **Model Klasifikasi Pilihan:** `LogisticRegression` dengan penalti L2, bobot kelas seimbang (`class_weight='balanced'`), nilai regularisasi optimal $C=10.0$, dan ambang batas probabilitas (*optimal probability threshold*) $P(Y=1) \ge 0.49$.
3. **Model Pembanding:** `LinearSVC` (Calibrated), `LightGBM`, dan `MultinomialNB`.

---

## 🎯 2. Alasan Pemilihan Metode

1. **Efisiensi Komputasi Tinggi pada Teks Berdimensi Tinggi:**
   TF-IDF menghasilkan matriks jarang (*sparse matrix*) yang sangat hemat memori (RAM) dan cepat dikomputasikan dibandingkan model berbasis *deep learning* atau *transformer* (seperti BERT), sehingga sangat ideal untuk dataset teks ulasan berskala besar.
2. **Penangkapan Konteks Negasi (N-gram 1-2):**
   Ulasan game sering kali menggunakan frasa negasi seperti *"not good"*, *"no fun"*, atau *"never buy"*. Penggunaan bigram memastikan sentimen negatif terdeteksi secara akurat tanpa terpotong menjadi kata positif *"good"*.
3. **Penanganan Ketidakseimbangan Kelas (*Class Imbalance Handling*):**
   Dataset memiliki rasio alami sekitar 87% Positif vs 13% Negatif. Parameter `class_weight='balanced'` memberikan penalti yang berbanding terbalik dengan frekuensi kelas, sehingga model tidak bias ke kelas mayoritas dan mampu mendeteksi komplain pemain dengan sensitivitas tinggi.
4. **Kalibrasi Probabilitas:**
   Logistic Regression menghasilkan estimasi probabilitas kontinu yang memungkinkan penyesuaian ambang batas (*threshold tuning*) untuk memaksimalkan *trade-off* presisi dan recall.

---

## 📊 3. Cara Pembagian Data Latih & Validasi

Eksperimen menerapkan dua strategi pembagian:
1. **Stratified Split (80:20) — *Strategi Utama*:**
   - **Data Latih (Train):** 3.585 ulasan (80%)
   - **Data Validasi (Validation):** 897 ulasan (20%)
   - **Alasan:** Menjaga rasio kelas minoritas (ulasan negatif ~13%) tetap identik dan representatif di kedua set data, sehingga evaluasi model tidak bias dan menghasilkan estimasi generalisasi yang stabil.
2. **Time-based Split (Train $\le 2018$, Validation $2019$) — *Strategi Pembanding*:**
   - **Data Latih:** 3.537 ulasan ($\le 2018$, 78.9%)
   - **Data Validasi:** 945 ulasan ($2019$, 21.1%)
   - **Alasan:** Mensimulasikan skenario nyata (*real-world deployment*) di mana model dilatih menggunakan data masa lalu untuk memprediksi data ulasan baru di masa depan tanpa kebocoran temporal (*temporal data leakage*).

---

## 📈 4. Hasil Evaluasi Skor Macro-F1

Evaluasi dilakukan pada data validasi (*Stratified Split*) menggunakan metrik **Macro-F1** untuk memastikan kedua kelas dinilai secara berimbang:

$$\text{Macro-F1} = \frac{\text{F1}_{\text{Negatif}} + \text{F1}_{\text{Positif}}}{2}$$

### Perbandingan Model & Hasil Tuning:

| Model / Konfigurasi | Akurasi | F1 (Negatif) | F1 (Positif) | Macro-F1 |
| :--- | :---: | :---: | :---: | :---: |
| Multinomial Naive Bayes | 87.63% | 0.0435 | 0.9339 | 0.4887 |
| LightGBM (`class_weight='balanced'`) | 92.31% | 0.7054 | 0.9568 | 0.8311 |
| LinearSVC (`class_weight='balanced'`) | 94.20% | 0.7554 | 0.9678 | 0.8616 |
| Logistic Regression Baseline ($C=1.0, \text{th}=0.50$) | 91.97% | 0.7188 | 0.9532 | 0.8360 |
| **Logistic Regression Tuned ($C=10.0, \text{th}=0.49$)** | **94.54%** | **0.7763** | **0.9689** | **0.8726** |

> **Kesimpulan:** Model terbaik adalah **TF-IDF + Logistic Regression ($C=10.0$, Threshold $=0.49$)** dengan skor **Macro-F1 sebesar 0.8726** dan **Akurasi 94.54%**, mampu mendeteksi 75% ulasan negatif dengan presisi 81%.

---

## 🚀 5. Cara Menjalankan & Memuat Model

```bash
# 1. Instal dependensi
pip install -r requirements.txt

# 2. Jalankan notebook (Restart & Run All)
jupyter notebook notebook.ipynb
```

```python
# Inferensi mandiri dengan file model tersimpan
import joblib

model = joblib.load('model_sentimen.joblib')
sample = ["Great graphics and smooth combat!", "Game crashed constantly, waste of money."]
print(model.predict(sample))  # Output: [1, 0] -> [Positif, Negatif]
```

---

## 📂 Struktur Repositori
- `notebook.ipynb`: Notebook utama (*Restart & Run All* terverifikasi).
- `video_games_sentiment_analysis.ipynb`: Salinan berkas analisis sentimen.
- `model_sentimen.joblib`: Berkas model serialisasi pipeline terbaik di root direktori.
- `model/`: Direktori penyimpanan model (`model_sentimen.joblib`, model pembanding, dan `model_config.json`).
- `requirements.txt`: Daftar pustaka dependensi dan versinya.
- `load_data.py`: Skrip modular untuk streaming, preprocessing, dan pipeline.
- `README.md`: Dokumentasi proyek ringkas.

**Tautan Repositori GitHub:** [https://github.com/Raihan220106/video-games-sentiment-analysis](https://github.com/Raihan220106/video-games-sentiment-analysis)
