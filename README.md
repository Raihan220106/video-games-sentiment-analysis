# 🎮 Video Games Sentiment Analysis

Repositori ini berisi implementasi pipeline **Natural Language Processing (NLP)** dan **Machine Learning** untuk menganalisis dan mengklasifikasikan sentimen ulasan video game (Positif dan Negatif) menggunakan dataset **Amazon Reviews 2023** kategori **Video_Games**.

Proyek ini mencakup alur lengkap:
1. **Persiapan Lingkungan & Repositori**
2. **Pemuatan Data Bertahap (Streaming / Chunking)** dengan filter timestamp $\le 31$ Desember 2019
3. **Pembersihan Data & Pelabelan Biner (Preprocessing)**
4. **Pembagian Data (Train-Validation Split)** dengan pendekatan *Time-based Split* & *Stratified Split*
5. **Pelatihan Model & Pipeline Klasifikasi (TF-IDF + Logistic Regression)**
6. **Evaluasi dan Pengujian Inferensi Real-time**

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
│   ├── model_sentimen.joblib                # Model Pipeline terlatih utama (Stratified Split)
│   └── model_sentimen_time_based.joblib     # Model Pipeline terlatih (Time-based Split)
│
├── model_sentimen.joblib                    # Salinan model utama di root direktori
├── load_data.py                             # Modul streaming, pra-pemrosesan, dan pembagian data
├── notebook.ipynb                           # Jupyter Notebook utama (alur lengkap ML & visualisasi)
├── video_games_sentiment_analysis.ipynb     # Salinan deskriptif untuk notebook analisis sentimen
├── requirements.txt                         # Daftar pustaka & dependensi
├── .gitignore                               # Aturan pengabaian berkas Git
└── README.md                                # Dokumentasi proyek lengkap
```

---

## 📊 Tahap 4: Pembagian Data (Train-Validation Split)

Untuk mengevaluasi performa model klasifikasi sentimen secara objektif dan mendalam, diterapkan dua strategi pembagian data:

### 1. Strategi Utama: Time-based Split (Skenario Dunia Nyata)
- **Data Latih (Train)**: Ulasan dengan tahun $\le 2018$ $\rightarrow$ **3.537 sampel (78.9%)**
  - Negatif (0): 489 sampel (13.8%)
  - Positif (1): 3.048 sampel (86.2%)
- **Data Validasi (Validation)**: Ulasan tahun $2019$ $\rightarrow$ **945 sampel (21.1%)**
  - Negatif (0): 91 sampel (9.6%)
  - Positif (1): 854 sampel (90.4%)

#### 📝 Alasan Pemilihan Time-based Split:
1. **Simulasi Skenario Produksi Nyata (*Real-world Deployment Scenario*)**: Di industri video game, model NLP dilatih menggunakan kumpulan ulasan historis masa lalu dan kemudian diuji untuk memprediksi ulasan yang baru masuk di masa mendatang.
2. **Pencegahan Kebocoran Temporal (*Temporal Data Leakage*)**: Pembagian acak berisiko membagi ulasan untuk game yang sama di masa depan ke dalam data latih, sehingga model mendapat keuntungan informasi yang tidak realistis. Time-based split menjamin evaluasi bebas dari kebocoran waktu.
3. **Pengujian Terhadap Pergeseran Tren & Bahasa (*Concept & Lexical Drift*)**: Tren industri game terus berubah (misalnya rilis konsol baru, istilah mekanik baru seperti *battle pass*, *ray-tracing*, atau *crossplay*). Time-based split menguji daya tahan model terhadap istilah-istilah ulasan generasi baru di tahun 2019.
4. **Rasio Alami Mendekati 80:20**: Distribusi waktu historis ($\le 2018$) vs tahun terbaru ($2019$) secara alami menghasilkan proporsi **78.9% : 21.1%**, sangat ideal sesuai standar pembelajaran mesin.

---

### 2. Strategi Pembanding: Stratified Split (80:20)
- **Data Latih (Train)**: **3.585 sampel (80.0%)**
  - Negatif (0): 464 sampel (12.94%)
  - Positif (1): 3.121 sampel (87.06%)
- **Data Validasi (Validation)**: **897 sampel (20.0%)**
  - Negatif (0): 116 sampel (12.94%)
  - Positif (1): 781 sampel (87.06%)

#### 📝 Alasan Pemilihan Stratified Split:
1. **Keseimbangan Proporsi Kelas (*Class Imbalance Preservation*)**: Dataset ulasan game memiliki ketidakseimbangan kelas (Positif ~87%, Negatif ~13%). Stratifikasi memastikan proporsi kelas minoritas tetap konsisten di data latih dan data uji.
2. **Stabilitas Metrik Evaluasi**: Menghindari bias acak di mana set validasi kekurangan ulasan negatif, sehingga evaluasi F1-score untuk kelas negatif lebih stabil dan terukur.

---

### 📈 Perbandingan Hasil Evaluasi Model

| Metrik Evaluasi | Time-based Split ($\le 2018$ vs $2019$) | Stratified Split (80:20) |
| :--- | :---: | :---: |
| **Akurasi Total** | **90.37%** | **92.31%** |
| **Precision (Negatif / Positif)** | 0.50 / 0.96 | 0.67 / 0.97 |
| **Recall (Negatif / Positif)** | 0.67 / 0.93 | 0.80 / 0.94 |
| **F1-Score (Negatif / Positif)** | 0.57 / 0.95 | 0.73 / 0.96 |
| **Macro Average F1** | **0.76** | **0.84** |

> **Catatan Analisis**: Performa pada *Stratified Split* sedikit lebih tinggi (92.31%) karena distribusi data uji identik dengan data latih. Namun, hasil *Time-based Split* (90.37%) membuktikan bahwa model tetap memiliki daya generalisasi yang tangguh saat menghadapi ulasan masa depan tanpa mengalami *overfitting*.

---

## 🧹 Tahap 3: Pembersihan Data & Pelabelan (Preprocessing)

1. **Pembersihan Baris Kosong & Netral**:
   - Membuang baris yang memiliki nilai `rating == 3` (ulasan berbobot netral).
   - Membuang baris yang memiliki nilai kosong (`NaN` atau string kosong) pada kolom `rating`, `title`, dan `text`.
2. **Pelabelan Target Biner (`label`)**:
   - **`rating` 1 atau 2 $\rightarrow$ `0` (Negatif)**
   - **`rating` 4 atau 5 $\rightarrow$ `1` (Positif)**
3. **Penggabungan Fitur Teks Utama (`full_text`)**:
   $$\text{full\_text} = \text{title} + \text{" "} + \text{text}$$
4. **Pembersihan Dasar (`clean_text`)**:
   - Menghapus tag HTML (`<.*?>`) dan tautan web URL (`http\S+|www\.\S+`).
   - Menghapus karakter tidak relevan dan simbol aneh, mempertahankan alfanumerik serta tanda baca dasar.
   - Menghapus *whitespace* berlebih (spasi ganda, newline) menjadi spasi tunggal yang rapi.

---

## 🛠️ Pustaka & Dependensi

- **Pandas** (`pandas`): Manipulasi, filter, dan penataan data tabular.
- **Scikit-Learn** (`scikit-learn`): Pembagian data terstratifikasi (`train_test_split`), ekstraksi fitur teks (`TfidfVectorizer`), pemodelan klasifikasi (`LogisticRegression`), dan kalkulasi metrik evaluasi.
- **NumPy** (`numpy`): Operasi array numerik dan penanganan missing values.
- **Joblib** (`joblib`): Serialisasi model machine learning (`.joblib`) untuk inferensi mandiri.
- **Requests & Zlib / Gzip** (`requests`, `zlib`, `gzip`): Streaming berkas `.jsonl.gz` skala besar tanpa membebani memori RAM.
- **Matplotlib & Seaborn** (`matplotlib`, `seaborn`): Visualisasi distribusi sentimen, tren rating, dan heatmap confusion matrix.

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
2. **Pengunduhan & Pemuatan Data Streaming (Filter Timestamp $\le$ 31 Des 2019)**
3. **Pembersihan Data & Pelabelan (Preprocessing)**
4. **Pembagian Data (Time-based Split & Stratified Split)**
5. **Feature Extraction & Modeling (TF-IDF + Logistic Regression)**
6. **Evaluasi Komparatif Model** (Akurasi, Precision, Recall, F1-Score, Confusion Matrix Heatmap)
7. **Penyimpanan Model** (`model/model_sentimen.joblib`)
8. **Uji Inferensi Real-time** dengan ulasan game baru

### Menggunakan Model Terlatih Secara Langsung (Inference Script)
```python
import joblib

# Muat model pipeline
model = joblib.load('model/model_sentimen.joblib')

# Contoh ulasan game baru (kombinasi judul & ulasan)
ulasan_baru = [
    "Masterpiece of RPG! The combat is smooth, story is deep, and visuals are stunning.",
    "Broken launch and terrible performance. Constant crashing and game breaking bugs."
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

## 👤 Penulis
- **Raihan Muhammad Riswandi** ([@Raihan220106](https://github.com/Raihan220106))
