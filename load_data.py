"""
Module Pemuatan, Pembersihan, Pembagian Data, Pipeline Pemodelan & Tuning Macro-F1
1. Streaming / chunking baris demi baris (json.loads) dengan filter timestamp <= 1577836799999 (31 Des 2019 UTC).
2. Pembersihan & Pelabelan (Preprocessing):
   - Buang rating == 3 dan baris kosong pada rating, title, text.
   - Buat label biner: rating 1, 2 -> 0 (Negatif); rating 4, 5 -> 1 (Positif).
   - Gabungkan title dan text menjadi full_text.
   - Bersihkan whitespace berlebih dan karakter tidak relevan menjadi clean_text.
3. Pembagian Data (Train-Validation Split):
   - Time-based Split: Ulasan <= 2018 (Train) vs Ulasan 2019 (Validation).
   - Stratified Split: Pembagian terstratifikasi 80:20.
4. Pipeline Scikit-Learn:
   - TfidfVectorizer (max_features=20000, ngram_range=(1,2), negasi dipertahankan).
   - Model klasifikasi: Logistic Regression, LinearSVC, MultinomialNB, LightGBM (class_weight='balanced').
5. Hyperparameter & Threshold Tuning:
   - Evaluasi f1_score(average='macro') dan penentuan ambang batas probabilitas optimal.
"""

import os
import gzip
import json
import zlib
import time
import datetime
import re
import requests
import numpy as np
import pandas as pd
from typing import Optional, Generator, Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

URL_VIDEO_GAMES = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz"
CUTOFF_TIMESTAMP = 1577836799999  # 2019-12-31 23:59:59.999 UTC

# Kata negasi yang dipertahankan dalam stop words
NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'nobody', 'none', 'nor', 'nothing', 'nowhere', 'hardly', 'barely', 'scarcely'}
CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS - NEGATION_WORDS)


def stream_reviews_from_url(
    url: str = URL_VIDEO_GAMES,
    max_records: Optional[int] = 10000,
    cutoff_timestamp: int = CUTOFF_TIMESTAMP,
    chunk_size: int = 65536
) -> Generator[Dict[str, Any], None, None]:
    """Streaming dataset .jsonl.gz baris demi baris dari URL dengan filter timestamp."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    
    buffer = ""
    yielded = 0

    for chunk in resp.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        decompressed = d.decompress(chunk)
        buffer += decompressed.decode("utf-8", errors="ignore")
        lines = buffer.split("\n")
        buffer = lines[-1]

        for line in lines[:-1]:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            ts = record.get("timestamp")
            
            if ts is not None and ts <= cutoff_timestamp:
                yield record
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return


def basic_clean_text(text: str) -> str:
    """Membersihkan teks ulasan dari tag HTML, URL, karakter tidak relevan, dan spasi berlebih."""
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s.,!?'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_reviews_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Menjalankan pra-pemrosesan: buang rating 3, label biner, gabung full_text, dan bersihkan teks."""
    df_clean = df.copy()
    df_clean["title"] = df_clean["title"].replace(r"^\s*$", np.nan, regex=True)
    df_clean["text"] = df_clean["text"].replace(r"^\s*$", np.nan, regex=True)
    df_clean = df_clean.dropna(subset=["rating", "title", "text"]).copy()
    
    df_clean = df_clean[df_clean["rating"] != 3].copy()
    
    df_clean["label"] = df_clean["rating"].apply(
        lambda r: 0 if r in [1, 2, 1.0, 2.0] else (1 if r in [4, 5, 4.0, 5.0] else np.nan)
    )
    df_clean = df_clean.dropna(subset=["label"]).copy()
    df_clean["label"] = df_clean["label"].astype(int)
    
    df_clean["full_text"] = df_clean["title"].astype(str).str.strip() + " " + df_clean["text"].astype(str).str.strip()
    df_clean["clean_text"] = df_clean["full_text"].apply(basic_clean_text)
    
    return df_clean


def build_sentiment_pipeline(model_type: str = "logistic", c_value: float = 10.0) -> Pipeline:
    """
    Membangun scikit-learn Pipeline dengan TfidfVectorizer (unigram & bigram, negasi dipertahankan)
    dan model klasifikasi pilihan: 'linearsvc', 'logistic', 'naive_bayes', atau 'lightgbm'.
    """
    tfidf = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words=CUSTOM_STOP_WORDS
    )
    
    if model_type == "logistic":
        clf = LogisticRegression(C=c_value, max_iter=1000, class_weight="balanced", random_state=42)
    elif model_type == "linearsvc":
        clf = CalibratedClassifierCV(LinearSVC(C=c_value, class_weight="balanced", max_iter=2000, random_state=42))
    elif model_type == "naive_bayes":
        clf = MultinomialNB(alpha=0.5)
    elif model_type == "lightgbm":
        clf = LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight="balanced", random_state=42, verbose=-1)
    else:
        raise ValueError(f"Model type '{model_type}' tidak dikenal. Pilih: logistic, linearsvc, naive_bayes, lightgbm.")

    return Pipeline([
        ("tfidf", tfidf),
        ("classifier", clf)
    ])


def find_optimal_threshold(
    y_true: np.ndarray,
    probs_positive: np.ndarray,
    threshold_range: Tuple[float, float, int] = (0.3, 0.8, 51)
) -> Tuple[float, float, pd.DataFrame]:
    """
    Mencari ambang batas (threshold) probabilitas kelas positif P(Y=1) yang memaksimalkan Macro-F1.
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], threshold_range[2])
    evals = []
    
    for t in thresholds:
        preds = (probs_positive >= t).astype(int)
        macro_f1 = f1_score(y_true, preds, average="macro")
        f1_neg = f1_score(y_true, preds, pos_label=0)
        f1_pos = f1_score(y_true, preds, pos_label=1)
        acc = accuracy_score(y_true, preds)
        evals.append({
            "threshold": round(t, 3),
            "macro_f1": macro_f1,
            "f1_negative": f1_neg,
            "f1_positive": f1_pos,
            "accuracy": acc
        })
        
    df_evals = pd.DataFrame(evals)
    best_row = df_evals.sort_values(by="macro_f1", ascending=False).iloc[0]
    return float(best_row["threshold"]), float(best_row["macro_f1"]), df_evals


if __name__ == "__main__":
    csv_path = os.path.join("data", "video_games_preprocessed.csv")
    df_clean = pd.read_csv(csv_path)

    X_train, X_val, y_train, y_val = train_test_split(
        df_clean["clean_text"], df_clean["label"], test_size=0.2, random_state=42, stratify=df_clean["label"]
    )
    
    pipe = build_sentiment_pipeline(model_type="logistic", c_value=10.0)
    pipe.fit(X_train, y_train)
    
    val_probs = pipe.predict_proba(X_val)[:, 1]
    best_th, best_mf1, _ = find_optimal_threshold(y_val, val_probs)
    print(f"Threshold Optimal: {best_th} | Macro-F1: {best_mf1:.4f}")
