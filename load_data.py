"""
Module Pemuatan & Pembersihan Data (Amazon Reviews 2023 - Video_Games)
1. Streaming / chunking baris demi baris (json.loads) dengan filter timestamp <= 1577836799999 (31 Des 2019 UTC).
2. Pembersihan & Pelabelan (Preprocessing):
   - Buang rating == 3 dan nilai kosong pada rating, title, text.
   - Buat label biner: rating 1, 2 -> 0 (Negatif); rating 4, 5 -> 1 (Positif).
   - Gabungkan title dan text menjadi full_text.
   - Bersihkan whitespace berlebih dan karakter tidak relevan menjadi clean_text.
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
from typing import Optional, Generator, Dict, Any

URL_VIDEO_GAMES = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Video_Games.jsonl.gz"
CUTOFF_TIMESTAMP = 1577836799999  # 2019-12-31 23:59:59.999 UTC


def stream_reviews_from_url(
    url: str = URL_VIDEO_GAMES,
    max_records: Optional[int] = 10000,
    cutoff_timestamp: int = CUTOFF_TIMESTAMP,
    chunk_size: int = 65536
) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming dataset .jsonl.gz langsung dari URL secara bertahap (chunking),
    mendekompresi on-the-fly, membaca baris demi baris dengan json.loads,
    serta memfilter ulasan berdasarkan timestamp <= cutoff_timestamp.
    """
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
            
            # Filter ulasan hanya dengan timestamp <= cutoff_timestamp
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
    """
    Menjalankan alur preprocessing:
    1. Buang rating == 3 dan nilai kosong pada rating, title, dan text.
    2. Pelabelan biner: 1, 2 -> 0 (Negatif); 4, 5 -> 1 (Positif).
    3. Gabungkan title dan text menjadi full_text.
    4. Pembersihan dasar menjadi clean_text.
    """
    df_clean = df.copy()
    
    # 1. Konversi string kosong ke NaN dan buang baris kosong
    df_clean["title"] = df_clean["title"].replace(r"^\s*$", np.nan, regex=True)
    df_clean["text"] = df_clean["text"].replace(r"^\s*$", np.nan, regex=True)
    df_clean = df_clean.dropna(subset=["rating", "title", "text"]).copy()
    
    # Buang rating 3 (netral)
    df_clean = df_clean[df_clean["rating"] != 3].copy()
    
    # 2. Buat label biner
    df_clean["label"] = df_clean["rating"].apply(
        lambda r: 0 if r in [1, 2, 1.0, 2.0] else (1 if r in [4, 5, 4.0, 5.0] else np.nan)
    )
    df_clean = df_clean.dropna(subset=["label"]).copy()
    df_clean["label"] = df_clean["label"].astype(int)
    
    # 3. Gabungkan title dan text menjadi full_text
    df_clean["full_text"] = df_clean["title"].astype(str).str.strip() + " " + df_clean["text"].astype(str).str.strip()
    
    # 4. Bersihkan teks dasar
    df_clean["clean_text"] = df_clean["full_text"].apply(basic_clean_text)
    
    return df_clean


if __name__ == "__main__":
    raw_csv = os.path.join("data", "video_games_reviews_2019.csv")
    if os.path.exists(raw_csv):
        print(f"Membaca data dari {raw_csv}...")
        df_raw = pd.read_csv(raw_csv)
    else:
        print("Data lokal belum ada, melakukan streaming dari web...")
        records = list(stream_reviews_from_url(max_records=5000))
        df_raw = pd.DataFrame(records)
        os.makedirs("data", exist_ok=True)
        df_raw.to_csv(raw_csv, index=False)
        
    print(f"Bentuk data awal: {df_raw.shape}")
    df_preprocessed = preprocess_reviews_dataframe(df_raw)
    print(f"Bentuk data setelah preprocessing: {df_preprocessed.shape}")
    print("\nDistribusi Label Biner:")
    print(df_preprocessed["label"].value_counts())
    
    out_csv = os.path.join("data", "video_games_preprocessed.csv")
    df_preprocessed.to_csv(out_csv, index=False)
    print(f"\nDataset hasil preprocessing tersimpan di: {out_csv}")
