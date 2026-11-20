"""
Module Pemuatan Data Streaming & Chunking (Amazon Reviews 2023 - Video_Games)
Membaca file .jsonl.gz baris demi baris (line-by-line) dan memfilter ulasan
dengan timestamp <= 1577836799999 (hingga 31 Desember 2019 UTC).
"""

import os
import gzip
import json
import zlib
import time
import datetime
import requests
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


def stream_reviews_from_local_file(
    file_path: str,
    max_records: Optional[int] = 10000,
    cutoff_timestamp: int = CUTOFF_TIMESTAMP
) -> Generator[Dict[str, Any], None, None]:
    """
    Membaca berkas lokal .jsonl.gz baris demi baris menggunakan gzip.open
    dan json.loads agar memori RAM tetap ringan.
    """
    yielded = 0
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
            record = json.loads(line_str)
            ts = record.get("timestamp")
            if ts is not None and ts <= cutoff_timestamp:
                yield record
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    break


def download_file_chunked(
    url: str = URL_VIDEO_GAMES,
    output_path: str = "data/Video_Games.jsonl.gz",
    chunk_size: int = 1024 * 1024,
    max_mb: Optional[int] = 50
) -> str:
    """
    Mengunduh berkas secara bertahap (chunked download) dengan batas ukuran tertentu (opsional).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    downloaded = 0
    max_bytes = max_mb * 1024 * 1024 if max_mb else None

    print(f"Mengunduh {url} secara chunked ke {output_path}...")
    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if max_bytes and downloaded >= max_bytes:
                    print(f"Batas {max_mb} MB tercapai, selesai mengunduh bagian sampel.")
                    break
    print(f"Tersimpan: {downloaded / (1024*1024):.2f} MB di {output_path}")
    return output_path


def load_filtered_dataset_to_df(
    max_records: int = 10000,
    source: str = "url",
    local_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Memuat ulasan Video Games tersaring ke dalam pandas DataFrame.
    """
    records = []
    streamer = (
        stream_reviews_from_url(max_records=max_records)
        if source == "url"
        else stream_reviews_from_local_file(local_path, max_records=max_records)
    )

    for item in streamer:
        text = item.get("text", "")
        if text and len(text.strip()) > 5:
            ts = item.get("timestamp")
            dt = datetime.datetime.fromtimestamp(ts / 1000.0, datetime.timezone.utc)
            rating = item.get("rating")
            
            # Label sentimen: Positif (1) jika rating >= 4.0, Negatif (0) jika rating <= 2.0
            sentiment = 1 if rating >= 4.0 else (0 if rating <= 2.0 else None)

            records.append({
                "rating": rating,
                "title": item.get("title", ""),
                "text": text.strip(),
                "timestamp": ts,
                "date": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "year": dt.year,
                "asin": item.get("asin", item.get("parent_asin", "")),
                "verified_purchase": item.get("verified_purchase", False),
                "helpful_vote": item.get("helpful_vote", 0),
                "sentiment": sentiment
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("=" * 70)
    print("DEMO PEMUATAN DATA STREAMING DENGAN FILTER TIMESTAMP <= 2019-12-31")
    print("=" * 70)
    start = time.time()
    df_filtered = load_filtered_dataset_to_df(max_records=5000, source="url")
    elapsed = time.time() - start

    output_csv = os.path.join("data", "video_games_reviews_2019.csv")
    df_filtered.to_csv(output_csv, index=False)

    print(f"Selesai memuat {len(df_filtered)} ulasan dalam {elapsed:.2f} detik.")
    print(f"Data disimpan ke: {output_csv}")
    print(f"Maksimum timestamp : {df_filtered['timestamp'].max()} (Batas: {CUTOFF_TIMESTAMP})")
    print(f"Rentang tanggal    : {df_filtered['date'].min()} s/d {df_filtered['date'].max()}")
    print("Verifikasi <= 2019 :", (df_filtered['timestamp'] <= CUTOFF_TIMESTAMP).all())
    print("\nDistribusi Sentimen:")
    print(df_filtered['sentiment'].value_counts(dropna=False))
