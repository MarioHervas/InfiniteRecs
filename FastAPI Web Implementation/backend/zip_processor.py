import io
import zipfile
from typing import Optional, Tuple

import pandas as pd


def extract_username_from_filename(filename: str) -> Optional[str]:
    base = filename.rsplit("/", 1)[-1]
    if base.endswith(".zip"):
        base = base[:-4]

    parts = base.split("-")
    if len(parts) < 3 or parts[0].lower() != "letterboxd":
        return None
    return parts[1]


def load_ratings_from_zip_bytes(zip_bytes: bytes,
                                 positive_threshold: float = 0.6,
                                 negative_threshold: float = 0.3
                                 ) -> Tuple[pd.DataFrame, dict]:
    
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        candidates = [n for n in z.namelist() if n.endswith("ratings.csv")]
        if not candidates:
            raise ValueError(
                f"No ratings.csv found!"
            )
        with z.open(candidates[0]) as f:
            df = pd.read_csv(f)

    if "Name" not in df.columns or "Rating" not in df.columns:
        raise ValueError(
            "Not a valid Letterboxd export"
        )

    df = df.dropna(subset=["Name", "Rating"]).copy()
    df["rating_norm"] = (df["Rating"] - 0.5) / (5.0 - 0.5)
    df["title_normalized"] = df["Name"].str.lower()

    stats = {
        "n_total": int(len(df)),
        "n_positive": int((df["rating_norm"] > positive_threshold).sum()),
        "n_negative": int((df["rating_norm"] < negative_threshold).sum()),
        "n_neutral": int(len(df) - (df["rating_norm"] > positive_threshold).sum() - (df["rating_norm"] < negative_threshold).sum()),
    }

    return df[["title_normalized", "rating_norm"]].copy(), stats


def load_seen_titles_from_zip_bytes(zip_bytes: bytes) -> set:
    seen = set()
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as z:
        candidates = [n for n in z.namelist() if n.endswith("watched.csv")]
        if not candidates:
            return seen
        with z.open(candidates[0]) as f:
            df = pd.read_csv(f)
    if "Name" not in df.columns:
        return seen
    df = df.dropna(subset=["Name"]).copy()
    seen = set(df["Name"].str.lower())
    return seen
