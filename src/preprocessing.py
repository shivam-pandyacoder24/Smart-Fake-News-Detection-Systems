"""Dataset loading and consistent text preprocessing.

The canonical dataset is Clément Bisaillon's Fake and Real News Dataset.
Expected source files: data/raw/Fake.csv and data/raw/True.csv.
Labels: 0 = Fake, 1 = Real.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
HTML_RE = re.compile(r"<[^>]+>")
NON_TEXT_RE = re.compile(r"[^a-z0-9\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize news text with the same function used for training and inference.

    Stop words are intentionally retained: negations and context words can be
    useful for this classification task. Punctuation and URLs are normalized,
    text is lowercased, and whitespace is collapsed.
    """
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value))
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" URL ", text)
    text = text.lower()
    text = NON_TEXT_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def _read_news_csv(path: Path, label: int, label_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dataset file: {path}. Follow data/README.md to download the dataset."
        )
    frame = pd.read_csv(path, low_memory=False)
    required = {"title", "text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    for optional in ("subject", "date"):
        if optional not in frame.columns:
            frame[optional] = ""
    frame = frame[["title", "text", "subject", "date"]].copy()
    frame["label"] = int(label)
    frame["label_name"] = label_name
    frame["source_file"] = path.name
    return frame


def load_source_dataset(data_dir: str | Path) -> pd.DataFrame:
    """Read Fake.csv and True.csv and return a single labeled dataframe."""
    data_dir = Path(data_dir)
    fake = _read_news_csv(data_dir / "Fake.csv", 0, "Fake")
    real = _read_news_csv(data_dir / "True.csv", 1, "Real")
    frame = pd.concat([fake, real], ignore_index=True)

    frame["title"] = frame["title"].fillna("").astype(str)
    frame["text"] = frame["text"].fillna("").astype(str)
    frame["subject"] = frame["subject"].fillna("Unknown").astype(str)
    frame["date"] = frame["date"].fillna("").astype(str)
    frame["combined_text"] = (frame["title"] + ". " + frame["text"]).str.strip()
    frame["clean_text"] = frame["combined_text"].map(clean_text)

    # Remove rows with no usable text and exact duplicate article text.
    frame = frame[frame["clean_text"].str.len() > 0].copy()
    before = len(frame)
    frame = frame.drop_duplicates(subset=["clean_text"], keep="first").reset_index(drop=True)
    frame.attrs["duplicates_removed"] = before - len(frame)
    frame.attrs["rows_before_deduplication"] = before
    return frame


def save_processed_dataset(frame: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["title", "text", "subject", "date", "label", "label_name", "clean_text", "source_file"]
    frame[cols].to_csv(output_path, index=False)
    metadata = {
        "duplicates_removed": int(frame.attrs.get("duplicates_removed", 0)),
        "rows_before_deduplication": int(frame.attrs.get("rows_before_deduplication", len(frame))),
        "rows_after_cleaning": int(len(frame)),
    }
    output_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def load_processed_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. Run `python scripts/prepare_data.py` first."
        )
    frame = pd.read_csv(path, low_memory=False)
    required = {"clean_text", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Processed dataset is missing columns: {sorted(missing)}")
    frame["clean_text"] = frame["clean_text"].fillna("").astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    frame = frame.dropna(subset=["label"])
    frame["label"] = frame["label"].astype(int)
    frame = frame[frame["label"].isin([0, 1])]
    frame = frame[frame["clean_text"].str.len() > 0]
    if frame.empty:
        raise ValueError("The processed dataset contains no usable labeled text rows.")
    metadata_path = path.with_suffix(".metadata.json")
    if metadata_path.exists():
        try:
            frame.attrs.update(json.loads(metadata_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return frame.reset_index(drop=True)
