"""Combine the downloaded Kaggle source files into one cleaned CSV."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import load_source_dataset, save_processed_dataset

frame = load_source_dataset(ROOT / "data" / "raw")
out = save_processed_dataset(frame, ROOT / "data" / "processed" / "news_dataset.csv")
print(f"Prepared {len(frame):,} articles -> {out}")
print(frame["label_name"].value_counts().to_string())
print(f"Duplicate articles removed: {frame.attrs.get('duplicates_removed', 0):,}")
