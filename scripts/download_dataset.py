"""Download and normalize the public Fake/Real News corpus.

Preferred source: the public Hugging Face mirror of the Kaggle Fake and Real
News Dataset. The mirror's target mapping is 0=Real, 1=Fake. This project
normalizes the output back to Fake.csv (label 0) and True.csv (label 1), as
required by the rest of the pipeline.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
HF_DATASET = "haoxianc/gilchr_fake-and-real-news-dataset"
KAGGLE_DATASET = "clmentbisaillon/fake-and-real-news-dataset"


def _write_hf_mirror() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The 'datasets' package is required for the Hugging Face mirror. Install requirements.txt.") from exc

    print(f"Loading public Hugging Face dataset: {HF_DATASET}")
    dataset = load_dataset(HF_DATASET, split="train")
    frame = dataset.to_pandas()
    if "content" in frame.columns and "text" not in frame.columns:
        frame = frame.rename(columns={"content": "text"})
    required = {"title", "text", "target"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset schema changed; missing columns: {sorted(missing)}")

    for column in ("subject", "date"):
        if column not in frame.columns:
            frame[column] = ""
    frame["target"] = frame["target"].astype(int)
    values = set(frame["target"].dropna().unique().tolist())
    if values != {0, 1}:
        raise ValueError(f"Expected binary target labels {{0, 1}}, received {values}")

    # Dataset card documents 0=Real and 1=Fake. Save in this project's schema.
    common = ["title", "text", "subject", "date"]
    fake = frame.loc[frame["target"] == 1, common].copy()
    real = frame.loc[frame["target"] == 0, common].copy()
    if min(len(fake), len(real)) < 100:
        raise ValueError("Unexpectedly small class count; refusing to train on an incomplete mirror.")
    fake.to_csv(RAW / "Fake.csv", index=False)
    real.to_csv(RAW / "True.csv", index=False)
    provenance = {
        "dataset_id": HF_DATASET,
        "mirror_of": "Kaggle Fake and Real News Dataset",
        "upstream_url": "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
        "mirror_url": f"https://huggingface.co/datasets/{HF_DATASET}",
        "mirror_license": "MIT (see dataset card and LICENSE on Hugging Face)",
        "mirror_label_mapping": {"0": "Real", "1": "Fake"},
        "project_label_mapping": {"0": "Fake", "1": "Real"},
        "rows": int(len(frame)),
        "fake_rows": int(len(fake)),
        "real_rows": int(len(real)),
    }
    (RAW / "dataset_source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Saved {len(fake):,} fake and {len(real):,} real articles to {RAW}")


def _write_kaggle() -> None:
    if not shutil.which("kaggle"):
        raise RuntimeError("Kaggle CLI is unavailable. Install/configure it or use the default Hugging Face download.")
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW), "--unzip"], cwd=ROOT, check=True)
    for name in ("Fake.csv", "True.csv"):
        if not (RAW / name).exists():
            raise FileNotFoundError(f"Kaggle download did not produce {name}")
    (RAW / "dataset_source.json").write_text(json.dumps({
        "dataset_id": KAGGLE_DATASET,
        "source_url": f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}",
        "project_label_mapping": {"0": "Fake", "1": "Real"},
        "download_method": "Kaggle CLI",
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["auto", "huggingface", "kaggle"], default="auto")
    args = parser.parse_args()
    if (RAW / "Fake.csv").exists() and (RAW / "True.csv").exists():
        print("Fake.csv and True.csv already exist in data/raw/. No download needed.")
        return 0

    if args.source in ("auto", "huggingface"):
        try:
            _write_hf_mirror()
            return 0
        except Exception as exc:
            if args.source == "huggingface":
                raise
            print(f"Hugging Face download failed: {exc}", file=sys.stderr)
            print("Trying Kaggle CLI fallback ...", file=sys.stderr)
    _write_kaggle()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"DATASET DOWNLOAD ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
