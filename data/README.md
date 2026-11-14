# Dataset source and download

This project uses the **Fake and Real News Dataset** (about 44.9k English-language articles).
The upstream dataset is published on Kaggle by Clément Bisaillon:

- Upstream: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
- Public Hugging Face mirror: https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset

The Hugging Face dataset card currently documents a 44.9k-row training split and these columns:
`title`, `text`, `subject`, `date`, `target`. Its mirror target mapping is `0 = Real`, `1 = Fake`.
The project's training convention is `0 = Fake`, `1 = Real`; `scripts/download_dataset.py`
explicitly converts the two classes into `data/raw/Fake.csv` and `data/raw/True.csv` so the
pipeline's label mapping remains correct.

## Download automatically

```bash
pip install -r requirements.txt
python scripts/download_dataset.py
```

The downloader uses the public Hugging Face mirror first and falls back to Kaggle CLI when
configured. No credentials are required for the public Hugging Face mirror. Kaggle fallback
requires a Kaggle API token, which must be stored outside this repository and must never be
committed.

## Expected local files

```text
data/raw/Fake.csv   # Fake articles
 data/raw/True.csv  # Real articles
```

The raw files are intentionally git-ignored to avoid committing a large third-party dataset.
The downloader writes `dataset_source.json` with source, license, counts, and label mapping.
Check the source dataset card and upstream terms before redistribution. The model report records
the dataset source and the dataset's limitations; model predictions are not factual verification.
