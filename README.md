# NewsGuard — Smart Fake News Detection System

**Major Project | AI/ML with Python | Python · NLP · TF-IDF · Scikit-learn · Streamlit**

NewsGuard is an educational text-classification system that takes a news headline or article and estimates whether its textual patterns resemble examples labeled **Fake** or **Real** in a labeled training dataset.

> **Responsible-use note:** This is not a definitive fact-checker. It learns patterns from its training corpus and cannot independently verify claims, citations, publisher identity, or breaking news. Predictions may be wrong.

## Automated project setup

The complete project source is held in the `source_bundle.*.b64` files so the repository can bootstrap from a clean initial commit. The included GitHub Actions workflow automatically:

1. Reconstructs and extracts the source package.
2. Installs the Python dependencies and a headless browser.
3. Downloads the labeled dataset from the public Hugging Face mirror.
4. Cleans and prepares the corpus, runs smoke checks and unit tests.
5. Trains and compares Logistic Regression with Multinomial Naive Bayes.
6. Generates held-out metrics, confusion matrices, charts and a refreshed PDF report.
7. Launches Streamlit, captures genuine application screenshots and records a short browser walkthrough.
8. Commits the trained model, fitted TF-IDF vectorizer, metrics, report, screenshots and demo video to `main`.

Watch the **Actions** tab for the initial bootstrap run. The detailed source files are expanded into their normal project folders by the workflow.

## Dataset and attribution

- Upstream: [Fake and Real News Dataset by Clément Bisaillon (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Public mirror: [haoxianc/gilchr_fake-and-real-news-dataset (Hugging Face)](https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset)
- Mirror fields: `title`, `text`, `subject`, `date`, `target`.
- Mirror label mapping: `0 = Real`, `1 = Fake`. The downloader deliberately converts this to the project mapping: `0 = Fake`, `1 = Real`.
- Raw CSVs are downloaded during the workflow and are not committed because they are large; source and download instructions are included in `data/README.md`.

The public dataset card reports approximately 44.9k examples and an MIT license for the mirror. Check the current dataset card and original Kaggle terms before redistributing the data.

## Machine-learning methodology

- Preprocessing handles missing fields, HTML, URLs, case, punctuation, whitespace and exact duplicate article text.
- A stratified 80/20 train/test split uses random seed 42.
- TF-IDF uses unigram and bigram features. The vectorizer is **fitted only on training text**; test and user text use the fitted transform.
- Logistic Regression and Multinomial Naive Bayes are trained on the same feature matrix and evaluated on the same held-out test set.
- Models are compared using accuracy, macro/weighted precision, recall and F1-score, per-class classification reports and confusion matrices.
- Selection is based on macro-F1, with accuracy as the tie-breaker.
- The selected classifier and fitted vectorizer are saved with Joblib. Streamlit loads them for inference without retraining on each submission.

## Local execution after source extraction

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py
python scripts/prepare_data.py
python scripts/smoke_check.py
pytest -q
python -m src.train_model
streamlit run app/streamlit_app.py
```

## Results and artifacts

Actual test results are saved to `outputs/evaluation_results/model_metrics.json` and `model_comparison.csv`; confusion matrices and charts are written to `outputs/visualizations/`. The chosen model, fitted vectorizer and metadata are written to `models/`. `Project_Report.pdf` is regenerated from the measured results.

**Do not quote accuracy or other performance numbers until the training workflow succeeds and the metrics JSON is committed.**

## Streamlit app and deployment

The application contains:
- **News Detector:** headline/article text input, empty-input validation and model prediction.
- **Model Dashboard:** held-out metrics, model comparison and charts.
- **Methodology & limits:** processing details, dataset caveats and responsible interpretation.

To deploy, use [Streamlit Community Cloud](https://share.streamlit.io/) and select `app/streamlit_app.py` as the main file after the training workflow has committed `models/model.joblib` and `models/tfidf_vectorizer.joblib`. Add the resulting public app URL to the README only after it is live and verified.

## Screenshots and demonstration

The workflow writes actual application screenshots to `screenshots/` and a browser demonstration recording to `demo/NewsGuard_Demo.webm`. A downloadable run artifact named `newsguard-training-evidence` is also retained for 30 days. A public video-hosting URL is not claimed until the recording is uploaded and the public/unlisted link is verified.

## Limitations

- Source-dataset labels are not independent fact-check verdicts.
- The historical, English-language news corpus has temporal, geographic and topic coverage limits.
- Satire, opinion, very short headlines, current events and unfamiliar topics may be misclassified.
- A classifier score is not a guarantee of truth or a calibrated measure of factual reliability.

## License and references

See `LICENSE` and `data/README.md` for project license and dataset attribution. The dataset itself is not redistributed here. Primary references: the dataset links above, [scikit-learn](https://scikit-learn.org/stable/), [Streamlit](https://docs.streamlit.io/) and [Joblib](https://joblib.readthedocs.io/).
