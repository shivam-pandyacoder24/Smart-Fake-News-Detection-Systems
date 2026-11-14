# NewsGuard — Smart Fake News Detection System

**Major Project | AI/ML with Python | Python · NLP · TF-IDF · Scikit-learn · Streamlit**

NewsGuard is an educational text-classification system that accepts a news headline or article and estimates whether its textual patterns resemble examples labeled **Fake** or **Real** in a labeled training corpus.

> **Responsible use:** This model is not a definitive fact-checker. It learns patterns from historical training examples and cannot independently verify claims, sources, citations or breaking news. Predictions can be wrong.

## Dataset and attribution

- Original source: [Fake and Real News Dataset by Clément Bisaillon (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Download mirror: [haoxianc/gilchr_fake-and-real-news-dataset (Hugging Face)](https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset)
- Fields include title, article text, subject, date and target label.
- The mirror's labels are `0 = Real`, `1 = Fake`; the downloader converts these into the application's mapping `0 = Fake`, `1 = Real`.
- The raw CSV is downloaded at build time and is not redistributed in this repository. Dataset attribution and download instructions are in `data/README.md`; provenance is recorded in `data/raw/dataset_source.json`.

## Machine-learning methodology

1. Handle missing content and labels, normalize text, and remove duplicate article records.
2. Explore class distribution and article-length patterns.
3. Split the cleaned corpus into stratified 80% train and 20% test partitions (random seed 42).
4. Fit TF-IDF on the training partition only; transform test and user-submitted text with the fitted vectorizer to prevent data leakage.
5. Train Logistic Regression and Multinomial Naive Bayes on the same feature matrix.
6. Compare accuracy, macro precision, macro recall, macro F1, classification reports and confusion matrices.
7. Select by macro F1, save the estimator and fitted TF-IDF vectorizer with Joblib, and load the saved artifacts in Streamlit without retraining at inference time.

## Measured held-out results

The held-out evaluation contains **7,766 articles**. The selected model is **Logistic Regression**, which achieved the higher macro F1.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **99.21%** | **99.23%** | **99.19%** | **99.21%** |
| Multinomial Naive Bayes | 96.05% | 96.03% | 96.01% | 96.02% |

The cleaned corpus contains **38,827 articles**: 17,901 Fake and 20,926 Real. The preprocessing step removed 6,071 duplicate articles. Detailed values, classification reports and confusion matrices are in `outputs/evaluation_results/`; charts are in `outputs/visualizations/`.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py --source huggingface
python scripts/prepare_data.py
python scripts/smoke_check.py
pytest -q
python -m src.train_model
streamlit run app/streamlit_app.py
```

## Project contents

- `app/streamlit_app.py`: interactive News Detector, Model Dashboard, and Methodology & Limitations views.
- `src/`: preprocessing, TF-IDF/model training, report generation and plotting.
- `scripts/`: dataset download, data preparation, smoke checks and application evidence capture.
- `models/model.joblib` and `models/tfidf_vectorizer.joblib`: selected trained estimator and fitted vectorizer.
- `outputs/evaluation_results/`: measured metrics and model comparison.
- `outputs/visualizations/`: class distribution, article lengths, comparison and confusion-matrix charts.
- `Project_Report.pdf`: project report generated from the actual held-out evaluation.
- `screenshots/` and `demo/`: genuine application screenshots and recorded browser walkthrough.

## Screenshots and demonstration

Application screenshots are in `screenshots/`. The browser demonstration is [demo/NewsGuard_Demo.webm](demo/NewsGuard_Demo.webm). A copy of the run evidence is also retained as the GitHub Actions artifact `newsguard-training-evidence`.

## Live deployment

The application is ready for Streamlit Community Cloud deployment. The repository owner must sign in at [share.streamlit.io](https://share.streamlit.io/), connect this repository, choose branch `main`, and set the main file to `app/streamlit_app.py`. **A live application URL is not claimed yet**; add it here only after deployment succeeds and the public URL is verified.

## Limitations

- Dataset labels are inherited from the source corpus and are not independent fact-check verdicts.
- The historical English-language corpus has time, topic, geographic and publisher coverage limits.
- Satire, opinion, short headlines, breaking news and unfamiliar subjects may be misclassified.
- A prediction reflects learned text patterns; it is not proof of truth or a substitute for source verification.

## References

- [Original Kaggle dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- [Hugging Face dataset mirror](https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset)
- [Scikit-learn documentation](https://scikit-learn.org/stable/)
- [Streamlit documentation](https://docs.streamlit.io/)
- [Joblib documentation](https://joblib.readthedocs.io/)
