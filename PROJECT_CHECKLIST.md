# Major project submission checklist

Verified by the successful GitHub Actions build, training, application-capture and artifact-validation steps:

- [x] Complete Python source code and Streamlit application
- [x] Dataset source, attribution, label mapping and download instructions
- [x] Preprocessing, EDA and duplicate/missing-value handling
- [x] TF-IDF fitted on training text only; no test-data leakage
- [x] Logistic Regression and Multinomial Naive Bayes implementations
- [x] Held-out accuracy, precision, recall, F1, classification reports and confusion matrices for both models
- [x] Model comparison and documented Logistic Regression selection
- [x] Saved Joblib classifier and fitted TF-IDF vectorizer
- [x] Evaluation results and visualizations
- [x] Project_Report.pdf regenerated using actual measured results
- [x] Genuine Streamlit screenshots, including empty-input validation and a sample prediction
- [x] Browser demonstration recording: [demo/NewsGuard_Demo.webm](demo/NewsGuard_Demo.webm)
- [ ] Live Streamlit Community Cloud deployment and public URL (requires repository-owner authorization)

**Measured test results:** Logistic Regression accuracy 99.21%, macro-F1 99.21%; Multinomial Naive Bayes accuracy 96.05%, macro-F1 96.02%. The cleaned corpus contains 38,827 articles (17,901 Fake; 20,926 Real), with 7,766 held-out test articles.

Raw dataset CSV files are downloaded by the documented script rather than redistributed. Dataset attribution and provenance are included in `data/README.md` and `data/raw/dataset_source.json`.

**Responsible interpretation:** NewsGuard is a text-pattern classifier, not a fact-checking authority. It may be incorrect, particularly for unfamiliar topics, satire and breaking news.
