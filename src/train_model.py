"""Train, compare, evaluate and persist the Smart Fake News models.

Run from the repository root: python -m src.train_model
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.preprocessing import load_source_dataset, load_processed_dataset, save_processed_dataset
from src.visualization import create_eda_visualizations, create_evaluation_visualizations
from src.report import generate_project_report

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_PATH = ROOT / "data" / "processed" / "news_dataset.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "outputs" / "evaluation_results"
VIZ_DIR = ROOT / "outputs" / "visualizations"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def _metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Fake", "Real"], output_dict=True, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": report,
    }


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    if not PROCESSED_PATH.exists():
        print("Preparing the combined dataset from data/raw/Fake.csv and True.csv ...")
        frame = load_source_dataset(RAW_DIR)
        save_processed_dataset(frame, PROCESSED_PATH)
    frame = load_processed_dataset(PROCESSED_PATH)

    if frame["label"].nunique() != 2:
        raise ValueError("Training requires both labels: 0=Fake and 1=Real.")
    frame["label_name"] = frame["label"].map({0: "Fake", 1: "Real"})
    X = frame["clean_text"].astype(str)
    y = frame["label"].astype(int)

    # Split first. The vectorizer is fitted only on X_train, never on test text.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    create_eda_visualizations(frame, VIZ_DIR)

    vectorizer = TfidfVectorizer(
        lowercase=False,  # clean_text already lowercases both training and user text
        ngram_range=(1, 2),
        max_features=80000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    estimators = {
        "Logistic Regression": LogisticRegression(C=2.0, max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    }

    scores = {}
    predictions = {}
    fitted = {}
    for name, estimator in estimators.items():
        print(f"Training {name} ...")
        estimator.fit(X_train_vec, y_train)
        y_pred = estimator.predict(X_test_vec)
        predictions[name] = y_pred
        fitted[name] = estimator
        scores[name] = _metrics(y_test, y_pred)
        print(f"  accuracy={scores[name]['accuracy']:.4f} | macro-F1={scores[name]['f1_macro']:.4f}")

    # Selection rule is explicit and reproducible: highest macro-F1, then accuracy.
    selected_name = max(scores, key=lambda n: (scores[n]["f1_macro"], scores[n]["accuracy"]))
    selected_model = fitted[selected_name]

    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.joblib", compress=3)
    joblib.dump(selected_model, MODEL_DIR / "model.joblib", compress=3)
    for name, estimator in fitted.items():
        file_name = "logistic_regression.joblib" if name == "Logistic Regression" else "multinomial_nb.joblib"
        joblib.dump(estimator, MODEL_DIR / file_name, compress=3)

    metadata = {
        "selected_model": selected_name,
        "label_map": {"0": "Fake", "1": "Real"},
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "total_rows": int(len(frame)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "tfidf_features": int(X_train_vec.shape[1]),
        "tfidf_config": {"ngram_range": [1, 2], "max_features": 80000, "min_df": 2, "max_df": 0.95, "sublinear_tf": True},
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_rule": "Highest macro-F1; accuracy breaks ties.",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_payload = {
        "dataset": {
            "rows_after_cleaning": int(len(frame)),
            "fake_count": int((frame["label"] == 0).sum()),
            "real_count": int((frame["label"] == 1).sum()),
            "duplicates_removed": int(frame.attrs.get("duplicates_removed", 0)),
        },
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "selected_model": selected_name,
        "models": scores,
        "metadata": metadata,
    }
    (RESULT_DIR / "model_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    rows = []
    for name, metric in scores.items():
        rows.append({"model": name, **{k: metric[k] for k in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted"]}})
    pd.DataFrame(rows).to_csv(RESULT_DIR / "model_comparison.csv", index=False)

    summary = ["# Model evaluation summary", "", f"**Selected model:** {selected_name}", "", "All metrics below are measured on a held-out, stratified test set.", "", "| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |", "|---|---:|---:|---:|---:|"]
    for name, metric in scores.items():
        summary.append(f"| {name} | {metric['accuracy']:.4f} | {metric['precision_macro']:.4f} | {metric['recall_macro']:.4f} | {metric['f1_macro']:.4f} |")
    summary += ["", "## Confusion matrix label order", "Rows are actual labels; columns are predictions. Order: Fake (0), Real (1).", ""]
    for name, metric in scores.items(): summary.append(f"- **{name}:** `{metric['confusion_matrix']}`")
    (RESULT_DIR / "evaluation_summary.md").write_text("\n".join(summary), encoding="utf-8")

    create_evaluation_visualizations(scores, y_test, predictions, VIZ_DIR, vectorizer, fitted["Logistic Regression"])
    generate_project_report(ROOT)

    print("\nTraining complete.")
    print(f"Selected model: {selected_name}")
    print(f"Artifacts: {MODEL_DIR}")
    print(f"Evaluation: {RESULT_DIR}")
    print(f"Report: {ROOT / 'Project_Report.pdf'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
