"""Publication-ready EDA and model evaluation visualizations."""
from __future__ import annotations
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

PALETTE = {"Fake": "#F47C7C", "Real": "#48D6C8"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def create_eda_visualizations(frame: pd.DataFrame, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    sns.set_theme(style="whitegrid", context="notebook")
    theme_bg, text_color, grid_color = "#101D31", "#EAF2FA", "#29415F"

    counts = frame["label_name"].value_counts().reindex(["Fake", "Real"], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=theme_bg)
    ax.set_facecolor(theme_bg)
    bars = ax.bar(counts.index, counts.values, color=[PALETTE[x] for x in counts.index], width=0.58)
    ax.set_title("Dataset label distribution", color=text_color, weight="bold", pad=16)
    ax.set_ylabel("Number of articles", color=text_color)
    ax.tick_params(colors=text_color)
    ax.grid(axis="y", color=grid_color, alpha=0.45)
    for spine in ax.spines.values(): spine.set_visible(False)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{value:,}", ha="center", va="bottom", color=text_color, fontsize=10)
    _save(fig, out_dir / "label_distribution.png")

    lengths = frame["clean_text"].str.split().str.len().clip(upper=500)
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=theme_bg)
    ax.set_facecolor(theme_bg)
    for label in (0, 1):
        values = lengths[frame["label"] == label]
        ax.hist(values, bins=45, alpha=0.64, color=PALETTE["Fake" if label == 0 else "Real"], label="Fake" if label == 0 else "Real", density=True)
    ax.set_title("Article length distribution", color=text_color, weight="bold", pad=16)
    ax.set_xlabel("Tokens per article (clipped at 500)", color=text_color)
    ax.set_ylabel("Density", color=text_color)
    ax.tick_params(colors=text_color)
    ax.legend(frameon=False, labelcolor=text_color)
    ax.grid(color=grid_color, alpha=0.35)
    for spine in ax.spines.values(): spine.set_visible(False)
    _save(fig, out_dir / "article_length_distribution.png")


def create_evaluation_visualizations(results: dict, y_test, predictions: dict, out_dir: str | Path, vectorizer=None, logistic_model=None) -> None:
    out_dir = Path(out_dir)
    theme_bg, text_color, grid_color = "#101D31", "#EAF2FA", "#29415F"
    sns.set_theme(style="whitegrid", context="notebook")

    names = list(results.keys())
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(metrics)); width = 0.34
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=theme_bg)
    ax.set_facecolor(theme_bg)
    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + (i - (len(names)-1)/2)*width, vals, width, label=name,
               color="#48D6C8" if "Logistic" in name else "#8B9EFF")
    ax.set_xticks(x, ["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1-score\n(macro)"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score", color=text_color)
    ax.set_title("Classifier performance comparison", color=text_color, weight="bold", pad=16)
    ax.tick_params(colors=text_color)
    ax.grid(axis="y", color=grid_color, alpha=0.4)
    ax.legend(frameon=False, labelcolor=text_color)
    for spine in ax.spines.values(): spine.set_visible(False)
    _save(fig, out_dir / "model_comparison.png")

    for name, pred in predictions.items():
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5.7, 4.8), facecolor=theme_bg)
        ax.set_facecolor(theme_bg)
        sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
                    xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"],
                    annot_kws={"size": 13, "weight": "bold", "color": "#071321"}, ax=ax)
        ax.set_title(f"{name} - confusion matrix", color=text_color, weight="bold", pad=14)
        ax.set_xlabel("Predicted label", color=text_color)
        ax.set_ylabel("Actual label", color=text_color)
        ax.tick_params(colors=text_color)
        _save(fig, out_dir / ("confusion_matrix_" + name.lower().replace(" ", "_") + ".png"))

    if vectorizer is not None and logistic_model is not None and hasattr(logistic_model, "coef_"):
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        weights = logistic_model.coef_[0]
        fake_idx = np.argsort(weights)[:12]
        real_idx = np.argsort(weights)[-12:][::-1]
        labels = list(feature_names[fake_idx][::-1]) + list(feature_names[real_idx])
        vals = list(weights[fake_idx][::-1]) + list(weights[real_idx])
        colors = [PALETTE["Fake"]]*12 + [PALETTE["Real"]]*12
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=theme_bg)
        ax.set_facecolor(theme_bg)
        ax.barh(labels, vals, color=colors)
        ax.axvline(0, color=text_color, linewidth=0.8, alpha=0.7)
        ax.set_title("Most influential TF-IDF terms (Logistic Regression)", color=text_color, weight="bold", pad=16)
        ax.set_xlabel("Learned coefficient (directional, not causal)", color=text_color)
        ax.tick_params(colors=text_color, labelsize=9)
        ax.grid(axis="x", color=grid_color, alpha=0.3)
        for spine in ax.spines.values(): spine.set_visible(False)
        _save(fig, out_dir / "top_logistic_terms.png")
