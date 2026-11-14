"""Create the submission PDF report. Re-run after training to embed real metrics."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether

NAVY = colors.HexColor("#0B1830")
TEAL = colors.HexColor("#18B6AB")
PALE = colors.HexColor("#EAF4F7")
INK = colors.HexColor("#24354B")
MUTED = colors.HexColor("#60748A")


def generate_project_report(root: str | Path) -> Path:
    root = Path(root)
    output = root / "Project_Report.pdf"
    metrics_path = root / "outputs/evaluation_results/model_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else None

    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=19*mm, leftMargin=19*mm, topMargin=19*mm, bottomMargin=18*mm, title="Smart Fake News Detection System", author="Shivam A Pandya")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CoverTitle2", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=27, leading=32, textColor=colors.white, alignment=TA_LEFT, spaceAfter=8))
    styles.add(ParagraphStyle(name="CoverSub2", parent=styles["Normal"], fontSize=12, leading=18, textColor=colors.HexColor("#D1E5F0")))
    styles.add(ParagraphStyle(name="H1x", parent=styles["Heading1"], fontSize=17, leading=21, textColor=NAVY, spaceBefore=10, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2x", parent=styles["Heading2"], fontSize=11.5, leading=15, textColor=TEAL, spaceBefore=8, spaceAfter=4))
    styles.add(ParagraphStyle(name="Bodyx", parent=styles["BodyText"], fontSize=9, leading=13.5, textColor=INK, spaceAfter=6))
    styles.add(ParagraphStyle(name="Smallx", parent=styles["BodyText"], fontSize=7.7, leading=11, textColor=MUTED, spaceAfter=4))
    styles.add(ParagraphStyle(name="Calloutx", parent=styles["BodyText"], fontSize=9, leading=13, textColor=NAVY, backColor=PALE, borderColor=TEAL, borderWidth=0.7, borderPadding=8, spaceBefore=6, spaceAfter=9))

    story = []
    cover_data = [
        [Paragraph("AI / MACHINE LEARNING PROJECT", styles["CoverSub2"])],
        [Spacer(1, 24*mm)],
        [Paragraph("Smart Fake News<br/>Detection System", styles["CoverTitle2"])],
        [Spacer(1, 7*mm)],
        [Paragraph("An explainable baseline NLP classification workflow using TF-IDF, Logistic Regression, Multinomial Naive Bayes and Streamlit.", styles["CoverSub2"])],
        [Spacer(1, 30*mm)],
        [Paragraph("PROJECT REPORT", ParagraphStyle("pr", parent=styles["CoverSub2"], fontSize=10, textColor=colors.HexColor("#48D6C8")))],
        [Paragraph("Student: Shivam A Pandya<br/>Program: AI/ML with Python Training Program<br/>Project: Major Project<br/>Report date: " + datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %B %Y"), styles["CoverSub2"])],
    ]
    cover = Table(cover_data, colWidths=[170*mm], rowHeights=[8*mm, 24*mm, 32*mm, 7*mm, 28*mm, 30*mm, 8*mm, 26*mm])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), NAVY), ("BOX", (0,0), (-1,-1), 0, NAVY),
        ("LEFTPADDING", (0,0), (-1,-1), 14*mm), ("RIGHTPADDING", (0,0), (-1,-1), 14*mm),
        ("TOPPADDING", (0,0), (-1,-1), 3*mm), ("BOTTOMPADDING", (0,0), (-1,-1), 3*mm),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story += [Spacer(1, 23*mm), cover, Spacer(1, 10*mm), Paragraph("Prepared in accordance with the Major Project requirements in the InternsElite AI/ML with Python project guidelines.", styles["Smallx"]), PageBreak()]

    def heading(s): story.append(Paragraph(s, styles["H1x"]))
    def sub(s): story.append(Paragraph(s, styles["H2x"]))
    def body(s): story.append(Paragraph(s, styles["Bodyx"]))

    heading("Abstract")
    body("The Smart Fake News Detection System is a supervised text-classification project that analyzes a supplied English-language news headline or article and estimates whether its textual patterns resemble examples labeled Fake or Real in the training dataset. The workflow combines data cleaning, exploratory analysis, TF-IDF feature extraction, Logistic Regression and Multinomial Naive Bayes model comparison, held-out evaluation, Joblib persistence and a Streamlit user interface.")
    body("This system is an educational screening aid, not an independent fact-checker. It does not verify sources, corroborate claims or establish whether a statement is true.")
    heading("1. Introduction and problem statement")
    body("Online information is published rapidly and readers may encounter misleading or fabricated claims. Natural Language Processing provides a way to examine patterns in the language of labeled examples. This project implements a reproducible binary classifier that predicts a likely label from article text and presents its output with evaluation context and limitations.")
    heading("2. Objectives and scope")
    for item in [
        "Collect and inspect a labeled real/fake news corpus.",
        "Handle missing text, normalize case and unnecessary characters, and remove duplicate articles.",
        "Explore label balance and article-length patterns with visualizations.",
        "Fit TF-IDF on training data only and transform held-out data using the fitted vectorizer.",
        "Train and compare Logistic Regression and Multinomial Naive Bayes.",
        "Evaluate both models with accuracy, precision, recall, F1-score and confusion matrices.",
        "Persist the selected model and vectorizer, then load them in an interactive Streamlit application.",
    ]:
        body("&#8226; " + item)
    heading("3. Dataset description and source")
    provenance_path = root / "data/raw/dataset_source.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    body("The project uses the Fake and Real News Dataset distributed as Fake.csv and True.csv. The public Hugging Face mirror used by the automated workflow identifies itself as a mirror of Gilchrist's Kaggle copy and links to the upstream Fake and Real News dataset by Clément Bisaillon. The mirror exposes 44,898 records with title, text, subject, date and target fields. Its dataset card documents target 0 = Real and target 1 = Fake; the downloader explicitly converts this to the project convention 0 = Fake and 1 = Real. The original raw CSV files are not redistributed in this repository; the workflow downloads them during training.")
    body("Upstream Kaggle source: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset. Public mirror and attribution: https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset. The mirror card lists an MIT license; always consult the current dataset card and upstream terms. The corpus is a historical English-language news collection, so temporal, geographic, language and topic coverage are limited.")
    if provenance:
        body(f"Recorded download provenance: <b>{provenance.get('rows', 'not recorded'):,}</b> rows; <b>{provenance.get('fake_rows', 'not recorded'):,}</b> Fake and <b>{provenance.get('real_rows', 'not recorded'):,}</b> Real articles. Mirror label mapping: {provenance.get('mirror_label_mapping', {})}; project mapping: {provenance.get('project_label_mapping', {})}.")
    heading("4. Data preprocessing")
    body("The preprocessing function decodes HTML entities, removes HTML tags, replaces URLs with a token, converts text to lowercase, normalizes punctuation to spaces and collapses repeated whitespace. Stop words are retained so that negation and contextual words are not removed by default. Empty records are discarded, and exact duplicate cleaned articles are removed. The same clean_text function is used for training examples and user-submitted text.")
    heading("5. Exploratory data analysis")
    body("The EDA step reports the number of usable records and Fake/Real class counts, and creates a label-distribution chart and article-length distribution chart. These visualizations are generated from the actual source files when the training workflow is executed.")
    for img in [root / "outputs/visualizations/label_distribution.png", root / "outputs/visualizations/article_length_distribution.png"]:
        if img.exists():
            story.append(Image(str(img), width=160*mm, height=82*mm, kind="proportional")); story.append(Spacer(1, 4*mm))
    heading("6. NLP and TF-IDF feature extraction")
    body("TF-IDF converts cleaned news documents into sparse numerical vectors using term frequency and inverse document frequency. The implementation uses unigram and bigram features, a maximum vocabulary size of 80,000, minimum document frequency 2, maximum document frequency 0.95 and sublinear term frequency scaling. The vectorizer is fitted only on the training split. Test documents and new application inputs are transformed with that fitted vectorizer, preventing test-set vocabulary leakage.")
    heading("7. Machine learning algorithms")
    sub("Logistic Regression")
    body("A linear classifier that estimates a decision boundary from TF-IDF features. It is a strong, interpretable baseline for sparse text features. Its learned coefficients can be visualized as directional associations, not causal explanations.")
    sub("Multinomial Naive Bayes")
    body("A probabilistic classifier commonly used for word-count and TF-IDF representations. It estimates class probabilities under a conditional-independence assumption and provides a useful comparison with Logistic Regression.")
    heading("8. Training and testing methodology")
    body("The workflow uses an 80/20 train-test split with stratification and a fixed random seed (42). Both models are trained on the same training TF-IDF matrix and evaluated once on the same held-out test matrix. Model selection uses the higher macro-F1 score; accuracy is the tie-breaker. The selected classifier and fitted TF-IDF vectorizer are saved independently with Joblib for inference without retraining.")
    heading("9. Model evaluation and comparison")
    if metrics:
        ds = metrics["dataset"]; sp = metrics["split"]
        body(f"After cleaning, the evaluated dataset contained <b>{ds['rows_after_cleaning']:,}</b> records ({ds['fake_count']:,} Fake and {ds['real_count']:,} Real). The split contained {sp['train_rows']:,} training and {sp['test_rows']:,} held-out test records.")
        table_data = [["Model", "Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]]
        for name, m in metrics["models"].items():
            table_data.append([name, f"{m['accuracy']:.4f}", f"{m['precision_macro']:.4f}", f"{m['recall_macro']:.4f}", f"{m['f1_macro']:.4f}"])
        t = Table(table_data, colWidths=[54*mm, 24*mm, 29*mm, 27*mm, 23*mm], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0,0),(-1,0), NAVY), ("TEXTCOLOR", (0,0),(-1,0), colors.white), ("BACKGROUND", (0,1),(-1,-1), PALE), ("TEXTCOLOR", (0,1),(-1,-1), INK), ("GRID", (0,0),(-1,-1),0.4,colors.HexColor("#B8C9D4")), ("FONTNAME", (0,0),(-1,0),"Helvetica-Bold"), ("FONTSIZE",(0,0),(-1,-1),7.5), ("ALIGN",(1,1),(-1,-1),"CENTER"), ("VALIGN",(0,0),(-1,-1),"MIDDLE"), ("TOPPADDING",(0,0),(-1,-1),7), ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
        story.append(t); story.append(Spacer(1, 5*mm))
        body(f"The selected model was <b>{metrics['selected_model']}</b>, determined by macro-F1 and the stated tie-break rule. Confusion matrices use row order Actual Fake/Real and column order Predicted Fake/Real.")
        for img in [root / "outputs/visualizations/model_comparison.png", root / "outputs/visualizations/confusion_matrix_logistic_regression.png", root / "outputs/visualizations/confusion_matrix_multinomial_naive_bayes.png", root / "outputs/visualizations/top_logistic_terms.png"]:
            if img.exists(): story.append(Image(str(img), width=155*mm, height=84*mm, kind="proportional")); story.append(Spacer(1, 3*mm))
    else:
        story.append(Paragraph("Evaluation is pending in this generated report. No accuracy, precision, recall, F1-score or confusion-matrix values are fabricated. Run <b>python scripts/download_dataset.py</b>, then <b>python scripts/prepare_data.py</b> and <b>python -m src.train_model</b>. The training script will populate the metrics, charts and this PDF with actual held-out test results.", styles["Calloutx"]))
        body("The evaluation script calculates accuracy, macro and weighted precision, macro and weighted recall, macro and weighted F1-score, per-class classification reports and confusion matrices for both classifiers.")
    heading("10. Streamlit application")
    body("The Streamlit application provides a text input area for a headline or full article, validates blank input, applies the exact training-time cleaner, transforms text with the saved TF-IDF vectorizer and predicts with the selected saved model. Where available, the interface displays the model-estimated class probability. It also provides a model dashboard with the test metrics, model comparison and generated charts.")
    heading("11. Model and vectorizer persistence")
    body("The training process writes models/model.joblib, models/tfidf_vectorizer.joblib and supporting model files. It also saves metadata describing the split, selected classifier, label mapping and vectorizer configuration. The Streamlit app loads these files at runtime and does not retrain on user requests.")
    story.append(PageBreak())
    heading("12. Limitations and responsible interpretation")
    for item in [
        "Dataset labels are inherited from the source collection and are not an independently verified ground truth for every article.",
        "The corpus is English-language and concentrated in 2015-2017 political/world news; performance may be weaker for other regions, topics, languages and current events.",
        "The model learns correlations in text and source-dataset patterns. It cannot browse, verify citations, confirm publisher identity or establish factual truth.",
        "Satire, opinion, short headlines, breaking news, paraphrases and unfamiliar topics can produce incorrect predictions.",
        "A probability score is the classifier's estimate, not a guarantee of correctness or a calibrated measure of truth.",
    ]: body("&#8226; " + item)
    heading("13. Future enhancements")
    body("Future work may include time-based and cross-domain evaluation, multilingual datasets, source credibility signals, claim-level retrieval and evidence checking, probability calibration, explainability for individual predictions, drift monitoring and human review for high-impact decisions.")
    heading("14. Conclusion")
    body("This project demonstrates an end-to-end classical NLP workflow for binary fake/real news text classification, from data preparation and TF-IDF through comparative model evaluation and a saved-model Streamlit application. The output should be interpreted as a learned text-pattern estimate and not as a definitive fact-checking verdict.")
    heading("References")
    for ref in [
        "Bisaillon, C. Fake and Real News Dataset. Kaggle. https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset",
        "Gilchrist dataset mirror. Fake and Real News Dataset. Hugging Face Datasets. https://huggingface.co/datasets/haoxianc/gilchr_fake-and-real-news-dataset",
        "scikit-learn documentation: TfidfVectorizer, LogisticRegression, MultinomialNB and classification metrics. https://scikit-learn.org/stable/",
        "Streamlit documentation. https://docs.streamlit.io/",
        "Joblib documentation. https://joblib.readthedocs.io/",
    ]: body("&#8226; " + ref)

    def page(canvas, doc_):
        canvas.saveState()
        w, h = A4
        if doc_.page > 1:
            canvas.setStrokeColor(colors.HexColor("#D7E3EA")); canvas.line(19*mm, h-13*mm, w-19*mm, h-13*mm)
            canvas.setFont("Helvetica-Bold", 7); canvas.setFillColor(NAVY); canvas.drawString(19*mm, h-10*mm, "SMART FAKE NEWS DETECTION SYSTEM")
            canvas.setFont("Helvetica", 7); canvas.setFillColor(MUTED); canvas.drawRightString(w-19*mm, 9*mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=page, onLaterPages=page)
    return output
