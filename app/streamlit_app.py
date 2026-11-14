"""Streamlit front-end for the Smart Fake News Detection System."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Streamlit executes this file from app/; add the repository root for src imports.
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import clean_text

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.joblib"
METRICS_PATH = ROOT / "outputs" / "evaluation_results" / "model_metrics.json"
VIZ_DIR = ROOT / "outputs" / "visualizations"

st.set_page_config(page_title="NewsGuard | Smart Fake News Detection", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ng-bg:#0A1324; --ng-card:#111F35; --ng-line:#263B57; --ng-teal:#48D6C8; --ng-text:#EAF2FA; --ng-muted:#9CB0C7; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(circle at 85% 2%, rgba(72,214,200,.10), transparent 27%), linear-gradient(150deg,#081321 0%,#0B1830 55%,#0A1428 100%); color:var(--ng-text); }
.block-container { padding-top:2rem; padding-bottom:3rem; max-width:1420px; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#101E34 0%,#0B172A 100%); border-right:1px solid #263B57; }
[data-testid="stMetric"] { background:#111F35; border:1px solid #263B57; border-radius:16px; padding:16px 18px; }
[data-testid="stMetricLabel"] { color:#9CB0C7 !important; }
[data-testid="stMetricValue"] { color:#EAF2FA !important; font-family:'Manrope',sans-serif; }
.stTextArea textarea { background:#0D1A2E !important; color:#EAF2FA !important; border:1px solid #36516F !important; border-radius:14px !important; }
.stButton>button, .stDownloadButton>button { border-radius:12px; border:1px solid #48D6C8; background:linear-gradient(135deg,#48D6C8,#2BAEA7); color:#071B2B; font-weight:800; min-height:2.7rem; }
.stButton>button:hover { border-color:#7CE9DF; color:#061626; filter:brightness(1.08); }
.ng-hero { border:1px solid #29435F; border-radius:24px; padding:28px 30px; background:linear-gradient(115deg,rgba(23,49,75,.96),rgba(15,34,57,.82) 58%,rgba(24,83,86,.46)); box-shadow:0 18px 55px rgba(0,0,0,.22); margin-bottom:22px; }
.ng-kicker { color:#48D6C8; font-weight:800; letter-spacing:.16em; font-size:.72rem; text-transform:uppercase; }
.ng-title { color:#F2F7FB; font-family:'Manrope',sans-serif; font-size:2.15rem; font-weight:800; line-height:1.12; margin:.4rem 0 .5rem 0; }
.ng-sub { color:#B8C9D9; max-width:760px; font-size:1rem; line-height:1.6; }
.ng-panel { background:#111F35; border:1px solid #263B57; border-radius:18px; padding:20px 22px; }
.ng-result { border-radius:18px; padding:22px; border:1px solid #34516B; background:#10223A; }
.ng-muted { color:#9CB0C7; font-size:.88rem; }
.ng-disclaimer { border-left:3px solid #E8B45A; background:rgba(232,180,90,.08); padding:12px 15px; border-radius:0 12px 12px 0; color:#D9E1EA; font-size:.87rem; }
hr { border-color:#263B57 !important; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)

@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))

model, vectorizer = load_artifacts()
metrics = load_metrics()

with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-weight:800;font-size:1.3rem;color:#EAF2FA'>🛡️ NewsGuard</div><div class='ng-muted'>SMART NEWS ANALYTICS</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Workspace", ["News detector", "Model dashboard", "Methodology & limits"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Model status**")
    if model is None or vectorizer is None:
        st.warning("Training required")
    else:
        st.success("Saved model loaded")
    if metrics:
        st.caption(f"Selected: {metrics.get('selected_model', 'N/A')}")
    st.markdown("<div class='ng-muted'>Educational NLP classifier<br/>Not a fact-checking authority.</div>", unsafe_allow_html=True)

st.markdown("""
<div class='ng-hero'>
  <div class='ng-kicker'>NLP • TF-IDF • MACHINE LEARNING</div>
  <div class='ng-title'>Read beyond the headline.</div>
  <div class='ng-sub'>A text-pattern screening tool that compares a news story with labeled examples and estimates whether it resembles the Fake or Real class learned during training.</div>
</div>
""", unsafe_allow_html=True)

if page == "News detector":
    left, right = st.columns([1.55, 0.85], gap="large")
    with left:
        st.subheader("Analyze a news story")
        st.markdown("<div class='ng-muted'>Paste a complete headline or article for a more informative text signal. Empty input is rejected.</div>", unsafe_allow_html=True)
        if "news_input" not in st.session_state:
            st.session_state.news_input = ""
        with st.form("detector_form"):
            text_input = st.text_area("News text", key="news_input", height=240, placeholder="Paste a headline or article here...", label_visibility="collapsed", max_chars=20000)
            submitted = st.form_submit_button("Analyze news", use_container_width=True)
        if submitted:
            if not text_input or not text_input.strip():
                st.error("Please enter a headline or article before running the analysis.")
            elif len(text_input.strip()) < 20:
                st.warning("This input is very short. A full headline or article may provide more useful context.")
            elif model is None or vectorizer is None:
                st.error("The saved model is not available yet. Download the dataset and run the training steps in the README.")
            else:
                cleaned = clean_text(text_input)
                if not cleaned:
                    st.error("The text did not contain usable words after preprocessing. Please try a different input.")
                else:
                    try:
                        features = vectorizer.transform([cleaned])
                        prediction = int(model.predict(features)[0])
                        label = "Fake" if prediction == 0 else "Real"
                        probs = model.predict_proba(features)[0] if hasattr(model, "predict_proba") else None
                        probability = float(probs[prediction]) if probs is not None else None
                        st.session_state.last_result = {"label": label, "probability": probability, "cleaned_length": len(cleaned.split())}
                    except Exception as exc:
                        st.error(f"The prediction could not be completed: {exc}")
        result = st.session_state.get("last_result")
        if result:
            st.markdown("---")
            if result["label"] == "Fake":
                title, color, icon = "Likely Fake pattern", "#F47C7C", "⚠️"
                desc = "The text resembles patterns associated with the Fake class in the training dataset."
            else:
                title, color, icon = "Likely Real pattern", "#48D6C8", "✓"
                desc = "The text resembles patterns associated with the Real class in the training dataset."
            st.markdown(f"<div class='ng-result'><div style='font-size:.78rem;letter-spacing:.14em;color:#9CB0C7;font-weight:800'>MODEL PREDICTION</div><div style='font-family:Manrope;font-size:1.75rem;font-weight:800;color:{color};margin:7px 0'>{icon} {title}</div><div style='color:#C5D4E1'>{desc}</div></div>", unsafe_allow_html=True)
            if result["probability"] is not None:
                st.metric("Model-estimated confidence", f"{result['probability']*100:.1f}%")
                st.progress(max(0.0, min(1.0, result["probability"])))
            st.caption(f"Analyzed {result['cleaned_length']:,} cleaned tokens. Confidence is a model score, not a probability that the story is factually true.")
    with right:
        st.markdown("<div class='ng-panel'><div class='ng-kicker'>HOW IT WORKS</div><h3 style='margin:.3rem 0 1rem 0'>From words to signal</h3><p>01 · Clean text consistently</p><p>02 · Convert words into TF-IDF features</p><p>03 · Predict with the selected classifier</p><p>04 · Review the output with its limitations</p><hr><div class='ng-muted'>For best results, provide the complete article rather than a short or ambiguous phrase.</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='ng-disclaimer' style='margin-top:16px'><b>Important</b><br/>This tool does not search the web, verify citations, check publisher credentials or confirm claims against evidence.</div>", unsafe_allow_html=True)

elif page == "Model dashboard":
    st.subheader("Model performance dashboard")
    if not metrics:
        st.info("Evaluation results will appear here after you run `python -m src.train_model` with the source dataset.")
    else:
        ds = metrics.get("dataset", {}); split = metrics.get("split", {})
        selected = metrics.get("selected_model", "Not selected")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Articles", f"{ds.get('rows_after_cleaning',0):,}")
        c2.metric("Fake records", f"{ds.get('fake_count',0):,}")
        c3.metric("Real records", f"{ds.get('real_count',0):,}")
        c4.metric("Selected model", selected)
        st.markdown("### Classifier comparison")
        records = []
        for name, row in metrics.get("models", {}).items():
            records.append({"Model": name, "Accuracy": row["accuracy"], "Precision (macro)": row["precision_macro"], "Recall (macro)": row["recall_macro"], "F1 (macro)": row["f1_macro"]})
        comparison = pd.DataFrame(records).set_index("Model")
        st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
        chart = VIZ_DIR / "model_comparison.png"
        if chart.exists(): st.image(str(chart), use_container_width=True)
        col1,col2 = st.columns(2)
        for col, name in zip([col1,col2], ["logistic_regression", "multinomial_naive_bayes"]):
            path = VIZ_DIR / f"confusion_matrix_{name}.png"
            if path.exists(): col.image(str(path), use_container_width=True)
        st.markdown("### Dataset exploration")
        a,b = st.columns(2)
        for col, file in zip([a,b], ["label_distribution.png", "article_length_distribution.png"]):
            path = VIZ_DIR / file
            if path.exists(): col.image(str(path), use_container_width=True)
        top_terms = VIZ_DIR / "top_logistic_terms.png"
        if top_terms.exists(): st.image(str(top_terms), use_container_width=True)
        st.caption(f"Held-out split: {split.get('train_rows',0):,} train / {split.get('test_rows',0):,} test. Random seed: {split.get('random_state',42)}.")

else:
    st.subheader("Methodology & responsible use")
    st.markdown("""
    <div class='ng-panel'>
    <h3>Training workflow</h3>
    <ol>
      <li>Load the labeled Fake.csv and True.csv files.</li>
      <li>Handle missing text, normalize text, remove empty rows and duplicate articles.</li>
      <li>Split the labeled data into stratified training and held-out test sets.</li>
      <li>Fit TF-IDF on training text only; transform test and user input with that fitted vectorizer.</li>
      <li>Train Logistic Regression and Multinomial Naive Bayes on the same feature matrix.</li>
      <li>Compare accuracy, macro precision, macro recall, macro F1 and confusion matrices.</li>
      <li>Select the classifier by macro F1 (accuracy breaks ties), then save it with Joblib.</li>
    </ol>
    <h3>What the score means</h3>
    <p>The model recognizes statistical text patterns in the training corpus. It does not establish truth. The label can be affected by dataset bias, topic, publisher style, time period, language, satire and missing context.</p>
    <h3>Limitations</h3>
    <ul>
      <li>The source corpus is largely English-language political/world news from 2015-2017.</li>
      <li>Labels are inherited from the dataset and may not represent perfect fact-checking ground truth.</li>
      <li>Breaking stories and unfamiliar topics may differ from training examples.</li>
      <li>Confidence is a model-estimated class score, not a guarantee of factual correctness.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div class='ng-muted' style='text-align:center'>NEWSGUARD · SMART FAKE NEWS DETECTION · ACADEMIC DEMONSTRATION ONLY</div>", unsafe_allow_html=True)
