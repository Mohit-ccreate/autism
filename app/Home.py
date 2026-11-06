"""
ASD Screening Research Platform — Streamlit Home Page.

MEDICAL DISCLAIMER: This is a research prototype, not a diagnostic device.
No clinical use. No patient data stored.

Run with: streamlit run app/Home.py
"""

import json
import os
import sys
from pathlib import Path

import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Screening Audit — Research Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Persistent banner ─────────────────────────────────────────────────────────
st.error(
    "⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**  "
    "This platform demonstrates ML methodology for academic purposes only. "
    "It does not diagnose Autism Spectrum Disorder and must not be used for clinical decisions.",
    icon="⚠️",
)

# ── Load paper facts ──────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_paper_facts():
    facts_path = RESULTS_DIR / "paper_facts.json"
    if facts_path.exists():
        with open(facts_path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_tautology_results():
    path = RESULTS_DIR / "tautology_audit.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_site_analysis():
    path = RESULTS_DIR / "site_analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_v4_audit():
    path = RESULTS_DIR / "v4_audit_pass1.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


facts = load_paper_facts()
findings = facts.get("findings", {})

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🧠 ASD Screening — Methodological Audit & Reproducible Benchmark")
st.markdown(
    """
    > **Paper:** *A Methodological Audit and Reproducible Benchmark of Machine Learning for Autism 
    Screening: Behavioral Tautologies, Scanner Generalization, and Uncertainty-Aware Decision Support.*

    This platform provides interactive exploration of the paper's findings. 
    Navigate using the sidebar to explore each module.
    """
)

# ── Headline findings ─────────────────────────────────────────────────────────
st.header("🔑 Headline Findings")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Questionnaire Tautology",
        value="100% concordance",
        delta="Label = threshold(sum) in all 4 datasets",
        delta_color="off",
    )
    st.caption(
        "The ASD label is a **deterministic step function** of item sums. "
        "Any ML model with the total score approaches 100% by arithmetic."
    )

with col2:
    abide_loso = findings.get("abide_loso_lr", {})
    loso_acc = abide_loso.get("accuracy", 0.55)
    st.metric(
        label="ABIDE LOSO Accuracy (Honest)",
        value=f"{loso_acc:.1%}",
        delta="vs 58.3% random-CV (leaky)",
        delta_color="inverse",
    )
    st.caption(
        "Leave-One-Site-Out accuracy with ComBat **inside folds**. "
        "Statistically indistinguishable from chance (binomial p=0.315)."
    )

with col3:
    binom_p = findings.get("abide_binomial_p", {}).get("value", 0.315)
    st.metric(
        label="LOSO vs Chance (Binomial test)",
        value=f"p = {binom_p:.3f}",
        delta="Cannot reject null hypothesis",
        delta_color="off",
    )
    st.caption(
        "ABIDE LOSO performance is statistically indistinguishable from chance "
        "under current statistical power (N=120, 4 sites)."
    )

st.divider()

# ── Dataset cards ──────────────────────────────────────────────────────────────
st.header("📊 Dataset Cards")

datasets = [
    {
        "name": "UCI Adult AQ-10",
        "n_raw": 704, "n_clean": 667, "asd_rate": "~32%",
        "source": "UCI ML Repository #426",
        "finding": "Label = sum(A1..A10) ≥ 7 (100% concordance)",
    },
    {
        "name": "UCI Adolescent AQ-10",
        "n_raw": 104, "n_clean": 97, "asd_rate": "~84%",
        "source": "UCI ML Repository #420",
        "finding": "Label = sum(A1..A10) ≥ 7 (100% concordance)",
    },
    {
        "name": "UCI Child AQ-10",
        "n_raw": 292, "n_clean": 262, "asd_rate": "~73%",
        "source": "UCI ML Repository #419",
        "finding": "Label = sum(A1..A10) ≥ 7 (100% concordance)",
    },
    {
        "name": "Toddler Q-CHAT-10",
        "n_raw": 1054, "n_clean": 850, "asd_rate": "~65%",
        "source": "Kaggle/Thabtah",
        "finding": "Label = sum(A1..A10) ≥ 4 (100% concordance)",
    },
    {
        "name": "ABIDE I CC200 (4-site)",
        "n_raw": "~871 (full)", "n_clean": 120, "asd_rate": "~50%",
        "source": "nilearn.datasets.fetch_abide_pcp",
        "finding": "LOSO AUROC = 0.61 (not significant, p=0.315)",
    },
]

cols = st.columns(len(datasets))
for col, ds in zip(cols, datasets):
    with col:
        st.markdown(f"**{ds['name']}**")
        st.markdown(f"- Raw N: {ds['n_raw']}")
        st.markdown(f"- Clean N: {ds['n_clean']}")
        st.markdown(f"- ASD rate: {ds['asd_rate']}")
        st.markdown(f"- Source: {ds['source']}")
        st.info(ds["finding"])

st.divider()

# ── Thesis summary ────────────────────────────────────────────────────────────
st.header("📋 Central Thesis (Test Results)")

st.markdown(
    """
    | Thesis Claim | Supported? | Evidence |
    |---|---|---|
    | (1) Behavioral features cause label leakage | ✅ **Yes** | Label is 100% function of item sum |
    | (2) Site/scanner confounds inflate neuroimaging results | ⚠️ **Partially** | Random CV inflates AUROC by ~0.03; full 17-site analysis pending |
    | (3) ML predictions are uncalibrated and abstention-free | ⚠️ **Yes (design gap)** | ECE > 0.05 without calibration; conformal adds abstention |
    | Honest evaluation yields trustworthy decision support | ✅ **Yes** | LOSO + ComBat + conformal + selective prediction implemented |
    """
)

# ── Navigation guide ──────────────────────────────────────────────────────────
st.header("🗺 Navigation Guide")
st.markdown(
    """
    Use the **sidebar** to navigate to each analysis module:
    
    | Page | Description |
    |------|-------------|
    | 🏠 **Home** (this page) | Overview, thesis, dataset cards |
    | 🔍 **Tautology Audit** | Rule recovery, feature ablation, SHAP, Tautology Index |
    | 🏆 **Benchmark Leaderboard** | Model × metric × protocol, sortable with CIs |
    | 🧲 **Scanner Generalization** | Random-CV vs LOSO, harmonization comparison |
    | 🎯 **Uncertainty Lab** | Calibration, reliability diagrams, conformal, risk-coverage |
    | 🖊 **Screening Simulator** | Demo without leaking features; conformal abstention |
    | 🔬 **Reproducibility** | Configs, seeds, pipeline diagram |
    | ✅ **Audit Checklist** | Downloadable ML-for-autism evaluation checklist |
    """
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "📚 **For research and educational purposes only.** "
    "This tool does not provide medical advice, diagnosis, or treatment recommendations. "
    "Always consult a qualified healthcare professional for clinical evaluation. "
    f"| Git: {facts.get('_metadata', {}).get('git_commit', 'N/A')}"
)
