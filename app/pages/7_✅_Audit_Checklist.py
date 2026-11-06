"""
Audit Checklist — Streamlit Page 7.

Downloadable 12-15 item ML-for-autism evaluation checklist
derived from the project's findings.

MEDICAL DISCLAIMER: Research prototype. Not a diagnostic device.
"""

import streamlit as st

st.set_page_config(page_title="Audit Checklist", page_icon="✅", layout="wide")
st.error("⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**", icon="⚠️")
st.title("✅ ML-for-Autism Evaluation Checklist")
st.markdown(
    "A systematic checklist for evaluating ML papers on autism screening/diagnosis. "
    "Derived from the methodological audit findings of this project."
)

CHECKLIST = [
    {
        "id": 1,
        "category": "Data",
        "item": "Subject-wise splits: Ensure no subject appears in both train and test sets.",
        "severity": "Critical",
        "finding": "Some published ABIDE analyses violate this when using session-wise splits.",
    },
    {
        "id": 2,
        "category": "Data",
        "item": "Leaking features removed: Verify that total score, sum columns, or directly label-derived features are excluded.",
        "severity": "Critical",
        "finding": "UCI ASD datasets: label = sum(A1..A10) ≥ τ. Including sum or score yields trivial ~100% accuracy.",
    },
    {
        "id": 3,
        "category": "Data",
        "item": "Duplicate analysis: Check for exact or near-duplicate rows that could inflate test performance.",
        "severity": "High",
        "finding": "UCI datasets contain 7-20% duplicate rows that boost performance if not removed before splitting.",
    },
    {
        "id": 4,
        "category": "Preprocessing",
        "item": "In-fold preprocessing: Scaling, imputation, feature selection fit ONLY on training folds.",
        "severity": "Critical",
        "finding": "Fitting scaler/imputer on full data before CV leaks test statistics into training.",
    },
    {
        "id": 5,
        "category": "Preprocessing",
        "item": "Harmonization inside folds: ComBat or site correction fitted on training sites only; test site transformed separately.",
        "severity": "Critical",
        "finding": "ComBat fit on all data before CV inflates site-decodability collapse to 0% (artifact).",
    },
    {
        "id": 6,
        "category": "Evaluation",
        "item": "Site-aware validation: Use LOSO or leave-site-group-out for multi-site neuroimaging datasets.",
        "severity": "Critical",
        "finding": "Random CV inflates LR by +3.3pp and BNT by +22.5pp due to scanner hardware leakage.",
    },
    {
        "id": 7,
        "category": "Evaluation",
        "item": "Multiple seeds: Report results across ≥5 random seeds to characterize variance.",
        "severity": "High",
        "finding": "Single-seed results have uncharacterized variance; can mislead model selection.",
    },
    {
        "id": 8,
        "category": "Evaluation",
        "item": "Significance testing: Use DeLong, McNemar, or paired bootstrap for AUROC differences; correct for multiple comparisons (Holm-Bonferroni).",
        "severity": "High",
        "finding": "Many published comparisons lack statistical tests; ABIDE LOSO results often not sig. different from chance.",
    },
    {
        "id": 9,
        "category": "Calibration",
        "item": "Calibration reported: Report ECE, Brier score, and reliability diagrams.",
        "severity": "Medium",
        "finding": "Raw LR on questionnaire data has ECE ~0.059; without calibration, confidence estimates are unreliable.",
    },
    {
        "id": 10,
        "category": "Uncertainty",
        "item": "Uncertainty quantification: Provide prediction intervals or conformal sets; flag uncertain cases for review.",
        "severity": "Medium",
        "finding": "Conformal prediction with α=0.10 flags ~4-6% of adult screening cases for expert review.",
    },
    {
        "id": 11,
        "category": "Confounds",
        "item": "Confound analysis: Check motion, age, sex, IQ correlation with diagnosis label; report partial correlations.",
        "severity": "High",
        "finding": "ABIDE: |r(motion,ASD)| = 0.14, |r(age,ASD)| = 0.18 — controlled but should be reported.",
    },
    {
        "id": 12,
        "category": "Generalization",
        "item": "External validation: Validate on a separate dataset (different sites, collection period, scanner).",
        "severity": "High",
        "finding": "ABIDE-II external validation attempted but requires institutional access; ABIDE-I 4-site → all 17 sites is a minimum.",
    },
    {
        "id": 13,
        "category": "Ethics",
        "item": "Clinical framing: Explicitly state the tool is a research prototype; avoid diagnostic claims.",
        "severity": "Critical",
        "finding": "Many apps and papers imply diagnostic capability from screening questionnaire results.",
    },
    {
        "id": 14,
        "category": "Reproducibility",
        "item": "Code and data: Provide code, configs, seeds, and data download instructions for full reproducibility.",
        "severity": "High",
        "finding": "Many published ABIDE papers cannot be reproduced due to undisclosed preprocessing choices.",
    },
    {
        "id": 15,
        "category": "Reporting",
        "item": "Report negative results: Document what did not work (p>>n regime, low-power settings) honestly.",
        "severity": "Medium",
        "finding": "BNT LOSO accuracy 45.8% (below chance) — a valid negative finding that characterizes the limits of deep learning on ABIDE.",
    },
]

import pandas as pd

df = pd.DataFrame(CHECKLIST)

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    sel_category = st.multiselect(
        "Filter by Category",
        sorted(df["category"].unique()),
        default=sorted(df["category"].unique()),
    )
with col2:
    sel_severity = st.multiselect(
        "Filter by Severity",
        ["Critical", "High", "Medium"],
        default=["Critical", "High", "Medium"],
    )

df_filtered = df[df["category"].isin(sel_category) & df["severity"].isin(sel_severity)]

# ── Display with severity color ───────────────────────────────────────────────
for _, row in df_filtered.iterrows():
    color_map = {"Critical": "🔴", "High": "🟠", "Medium": "🟡"}
    icon = color_map.get(row["severity"], "⚪")
    with st.expander(f"{icon} [{row['category']}] {row['item'][:60]}... | **{row['severity']}**"):
        st.markdown(f"**Checklist Item #{row['id']}:** {row['item']}")
        st.markdown(f"**Category:** {row['category']} | **Severity:** {row['severity']}")
        st.markdown(f"**Evidence from this audit:** {row['finding']}")

st.divider()

# ── Download ──────────────────────────────────────────────────────────────────
csv_buf = df[["id", "category", "severity", "item", "finding"]].to_csv(index=False)
st.download_button(
    "⬇ Download Checklist as CSV",
    data=csv_buf,
    file_name="ml_autism_audit_checklist.csv",
    mime="text/csv",
)

md_lines = ["# ML-for-Autism Evaluation Checklist\n",
            "Derived from: *A Methodological Audit and Reproducible Benchmark of Machine Learning for Autism Screening*\n",
            "\n| # | Category | Severity | Checklist Item | Evidence |\n",
            "|---|----------|----------|----------------|----------|\n"]
for _, row in df.iterrows():
    md_lines.append(f"| {row['id']} | {row['category']} | {row['severity']} | {row['item']} | {row['finding']} |\n")

st.download_button(
    "⬇ Download Checklist as Markdown",
    data="".join(md_lines),
    file_name="ml_autism_audit_checklist.md",
    mime="text/markdown",
)

st.divider()
st.caption("Research prototype. This checklist is for academic and educational use only.")
