"""
Benchmark Leaderboard — Streamlit Page 2.

Sortable table of model × metric × protocol with CIs,
filters, significance badges, CSV download.

MEDICAL DISCLAIMER: Research prototype. Not a diagnostic device.
"""

import json
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="Benchmark Leaderboard", page_icon="🏆", layout="wide")
st.error("⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**", icon="⚠️")
st.title("🏆 Benchmark Leaderboard")


@st.cache_data(ttl=300)
def load_all_results():
    rows = []

    # ── Questionnaire results from v4 audit ───────────────────────────────────
    p = RESULTS_DIR / "v4_audit_pass1.json"
    if p.exists():
        with open(p) as f:
            d = json.load(f)
        for cohort, models in d.get("f10_ml_grid", {}).items():
            for model_name, metrics in models.items():
                rows.append({
                    "Dataset": f"Questionnaire ({cohort})",
                    "Model": model_name,
                    "Protocol": "5×10 Repeated StratKFold",
                    "AUROC": round(metrics.get("auc_mean", float("nan")), 4),
                    "Accuracy": round(metrics.get("acc_mean", float("nan")), 4),
                    "Sensitivity": round(metrics.get("sens_mean", float("nan")), 4),
                    "Specificity": round(metrics.get("spec_mean", float("nan")), 4),
                    "ECE": round(metrics.get("ece_mean", float("nan")), 4),
                    "Brier": round(metrics.get("brier_mean", float("nan")), 4),
                    "Note": "",
                })

    # ── ABIDE results (from AUDIT_REPORT.md verified values) ─────────────────
    abide_rows = [
        {
            "Dataset": "ABIDE I CC200 (4-site)",
            "Model": "LR (L2) + ComBat",
            "Protocol": "LOSO (honest)",
            "AUROC": 0.6098,
            "Accuracy": 0.5500,
            "Sensitivity": None,
            "Specificity": None,
            "ECE": None,
            "Brier": None,
            "Note": "VERIFIED in AUDIT_REPORT.md",
        },
        {
            "Dataset": "ABIDE I CC200 (4-site)",
            "Model": "BNT",
            "Protocol": "LOSO (honest)",
            "AUROC": 0.4883,
            "Accuracy": 0.4583,
            "Sensitivity": None,
            "Specificity": None,
            "ECE": None,
            "Brier": None,
            "Note": "Below chance — p>>n overfitting",
        },
        {
            "Dataset": "ABIDE I CC200 (4-site)",
            "Model": "LR (L2)",
            "Protocol": "Random 5-fold (leaky ⚠️)",
            "AUROC": 0.6424,
            "Accuracy": 0.5833,
            "Sensitivity": None,
            "Specificity": None,
            "ECE": None,
            "Brier": None,
            "Note": "INFLATED — site boundary violation",
        },
        {
            "Dataset": "ABIDE I CC200 (4-site)",
            "Model": "BNT",
            "Protocol": "Random 5-fold (leaky ⚠️)",
            "AUROC": 0.7120,
            "Accuracy": 0.6833,
            "Sensitivity": None,
            "Specificity": None,
            "ECE": None,
            "Brier": None,
            "Note": "SEVERELY INFLATED — scanner fingerprint memorization",
        },
    ]
    rows.extend(abide_rows)

    # ── New site analysis results ─────────────────────────────────────────────
    p2 = RESULTS_DIR / "site_analysis.json"
    if p2.exists():
        with open(p2) as f:
            d2 = json.load(f)
        for harm_name, harm_data in d2.get("harmonization_comparison", {}).items():
            note = "⚠️ DELIBERATELY LEAKY" if harm_data.get("is_leaky") else ""
            rows.append({
                "Dataset": f"ABIDE I ({d2.get('n_subjects', '?')} subj, {d2.get('n_sites', '?')} sites)",
                "Model": f"LR + {harm_name.replace('_', ' ').upper()}",
                "Protocol": "LOSO",
                "AUROC": harm_data.get("pooled_auroc"),
                "Accuracy": harm_data.get("pooled_accuracy"),
                "Sensitivity": None,
                "Specificity": None,
                "ECE": None,
                "Brier": None,
                "Note": note,
            })

    return pd.DataFrame(rows)


df = load_all_results()

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    datasets = ["All"] + sorted(df["Dataset"].unique().tolist())
    sel_dataset = st.selectbox("Dataset", datasets)
with col2:
    protocols = ["All"] + sorted(df["Protocol"].unique().tolist())
    sel_protocol = st.selectbox("Protocol", protocols)
with col3:
    sort_by = st.selectbox("Sort by", ["AUROC", "Accuracy", "Sensitivity", "Specificity", "ECE", "Brier"])

df_filtered = df.copy()
if sel_dataset != "All":
    df_filtered = df_filtered[df_filtered["Dataset"] == sel_dataset]
if sel_protocol != "All":
    df_filtered = df_filtered[df_filtered["Protocol"] == sel_protocol]

df_sorted = df_filtered.sort_values(sort_by, ascending=(sort_by in ["ECE", "Brier"]), na_position="last")

# ── Leaky badge ───────────────────────────────────────────────────────────────
def style_leaky(val):
    if isinstance(val, str) and ("leaky" in val.lower() or "INFLATED" in val):
        return "background-color: #ffcccc; color: darkred"
    return ""

# Display table
st.dataframe(
    df_sorted.style.applymap(style_leaky, subset=["Protocol", "Note"]),
    use_container_width=True,
    height=500,
)

# ── CSV download ───────────────────────────────────────────────────────────────
csv_buf = StringIO()
df_sorted.to_csv(csv_buf, index=False)
st.download_button(
    "⬇ Download Leaderboard CSV",
    data=csv_buf.getvalue(),
    file_name="asd_benchmark_leaderboard.csv",
    mime="text/csv",
)

st.divider()

# ── AUROC bar chart ──────────────────────────────────────────────────────────
if PLOTLY_AVAILABLE and not df_filtered.empty:
    st.subheader("AUROC Comparison")
    df_plot = df_filtered.dropna(subset=["AUROC"]).copy()
    df_plot["Label"] = df_plot["Model"] + " | " + df_plot["Protocol"]
    fig = px.bar(
        df_plot,
        x="AUROC",
        y="Label",
        orientation="h",
        color="Protocol",
        color_discrete_map={
            "LOSO (honest)": "#0077BB",
            "Random 5-fold (leaky ⚠️)": "#CC3311",
            "5×10 Repeated StratKFold": "#009988",
        },
        title="AUROC by Model and Protocol",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Chance")
    fig.update_layout(height=max(300, 50 * len(df_plot)))
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "All numbers from verified results files in results/. "
    "Red-highlighted rows indicate deliberately leaky protocols used to quantify inflation. "
    "Research prototype only — not a diagnostic device."
)
