"""
Benchmark Leaderboard — Streamlit Page 2.

Comprehensive sortable table of model × metric × protocol across
Leak-Free Tabular Cohorts and Full ABIDE I Neuroimaging.
Reads exclusively from results/benchmark_leaderboard.json and results/site_analysis.json.

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
st.markdown(
    "Rigorous comparative evaluation across model families. "
    "All results are loaded dynamically from reproducible experiment ledgers in `results/`."
)


@st.cache_data(ttl=300)
def load_all_results():
    rows = []

    # 1. Load Tabular and ABIDE from benchmark_leaderboard.json
    bm_path = RESULTS_DIR / "benchmark_leaderboard.json"
    if bm_path.exists():
        with open(bm_path, "r", encoding="utf-8") as f:
            bm_data = json.load(f)

        # Tabular leak-free models
        tab_data = bm_data.get("tabular_leak_free", {})
        for cohort, cohort_info in tab_data.items():
            n = cohort_info.get("n_samples", "?")
            for model_name, metrics in cohort_info.get("models", {}).items():
                rows.append({
                    "Dataset": f"Tabular {cohort.capitalize()} (N={n}, Demographics Only)",
                    "Model": model_name,
                    "Protocol": "5-Fold CV (Leak-Free)",
                    "AUROC": round(metrics.get("auroc", float("nan")), 4),
                    "Accuracy": round(metrics.get("accuracy", float("nan")), 4),
                    "Balanced Accuracy": round(metrics.get("balanced_accuracy", float("nan")), 4),
                    "Sensitivity": round(metrics.get("sensitivity", float("nan")), 4),
                    "Specificity": round(metrics.get("specificity", float("nan")), 4),
                    "Brier": round(metrics.get("brier_score", float("nan")), 4),
                    "Note": "Leak-free demographics/history only",
                })

        # ABIDE LOSO models
        abide_data = bm_data.get("abide_full", {})
        n_sub = abide_data.get("n_subjects", 854)
        n_sites = abide_data.get("n_sites", 20)
        loso_models = abide_data.get("loso_benchmark", {})
        for model_name, metrics in loso_models.items():
            rows.append({
                "Dataset": f"ABIDE I Full (N={n_sub}, {n_sites} sites)",
                "Model": model_name,
                "Protocol": "LOSO (Honest)",
                "AUROC": round(metrics.get("auroc", float("nan")), 4),
                "Accuracy": round(metrics.get("accuracy", float("nan")), 4),
                "Balanced Accuracy": round(metrics.get("balanced_accuracy", float("nan")), 4),
                "Sensitivity": round(metrics.get("sensitivity", float("nan")), 4),
                "Specificity": round(metrics.get("specificity", float("nan")), 4),
                "Brier": round(metrics.get("brier_score", float("nan")), 4),
                "Note": "In-fold ComBat harmonization across 20 sites",
            })

        # ABIDE Random CV models (site-leaking)
        rand_models = abide_data.get("random_cv_benchmark", {})
        for model_name, metrics in rand_models.items():
            rows.append({
                "Dataset": f"ABIDE I Full (N={n_sub}, {n_sites} sites)",
                "Model": model_name,
                "Protocol": "Random 5-fold (Site-Leaking ⚠️)",
                "AUROC": round(metrics.get("auroc", float("nan")), 4),
                "Accuracy": round(metrics.get("accuracy", float("nan")), 4),
                "Balanced Accuracy": round(metrics.get("balanced_accuracy", float("nan")), 4),
                "Sensitivity": round(metrics.get("sensitivity", float("nan")), 4),
                "Specificity": round(metrics.get("specificity", float("nan")), 4),
                "Brier": round(metrics.get("brier_score", float("nan")), 4),
                "Note": "⚠️ Cross-site scanner leakage present",
            })

    # 2. Load Harmonization arms from site_analysis.json
    sa_path = RESULTS_DIR / "site_analysis.json"
    if sa_path.exists():
        with open(sa_path, "r", encoding="utf-8") as f:
            sa_data = json.load(f)
        n_sub = sa_data.get("n_subjects", 854)
        n_sites = sa_data.get("n_sites", 20)
        for harm_name, harm_info in sa_data.get("harmonization_comparison", {}).items():
            is_leaky = harm_info.get("is_leaky", False)
            rows.append({
                "Dataset": f"ABIDE I Full (N={n_sub}, {n_sites} sites)",
                "Model": f"LR + {harm_name.replace('_', ' ').upper()}",
                "Protocol": "LOSO (Leaky ⚠️)" if is_leaky else "LOSO (Honest)",
                "AUROC": round(harm_info.get("pooled_auroc", float("nan")), 4),
                "Accuracy": round(harm_info.get("pooled_accuracy", float("nan")), 4),
                "Balanced Accuracy": round(harm_info.get("pooled_bacc", float("nan")), 4),
                "Sensitivity": round(harm_info.get("pooled_sensitivity", float("nan")), 4),
                "Specificity": round(harm_info.get("pooled_specificity", float("nan")), 4),
                "Brier": float("nan"),
                "Note": "⚠️ ComBat fit before CV (deliberately leaky contrast)" if is_leaky else "Harmonization comparison arm",
            })

    return pd.DataFrame(rows)


df = load_all_results()

if df.empty:
    st.warning("No benchmark results found. Run python run_all.py to generate benchmarks.")
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Model Evaluations", len(df))
c2.metric("Datasets / Cohorts", df["Dataset"].nunique())
c3.metric("Protocols Evaluated", df["Protocol"].nunique())
c4.metric("Clinical Sites (ABIDE)", 20)

st.divider()

# ── Filters ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    datasets = ["All"] + sorted(df["Dataset"].unique().tolist())
    sel_dataset = st.selectbox("Dataset Filter", datasets)
with col2:
    protocols = ["All"] + sorted(df["Protocol"].unique().tolist())
    sel_protocol = st.selectbox("Evaluation Protocol", protocols)
with col3:
    sort_by = st.selectbox("Sort by Metric", ["AUROC", "Accuracy", "Balanced Accuracy", "Sensitivity", "Specificity", "Brier"])

df_filtered = df.copy()
if sel_dataset != "All":
    df_filtered = df_filtered[df_filtered["Dataset"] == sel_dataset]
if sel_protocol != "All":
    df_filtered = df_filtered[df_filtered["Protocol"] == sel_protocol]

df_sorted = df_filtered.sort_values(sort_by, ascending=(sort_by in ["Brier"]), na_position="last")


def style_leaky(val):
    if isinstance(val, str) and ("leaky" in val.lower() or "⚠️" in val):
        return "background-color: #ffcccc; color: darkred; font-weight: bold"
    return ""


# Display table
st.dataframe(
    df_sorted.style.applymap(style_leaky, subset=["Protocol", "Note"]),
    use_container_width=True,
    height=550,
)

# ── CSV download ───────────────────────────────────────────────────────────────
csv_buf = StringIO()
df_sorted.to_csv(csv_buf, index=False)
st.download_button(
    "⬇ Download Filtered Leaderboard (CSV)",
    data=csv_buf.getvalue(),
    file_name="asd_benchmark_leaderboard.csv",
    mime="text/csv",
)

st.divider()

# ── Visualization ─────────────────────────────────────────────────────────────
if PLOTLY_AVAILABLE and not df_filtered.empty:
    st.subheader("📊 Performance Distribution by Protocol")
    df_plot = df_filtered.dropna(subset=["AUROC"]).copy()
    df_plot["Label"] = df_plot["Model"] + " (" + df_plot["Dataset"].str.split(" ").str[0] + ")"
    fig = px.bar(
        df_plot,
        x="AUROC",
        y="Label",
        orientation="h",
        color="Protocol",
        color_discrete_map={
            "LOSO (Honest)": "#0077BB",
            "Random 5-fold (Site-Leaking ⚠️)": "#CC3311",
            "5-Fold CV (Leak-Free)": "#009988",
            "LOSO (Leaky ⚠️)": "#EE7733",
        },
        title="AUROC by Model and Evaluation Protocol",
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Chance Level (0.50)")
    fig.update_layout(height=max(350, 40 * len(df_plot)), xaxis_range=[0.35, 0.85])
    st.plotly_chart(fig, use_container_width=True)

st.caption("All metrics traceable to results/benchmark_leaderboard.json and results/site_analysis.json.")
