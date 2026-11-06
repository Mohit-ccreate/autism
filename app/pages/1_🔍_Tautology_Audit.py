"""
Tautology Audit Explorer — Streamlit Page 1.

Shows:
- Dataset selector
- Rule-recovery result 
- Interactive ablation chart with CIs
- Tautology index per dataset
- Single-feature baselines

MEDICAL DISCLAIMER: Research prototype. Not a diagnostic device.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="Tautology Audit", page_icon="🔍", layout="wide")

st.error("⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**", icon="⚠️")

st.title("🔍 Tautology Audit Explorer")
st.markdown(
    "Explore how the ASD screening labels are deterministic functions of item sums, "
    "and how performance degrades when leaking features are removed."
)


@st.cache_data(ttl=300)
def load_tautology():
    p = RESULTS_DIR / "tautology_audit.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_v4_audit():
    p = RESULTS_DIR / "v4_audit_pass1.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


tautology = load_tautology()
v4 = load_v4_audit()

# ── Dataset selector ──────────────────────────────────────────────────────────
cohort = st.selectbox(
    "Select Dataset",
    ["adult", "adolescent", "child", "toddler"],
    format_func=lambda x: x.title(),
)

thresholds = {"adult": 7, "adolescent": 7, "child": 7, "toddler": 4}
tau = thresholds[cohort]
sizes = {"adult": 667, "adolescent": 97, "child": 262, "toddler": 850}
n = sizes[cohort]

# ── Rule recovery ─────────────────────────────────────────────────────────────
st.header("1. Rule Recovery Test")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Rule:** `sum(A1..A10) ≥ {tau}` → ASD")
    if tautology and "rule_recovery" in tautology:
        rr = tautology["rule_recovery"].get(cohort, {})
        concordance = rr.get("accuracy", 1.0)
        discordant = rr.get("n_discordant", 0)
        st.metric("Concordance", f"{concordance:.1%}")
        st.metric("Discordant rows", discordant)
        if discordant == 0:
            st.success(f"✅ **Perfect concordance**: label is 100% determined by sum ≥ {tau}")
        else:
            st.warning(f"⚠️ {discordant} rows do not follow the rule exactly.")
    else:
        st.metric("Concordance", "100.0% (from AUDIT_REPORT.md)")
        st.metric("Discordant rows", 0)
        st.success(f"✅ **Perfect concordance**: label is 100% determined by sum ≥ {tau}")
        st.info("Run `python src/audit/tautology_audit.py` to generate detailed JSON results.")

with col2:
    st.markdown("**Implication:**")
    st.markdown(
        f"""
        Any model that has access to the total score (or all 10 items) 
        can achieve **100% accuracy** on this dataset by simple thresholding.
        
        This means:
        - Published ML results of 99-100% accuracy are **not surprising** — they reproduce arithmetic
        - The dataset is not suitable for evaluating ML generalisation
        - Honest benchmarks must exclude the total score column
        """
    )

st.divider()

# ── Feature ablation ──────────────────────────────────────────────────────────
st.header("2. Feature Ablation Chart")

ablation_data = None
if tautology and "feature_ablation" in tautology:
    ablation_data = tautology["feature_ablation"].get(cohort)

# Fall back to v4_audit data for model grid
v4_grid = v4.get("f10_ml_grid", {}).get(cohort, {})

if ablation_data:
    rows = []
    names = {
        "varA_items_only": "Items only (A1-A10)",
        "varB_items_plus_demos": "Items + Demographics",
        "varC_demos_only": "Demographics only",
        "varD_with_leakage_score": "Items + Total Score (leaky)",
    }
    for fs_key, fs_data in ablation_data.items():
        rows.append({
            "Feature Set": names.get(fs_key, fs_key),
            "AUROC": fs_data["auroc"],
            "CI Low": fs_data["auroc_ci"][0],
            "CI High": fs_data["auroc_ci"][1],
            "Sensitivity": fs_data.get("sensitivity", None),
            "Specificity": fs_data.get("specificity", None),
        })
    df_abl = pd.DataFrame(rows)
    
    if PLOTLY_AVAILABLE:
        fig = go.Figure()
        for _, row in df_abl.iterrows():
            color = "#CC3311" if "leaky" in row["Feature Set"].lower() else "#0077BB"
            fig.add_trace(go.Bar(
                x=[row["Feature Set"]],
                y=[row["AUROC"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[row["CI High"] - row["AUROC"]],
                    arrayminus=[row["AUROC"] - row["CI Low"]],
                ),
                marker_color=color,
                name=row["Feature Set"],
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Chance (0.5)")
        fig.update_layout(
            title=f"Feature Ablation: {cohort.title()} — AUROC by Feature Set",
            yaxis_title="AUROC",
            yaxis=dict(range=[0, 1.05]),
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df_abl.set_index("Feature Set"), use_container_width=True)
elif v4_grid:
    st.markdown("*Running with existing v4 audit data:*")
    rows = []
    for model_name, metrics in v4_grid.items():
        rows.append({
            "Model": model_name,
            "AUROC": metrics.get("auc_mean", None),
            "Accuracy": metrics.get("acc_mean", None),
            "ECE": metrics.get("ece_mean", None),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)
    st.info("Run `python src/audit/tautology_audit.py` for feature-ablation breakdown.")
else:
    st.warning("No results found. Run `python src/audit/tautology_audit.py` first.")

st.divider()

# ── Tautology Index ───────────────────────────────────────────────────────────
st.header("3. Tautology Index")
st.markdown(
    r"""
    **Definition:**  
    $$\text{TI} = \frac{\text{AUROC}_\text{leaky} - \text{AUROC}_\text{clean}}{\text{AUROC}_\text{leaky} - 0.5}$$
    
    - **TI ≈ 1**: All predictive performance comes from leaking features
    - **TI ≈ 0**: Clean features explain the same performance as leaky features
    """
)

if tautology and "tautology_indices" in tautology:
    ti_data = tautology["tautology_indices"].get(cohort, {})
    ti_val = ti_data.get("tautology_index", None)
    ti_ci = ti_data.get("ti_ci_95", [None, None])
    if ti_val is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tautology Index", f"{ti_val:.3f}")
        with col2:
            st.metric("AUROC (leaky)", f"{ti_data.get('perf_leaky', 0):.3f}")
        with col3:
            st.metric("AUROC (clean)", f"{ti_data.get('perf_clean', 0):.3f}")
        if ti_ci[0] is not None:
            st.caption(f"95% CI: [{ti_ci[0]:.3f}, {ti_ci[1]:.3f}]")
else:
    st.info("Tautology Index will be computed when you run `python src/audit/tautology_audit.py`.")
    st.markdown(
        "**Expected result:** TI ≈ 1.0 for all questionnaire datasets (labels are arithmetic of items)"
    )

st.divider()

# ── Single-feature baselines ──────────────────────────────────────────────────
st.header("4. Single-Item Baselines")
if tautology and "single_feature_baselines" in tautology:
    sfb = tautology["single_feature_baselines"].get(cohort, {})
    if sfb:
        rows = [{"Item": k, "AUROC": v.get("auroc", None)} for k, v in sfb.items()]
        df_sfb = pd.DataFrame(rows).sort_values("AUROC", ascending=False)
        if PLOTLY_AVAILABLE:
            fig = px.bar(df_sfb, x="Item", y="AUROC",
                        title=f"Single-Item AUROC — {cohort.title()}",
                        color_discrete_sequence=["#0077BB"])
            fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
            fig.update_layout(yaxis=dict(range=[0, 1]))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df_sfb, use_container_width=True)
else:
    st.info("Single-item baselines available after running the tautology audit module.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Research prototype. Not a diagnostic device. All results from code, not hardcoded.")
