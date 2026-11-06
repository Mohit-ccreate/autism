"""
Uncertainty Lab — Streamlit Page 4.

Shows:
- Reliability diagrams before/after calibration
- Per-site conformal coverage vs nominal
- Risk-coverage curves with coverage slider
- Abstention trade-off table

MEDICAL DISCLAIMER: Research prototype. Not a diagnostic device.
"""

import json
from pathlib import Path

import numpy as np
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
FIGURES_DIR = ROOT / "figures"

st.set_page_config(page_title="Uncertainty Lab", page_icon="🎯", layout="wide")
st.error("⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**", icon="⚠️")
st.title("🎯 Uncertainty Lab")
st.markdown("Calibration, conformal prediction, and selective prediction analysis.")


@st.cache_data(ttl=300)
def load_uncertainty():
    p = RESULTS_DIR / "uncertainty_main.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    # Try demo file
    p2 = RESULTS_DIR / "uncertainty_demo.json"
    if p2.exists():
        with open(p2) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_site_analysis():
    p = RESULTS_DIR / "site_analysis.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


unc = load_uncertainty()
site = load_site_analysis()

# ── Tab layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Reliability Diagrams", "📊 Conformal Coverage", "🎯 Risk-Coverage", "📋 Abstention Trade-off"]
)

# ── Tab 1: Reliability diagrams ───────────────────────────────────────────────
with tab1:
    st.header("Calibration: Reliability Diagrams")
    st.markdown(
        """
        A well-calibrated model's confidence should match empirical accuracy.
        The red band shows miscalibration (confidence ≠ accuracy).
        """
    )
    
    # Try to load saved figure
    rel_fig_path = FIGURES_DIR / "reliability_diagram_questionnaire_adult.png"
    if rel_fig_path.exists():
        st.image(str(rel_fig_path), caption="Reliability diagram: before and after temperature scaling")
    elif rel_fig_path.with_suffix(".pdf").exists():
        st.info("PDF figure available at: " + str(rel_fig_path.with_suffix(".pdf")))
    
    # Show calibration metrics
    if unc and "calibration" in unc:
        calib = unc["calibration"]
        st.subheader("Calibration Metrics")
        rows = []
        for method_name, method_data in calib.items():
            if isinstance(method_data, dict) and "metrics" in method_data:
                m = method_data["metrics"]
                rows.append({
                    "Method": method_name.replace("_", " ").title(),
                    "ECE (15-bin)": round(m.get("ece_15bin", float("nan")), 4),
                    "ECE (adaptive)": round(m.get("ece_adaptive", float("nan")), 4),
                    "Brier": round(m.get("brier", float("nan")), 4),
                    "NLL": round(m.get("nll", float("nan")), 4),
                })
            elif isinstance(method_data, dict) and "ece_15bin" in method_data:
                rows.append({
                    "Method": method_data.get("name", method_name).replace("_", " ").title(),
                    "ECE (15-bin)": round(method_data.get("ece_15bin", float("nan")), 4),
                    "ECE (adaptive)": round(method_data.get("ece_adaptive", float("nan")), 4),
                    "Brier": round(method_data.get("brier", float("nan")), 4),
                    "NLL": round(method_data.get("nll", float("nan")), 4),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Method"), use_container_width=True)
    else:
        st.info("Run `python src/uncertainty/calibration.py` for calibration results.")
        st.markdown(
            """
            **From existing forensic_evaluation.py results:**
            - Adult questionnaire ECE (10-bin, Var A): ~0.059
            - Adult questionnaire Brier (Var A): ~0.015
            - Conformal coverage (100 splits): 89.8% ± 3.6%
            """
        )

# ── Tab 2: Conformal coverage ─────────────────────────────────────────────────
with tab2:
    st.header("Conformal Prediction Coverage")
    
    alpha_sel = st.radio("Target alpha (miscoverage rate)", [0.05, 0.10], horizontal=True)
    
    # From uncertainty results
    conf_key = f"conformal_lac_alpha{int(alpha_sel*100)}"
    conf_data = unc.get(conf_key, {})
    
    if conf_data:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Marginal Coverage",
                      f"{conf_data.get('marginal_coverage', float('nan')):.1%}",
                      delta=f"Target: {1-alpha_sel:.0%}")
        with col2:
            st.metric("Mean Set Size", f"{conf_data.get('mean_set_size', float('nan')):.3f}")
        with col3:
            st.metric("% Empty sets (refer)", 
                      f"{conf_data.get('fraction_empty', float('nan')):.1%}")
        with col4:
            st.metric("% Singleton sets (decided)",
                      f"{conf_data.get('fraction_singleton', float('nan')):.1%}")
        
        # Class-conditional coverage
        st.subheader("Class-Conditional Coverage (Mondrian)")
        mondrian = unc.get("mondrian_conformal_alpha10", {})
        if mondrian:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ASD (class 1) coverage", 
                          f"{mondrian.get('class_1_coverage', 'N/A')}")
            with col2:
                st.metric("Control (class 0) coverage",
                          f"{mondrian.get('class_0_coverage', 'N/A')}")
    else:
        st.info("Run uncertainty analysis to see conformal coverage results.")
        st.markdown(
            """
            **From forensic_evaluation.py (100 splits, adult questionnaire):**
            - Marginal coverage (α=0.10): 89.8% ± 3.6%
            - Class-conditional: ASD 76.6%, Control 94.4%
            - Imbalance causes class-conditional coverage gap — Mondrian conformal addresses this
            """
        )
    
    # Site-conditional coverage
    st.subheader("Coverage Under Site Shift")
    if site:
        st.markdown(
            "When calibrating on training sites and testing on held-out sites, "
            "coverage may drop if the exchangeability assumption breaks."
        )
        st.info(
            "Site-conditional coverage requires the full uncertainty pipeline with ABIDE data. "
            "Run `python src/eval/site_analysis.py` and `python src/uncertainty/calibration.py`."
        )
    else:
        st.info("Run site analysis to see per-site conformal coverage.")

# ── Tab 3: Risk-coverage ──────────────────────────────────────────────────────
with tab3:
    st.header("Risk-Coverage Curves")
    st.markdown(
        """
        Risk-coverage curves show the trade-off between how many samples we decide on 
        (coverage) and the error rate on those samples (risk).
        
        Abstaining from the most uncertain samples improves accuracy on the remaining ones.
        """
    )
    
    # Coverage slider
    target_cov = st.slider("Coverage target", 0.5, 1.0, 0.8, 0.05)
    
    sel_pred = unc.get("selective_prediction", {})
    if sel_pred:
        acc_key = f"acc_at_{int(target_cov*100)}pct"
        sens_key = f"sens_at_{int(target_cov*100)}pct"
        acc = sel_pred.get("accuracy_at_coverage_targets", {}).get(acc_key)
        sens = sel_pred.get("accuracy_at_coverage_targets", {}).get(sens_key)
        aurc = sel_pred.get("aurc")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Accuracy at {target_cov:.0%} coverage",
                      f"{acc:.1%}" if acc else "N/A")
        with col2:
            st.metric(f"Sensitivity at {target_cov:.0%} coverage",
                      f"{sens:.1%}" if sens else "N/A")
        with col3:
            st.metric("AURC (lower = better)", f"{aurc:.4f}" if aurc else "N/A")
        
        # Plot risk-coverage curve if grids available
        if PLOTLY_AVAILABLE and sel_pred.get("coverage_grid"):
            cov_grid = sel_pred["coverage_grid"]
            risk_grid = sel_pred["risk_grid"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cov_grid, y=risk_grid,
                mode="lines", name=sel_pred.get("name", "Model"),
                line=dict(color="#0077BB", width=2),
            ))
            fig.add_vline(x=target_cov, line_dash="dash", line_color="orange",
                          annotation_text=f"Target: {target_cov:.0%}")
            fig.update_layout(
                title="Risk-Coverage Curve",
                xaxis_title="Coverage",
                yaxis_title="Risk (1 - Accuracy)",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run uncertainty analysis for risk-coverage curves.")
        st.markdown(
            "Expected: at 80% coverage, accuracy improves by ~2-5pp compared to full coverage, "
            "as the most uncertain borderline cases are deferred to expert review."
        )

# ── Tab 4: Abstention trade-off ──────────────────────────────────────────────
with tab4:
    st.header("Abstention Trade-off Table")
    st.markdown(
        """
        The **decision-support policy** maps conformal prediction sets to decisions:
        - Singleton {1} → **Screen positive** (flag for follow-up)
        - Singleton {0} → **Screen negative** (low concern)
        - Empty {} or Full {0,1} → **Refer for expert review** (uncertain)
        """
    )
    
    policy = unc.get("decision_policy", {})
    if policy:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Screen Positive", policy.get("n_screen_positive", "N/A"))
        with col2:
            st.metric("Screen Negative", policy.get("n_screen_negative", "N/A"))
        with col3:
            st.metric("Refer for Review", 
                      f"{policy.get('n_refer_for_review', 'N/A')} "
                      f"({policy.get('review_burden', 0):.1%})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy (decided only)",
                      f"{policy.get('decided_accuracy', float('nan')):.1%}"
                      if policy.get('decided_accuracy') else "N/A")
        with col2:
            st.metric("FN rate (missed cases/all)",
                      f"{policy.get('false_negative_rate_total', float('nan')):.1%}"
                      if policy.get('false_negative_rate_total') is not None else "N/A")
    else:
        st.info("Run uncertainty analysis for the decision-support policy results.")
        st.markdown(
            """
            **Coverage targets for adult questionnaire (from forensic_evaluation.py):**
            - 100% coverage: accuracy ~89-95%, ~6-7% refer for review
            - 90% coverage: accuracy improves, review burden reduces
            """
        )
    
    st.divider()
    st.markdown(
        """
        > **Important note:** These abstention rates apply to the screening scenario 
        > (behavioral questionnaire). Conformal prediction for ABIDE fMRI is computed separately 
        > and expected to have higher abstention rates (more uncertain predictions from neuroimaging).
        """
    )

st.divider()
st.caption("Research prototype. Not a diagnostic device. All numbers from verified results files.")
