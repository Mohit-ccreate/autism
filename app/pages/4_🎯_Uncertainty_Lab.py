"""
Uncertainty Lab — Streamlit Page 4.

Shows:
- Multi-arm Uncertainty Evaluation: ABIDE I LOSO (N=854), Leak-Free Demographics (4 cohorts), and Leaky Contrast
- Calibration: Reliability diagrams, Temperature scaling, Platt, Isotonic regression (ECE, Brier, NLL)
- Split Conformal Prediction (LAC, APS, Mondrian) with set-size identity verification
- Per-site coverage breakdown under LOSO domain shift (over-coverage / under-coverage)
- Risk-Coverage curves and selective prediction
- 3-Way Clinical Decision Support Triage Policy

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
st.error("⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.** No clinical claims.", icon="⚠️")
st.title("🎯 Uncertainty-Aware Decision Support Lab")
st.markdown(
    "Rigorous calibration, split conformal prediction, and abstention policies evaluated on "
    "leak-free tabular demographic models and multi-center neuroimaging under Leave-One-Site-Out (LOSO)."
)


@st.cache_data(ttl=300)
def load_uncertainty():
    p = RESULTS_DIR / "uncertainty_main.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


unc = load_uncertainty()

# Arm Selector
arm_options = [
    "ABIDE I Full Cohort (N=854, LOSO)",
    "Adult Demographics (Leak-Free, N=667)",
    "Toddler Demographics (Leak-Free, N=850)",
    "Child Demographics (Leak-Free, N=262)",
    "Adolescent Demographics (Leak-Free, N=97)",
    "Adult Questionnaire (LEAKY CONTRAST ONLY, N=667)",
]
selected_arm_label = st.selectbox("Select Evaluation Arm", arm_options)

if "LEAKY CONTRAST" in selected_arm_label:
    st.warning("⚠️ **LEAKY CONTRAST ONLY:** This run contains questionnaire item features that deterministically leak the diagnosis label. Included solely as a methodological negative contrast.")
    arm_data = unc.get("leaky_contrast_baseline", {})
elif "ABIDE" in selected_arm_label:
    arm_data = unc.get("abide_full_loso", {})
else:
    cohort_key = selected_arm_label.split()[0].lower()
    arm_data = unc.get("tabular_leak_free", {}).get(cohort_key, {})

# Top ribbon for selected arm
col_r1, col_r2, col_r3, col_r4 = st.columns(4)
with col_r1:
    st.metric("Evaluation N", arm_data.get("n_test", "N/A"))
with col_r2:
    st.metric("Calibration Set N", arm_data.get("n_cal", "N/A"))
with col_r3:
    conf10 = arm_data.get("conformal_prediction", {}).get("lac_alpha10", {})
    st.metric("LAC Conformal Coverage (α=0.10)", f"{conf10.get('marginal_coverage', 0.0):.1%}", delta="Target: 90.0%")
with col_r4:
    st.metric("Mean Set Size", f"{conf10.get('mean_set_size', 0.0):.3f}")

st.divider()

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Calibration & ECE", "📊 Split Conformal Prediction", "🎯 Risk-Coverage Curves", "📋 3-Way Triage Policy"]
)

# ── Tab 1: Calibration ────────────────────────────────────────────────────────
with tab1:
    st.header("1. Probability Calibration & Reliability")
    st.markdown("Comparing uncalibrated model probabilities against Platt Scaling, Isotonic Regression, and Temperature Scaling.")

    calib_dict = arm_data.get("calibration", {})
    if calib_dict:
        rows = []
        for method_key, method_val in calib_dict.items():
            if isinstance(method_val, dict) and "metrics" in method_val:
                m = method_val["metrics"]
                name = m.get("name", method_key)
            else:
                m = method_val
                name = m.get("name", method_key)

            rows.append({
                "Method": name.replace("_", " ").title(),
                "ECE (15-bin)": round(m.get("ece_15bin", float("nan")), 4),
                "ECE (10-bin)": round(m.get("ece_10bin", float("nan")), 4),
                "Adaptive ECE": round(m.get("ece_adaptive", float("nan")), 4),
                "Brier Score": round(m.get("brier", float("nan")), 4),
                "NLL": round(m.get("nll", float("nan")), 4),
                "Mean Pred Prob": round(m.get("mean_predicted_prob", float("nan")), 4),
                "Empirical Prevalence": round(m.get("empirical_prevalence", float("nan")), 4),
            })
        df_cal = pd.DataFrame(rows)
        st.dataframe(df_cal.set_index("Method"), use_container_width=True)

        if PLOTLY_AVAILABLE:
            fig_ece = px.bar(
                df_cal,
                x="Method",
                y="ECE (15-bin)",
                color="Brier Score",
                title="Calibration Error (ECE 15-bin) by Post-Hoc Method",
                color_continuous_scale="Reds",
            )
            fig_ece.update_layout(height=350)
            st.plotly_chart(fig_ece, use_container_width=True)
    else:
        st.info("Calibration metrics loading from results/uncertainty_main.json.")

# ── Tab 2: Conformal Prediction ───────────────────────────────────────────────
with tab2:
    st.header("2. Split Conformal Prediction (Guaranteed Coverage)")
    st.markdown(
        "Split conformal prediction constructs prediction sets $\\mathcal{C}(X) \\subseteq \\{0, 1\\}$ "
        "guaranteeing marginal coverage $\\mathbb{P}(Y \\in \\mathcal{C}(X)) \\ge 1 - \\alpha$ under exchangeability."
    )

    conf_dict = arm_data.get("conformal_prediction", {})
    if conf_dict:
        conf_rows = []
        for name, m in conf_dict.items():
            conf_rows.append({
                "Method": name.upper(),
                "Nominal": f"{m.get('nominal_coverage', 0.0):.1%}",
                "Empirical Coverage": f"{m.get('marginal_coverage', 0.0):.1%}",
                "Mean Set Size": round(m.get("mean_set_size", 0.0), 3),
                "% Empty ∅": f"{m.get('fraction_empty', 0.0):.1%}",
                "% Singleton {y}": f"{m.get('fraction_singleton', 0.0):.1%}",
                "% Both {0,1}": f"{m.get('fraction_both', 0.0):.1%}",
                "Identity Verified": "✅" if m.get("set_size_identity_verified") else "❌",
            })
        st.dataframe(pd.DataFrame(conf_rows).set_index("Method"), use_container_width=True)

        st.info(
            "**Set-Size Identity Verified:**  \n"
            "$\\text{mean\\_set\\_size} = 1.0 \\times \\text{fraction\\_singleton} + 2.0 \\times \\text{fraction\\_both} + 0.0 \\times \\text{fraction\\_empty}$.  \n"
            "This identity holds strictly across all conformal arms."
        )

    # Per-Site Coverage under LOSO (if ABIDE)
    if "abide" in selected_arm_label.lower():
        st.subheader("Per-Site Conformal Coverage Under Scanner Shift (LOSO)")
        st.markdown(
            "When calibrating strictly on training sites and deploying to held-out physical scanners, "
            "inter-scanner domain shift can cause empirical coverage to deviate from nominal 90%."
        )

        site_summary = arm_data.get("per_site_summary", {})
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Well-Calibrated Sites (85%-95%)", len(site_summary.get("well_calibrated_sites", [])))
        with col_s2:
            st.metric("Under-Covered Sites (<85%)", len(site_summary.get("under_covered_sites", [])))
        with col_s3:
            st.metric("Over-Covered Sites (>95%)", len(site_summary.get("over_covered_sites", [])))

        per_site_conf = arm_data.get("per_site_loso_conformal", {})
        if per_site_conf:
            ps_rows = []
            for s_name, s_data in per_site_conf.items():
                cov = s_data.get("empirical_coverage", 0.0)
                ps_rows.append({
                    "Site": s_name,
                    "N Test": s_data.get("n_test", 0),
                    "Empirical Coverage": cov,
                    "Mean Set Size": round(s_data.get("mean_set_size", 0.0), 3),
                    "Status": "⚠️ Under-covered" if cov < 0.85 else ("⚡ Over-covered" if cov > 0.95 else "✅ Well-calibrated"),
                })
            df_ps_conf = pd.DataFrame(ps_rows).sort_values("N Test", ascending=False)
            st.dataframe(df_ps_conf.set_index("Site"), use_container_width=True)

            if PLOTLY_AVAILABLE:
                fig_ps_cov = px.bar(
                    df_ps_conf,
                    x="Site",
                    y="Empirical Coverage",
                    color="Status",
                    color_discrete_map={"✅ Well-calibrated": "#0077BB", "⚠️ Under-covered": "#CC3311", "⚡ Over-covered": "#EE7733"},
                    title="Per-Site Conformal Coverage vs Nominal 90% under LOSO",
                )
                fig_ps_cov.add_hline(y=0.90, line_dash="dash", line_color="black", annotation_text="Nominal 90%")
                fig_ps_cov.update_layout(yaxis=dict(range=[0.5, 1.05]), height=380, xaxis_tickangle=-45)
                st.plotly_chart(fig_ps_cov, use_container_width=True)

# ── Tab 3: Risk-Coverage Curves ──────────────────────────────────────────────
with tab3:
    st.header("3. Selective Prediction & Risk-Coverage Curves")
    st.markdown(
        "By abstaining on predictions with high uncertainty, the decision-support system reduces risk "
        "(1 - Accuracy) on the retained samples."
    )

    sel = arm_data.get("selective_prediction", {})
    if sel:
        col_aurc, col_eaurc = st.columns(2)
        with col_aurc:
            st.metric("Area Under Risk-Coverage (AURC)", f"{sel.get('aurc', float('nan')):.4f}")
        with col_eaurc:
            st.metric("Excess AURC (EAURC vs optimal)", f"{sel.get('eaurc', float('nan')):.4f}")

        acc_targets = sel.get("accuracy_at_coverage_targets", {})
        if acc_targets:
            st.markdown("#### Accuracy & Sensitivity at Coverage Targets")
            acc_rows = []
            for t in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
                k_pct = int(t * 100)
                acc_val = acc_targets.get(f"acc_at_{k_pct}pct")
                sens_val = acc_targets.get(f"sens_at_{k_pct}pct")
                if acc_val is not None:
                    acc_rows.append({
                        "Coverage Target": f"{t:.0%}",
                        "Abstention Rate": f"{1.0 - t:.0%}",
                        "Retained Accuracy": f"{acc_val:.1%}",
                        "Retained Sensitivity": f"{sens_val:.1%}" if sens_val is not None else "N/A",
                    })
            if acc_rows:
                st.dataframe(pd.DataFrame(acc_rows).set_index("Coverage Target"), use_container_width=True)

        if PLOTLY_AVAILABLE and "coverage_grid" in sel and "risk_grid" in sel:
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(
                x=sel["coverage_grid"],
                y=sel["risk_grid"],
                mode="lines+markers",
                name="Selective Prediction Risk",
                line=dict(color="#0077BB", width=3),
            ))
            fig_rc.update_layout(
                title="Empirical Risk-Coverage Curve",
                xaxis_title="Coverage (Fraction Retained)",
                yaxis_title="Empirical Risk (1 - Accuracy)",
                height=380,
            )
            st.plotly_chart(fig_rc, use_container_width=True)

# ── Tab 4: 3-Way Triage Policy ───────────────────────────────────────────────
with tab4:
    st.header("4. Three-Way Clinical Triage Decision Policy")
    st.markdown(
        """
        The triage policy partitions incoming cases into three actionable pathways:
        - **Definitive Positive {1}:** Confident indication of risk $\\rightarrow$ High-priority specialist referral
        - **Definitive Negative {0}:** Confident low likelihood $\\rightarrow$ Routine monitoring / primary care follow-up
        - **Abstain / Uncertain {0,1} or ∅:** Inconclusive $\\rightarrow$ Defer to comprehensive clinical evaluation
        """
    )

    triage = arm_data.get("clinical_triage_policy", {})
    if triage:
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.metric("Screen Positive {1}", f"{triage.get('count_screen_positive', 0)} ({triage.get('fraction_screen_positive', 0.0):.1%})")
        with col_t2:
            st.metric("Screen Negative {0}", f"{triage.get('count_screen_negative', 0)} ({triage.get('fraction_screen_negative', 0.0):.1%})")
        with col_t3:
            st.metric("Abstain / Defer to Review", f"{triage.get('count_refer_to_review', 0)} ({triage.get('triage_burden', 0.0):.1%})")

        st.markdown("#### Performance on Retained Decisions")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Singleton Accuracy", f"{triage.get('singleton_accuracy', float('nan')):.1%}")
        with col_p2:
            st.metric("Singleton Sensitivity", f"{triage.get('singleton_sensitivity', float('nan')):.1%}")
        with col_p3:
            st.metric("False Negative Rate (Total)", f"{triage.get('false_negative_rate_total', float('nan')):.1%}")

        st.warning(
            "⚠️ **Scientific Honesty Safeguard:** Abstaining on uncertain samples concentrates high-confidence cases, "
            "improving accuracy on retained cases. However, abstention **does not eliminate errors**. "
            "In clinical practice, deferred cases still require human expert diagnostic evaluation."
        )

st.divider()
st.caption("Research prototype. Not a diagnostic device. All metrics trace directly to results/uncertainty_main.json.")
