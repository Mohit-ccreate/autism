"""
Scanner Generalization — Streamlit Page 3.

Shows:
- Random-CV vs LOSO accuracy/AUROC comparison
- Per-site accuracy gaps
- Site probe result (connectivity site decodability)
- Harmonization comparison including leaky-ComBat arm
- Confound analysis (motion/age/sex)

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

st.set_page_config(page_title="Scanner Generalization", page_icon="🧲", layout="wide")
st.error("⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**", icon="⚠️")
st.title("🧲 Scanner / Site Generalization")


@st.cache_data(ttl=300)
def load_site_analysis():
    p = RESULTS_DIR / "site_analysis.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(ttl=300)
def load_v4_audit():
    p = RESULTS_DIR / "v4_audit_pass1.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


site = load_site_analysis()
v4 = load_v4_audit()

# ── Random CV vs LOSO ─────────────────────────────────────────────────────────
st.header("1. Random CV vs LOSO — The Generalization Gap")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Existing verified results (AUDIT_REPORT.md):**")
    comparison_data = {
        "Model": ["LR + ComBat", "BNT", "Linear SVM", "2D CNN", "BrainGAT"],
        "LOSO (honest)": [0.5500, 0.4583, 0.5167, 0.5167, 0.5167],
        "LOSO AUROC": [0.6098, 0.4883, None, 0.5612, 0.5584],
        "Random CV (leaky)": [0.5833, 0.6833, None, None, None],
        "Gap (Random-LOSO)": [0.0333, 0.2250, None, None, None],
    }
    df_cmp = pd.DataFrame(comparison_data)
    st.dataframe(df_cmp.set_index("Model"), use_container_width=True)

with col2:
    if PLOTLY_AVAILABLE:
        models = ["LR + ComBat", "BNT"]
        loso_acc = [0.5500, 0.4583]
        rand_acc = [0.5833, 0.6833]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="LOSO (honest)", x=models, y=loso_acc,
                             marker_color="#0077BB"))
        fig.add_trace(go.Bar(name="Random 5-fold (leaky ⚠️)", x=models, y=rand_acc,
                             marker_color="#CC3311"))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Chance")
        fig.update_layout(barmode="group",
                          title="Accuracy: LOSO vs Random-CV",
                          yaxis=dict(range=[0, 0.75], title="Accuracy"),
                          height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Install plotly for interactive charts.")

st.markdown(
    """
    > **Key finding:** Random CV inflates accuracy by **+3.3pp (LR)** and **+22.5pp (BNT)**.  
    > BNT memorizes scanner hardware fingerprints under random CV (site signatures leak across folds).  
    > LOSO accuracy is **indistinguishable from chance** (binomial p=0.315).
    """
)

st.divider()

# ── Harmonization comparison ──────────────────────────────────────────────────
st.header("2. Harmonization Comparison")

harm_comp = site.get("harmonization_comparison", {})

if harm_comp:
    rows = []
    for harm_name, harm_data in harm_comp.items():
        leaky = harm_data.get("is_leaky", False)
        rows.append({
            "Harmonization": harm_name.replace("_", " ").title() + (" ⚠️ LEAKY" if leaky else ""),
            "Pooled AUROC": harm_data.get("pooled_auroc"),
            "Pooled Accuracy": harm_data.get("pooled_accuracy"),
            "Leaky": leaky,
        })
    df_harm = pd.DataFrame(rows)

    if PLOTLY_AVAILABLE:
        colors = ["#CC3311" if r["Leaky"] else "#0077BB" for _, r in df_harm.iterrows()]
        fig = go.Figure()
        for i, (_, row) in enumerate(df_harm.iterrows()):
            fig.add_trace(go.Bar(
                name=row["Harmonization"],
                x=[row["Harmonization"]],
                y=[row["Pooled AUROC"]],
                marker_color="#CC3311" if row["Leaky"] else "#0077BB",
                showlegend=False,
            ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.update_layout(title="LOSO AUROC by Harmonization Strategy",
                          yaxis=dict(range=[0, 1], title="Pooled AUROC"),
                          height=350)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df_harm.drop("Leaky", axis=1).set_index("Harmonization"),
                     use_container_width=True)
    st.warning(
        "⚠️ The **LEAKY** arm (ComBat fit on all data before CV) is included solely to quantify "
        "inflation from this methodological error. It MUST NOT be used for real research."
    )
else:
    st.info("Run `python src/eval/site_analysis.py` to generate harmonization comparison.")
    st.markdown(
        """
        **Expected results (from prior 4-site analysis):**
        | Harmonization | AUROC |
        |---|---|
        | No harmonization | ~0.57 |
        | Site regression | ~0.58 |
        | ComBat (correct) | 0.6098 |
        | ComBat (leaky ⚠️) | ~0.65+ (inflated) |
        """
    )

st.divider()

# ── Site probe ────────────────────────────────────────────────────────────────
st.header("3. Site Probe: Can we predict scanner site from connectivity?")

probe = site.get("site_probe", {})
if probe:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Connectivity → Site Accuracy",
                  f"{probe.get('connectivity_site_pred_accuracy', float('nan')):.1%}")
    with col2:
        st.metric("Confounds (age/sex/motion) → Site",
                  f"{probe.get('confound_site_pred_accuracy', float('nan')):.1%}")
    with col3:
        st.metric("Base rate (most common site)", 
                  f"{probe.get('base_rate_accuracy', float('nan')):.1%}")
    st.markdown(
        "If connectivity features strongly predict site, they carry scanner hardware signatures "
        "that can inflate cross-fold performance."
    )
else:
    st.metric("Pre-ComBat Site Prediction (5-fold CV)", "73.3% (from v4 audit)")
    st.metric("Post-ComBat", "Deleted (artifact — in-sample ComBat)")
    st.info("Run `python src/eval/site_analysis.py` for updated site probe.")

st.divider()

# ── Confound analysis ─────────────────────────────────────────────────────────
st.header("4. Confound Analysis (Motion / Age / Sex)")

confounds = site.get("confound_models", {})
if confounds:
    st.metric("Confound-only LOSO AUROC", 
              f"{confounds.get('loso_auroc', float('nan')):.4f}")
    st.markdown("**Partial correlations with ASD label:**")
    if "partial_correlations_with_asd" in confounds:
        rows = [{"Confound": k, "r": v.get("r"), "p": v.get("p")}
                for k, v in confounds["partial_correlations_with_asd"].items()]
        st.dataframe(pd.DataFrame(rows).set_index("Confound"), use_container_width=True)
else:
    # From existing forensic_evaluation results
    st.markdown("**From AUDIT_REPORT.md verified values:**")
    conf_data = {
        "Confound": ["Mean FD (head motion)", "Age at scan"],
        "Pearson r with ASD": [-0.1399, -0.1839],
        "Threshold": ["|r| < 0.15", "|r| < 0.25"],
        "Status": ["✅ PASS", "✅ PASS"],
    }
    st.dataframe(pd.DataFrame(conf_data).set_index("Confound"), use_container_width=True)

st.divider()
st.caption(
    "ABIDE-II external validation: SKIPPED (access requires institutional account). "
    "Full 17-site analysis: requires complete nilearn cache download. "
    "Research prototype only."
)
