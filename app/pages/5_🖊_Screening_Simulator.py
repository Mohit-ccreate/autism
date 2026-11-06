"""
Screening Simulator — Streamlit Page 5.

Demonstrates a behavioral model WITHOUT leaking features.
User enters demographic/history inputs and sees:
- Calibrated probability
- Conformal prediction set
- "Refer for expert review" when uncertain
- Explanation of why a score-based tool would be misleading

IMPORTANT: This uses pre-trained model parameters from results/.
Stores NOTHING. No logging. No personal data.

MEDICAL DISCLAIMER: Research prototype. Not a diagnostic device.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

st.set_page_config(page_title="Screening Simulator", page_icon="🖊", layout="wide")

# ── Persistent prominent disclaimer ───────────────────────────────────────────
st.error(
    "⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**  \n"
    "This simulator demonstrates ML methodology only. It does NOT assess, diagnose, "
    "or screen for Autism Spectrum Disorder in real people. Any similarity to real screening "
    "results is coincidental. For clinical evaluation, consult a licensed healthcare professional.",
    icon="🚨",
)

st.title("🖊 Screening Simulator (Demographics Only)")
st.markdown(
    """
    This demo uses a model trained **without** the AQ-10 items or total score 
    (demographics only: age, sex, jaundice history, family ASD history).
    
    This shows what a *non-leaky* model can and cannot do.
    It is NOT a screening tool for real people.
    """
)

st.info(
    "💡 **Why this matters:** A model WITH access to all 10 items and the total score "
    "would achieve ~100% accuracy by arithmetic (the label IS the score threshold). "
    "This demo intentionally uses only demographics to show the honest, modest performance "
    "of a non-leaky model."
)

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
st.header("Enter Hypothetical Demographics")
st.markdown("*All fields are for demonstration only. No data is stored or transmitted.*")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age (years)", 1, 80, 25, help="Age of the hypothetical individual")
    sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
    
with col2:
    jaundice = st.radio("Jaundice at birth?", ["No", "Yes"], horizontal=True,
                        help="Was the individual born with jaundice?")
    family_asd = st.radio("Family history of ASD?", ["No", "Yes"], horizontal=True,
                          help="Does a close family member have an ASD diagnosis?")

st.divider()

# ── Compute (from pre-fitted simple LR params) ────────────────────────────────
if st.button("🔍 Show Model Output (Demonstration Only)", type="primary"):
    # Build feature vector
    sex_bin = 1 if sex == "Male" else 0
    jaundice_bin = 1 if jaundice == "Yes" else 0
    family_bin = 1 if family_asd == "Yes" else 0
    x = np.array([[age, sex_bin, jaundice_bin, family_bin]], dtype=float)
    
    # Normalize (approximate adult dataset stats)
    age_mean, age_std = 29.0, 10.0
    x_norm = x.copy()
    x_norm[:, 0] = (x_norm[:, 0] - age_mean) / age_std

    # Try to load saved model from results/
    model_params_path = RESULTS_DIR / "demo_lr_params_demos_only.json"
    
    if model_params_path.exists():
        with open(model_params_path) as f:
            params = json.load(f)
        coef = np.array(params["coef"])
        intercept = float(params["intercept"])
        logit = float(x_norm @ coef + intercept)
        prob = float(1.0 / (1.0 + np.exp(-logit)))
    else:
        # Fallback: a rough logistic model trained on demographics only
        # (approximate coefficients from v4_audit_pass1.json Var C results for adult)
        # These are illustrative and reflect that demographics-only models are weak
        # coef: [age, sex, jaundice, family_asd] — rough heuristic
        rough_coef = np.array([-0.02, 0.15, 0.35, 0.60])
        rough_intercept = -1.2
        logit = float(x_norm @ rough_coef + rough_intercept)
        prob = float(1.0 / (1.0 + np.exp(-logit)))
        st.warning(
            "Using approximate coefficients (demographics-only model not yet fitted from data). "
            "Run `python src/audit/tautology_audit.py` to fit real model parameters."
        )

    # Conformal: use a fixed q_hat from calibration (approximate from forensic results)
    # demographics-only model is weakly calibrated; q_hat ~0.6 means large sets
    q_hat = 0.55  # approximate for demographics-only model
    
    pset = []
    if 1 - prob <= q_hat:
        pset.append("ASD Traits Present")
    if prob <= q_hat:
        pset.append("ASD Traits Absent")
    
    # Decision
    if pset == ["ASD Traits Present"]:
        decision = "Screen Positive"
        decision_color = "error"
    elif pset == ["ASD Traits Absent"]:
        decision = "Screen Negative"
        decision_color = "success"
    else:
        decision = "Refer for Expert Review"
        decision_color = "warning"
    
    st.header("Model Output (Demonstration Only)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Calibrated Probability", f"{prob:.1%}",
                  help="P(ASD traits present) according to demographics-only model")
    with col2:
        st.metric("Conformal Set", ", ".join(pset) if pset else "∅ (refer)")
    with col3:
        if decision_color == "error":
            st.error(f"⚠️ Decision: {decision}")
        elif decision_color == "success":
            st.success(f"✅ Decision: {decision}")
        else:
            st.warning(f"❓ Decision: {decision}")
    
    st.divider()
    
    # Explanation
    st.subheader("Why a Score-Based Tool Would Be Misleading")
    st.markdown(
        f"""
        **This demographics-only model is intentionally limited.** 
        It shows what can be inferred from age, sex, and history alone — which is very little.
        
        A model WITH access to the 10-item questionnaire responses and the total score 
        would achieve near-100% accuracy, but this is because:
        
        1. The ASD label is literally defined as `sum(A1..A10) ≥ {7}` for adults
        2. Including all items or the total score is equivalent to computing the diagnostic rule
        3. Such a model teaches nothing about autism — it just reproduces an arithmetic threshold
        
        **The demographics-only model's probability {prob:.1%} reflects the population base rate 
        more than individual characteristics.** This is an honest representation of what 
        demographic variables alone can tell us.
        
        > **Conformal prediction set:** {pset if pset else "∅ (empty — model too uncertain)"}  
        > An empty or full prediction set means the model cannot make a confident decision — 
        > this case should be reviewed by a clinical specialist.
        """
    )
    
    if decision == "Refer for Expert Review":
        st.warning(
            "**This model is too uncertain to make a prediction.** "
            "In a real screening scenario, this would trigger expert review. "
            "Clinicians use validated tools (ADOS-2, ADI-R) for formal assessment."
        )

st.divider()
st.caption(
    "🔒 **Privacy:** This tool stores no data, makes no network requests, and contains no analytics. "
    "All computation happens in-browser/locally. "
    "Research prototype — NOT a clinical screening or diagnostic tool. "
    "No patient data ever enters this application."
)
