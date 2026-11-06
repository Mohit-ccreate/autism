"""
Reproducibility & Methods — Streamlit Page 6.

Shows configs, seeds, pipeline diagram, links to README and audit.

MEDICAL DISCLAIMER: Research prototype. Not a diagnostic device.
"""

import json
import subprocess
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

st.set_page_config(page_title="Reproducibility", page_icon="🔬", layout="wide")
st.error("⚠️ **RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL.**", icon="⚠️")
st.title("🔬 Reproducibility & Methods")


def get_git_hash():
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT))
        return r.stdout.strip()
    except Exception:
        return "N/A"


st.header("Run Information")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Git Commit", get_git_hash()[:12])
with col2:
    facts_path = RESULTS_DIR / "paper_facts.json"
    if facts_path.exists():
        with open(facts_path) as f:
            facts = json.load(f)
        st.metric("Results Generated", facts.get("_metadata", {}).get("generated_at", "N/A")[:10])
    else:
        st.metric("Results Generated", "Not yet generated")
with col3:
    st.metric("Branch", "paper-revision")

st.divider()

# ── Seeds ──────────────────────────────────────────────────────────────────────
st.header("Random Seeds")
st.markdown(
    """
    All experiments use seeds from the config:
    - **Primary seed (original results):** 42
    - **Extension seeds:** 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    - Seeds logged in every run to `results/RUN_LOG.md`
    """
)
st.code("seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]", language="yaml")

st.divider()

# ── Configs ────────────────────────────────────────────────────────────────────
st.header("Configuration Files")

config_path = ROOT / "configs" / "main.yaml"
if config_path.exists():
    with open(config_path) as f:
        config_text = f.read()
    st.code(config_text, language="yaml")

st.divider()

# ── Pipeline diagram ──────────────────────────────────────────────────────────
st.header("Pipeline Diagram")

pipeline_path = ROOT / "figures" / "pipeline_schematic.png"
if pipeline_path.exists():
    st.image(str(pipeline_path), caption="End-to-end pipeline schematic")
else:
    st.markdown(
        """
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │  ASD Screening Audit Pipeline                               │
        ├──────────────────┬──────────────────────────────────────────┤
        │  BEHAVIORAL DATA │  NEUROIMAGING (ABIDE I)                  │
        │                  │                                          │
        │  UCI Adult/Ado/  │  CC200 ROI timeseries                    │
        │  Child/Toddler   │  → Pearson correlation                   │
        │                  │  → Fisher-z transform                    │
        │  Leakage audit   │                                          │
        │  (100% tautology │  LOSO CV (site-aware)                    │
        │   confirmed)     │  ├── ComBat (inside fold)               │
        │                  │  ├── Feature selection (inside fold)     │
        │  Feature ablation│  ├── LR, SVM, RF, LightGBM             │
        │  Var A: items    │  └── Random-CV (leaky, for comparison)   │
        │  Var B: +demos   │                                          │
        │  Var C: demos    │  Harmonization comparison:               │
        │  Var D: +score   │  None / Regression / ComBat / Leaky     │
        │  (leaky)         │                                          │
        ├──────────────────┴──────────────────────────────────────────┤
        │  UNCERTAINTY (Module 4)                                     │
        │  Calibration: ECE, Brier, NLL                               │
        │  Post-hoc: temperature scaling, isotonic, Platt             │
        │  Conformal: LAC, APS, Mondrian                              │
        │  Selective: risk-coverage curves, AURC                      │
        │  Policy: screen+ / screen- / refer                          │
        ├─────────────────────────────────────────────────────────────┤
        │  OUTPUTS                                                    │
        │  results/paper_facts.json  → verified numbers               │
        │  figures/*.pdf + *.png     → publication-quality            │
        │  app/                      → Streamlit demo                 │
        │  AUDIT.md                  → leakage inventory              │
        └─────────────────────────────────────────────────────────────┘
        ```
        """
    )
    st.info("Run `python src/viz/generate_figures.py` to generate the pipeline schematic.")

st.divider()

# ── Reproduce commands ─────────────────────────────────────────────────────────
st.header("Reproduction Commands")
st.code(
    """
# 1. Activate virtual environment (Windows)
.\\venv\\Scripts\\Activate.ps1

# 2. Run original pipeline (questionnaire + ABIDE baseline)
python run_all.py

# 3. Run extended pipeline (new modules)
python run_extended.py

# 4. Launch Streamlit demo
streamlit run app/Home.py

# 5. Run all tests
python -m pytest tests/ -v

# 6. Generate figures only
python src/viz/generate_figures.py
""",
    language="bash",
)

st.header("Runtime Estimates")
st.markdown(
    """
    | Pipeline | Estimated Time | Hardware |
    |----------|---------------|---------|
    | `run_all.py` (original) | ~3.5 minutes | CPU |
    | `run_extended.py` (new modules) | ~5-15 minutes | CPU |
    | Full ABIDE download | 30-60 minutes | Network + CPU |
    | Full 17-site LOSO | ~20 minutes | CPU |
    """
)

st.divider()
st.markdown("[📄 README](file:///" + str(ROOT / "README.md") + ")")
st.markdown("[🔍 AUDIT.md](file:///" + str(ROOT / "AUDIT.md") + ")")
st.markdown("[📋 PROGRESS.md](file:///" + str(ROOT / "PROGRESS.md") + ")")
