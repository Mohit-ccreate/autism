# Autism Spectrum Disorder (ASD) Honest Clinical Screening & Brain Connectivity Research Platform

> [!IMPORTANT]
> **CRITICAL CLINICAL & SAFETY DISCLAIMER:**  
> This software is an **academic research and preliminary screening tool**. It **DOES NOT** diagnose Autism Spectrum Disorder (ASD), provide medical advice, or replace an evaluation by a qualified clinician (such as a developmental pediatrician, child psychiatrist, or clinical psychologist using gold-standard instruments like ADOS-2 or ADI-R).  
> The system strictly assesses probabilistic behavioral patterns and neural functional connectivity; it never outputs definitive diagnostic assertions (e.g., asserting or excluding a formal autism diagnosis).

---

## 1. What Is This Project?

This repository contains a **complete, transparent, and reproducible clinical-AI screening platform** for Autism Spectrum Disorder (ASD). 

Unlike standard machine learning tutorials or published papers that report misleading 99%–100% accuracy, this project:
1. **Uncovers and proves the hidden leakage** in popular UCI and Kaggle autism datasets (where the target label was simply an arithmetic step function of the sum of questions: $\sum A_i \ge 7 \iff \text{Yes}$).
2. **Evaluates honest non-deterministic tasks** (rapid 5-item screeners, demographic baselines, and cross-age unified models).
3. **Implements an honest neuroimaging benchmark** on the Autism Brain Imaging Data Exchange (ABIDE I) resting-state fMRI dataset using **Leave-One-Site-Out (LOSO) Cross-Validation with fold-internal ComBat site harmonization**.
4. **Conducts a deep learning research lab** evaluating Brain Network Transformers (BNT), Vision Transformers (ViT), 2D CNNs, and Graph Attention Networks (BrainGAT) alongside self-supervised masked region pretraining, multimodal fusion with graceful degradation, and test-time site adaptation (CORAL).
5. **Applies strict scientific safeguards** verifying label-shuffle permutation collapse to chance (45.83%), scanner site hardware neutralization (predictability drops from 94.2% to 41.7%), and head motion artifact decorrelation ($|r| < 0.15$).
6. **Delivers a 100% private, client-side browser application** (PWA) with an interactive 3D neural attention explorer, experiment arena toggle, model disagreement inspector, calibrated risk gauge, conformal prediction bounds, strengths profiler, and printable clinical consultation reports.
7. **Compiles an auto-generated scientific paper pre-print draft** (`reports/paper_draft.md`) and executive research summary (`reports/v2_summary.md`) populated directly from clean data files.

---

## 2. Non-Negotiable Scientific Rules

1. **Absolute Honesty:** Every number in the UI, model card, leaderboard, paper draft, and README comes directly from clean training runs recorded in `reports/metrics.json` and `reports/lab_metrics.json`.
2. **Not a Diagnosis:** Language strictly uses screening phrasing (*"your answers show a higher/lower pattern of traits commonly seen in screening; a specialist can tell you more"*).
3. **No Fabricated Data or Features:** Zero synthetic rows. Severity prediction (level 1/2/3) and webcam face-image diagnosis are deliberately omitted as scientifically invalid on this data.
4. **Transparent Negative Findings:** Deep models overparameterized in the $p \gg n$ regime (BNT 45.83%) and 2D CNNs on raw correlation matrices are documented with scientific rationale rather than swept under the rug.
5. **Verified Sources Only:** All clinical screening questionnaires (Q-CHAT-10, AQ-10) and scientific papers are verified with DOIs and active links.
6. **100% Local Privacy:** Runs completely in the browser. Zero patient data, cookies, or analytics are sent to external servers.
7. **CPU-Friendly & GPU-Enabled:** Standard CPU executes the complete pipeline locally in ~3.5 minutes; a dedicated Google Colab notebook (`notebooks/Phase4_Deep_Research_Lab.ipynb`) is provided for scaled GPU training.

---

## 3. Real Performance Benchmarks (From Our Validated Runs)

All numbers below come directly from `reports/metrics.json` and `reports/lab_metrics.json` generated on untouched test sets:

### Questionnaire Screening Models (Untouched 20% Test Sets)

| Cohort | Task Variant | Architecture | Accuracy [95% Bootstrap CI] | Sensitivity | Specificity | ROC-AUC | Conformal Uncertainty Rate |
|---|---|---|---|---|---|---|---|
| **Adult (AQ-10)** | **Abbreviated (5 Items)** | Stacking Ensemble | **93.28%** [87.3% – 97.0%] | 85.71% | 95.96% | **0.9739** | 6.7% flagged for referral |
| **Adult (AQ-10)** | Demographics Only | Logistic Regression | 74.63% [66.4% – 82.1%] | 14.29% | 95.96% | 0.6837 | Negative control baseline |
| **Toddler (Q-CHAT-10)** | **Abbreviated (5 Items)** | Stacking Ensemble | **88.24%** [82.9% – 92.9%] | 95.16% | 69.57% | **0.9353** | 7.6% flagged for referral |
| **Toddler (Q-CHAT-10)** | Demographics Only | CatBoost | 73.53% [67.1% – 80.0%] | 96.77% | 10.87% | 0.6329 | Negative control baseline |
| **Unified Cross-Age** | **10 Items (Multi-Cohort)** | Stacking Ensemble | **85.37%** [81.6% – 88.8%] | 80.41% | 90.66% | **0.9379** | 5.1% flagged for referral |
| **Unified Cross-Age** | **Abbreviated (5 Items)** | CatBoost | **76.86%** [72.3% – 81.1%] | 74.74% | 79.12% | **0.8543** | 6.4% flagged for referral |
| **Adult / Child / Toddler** | 10 Items Full | Logistic Regression | 100.00% [Deterministic] | 100.00% | 100.00% | 1.0000 | *Replicates Allison cut-off rule ($\sum A_i \ge 7$)* |

### Full ABIDE I Multi-Center Benchmark ($N=854$ across all 20 Clinical Sites, CC200 Atlas)

From the full-cohort audit in `results/benchmark_leaderboard.json` and `results/site_analysis.json`:

| Architecture / Model | Evaluation Protocol | Accuracy [95% CI] | ROC-AUC [95% CI] | Scientific & Clinical Findings |
|---|---|---|---|---|
| **Random Forest** | **Leave-One-Site-Out (LOSO) CV** | **65.11%** | **0.7003** [$0.6648, 0.7346$] | Best-performing tree ensemble on functional connectomes. |
| **LightGBM** | **Leave-One-Site-Out (LOSO) CV** | **64.64%** | **0.6992** [$0.6631, 0.7336$] | Gradient boosting robust to inter-scanner feature scaling. |
| **Extra Trees** | **Leave-One-Site-Out (LOSO) CV** | **63.11%** | **0.6867** [$0.6495, 0.7217$] | Extremely randomized trees maintain high out-of-site stability. |
| **Heinsfeld Autoencoder** | **Leave-One-Site-Out (LOSO) CV** | **62.88%** | **0.6820** [$0.6453, 0.7175$] | Deep representation learning with LayerNorm; beats linear baselines. |
| **Linear SVM** | **Leave-One-Site-Out (LOSO) CV** | **63.11%** | **0.6765** [$0.6390, 0.7143$] | Maximum margin boundary with fold-internal feature selection. |
| **Logistic Regression (L2) + ComBat** | **Leave-One-Site-Out (LOSO) CV** | **61.71%** | **0.6593** [$0.6215, 0.6962$] | **Headline Baseline:** Permutation $p < 0.0001$, Binomial $p = 7.90 \times 10^{-12}$. |
| **Small MLP** | **Leave-One-Site-Out (LOSO) CV** | **62.41%** | **0.6671** [$0.6288, 0.7042$] | Two-layer perceptron with dropout. |
| **Tangent Space Logistic** | **Leave-One-Site-Out (LOSO) CV** | 58.78% | 0.6092 [$0.5707, 0.6481$] | Riemannian log-Euclidean projection. |
| **Random 5-Fold CV (LR)** | 5-Fold Random Split | 62.72% | 0.6728 [$0.6561, 0.6891$] | Generalization gap is $+0.0135$ across 5 seeds. |

### Preliminary 4-Site Subset ($N=120$) & Statistical Reconciliation

| Architecture / Model | Evaluation Protocol | Accuracy [95% CI] | ROC-AUC | Balanced Acc | Scientific & Clinical Findings |
|---|---|---|---|---|---|
| **Logistic Regression (L2) + ComBat** | **Leave-One-Site-Out (LOSO) CV** | **55.00%** [47.0% – 63.0%] | **0.6098** | **54.67%** | Statistically indistinguishable from chance on $N=120$ (perm $p=0.215$, binom $p=0.315$). |
| **Brain Network Transformer (BNT)** | **Leave-One-Site-Out (LOSO) CV** | **45.83%** [38.0% – 54.0%] | 0.4883 | 45.86% | Deep attention over 200 ROIs collapses under out-of-site deployment. |
| **Connectivity ViT** | **Leave-One-Site-Out (LOSO) CV** | 51.67% [43.0% – 60.0%] | 0.4964 | 51.40% | Non-local 14x14 patch attention lacks translation invariance on connectomes. |
| **Small 2D CNN** | **Leave-One-Site-Out (LOSO) CV** | 51.67% [43.0% – 60.0%] | 0.5612 | 51.80% | Grid convolution violates atlas node permutation invariance. |
| **BNT (Random CV)** | 5-Fold Random Split | 68.33% [59.0% – 77.0%] | 0.7120 | 67.80% | **Severe Leakage Exploitation:** Deep model memorizes scanner hardware fingerprints. |

### Scientific Safeguards Verification Matrix

| Safeguard Protocol | Target Tested | Real Value | Target Threshold | Status |
|---|---|---|---|---|
| **Full ABIDE Permutation Test** | Within-site label permutation test on $N=854$ (1,000 permutations) | **p < 0.0001** | $p < 0.05$ (Valid non-chance signal) | **PASS** |
| **Site Hardware Probe** | Decoding scanner site from functional connectomes | **64.87%** | Chance = 19.91% (NYU base rate) | **CONFIRMED** |
| **Harmonization Test** | Paired bootstrap $\Delta$ (Leaky vs In-Fold ComBat) | **+0.0156** [$-0.0013, 0.0328$] | $p_{\text{holm}} = 0.2840$ (No detectable difference) | **VERIFIED** |
| **Conformal Identity Check** | $\text{mean\_set\_size} = 1\cdot s + 2\cdot b + 0\cdot e$ | **Verified (<1e-6)** | Must hold exactly | **PASS** |
| **Head Motion Artifact Audit** | Pearson correlation with Mean Framewise Displacement (FD) | **r = -0.1399** | $\|r\| < 0.15$ (No motion artifact tracking) | **PASS** |
| **Age Confound Audit** | Pearson correlation with subject age at scan | **r = -0.1839** | $\|r\| < 0.25$ (Demographic confound controlled) | **PASS** |

---

## 4. Beginner's Quick Start Guide

### Step 1: Open PowerShell in the Project Directory
```powershell
cd C:\Users\mohit\OneDrive\Desktop\asd-project
```

### Step 2: Activate the Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(Or in Windows Command Prompt: `venv\Scripts\activate.bat`)*

### Step 3: Run the Complete Pipeline (One Command)
To execute all 5 phases end-to-end (ingestion, preprocessing, questionnaire models, ABIDE baseline, research lab, paper compilation, and the full 23-test suite):
```powershell
python run_all.py
```
**Expected Runtime:** ~3.5 minutes on a standard CPU.

### Step 4: Run the Complete Pytest Verification Suite
To run all 24 scientific, clinical, and software integrity tests:
```powershell
python -m pytest tests/ -v
```

### Step 5: Reproduce the IEEE Conference Paper Audit & Figures
To run the full forensic audit pipeline, verify empirical metrics, and regenerate vector figures:
```powershell
# 1. Run the forensic evaluation & generate high-resolution figures (PDF + PNG)
python src/forensic_evaluation.py

# 2. Compute extended statistical benchmarks (power analysis, permutations, Holm correction)
python src/compute_extended_audit.py

# 3. Generate machine-readable and human-readable audit and consistency reports
python scripts/generate_audit_reports.py

# 4. Compile IEEE LaTeX manuscript (Overleaf or local TeX distribution)
# Upload main.tex, references.bib, and figures/ to Overleaf, or run:
# pdflatex main && bibtex main && pdflatex main && pdflatex main
```

### Step 6: Launch the Research Demonstration Applications
The repository provides two demonstration interfaces:

#### Option A: Streamlit Multi-Page Research Platform (Recommended for Faculty Review)
```powershell
streamlit run app/Home.py
```
Opens an interactive research workbench at **http://localhost:8501** featuring:
- **Tautology Explorer:** Dataset selector, rule recovery check, interactive ablation plots, and Tautology Index.
- **Benchmark Leaderboard:** Sortable table across models, protocols, and metrics with CSV export.
- **Scanner Generalization:** LOSO vs. Random-CV comparison, site probe, and ComBat harmonization (including leaky arm).
- **Uncertainty Lab:** Reliability diagrams, per-site conformal coverage, and risk-coverage curves.
- **Screening Simulator:** Demographics-only non-leaky model with conformal uncertainty intervals and triage policy.
- **Reproducibility & Methods:** Verified parameters, seeds, configurations, and pipeline schematics.
- **Audit Checklist:** 15-item ML-for-autism peer-review evaluation checklist with CSV/Markdown download.

#### Option B: Client-Side Browser PWA (Offline In-Browser Scoring & 3D Brain)
```powershell
python -m http.server 8000 --directory webapp
```
Then open **[http://localhost:8000](http://localhost:8000)** in any browser.

---

## 5. Verified Data Lineage & Checksums

All datasets are verified with SHA-256 integrity checksums:

| Dataset | File Path | Records (Raw $\to$ Dedup) | Size (Bytes) | SHA-256 Checksum | Source / Citation |
|---|---|---|---|---|---|
| **UCI Adult AQ-10** | `data/uci_adult.csv` | $704 \to 667$ | 56,148 | `657021cddb337358640470cea758b72087bc4519e1234f7bc59c2b0160a0060d` | UCI Repository #426 (Thabtah 2017) |
| **UCI Adolescent AQ-10** | `data/uci_adolescent.csv` | $104 \to 97$ | 8,710 | `4753e2ce2c791c32181ad9413c55b6556828b9578a22d166d0cef7caaf93c89d` | UCI Repository #420 (Thabtah 2017) |
| **UCI Child AQ-10** | `data/uci_child.csv` | $292 \to 262$ | 23,270 | `aff0928863f6c930d4102bda422fb46b4a337356ef431ddfacb3bd07925f2aa2` | UCI Repository #419 (Thabtah 2017) |
| **Toddler Q-CHAT-10** | `data/toddler_qchat10.csv` | $1,054 \to 850$ | 70,146 | `818d604290cdae767f40a1bfcc2ca5634f1ba9ed1c31bfd41aa4b450943d973f` | Kaggle / Thabtah (2018) |
| **ABIDE I CC200 Cache** | `data/abide_cache.npz` | $120$ (4 sites) | 34,656,831 | `33e743f746e0fe883f7401eb0f7b2639bc3f85af88c997e786332d7f3ef1dfbc` | PCP / Nilearn (Di Martino et al. 2014) |

---

## 6. Project & Folder Structure

```text
asd-project/
├── app/                           # Streamlit multi-page research platform
│   ├── Home.py                    # Main overview, thesis, and headline metrics
│   └── pages/                     # Interactive exploratory modules
│       ├── 1_🔍_Tautology_Audit.py
│       ├── 2_🏆_Benchmark_Leaderboard.py
│       ├── 3_🧲_Scanner_Generalization.py
│       ├── 4_🎯_Uncertainty_Lab.py
│       ├── 5_🖊_Screening_Simulator.py
│       ├── 6_🔬_Reproducibility.py
│       └── 7_✅_Audit_Checklist.py
├── configs/                       # Pinned YAML experiment configurations
│   ├── main.yaml                  # Global reproducibility seeds, paths, CV
│   └── models.yaml                # Model hyperparameter search grids
├── data/                          # Raw and processed clinical & neuroimaging data
│   ├── processed/                 # Deduplicated, stratified 80/20 train/test CSVs
│   ├── abide/                     # ABIDE CC200 .1D resting-state timeseries
│   └── abide_cache.npz            # Verified 120-subject CC200 correlation cache
├── docs/                          # Audit templates and demonstration scripts
│   ├── lit_audit_template.csv     # Empty literature meta-audit schema
│   └── DEMO_SCRIPT.md             # 5-minute faculty presentation order
├── figures/                       # Publication-quality figures (vector PDF + 300-dpi PNG)
│   ├── pipeline_schematic.[pdf/png]
│   ├── tautology_ablation.[pdf/png]
│   ├── loso_gap.[pdf/png]
│   ├── site_probe_confounds.[pdf/png]
│   ├── harmonization_comparison.[pdf/png]
│   ├── reliability_diagrams.[pdf/png]
│   ├── conformal_coverage_per_site.[pdf/png]
│   └── risk_coverage_curves.[pdf/png]
├── results/                       # Empirical artifacts & single sources of truth
│   ├── paper_facts.json           # Machine-readable ledger of all verified values
│   ├── SUMMARY.md                 # Executive research summary & finding tables
│   ├── SUSPICIOUS.md              # Diagnostics for suspicious/unexpected results
│   ├── RUN_LOG.md                 # Complete chronological execution history
│   ├── tautology_audit.json       # Module 1 empirical audit data
│   ├── site_analysis.json         # Module 3 scanner & harmonization results
│   ├── uncertainty_main.json      # Module 4 calibration & conformal results
│   └── tables/                    # Exported LaTeX (.tex) and CSV (.csv) tables
├── src/                           # Core research Python modules
│   ├── audit/                     # Module 1: Tautology audit & feature ablation
│   ├── data/                      # Ingestion, cleaning, and deduplication
│   ├── eval/                      # Module 3: Site generalization, LOSO, ComBat
│   ├── uncertainty/               # Module 4: Calibration, conformal, selective prediction
│   └── viz/                       # Vector figure generation
├── tests/                         # Automated test suite (32/32 tests passing)
│   ├── test_audit_extension.py    # Quality gates: isolation, conformal, figures
│   ├── test_phase1.py             # Zero leakage & train/test isolation
│   ├── test_phase2.py             # Non-diagnostic phrasing & detector checks
│   ├── test_phase3.py             # PWA assets & PSI drift monitoring
│   └── test_phase4_5.py           # Deep lab parity & scientific safeguards
├── webapp/                        # Client-side offline browser PWA
├── AUDIT.md                       # Comprehensive methodological audit inventory
├── PROGRESS.md                    # Live checkpointing and phase tracker
├── run_all.py                     # Master execution script (Phases 1-5)
├── run_extended.py                # Extended audit orchestrator (Modules 1-6)
├── Makefile                       # One-command execution targets
├── Dockerfile                     # Reproducible container deployment
├── requirements.txt               # Pinned Python package dependencies (UTF-8)
└── README.md                      # This comprehensive documentation
```

---

## 7. Automated Quality Gates & Verification

Run the complete 32-test verification suite:
```powershell
python -m pytest tests/ -v
```
All **32 tests** verify:
1. **Zero Overlap & Isolation (8 tests):** Strictly 0% train/test overlap, in-fold scaler and imputer fitting, in-fold ComBat site parameter isolation.
2. **Scientific Safeguards (8 tests):** Label-shuffle permutation collapses to chance ($45.8\% - 50.0\%$), scanner hardware neutralization drop ($94.2\% \to 41.7\%$), motion confound decorrelation ($|r| < 0.15$), age confound decorrelation ($|r| < 0.25$).
3. **Uncertainty & Conformal Validity (4 tests):** Marginal coverage $\ge 1 - \alpha$ on validation splits, ECE non-zero monotonic calibration, selective prediction risk monotonicity.
4. **Reproducibility & Publication Schema (6 tests):** All 8 publication figures exist in both PDF and PNG format; all 5 main tables exist in both `.tex` and `.csv`; `results/paper_facts.json` schema validation.
5. **Software & PWA Integrity (6 tests):** Client-side PWA manifest, service worker caching, browser parameter vector distillation footprint ($< 10$ KB), JavaScript syntax integrity, ethical warning banners.

---

## 8. License, Data Use Acknowledgments, & Citation

### Data Use Acknowledgments
- **ABIDE I:** Autism Brain Imaging Data Exchange (PCP release). We acknowledge the contributing clinical sites (OHSU, OLIN, PITT, SDSU) and consortium investigators (Di Martino et al., 2014; Craddock et al., 2013). Data usage follows the 1000 Functional Connectomes Project terms.
- **UCI Screening Datasets:** Dr. Fadi Thabtah (Adult ID 426, Adolescent ID 420, Child ID 419) and Kaggle Toddler Q-CHAT-10. Usage follows Creative Commons Attribution 4.0 International (CC BY 4.0).

### Ethical & Regulatory Disclaimer
> This software is an academic research prototype and preliminary methodology auditor. It is **NOT** a Medical Device (SaMD), does not provide medical advice, and is not cleared or approved by the FDA, EMA, or any regulatory body. It strictly evaluates statistical properties of machine learning models and must never be used for formal diagnostic determinations.

### Citation
```bibtex
@article{autism_screening_audit_2026,
  title={A Methodological Audit and Reproducible Benchmark of Machine Learning for Autism Screening: Behavioral Tautologies, Scanner Generalization, and Uncertainty-Aware Decision Support},
  author={Research Team},
  journal={arXiv preprint},
  year={2026}
}
```
