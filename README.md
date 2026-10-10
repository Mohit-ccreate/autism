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

### ABIDE I Neuroimaging Benchmarks (120 Subjects across 4 Clinical Sites, Craddock CC200 Atlas)

| Architecture / Model | Evaluation Protocol | Accuracy [95% CI] | ROC-AUC | Balanced Acc | Scientific & Clinical Findings |
|---|---|---|---|---|---|
| **Logistic Regression (L2) + ComBat** | **Leave-One-Site-Out (LOSO) CV** | **55.00%** [47.0% – 63.0%] | **0.6098** | **54.67%** | **Headline Baseline:** Simple regularized boundary generalizes best to unseen scanners. |
| **Brain Network Transformer (BNT)** | **Leave-One-Site-Out (LOSO) CV** | **45.83%** [38.0% – 54.0%] | 0.4883 | 45.86% | Deep attention over 200 ROIs overfits sample noise in the $p \gg n$ regime. |
| **Connectivity ViT** | **Leave-One-Site-Out (LOSO) CV** | **51.67%** [43.0% – 60.0%] | 0.4964 | 51.40% | Non-local 14x14 patch attention; matrix geometry lacks translation invariance. |
| **Small 2D CNN** | **Leave-One-Site-Out (LOSO) CV** | **51.67%** [43.0% – 60.0%] | 0.5612 | 51.80% | Spatial grid assumption violates atlas permutation invariance. |
| **BrainGAT (Graph Attention)** | **Leave-One-Site-Out (LOSO) CV** | **51.67%** [43.0% – 60.0%] | 0.5584 | 51.60% | Graph attention on sparse top positive correlation edges. |
| **Logistic Regression (Random CV)** | 5-Fold Random Split | 58.33% [50.0% – 66.0%] | 0.6424 | 58.10% | **Inflated:** Scanner hardware signatures leak across train and test folds. |
| **BNT (Random CV)** | 5-Fold Random Split | 68.33% [59.0% – 77.0%] | 0.7120 | 67.80% | **Severe Leakage Exploitation:** Deep model memorizes scanner hardware fingerprints. |

### Scientific Safeguards Verification Matrix

| Safeguard Protocol | Target Tested | Real Value | Target Threshold | Status |
|---|---|---|---|---|
| **Label-Shuffle Permutation** | Out-of-fold cross-validation accuracy on permuted labels | **45.83%** | Must collapse to chance (50% $\pm$ 8%) | **PASS** |
| **Scanner Hardware Neutralization** | Site predictability before vs after ComBat | **94.2% → 41.7%** ($\Delta = -52.5\%$) | Predictability must drop significantly | **PASS** |
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

### Step 6: Open the Web Application
Start the local web server:
```powershell
python -m http.server 8000 --directory webapp
```
Then open your web browser to:
👉 **[http://localhost:8000](http://localhost:8000)**

*(Alternatively, you can directly double-click `webapp/index.html` to open it in any modern browser!)*

---

## 5. Web Application Feature Map (All 5 Phases)

The web application includes 15 distinct clinical and research tabs:
1. **Screening Questionnaire (`#tab-screening`):** Step-by-step 10-item cards for Toddler, Child, Adolescent, and Adult cohorts, with picture cues, rapid keyboard shortcuts (`1`/`0`, `y`/`n`), and real-time 3D brain mesh.
2. **Results & Explainability (`#tab-results`):** Calibrated risk gauge with 95% bootstrap CI, 90% conformal uncertainty state, counterfactual flips (*"what would change this result"*), model confidence breakdown, and printable appointment PDF.
3. **Audience Perspectives:** Toggle between **Adult Self-Report**, **Parent / Caregiver**, and **Clinician / Specialist** interpretations.
4. **Strengths & Support Profile (`#tab-strengths`):** Interactive sensory, communication, and cognitive reflection tool generating a shareable **"About Me" Profile Card** for school, workplace, or doctor appointments.
5. **Private Local History (`#tab-history`):** 100% client-side, zero-telemetry assessment timeline with one-click JSON/CSV export, individual deletion, and complete data wipe.
6. **Evidence-Based Learning Hub (`#tab-learning`):** Plain-language guides citing WHO ICD-11, NICE Guidelines (CG142/CG170), and peer-reviewed autism literature.
7. **Clinician & Researcher Mode (`#tab-clinician`):** Cohort triage risk distribution histogram (Chart.js) across 5 strata and downloadable Clinician Batch Summary & Model Governance PDF (`Clinical_Batch_Triage_Summary.pdf`).
8. **3D Neural Attention Explorer (`#tab-brain-explorer`):** Interactive Three.js 3D brain mesh with 200 CC200 ROI nodes. Hover over key Brain Network Transformer attention hubs (PCC, mPFC, STG, Insula, dlPFC, Fusiform, Precuneus) to highlight connections and Yeo networks. Features prominent ethical warning: *"WHAT THE MODEL USED, NOT WHERE AUTISM IS"*.
9. **Experiment Arena (`#tab-arena`):** Interactive toggle comparing **Honest LOSO CV** against **Inflated Random CV**, visually demonstrating the impact of scanner hardware leakage.
10. **Model vs. Model Disagreement Inspector (`#tab-disagreement`):** Interactive pairwise comparator (LR vs BNT, LR vs BrainGAT, BNT vs BrainGAT) showing concordance rates (75.0% – 82.5%) and case-level conflict analysis.
11. **Self-Supervised Data-Efficiency Curve (`#tab-efficiency`):** Dynamic Chart.js line plot illustrating model performance from scratch vs. self-supervised pretraining across sample fractions (N=30 to N=120).
12. **Robustness & Scientific Safeguards Matrix (`#tab-robustness`):** Verification matrix documenting missing modality dropout resilience, 2.4 KB browser distillation, and all 4 passed scientific safeguards.
13. **"What We Tried That Didn't Work" (`#tab-negative`):** Scientific negative findings gallery explaining why 2D CNNs, small-cohort deep transformers, and random splits fail.
14. **In-Browser Dataset Leakage Detector (`#tab-detector`):** Drag-and-drop any screening dataset CSV into the browser to check for total score leakage and deterministic threshold traps.
15. **Honest Leaderboard & Model Transparency (`#tab-leaderboard`, `#tab-transparency`):** Real-time metrics feed from `metrics.json` and subgroup fairness audits.

---

## 6. Project & Folder Structure

```text
asd-project/
├── data/                          # Raw and processed datasets
│   ├── processed/                 # Cleaned, deduplicated train/test CSVs
│   ├── abide/                     # ABIDE I CC200 .1D resting-state time series
│   ├── abide_cache.npz            # Cached 120-subject CC200 matrices and metadata
│   ├── uci_adult.csv              # Adult AQ-10 screening records
│   ├── uci_adolescent.csv         # Adolescent AQ-10 screening records
│   ├── uci_child.csv              # Child AQ-10 screening records
│   └── toddler_qchat10.csv        # Toddler Q-CHAT-10 screening records
├── src/                           # Production Python pipeline modules
│   ├── data_ingestion.py          # Automated downloading with SHA-256 checks
│   ├── data_preprocessing.py      # Cleaning, deduplication, and stratified splits
│   ├── train_questionnaire.py     # Model training, Optuna tuning, and calibration
│   ├── train_abide.py             # ABIDE connectivity, ComBat, and LOSO CV
│   ├── export_metrics.py          # Compiles metrics.json and markdown reports
│   ├── leakage_check.py           # Standalone dataset leakage & integrity auditor
│   ├── drift_detector.py          # Population Stability Index (PSI) drift monitoring
│   ├── research_lab.py            # Phase 4 deep architectures, SSL, & safeguards
│   └── export_paper_draft.py      # Phase 5 scientific paper draft & summary compiler
├── notebooks/                     # Interactive research notebooks
│   └── Phase4_Deep_Research_Lab.ipynb # Standalone Google Colab GPU notebook (12 cells)
├── models/                        # Exported models & browser parameters
│   ├── models_export.json         # Client-side parameter vectors for the browser
│   └── *.onnx                     # ONNX exported model graphs
├── reports/                       # Scientific documentation & single source of truth
│   ├── metrics.json               # Single Source of Truth for Phase 1-3 metrics
│   ├── lab_metrics.json           # Single Source of Truth for Phase 4 research metrics
│   ├── paper_draft.md             # Peer-reviewed journal pre-print draft
│   ├── v2_summary.md              # Executive research summary & system audit
│   ├── model_card.md              # Full clinical AI model card (Mitchell et al.)
│   ├── honest_leaderboard.md      # Comprehensive model rankings
│   ├── comparison.md              # Comparison with published literature
│   ├── leakage_investigation.md   # Mathematical proof of target threshold trap
│   ├── licenses.md                # Dataset, questionnaire, and software licenses
│   ├── data_sources.md            # Data lineage, row counts, and checksums
│   └── plots/                     # Calibration curves and SHAP importance figures
├── webapp/                        # Self-contained browser application (PWA)
│   ├── index.html                 # Accessible, futuristic glassmorphic UI
│   ├── style.css                  # Dark glassmorphism, responsive, sensory-friendly
│   ├── app.js                     # 3D brain, client-side inference, PDF export, assistant
│   ├── manifest.json              # PWA installation manifest
│   ├── sw.js                      # PWA offline service worker caching
│   ├── icon-192.png               # 192x192 PWA application icon
│   ├── icon-512.png               # 512x512 PWA application icon
│   ├── metrics.json               # Live metrics feed
│   ├── lab_metrics.json           # Live research lab metrics feed
│   ├── models_export.json         # Trained parameters for client scoring
│   └── vendor/                    # Three.js, Chart.js, GSAP, jsPDF (offline-ready)
├── tests/                         # Automated test suite (24/24 passing)
│   ├── test_phase1.py             # Zero leakage, zero overlap, metric truth
│   ├── test_phase2.py             # Non-diagnosis wording, detector, help links, risk calibration
│   ├── test_phase3.py             # PWA assets, PSI drift, JS syntax, assistant
│   └── test_phase4_5.py           # Deep lab parity, safeguards, paper, Colab nb
├── run_all.py                     # Master one-command execution script
├── requirements.txt               # Pinned Python package dependencies
├── CHANGELOG.md                   # Running log of changes across all phases
└── README.md                      # This beginner-friendly documentation
```

---

## 7. How to Run the Verification Tests

To run the automated scientific and software integrity checks across all 5 phases:
```powershell
python -m pytest tests/ -v
```
All **24 tests** verify:
- **Phase 1 Tests (5 tests):** Confirms no total score column exists in feature sets, 0% train/test overlap, metric traceability, honest ABIDE LOSO scores, and local offline vendor libraries.
- **Phase 2 Tests (5 tests):** Confirms strict absence of definitive diagnostic assertions, accurate dataset leakage detection, UI numbers tracing to `metrics.json`, help directory verification dates (Sep 2026), and non-zero monotonic clinical screening risk calibration.
- **Phase 3 Tests (6 tests):** Confirms PWA manifest and service worker validity, PSI drift calculation accuracy for stable vs drifted cohorts, webapp JavaScript syntax via `node --check`, Grounded Assistant crisis safety rules and diagnostic refusals, and DOM elements in `index.html`.
- **Phase 4 & 5 Tests (8 tests):** Confirms parity between `reports/` and `webapp/` `lab_metrics.json`, label-shuffle collapse to chance (45.83%), scanner neutralization drop, head motion and age confound controls, honest deep model LOSO constraints (<60%), multimodal graceful degradation, browser distillation footprint (<10 KB), paper draft and summary generation, Colab GPU notebook structure, and ethical warning banners.
