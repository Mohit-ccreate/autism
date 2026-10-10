# Project V2 Scientific Research Summary & System Audit

**Repository:** `asd-project`  
**Platform:** NEURO-SCREEN AI (Honest Multi-Modal Clinical Screening Research Platform)  
**Verification Date:** 2026-09-30  
**Status:** All 5 Phases Fully Implemented & Verified  

---

## Executive Summary

This platform represents a complete, transparent, and honest implementation of clinical-AI research for Autism Spectrum Disorder (ASD). It addresses the systemic flaws in published literature by eliminating data leakage, incorporating Leave-One-Site-Out cross-validation, applying Platt probability calibration, providing conformal prediction ambiguity bounds, and delivering a 100% private, client-side browser application with multi-modal accessibility.

---

## Key Performance Findings Across All Phases

### 1. Questionnaire Screening (Untouched 20% Test Sets)
- **Adult (AQ-10) Abbreviated 5 Items:** Accuracy = **93.28%**, ROC-AUC = **0.9739**, Sensitivity = 82.86%, Specificity = 96.97%.
- **Toddler (Q-CHAT-10) Abbreviated 5 Items:** Accuracy = **88.24%**, ROC-AUC = **0.9352**, Sensitivity = 91.94%, Specificity = 78.26%.
- **Unified Cross-Age Model:** Accuracy = **85.37%**, ROC-AUC = **0.9379**.
- **Negative Control (Demographics Only):** Accuracy = 74.63%, ROC-AUC = 0.6837.
- **10-Item Full Questionnaire:** Replicates the deterministic cutoff rule ($\sum A_i \ge 7 \iff 	ext{Yes}$), proving papers claiming 100% accuracy fit an adder rather than clinical ASD patterns.

### 2. ABIDE I Neuroimaging (120 Subjects across 4 Clinical Centers)
- **Honest Leave-One-Site-Out (LOSO) CV + ComBat:** Logistic Regression Accuracy = **55.00%**, ROC-AUC = **0.6098**.
- **Inflated 5-Fold Random Split Baseline:** Accuracy = **58.33%**, ROC-AUC = **0.6424** (Scanner site leakage across folds).
- **Brain Network Transformer (BNT):** Accuracy = **45.83%**, ROC-AUC = **0.4883**.
- **Vision Transformer (ViT):** Accuracy = **51.67%**.
- **Small 2D CNN:** Accuracy = **51.67%**.
- **Brain Graph Attention Network (BrainGAT):** Accuracy = **51.67%**, ROC-AUC = **0.5584**.

### 3. Scientific Safeguards
- **Permutation Test:** Out-of-fold accuracy on shuffled labels collapses to chance (**45.83%**).
- **Scanner Site Neutralization:** Site predictability drops from **94.2%** to **41.7%** after ComBat harmonization.
- **Motion Artifact Check:** Predicted likelihood correlation with head motion framewise displacement is negligible ($r = -0.1399$).

---

## What We Tried That Didn't Work

In accordance with scientific transparency guidelines, we document experiments, architectures, and design approaches that proved ineffective or scientifically invalid:
1. **2D CNNs on Raw Connectivity Matrices:** 2D CNN kernels assume spatial grid translation invariance. Brain atlas regions have arbitrary ordering; permuting ROI rows changes CNN output while representing the exact same connectome.
2. **Deep Overparameterized Transformers on Small Cohorts:** The Brain Network Transformer (BNT) achieved 45.83% accuracy under LOSO CV compared to 55.00% for regularized Logistic Regression, demonstrating that high-capacity attention models overfit sample noise in the $p \gg n$ regime.
3. **Evaluating Permutation Tests In-Sample:** In-sample evaluation on permuted labels still fits sample-specific noise (70%+ accuracy), while out-of-fold cross-validation correctly collapses to chance (45.83%).
4. **Random Train/Test Splits on Multi-Center fMRI:** Random K-Fold CV leaks scanner hardware artifacts across folds, inflating accuracy to 58.3% - 68.3% while failing on unseen scanners.

---

## System Architecture

```text
asd-project/
├── data/                      # Cleaned, deduplicated train/test CSVs & ABIDE CC200 cache
├── src/
│   ├── data_ingestion.py      # Automated data downloader with SHA-256 checks
│   ├── data_preprocessing.py  # Deduplication and stratified partitions
│   ├── train_questionnaire.py # Optuna tuning, Platt calibration, conformal bounds
│   ├── train_abide.py         # CC200 functional connectivity & ComBat harmonization
│   ├── leakage_check.py       # Standalone dataset leakage & integrity detector
│   ├── drift_detector.py      # Population Stability Index (PSI) drift engine
│   ├── research_lab.py        # Deep architectures, SSL pretraining & safeguards
│   └── export_paper_draft.py  # Dynamic paper draft & executive summary compiler
├── notebooks/
│   └── Phase4_Deep_Research_Lab.ipynb # Standalone Google Colab GPU notebook
├── reports/
│   ├── metrics.json           # Single source of truth for Phase 1-3 metrics
│   ├── lab_metrics.json       # Single source of truth for Phase 4 research lab metrics
│   ├── model_card.md          # Full clinical AI model card
│   ├── honest_leaderboard.md  # Comprehensive model comparison table
│   ├── comparison.md          # Comparison against published literature
│   ├── leakage_investigation.md # Mathematical proof of target threshold trap
│   ├── paper_draft.md         # Journal pre-print publication draft
│   └── v2_summary.md          # This executive research summary
├── webapp/                    # 100% offline client-side web application (PWA)
└── tests/                     # Automated test suite (Phase 1, 2, 3, 4, 5)
```
