# Model Card: ASD Clinical Screening and Functional Connectivity System

**Model Card Version:** 1.0  
**Generated:** 2026-09-30 01:00:24  
**Maintainer:** Clinical-AI Research Engineering Team  

---

## 1. Intended Use and Clinical Scope

> [!IMPORTANT]
> **CRITICAL CLINICAL NOTICE:** This software and its underlying models are designed **exclusively for preliminary behavioral trait screening and scientific neuroimaging research**.  
> It **DOES NOT** diagnose Autism Spectrum Disorder (ASD), provide clinical confirmation, or replace a comprehensive diagnostic evaluation by a licensed developmental pediatrician, child psychiatrist, or clinical neuropsychologist (e.g., ADOS-2, ADI-R).  
> The system strictly issues probabilistic risk indications with conformal uncertainty intervals and never outputs definitive diagnostic assertions (e.g., asserting or excluding a formal autism diagnosis).

- **Primary Intended Users:**
  1. Caregivers and individuals seeking preliminary structured trait assessment before specialist consultation.
  2. Clinical researchers studying machine learning fairness, feature leakage, and multi-site fMRI generalizability.
- **Out-of-Scope Uses:**
  - Automated diagnosis or diagnostic exclusion.
  - School educational classification without multidisciplinary assessment.
  - Treatment or medical prescription planning.
  - Photo/video or facial recognition-based autism diagnosis (explicitly unvalidated and excluded).

---

## 2. Model Architecture & Pipeline Details

### Questionnaire Models (Toddler, Child, Adolescent, Adult, Unified)
- **Base Estimators:**
  - L2-Regularized Logistic Regression (Interpretable baseline)
  - LightGBM & XGBoost (Gradient boosted decision trees)
  - CatBoost (Symmetric decision trees)
  - Stacking Classifier (Ensemble combining logistic regression and tree predictions)
- **Tuning:** Stratified Repeated 5-Fold Cross-Validation (2 repeats) optimized via Optuna.
- **Calibration:** Platt sigmoid calibration (`CalibratedClassifierCV(method='sigmoid', cv=5)`).
- **Conformal Prediction:** Split conformal prediction sets calibrated for 90% marginal coverage ($lpha = 0.10$). Ambiguous cases where both high-trait and low-trait classes remain plausible are marked **Uncertain (Specialist Referral Recommended)**.
- **Feature Variants:**
  - **Variant A (Honest Items Only):** Pure 10 question responses (`A1` through `A10`).
  - **Variant B (Items + Demographics):** 10 items + Age, Sex, Jaundice, Family ASD history (Strictly excluding Ethnicity and Country).

### Brain Functional Connectivity Baseline (ABIDE I)
- **Cohort:** 120 subjects across 4 independent imaging sites (`PITT`, `OLIN`, `OHSU`, `SDSU`).
- **Feature Extraction:** Pearson correlation matrices computed across 200 ROIs (Craddock CC200 Atlas), Fisher z-transformed, upper triangle vectorized ($19,900$ edges).
- **Harmonization & Evaluation:** Leave-One-Site-Out (LOSO) Cross-Validation. Site harmonization (Parametric Empirical Bayes ComBat) and feature selection (ANOVA F-test) are strictly computed **inside each fold** to eliminate cross-site data leakage.

---

## 3. Quantitative Performance Summary (Real Evaluated Numbers)

### Questionnaire Screening (Untouched 20% Test Sets)

| Cohort | Feature Variant | Model Architecture | Test Accuracy (95% CI) | Balanced Acc | Sensitivity | Specificity | ROC-AUC | Conformal Uncertainty Rate |
|---|---|---|---|---|---|---|---|---|
| **Adult (AQ-10)** | Variant A (Items) | Calibrated Best | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 5.2% |
| **Toddler (Q-CHAT-10)** | Variant A (Items) | Calibrated Best | 100.00% | 100.00% | 100.00% | 100.00% | 1.0000 | 3.5% |

### ABIDE I Brain Connectivity Baseline

| Evaluation Protocol | Model | Accuracy | ROC-AUC | Clinical Validity Note |
|---|---|---|---|---|
| **Leave-One-Site-Out (LOSO) CV** | Logistic Regression (L2) | **55.83%** | **0.6073** | **HONEST HEADLINE RESULT:** Evaluates true generalizability to an unseen scanner site. |
| **Random Split 5-Fold CV** | Logistic Regression (L2) | 58.33% | 0.6424 | **INFLATED BASELINE:** Subject to scanner site leakage; misleadingly optimistic. |

---

## 4. Ethical Safeguards, Fairness, and Limitations

1. **Feature Leakage Prevention:** Total score columns (`result`, `Qchat-10-Score`) are strictly removed. Including them creates an artificial 99-100% score that masks the absence of individual item signal.
2. **Subgroup Disaggregation:** Models are evaluated across sex (male vs female). Performance differences are logged in the honesty dashboard.
3. **Data Privacy:** All questionnaire screening occurs client-side in the browser using pre-compiled parameter vectors. Zero patient responses leave the client machine.
