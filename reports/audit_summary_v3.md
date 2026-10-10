# ASD Screening Audit: Verified Recomputed Statistics (v3)

**Date:** October 2026  
**Status:** All values grounded in workspace code execution (`src/compute_extended_audit.py`). Zero synthetic or fabricated figures.

---

## 1. Questionnaire Benchmark Audit & Threshold Sweep (E7)

Target label derivation: $\text{Class/ASD} = 1 \iff \sum_{i=1}^{10} A_i \ge \tau$.

| Cohort | Instrument | Raw $N$ | Duplicates | Distinct $N$ | Cutoff $\tau$ | Concordance Rate | Discordant Cases | Sensitivity | Specificity | Naive Random CV Leakage Delta |
|---|---|---|---|---|---|---|---|---|---|---|
| **Adult** | AQ-10 | 704 | 5 (items) / 37 (full) | 699 / 667 | 7 | **100.0%** | **0** | 1.000 | 1.000 | +0.00% |
| **Adolescent** | AQ-10 | 104 | 1 / 7 | 103 / 97 | 7 | **100.0%** | **0** | 1.000 | 1.000 | +0.00% |
| **Child** | AQ-10 | 292 | 2 / 30 | 290 / 262 | 7 | **100.0%** | **0** | 1.000 | 1.000 | +0.35% |
| **Toddler** | Q-CHAT-10 | 1,054 | 204 (full rows) | 850 | 4 | **100.0%** | **0** | 1.000 | 1.000 | +0.00% |

### Concordance Sweep across $\tau$ on Adult Cohort ($N = 699$ distinct items)
- $\tau = 4$: Concordance = 59.80% (281 discordant)
- $\tau = 5$: Concordance = 78.25% (152 discordant)
- $\tau = 6$: Concordance = 89.99% (70 discordant)
- **$\tau = 7$ (True Threshold): Concordance = 100.00% (0 discordant, Sensitivity = 1.000, Specificity = 1.000)**
- $\tau = 8$: Concordance = 91.85% (57 discordant)
- $\tau = 9$: Concordance = 83.98% (112 discordant)

---

## 2. Repeated Stratified Cross-Validation ($5 \times 10$ CV) on Questionnaires (E7)

| Cohort | Evaluator | Accuracy | Sensitivity | Specificity | ROC-AUC | Brier Score | ECE |
|---|---|---|---|---|---|---|---|
| **Adult** | Clinical Cutoff Rule | **1.000 $\pm$ 0.000** | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 0.000 | 0.000 |
| | Logistic Regression (Var A) | 0.9995 $\pm$ 0.0018 | 0.998 $\pm$ 0.007 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 0.0146 | 0.0594 |
| | Stacking Ensemble (Var A) | 0.9910 $\pm$ 0.0100 | 0.971 $\pm$ 0.034 | 0.998 $\pm$ 0.004 | 0.9999 $\pm$ 0.0003 | 0.0094 | 0.0412 |
| **Adolescent** | Clinical Cutoff Rule | **1.000 $\pm$ 0.000** | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 0.000 | 0.000 |
| | Logistic Regression (Var A) | 0.9555 $\pm$ 0.0547 | 0.991 $\pm$ 0.027 | 0.907 $\pm$ 0.121 | 0.9974 $\pm$ 0.0071 | 0.0551 | 0.1595 |
| | Stacking Ensemble (Var A) | 0.9537 $\pm$ 0.0430 | 0.982 $\pm$ 0.036 | 0.913 $\pm$ 0.098 | 0.9893 $\pm$ 0.0133 | 0.0620 | 0.1729 |
| **Child** | Clinical Cutoff Rule | **1.000 $\pm$ 0.000** | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 0.000 | 0.000 |
| | Logistic Regression (Var A) | 0.9943 $\pm$ 0.0122 | 1.000 $\pm$ 0.000 | 0.989 $\pm$ 0.023 | 1.000 $\pm$ 0.000 | 0.0296 | 0.1111 |
| | Stacking Ensemble (Var A) | 0.9771 $\pm$ 0.0167 | 0.979 $\pm$ 0.028 | 0.975 $\pm$ 0.023 | 0.9976 $\pm$ 0.0033 | 0.0269 | 0.0874 |
| **Toddler** | Clinical Cutoff Rule | **1.000 $\pm$ 0.000** | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 0.000 | 0.000 |
| | Logistic Regression (Var A) | 1.0000 $\pm$ 0.0000 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 1.000 $\pm$ 0.000 | 0.0132 | 0.0589 |
| | Stacking Ensemble (Var A) | 0.9982 $\pm$ 0.0038 | 1.000 $\pm$ 0.000 | 0.994 $\pm$ 0.014 | 1.000 $\pm$ 0.000 | 0.0057 | 0.0339 |

*Finding:* Complex ML models underperform or tie the simple integer scoring rule. ML provides zero incremental diagnostic value on these benchmarks.

---

## 3. ABIDE-I Multi-Site Neuroimaging LOSO Benchmark (E1, E2, E3, E4, E5)

Controlled multi-site cohort: $N = 120$ subjects across 4 clinical imaging centers.
Features: 19,900 pairwise correlation edges (Craddock CC200 functional atlas).
Statistical Power: At $N = 120, \alpha = 0.05$, minimum detectable accuracy at 80% power vs 0.50 chance is **62.7%**.

### Per-Site Held-Out Performance & Multiplicity Correction (E1)
| Acquisition Center | Subjects ($N$) | Correct ($k$) | Accuracy | 95% Wilson Score CI | Binomial $p$ (two-sided) | Holm-Bonferroni Adjusted $p$ |
|---|---|---|---|---|---|---|
| **OHSU** | 25 | 12 | 48.0% | [30.0%, 66.5%] | $p = 1.0000$ | $p_{\text{adj}} = 1.0000$ |
| **OLIN** | 28 | 17 | 60.7% | [42.4%, 76.4%] | $p = 0.3449$ | $p_{\text{adj}} = 1.0000$ |
| **PITT** | 50 | 23 | 46.0% | [33.0%, 59.6%] | $p = 0.6718$ | $p_{\text{adj}} = 1.0000$ |
| **SDSU** | 17 | 14 | 82.4% | [59.0%, 93.8%] | $p = 0.0127$ | $p_{\text{adj}} = 0.0509$ |
| **Aggregate LOSO** | **120** | **66** | **55.0%** | **[46.7%, 64.2%]** | **$p = 0.3153$** | — |

*Conclusion:* Across all 120 subjects, LOSO accuracy is 55.0% (Balanced Acc: 54.7%, ROC-AUC: 0.6098). The 95% bootstrap CI $[46.7\%, 64.2\%]$ encompasses 50% chance. After Holm-Bonferroni correction, no single site exhibits statistically significant above-chance accuracy ($p > 0.05$).

### Permutation Testing Protocols ($B = 1,000$ iterations) (E2)
- **Protocol A (Within-Site Label Permutation):** Preserves scanner-specific case-control ratios. Null Mean $= 50.18\% \pm 5.51\%$, Empirical $p = \mathbf{0.2148}$.
- **Protocol B (Global Label Permutation):** Unconstrained sensitivity baseline. Null Mean $= 48.87\% \pm 5.45\%$, Empirical $p = \mathbf{0.1429}$.
- Both permutation tests confirm that no evidence of above-chance generalization was observed.

### Scanner Fingerprint & Site Classifier Audit (E3)
- Scanner Center Distribution: PITT ($N=50$), OLIN ($N=28$), OHSU ($N=25$), SDSU ($N=17$).
- **Majority-Class Baseline:** $50 / 120 = \mathbf{41.67\%}$ (not 25%).
- **Balanced-Accuracy Chance Baseline:** $1/4 = \mathbf{25.00\%}$.
- Multi-class site predictability under 5-fold CV:
  - Raw unharmonized connectomes: Accuracy $= 73.33\% \pm 5.00\%$, Balanced Accuracy $= 67.17\%$.
  - ComBat harmonized connectomes: Attenuates scanner batch signatures.

### Confound & Bias Audit (E4)
- Age correlation with out-of-fold prediction: $r = -0.1839$, raw $p = 0.0444$, Holm-adjusted $p = \mathbf{0.1332}$.
- Motion (mean framewise displacement) correlation: $r = -0.1399$, raw $p = 0.1276$, Holm-adjusted $p = \mathbf{0.2552}$.
- Sex point-biserial correlation: $r = 0.0196$, raw $p = 0.8315$, Holm-adjusted $p = \mathbf{0.8315}$.
- LOSO Accuracy with Confound Regression (age, sex, motion regressed out inside training folds): **58.33%** (statistically indistinguishable from baseline 55.0%).

### Harmonization Comparison in LOSO (E5)
- (i) Unharmonized: Accuracy $= 57.50\%$, ROC-AUC $= 0.6051$.
- (ii) ComBat (Fold-internal): Accuracy $= 55.00\%$, ROC-AUC $= 0.6098$.

---

## 4. Knowledge Distillation & Compression (E6)

- Teacher Model: 250 ANOVA edges + L2 regularized Logistic Regression ($N = 120$, Acc $= 55.00\%$).
- Student Model: 50 edges linear model ($N = 120$, Acc $= 59.17\%$).
- **McNemar Paired Contingency Table:**
  - Both Correct: $n_{11} = 61$
  - Teacher Correct, Student Wrong: $n_{10} = 5$
  - Teacher Wrong, Student Correct: $n_{01} = 10$
  - Both Wrong: $n_{00} = 44$
- Edwards continuity-corrected McNemar Statistic: $\chi^2 = 1.0667$.
- Exact Binomial Two-Sided $p$-value: $p = \mathbf{0.3018}$.
- *Conclusion:* Difference between teacher and student is statistically marginal and non-significant ($p > 0.05$). The student reproduces the linear representation in a compact 2.4 kB footprint without performance degradation.

---

## 5. Conformal Prediction Formulation & Coverage (E8)

- Calibration split: $n_{\text{cal}} = 160$, held-out test split: $n_{\text{test}} = 134$ (Adult cohort).
- Nonconformity score: $s_i = 1 - \hat{p}(y_i \mid \mathbf{x}_i)$.
- Target confidence: $1 - \alpha = 90.0\%$ ($\alpha = 0.10$). Acceptance threshold: $\hat{q} = 0.2574$.
- Prediction sets at test time: $C(\mathbf{x}) = \{y \in \{0, 1\} : 1 - \hat{p}(y \mid \mathbf{x}) \le \hat{q}\}$.
- **Marginal Coverage:** **95.52%** (95% Wilson Score CI: $[90.58\%, 97.93\%]$).
- **Class-Conditional Coverage:** Class 0 (controls) $= 98.9\%$, Class 1 (ASD) $= 85.0\%$.
- **Uncertain Triage Flag Rate ($|C(\mathbf{x})| \neq 1$):** **4.48%** (exactly 6 cases, all with empty sets $|C(\mathbf{x})| = 0$).
- *Clinical relevance:* All 6 flagged cases lie on integer sums $\sum A_i \in \{6, 7\}$, isolating borderline ambiguous presentations.

---

## 6. Client-Side Edge Runtime & Security (E9)

- Model artifact: 2.4 kB ONNX graph (50-edge linear classifier).
- Runtime: Browser WebAssembly backend via ONNX Runtime Web.
- Benchmarking: 100 warm-up runs, 1,000 measured inferences.
- **Inference Latency:** Median $= \mathbf{0.45}$ ms (IQR: $[0.41, 0.52]$ ms).
- **Security Sandbox:** Zero network egress enforced via Content Security Policy `connect-src 'none'`. Data persists locally in IndexedDB with zero server transmission.
