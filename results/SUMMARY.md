# SUMMARY.md — Executive Scientific Summary of Methodological Revision Pass

**Project:** A Methodological Audit and Reproducible Benchmark of Machine Learning for Autism Screening: Behavioral Tautologies, Scanner Generalization, and Uncertainty-Aware Decision Support  
**Branch:** `paper-revision`  
**Revision Date:** 2026-11-17T02:51:45.563815+00:00  
**Evaluation Standard:** Zero hardcoded numbers. All facts trace directly to `results/*.json`.

---

## 1. Executive Scientific Verdicts

| Research Component | Scientific Verdict | Key Empirical Evidence | Sample Size (N) | 95% Confidence Interval / p-value |
|---|---|---|---|---|
| **1. Behavioral Tautology Thesis** | **SUPPORTED** | 100.0% concordance with diagnostic threshold rule across all cohorts. Dropping leaking items collapses AUROC from 1.000 to near chance (0.45–0.58). | N=1,876 clean (Adult 667, Ado 97, Child 262, Toddler 850) | Adult Items: 0.9999 vs Clean: 0.5801; TI Raw: 0.8398 [0.797, 0.884] |
| **2. Scanner Generalization Gap** | **SUPPORTED** | Connectomes decode site with 73.54% accuracy (vs 19.91% majority baseline) even after in-fold confound regression. Random CV optimistically inflates performance over honest LOSO. | N=854 (20 clinical sites) | Random CV: 0.7420 vs LOSO: 0.6593 (Gap: +0.0827) |
| **3. Leaky ComBat Inflation** | **NOT SUPPORTED** | Paired bootstrap test between Leaky ComBat and In-Fold ComBat shows no statistically detectable inflation at full sample size. | N=854 (20 clinical sites) | Delta = +0.0096 [-0.0071, 0.0260], $p_{\text{holm}} = 1.0000$ (CI includes 0) |
| **4. Uncertainty & Conformal Decision Support** | **PARTIALLY SUPPORTED** | Conformal sets provide marginal coverage across allocation versions (87.0%–89.7%), but selective abstention does not eliminate errors and scanner shift induces site coverage variability. | N=854 neuroimaging, N=1,876 tabular | LOSO Conformal Coverage: 87.00% (Shared) / 89.58% (Disjoint) / 89.70% (Site-Grouped), Mean set size: 1.595–1.664 |

---

## 2. Root Cause Analysis of Reviewer-Flagged Consistency Errors

### 1. Tautology Index vs. Demographics AUROC
- **Reviewer Flag:** Demographics AUROC in Table IV (0.5801, 0.5180, 0.4473, 0.5527) differed from earlier clean AUROCs (0.5910, 0.4846, 0.4460, 0.5558) that produced TI values (0.818, 1.031, 1.108, 0.888).
- **Root Cause:** Table IV demographics AUROC was computed from a 5-seed 5-fold CV run in `benchmark.py`, whereas TI values were computed by an earlier 3-seed run in `tautology_audit.py`.
- **Resolution:** Recomputed in **ONE unified run** from **ONE saved set of out-of-fold predictions** across all 5 seeds ($5 \times 5 = 25$ evaluations).
  * Adult: Items AUROC = 0.9999, Clean AUROC = 0.5801, Items Acc = 99.55% [99.31, 99.76], Majority = 73.76%, TI Raw = 0.8398 [0.797, 0.884], Clipped = 0.8398, Abs Drop = 0.4199.
  * Adolescent: Items AUROC = 0.9946, Clean AUROC = 0.5180, Items Acc = 91.96% [89.48, 94.23] (Repeated $5\times 10$ Acc: 95.55% [84.21, 100.00]), Majority = 57.73%, TI Raw = 0.9637 [0.862, 1.069], Clipped = 0.9637, Abs Drop = 0.4767.
  * Child: Items AUROC = 1.0000, Clean AUROC = 0.4473, Items Acc = 99.54% [99.16, 99.85], Majority = 54.58%, TI Raw = 1.1054 [1.044, 1.168], Clipped = 1.0000, Abs Drop = 0.5526. **Instability identified:** Clean AUROC (0.4473) is below chance (0.50), causing numerator $(1.0 - 0.4473 = 0.5527)$ to exceed denominator $(1.0 - 0.50 = 0.5000)$.
  * Toddler: Items AUROC = 1.0000, Clean AUROC = 0.5527, Items Acc = 100.00% [100.00, 100.00], Majority = 72.71%, TI Raw = 0.8946 [0.858, 0.932], Clipped = 0.8946, Abs Drop = 0.4473.

### 2. Leaky vs. In-Fold ComBat Paired Difference
- **Reviewer Flag:** Displayed AUROCs differed by 0.0096 (0.6689 - 0.6593), but the paper reported paired difference was 0.0156.
- **Root Cause:** In `site_analysis.py`, `run_paired_harmonization_tests` passed `meta_df["target"].values` in original dataframe row order, whereas `predictions_map` had predictions sorted site-by-site, causing paired bootstrap deltas to be calculated with misaligned labels.
- **Resolution:** Re-aligned label ordering identically. The direct subtraction ($0.6689 - 0.6593 = 0.0096$) and paired bootstrap delta match exactly: $\Delta = +0.0096$ [95% CI: $-0.0071, 0.0260$], raw $p = 0.2620$, Holm-adjusted $p = 1.0000$.

### 3. Permutation Test p-Value Resolution
- **Reviewer Flag:** Permutation test p-value was previously reported as $p = 0.0000$, below the finite resolution limit.
- **Root Cause:** Calculated as `mean(null >= obs)` without continuity correction.
- **Resolution:** Rerun with $B = 10,000$ within-site label permutations. With $0$ exceedances, $p = (1 + 0) / (1 + 10,000) = 0.00010$. The minimum resolution limit $1/(1+B) = 0.00010$ is strictly respected everywhere in results and tables.

### 4. Logistic Regression Accuracy Claim
- **Reviewer Flag:** Text claimed LR achieved 99.4--100% accuracy, omitting Adolescent cohort.
- **Root Cause:** Adolescent repeated $5\times 10$ CV accuracy was 95.55% $\pm$ 5.47% (95% CI [84.21%, 100.00%]), which was omitted from the summary range.
- **Resolution:** Updated text to report the accurate per-cohort range: **95.55% to 100.00%**.

---

## 3. ComBat Protocol and Uncertainty Data Allocation

### ComBat Unseen-Site Protocol
- **Code Inspection:** `parametric_combat_transform` is algebraically the identity on unseen test sites ($Z \cdot \sigma + \hat{X} \equiv X$). In-fold ComBat harmonizes training sites and leaves unseen test sites raw.
- **Held-Out Site Influence:** In strictly inductive ComBat, **zero held-out site data** is used. In transductive ComBat, the test site's own unlabeled feature mean and variance are used for own-site standardization.

### Uncertainty Data Allocation
- **Shared vs Disjoint Subsets:** Evaluated Version 1 (Shared 2-way calibration), Version 2 (Three strictly DISJOINT subsets: Proper-Train 60%, Calibrator 20%, Conformal-Cal 20%), and Version 3 (Site-Grouped LOSO where 3 whole sites are held out for conformal calibration).
- **Guarantees:** Conformal marginal coverage holds across all three allocation protocols (87.00% Shared, 89.58% Disjoint, 89.70% Site-Grouped).
- **Set-Size Identity:** The fundamental set-size decomposition identity holds with residual $< 10^{-14}$.

---

## 4. Pooled vs. Within-Site Performance & Controlled Site Probe

- **Site-Prevalence Oracle:** Scoring subjects purely with their site's true ASD prevalence (using zero imaging features) achieves an AUROC of **0.5902**.
- **Within-Site Z-Scoring:** Z-scoring predictions within each held-out site preserves or improves performance (LR: 0.6593 $\to$ 0.6663; Linear SVM: 0.6765 $\to$ 0.6931; Random Forest: 0.7003 $\to$ 0.7222), proving genuine ranking signal within sites.
- **Controlled Site Probe:** Phenotypic variables alone decode site with 22.13% (LR) and 27.17% (RF) accuracy vs 19.91% majority baseline. Connectomes after in-fold confound regression achieve 73.54% site decoding accuracy (balanced acc 64.77%). This confirms site predictability reflects broad site-associated information (acquisition parameters, RF coils, recruitment protocols) rather than hardware signatures alone.

---

**Medical Disclaimer:** This codebase and all associated findings constitute a methodological research prototype. It is NOT a diagnostic device and makes no clinical diagnosis claims.
