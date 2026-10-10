# CHANGELOG: Revision v3 -> v4 Mechanical Diff Ledger

This ledger details every metric, table cell, and methodological specification that differs between Revision v3 and Revision v4, generated directly from `results/*.json`, `reports/*.json`, and workspace code diffs.

---

## 1. Mechanical Diff Table (v3 vs. v4)

| Category / Table Cell | Revision v3 Value | Revision v4 Final Value | JSON Key / Log Source | Mechanical Reason for Difference |
|:---|:---|:---|:---|:---|
| **Adolescent LR Sensitivity** | 1.000 (single split) | 0.991 $\pm$ 0.027 | `v4_audit_pass1.json["f10_ml_grid"]["adolescent"]["Logistic Regression (Var A)"]["sens_mean"]` | Evaluated across repeated $5 \times 10$ Stratified CV (50 folds) on $N = 97$ deduplicated records instead of a single split. |
| **Adolescent LR Specificity** | 1.000 (single split) | 0.907 $\pm$ 0.121 | `v4_audit_pass1.json["f10_ml_grid"]["adolescent"]["Logistic Regression (Var A)"]["spec_mean"]` | Reflects true cross-validation variance across small negative class sample size in 50 folds. |
| **Adolescent LR Brier Score** | 0.0000 (single split) | 0.0551 | `v4_audit_pass1.json["f10_ml_grid"]["adolescent"]["Logistic Regression (Var A)"]["brier_mean"]` | Computed out-of-fold probability calibration error over 50 folds. |
| **Adult CatBoost Brier Score** | 0.0035 (single split) | 0.0596 | `v4_audit_pass1.json["f10_ml_grid"]["adult"]["CatBoost (Var A)"]["brier_mean"]` | $5 \times 10$ Stratified CV on $N = 667$ deduplicated adult records. |
| **Adult CatBoost ECE** | Unreported / 0.0000 | 0.1328 | `v4_audit_pass1.json["f10_ml_grid"]["adult"]["CatBoost (Var A)"]["ece_mean"]` | Expected Calibration Error formally audited across 50 folds. |
| **Toddler LightGBM Accuracy** | ~95.8\,\% (raw $N=1{,}054$) | 94.33 $\pm$ 1.97\,\% | `v4_audit_pass1.json["f10_ml_grid"]["toddler"]["LightGBM (Var A)"]["acc_mean"]` | Deduplication purged 204 identical records ($1,054 \to 850$), removing duplicate-induced test memorization. |
| **Toddler XGBoost Accuracy** | ~95.5\,\% (raw $N=1{,}054$) | 94.05 $\pm$ 1.95\,\% | `v4_audit_pass1.json["f10_ml_grid"]["toddler"]["XGBoost (Var A)"]["acc_mean"]` | Deduplication purged 204 duplicate records ($1,054 \to 850$). |
| **Toddler CatBoost Accuracy** | ~96.2\,\% (raw $N=1{,}054$) | 95.33 $\pm$ 1.75\,\% | `v4_audit_pass1.json["f10_ml_grid"]["toddler"]["CatBoost (Var A)"]["acc_mean"]` | Deduplication purged 204 duplicate records ($1,054 \to 850$). |
| **Adult Learning Curve Min $N$** | Unreported / 53 | $N = 42$ (94.15\,\%) | `v4_audit_pass1.json["f10_learning_curve"][0]` | Minimum fraction sweep calibrated on $N = 667$ adult cohort. |
| **fMRI Total Interval** | [89.55\,\%, 98.51\,\%] (conformal) | Wilson: [46.1\,\%, 63.6\,\%]; Bootstrap: [46.67\,\%, 64.17\,\%] | `extended_audit_metrics.json["loso_observed"]["pooled"]` | Corrected interval reporting: purged conformal coverage intervals from Table IV and properly separated Wilson and bootstrap CIs. |
| **fMRI Permutation SD (Within)** | 5.12\,\% (unseeded) | 5.51\,\% (seed 42) | `v4_amendments_output.json["a2_permutation"]["within_site_sd"]` | Rerun with logged seed 42 over $B = 1{,}000$ label permutations ($p = 0.2148$). |
| **fMRI Permutation SD (Global)** | 4.96\,\% (unseeded) | 5.45\,\% (seed 42) | `v4_amendments_output.json["a2_permutation"]["global_sd"]` | Rerun with logged seed 42 over $B = 1{,}000$ label permutations ($p = 0.1429$). |
| **Pre-ComBat Site Decoding Acc.** | 74.17\,\% (unseeded) | 73.33 $\pm$ 5.00\,\% | `v4_audit_pass1.json["f6_site_classifier"]["pre_combat_acc"]` | Evaluated under logged 5-fold cross-validation with seed 42 (balanced accuracy 67.17\,\%). |
| **Post-ComBat Site Decoding** | 0.00\,\% / Collapsed | DELETED from text | Per rule A1 | In-sample ComBat with site labels induces negative test covariance; artifactual 0.00% accuracy sentence deleted. |
| **Conformal Coverage Reporting** | 95.5\,\% (single split) | 89.8 $\pm$ 3.6\,\% marginal (94.4\,\% TD vs 76.6\,\% ASD) | `v4_amendments_output.json["a9_conformal_100_splits"]` | Expanded evaluation from a single representative split to 100 independent random splits. |
| **Edge Hardware / Latency Claims** | 0.45 ms on i7-11800H, Chrome v122 | DELETED; future work | Per rule E1 | `results/edge_benchmark.json` does not exist on disk; edge deployment left as future work. |
| **Abraham et al. Benchmark** | $N = 1,112$, "inflation" claim | $N = 871$, 10-fold 66.8\,\%, leave-center-out 67.0\,\% | `scripts/lit_check.csv` (NeuroImage 147:736-745, p. 740) | Verified against published paper: random CV did not inflate accuracy over leave-center-out; inflation claim deleted. |
| **Allison et al. Cutoffs** | Adolescent 0.85/0.83, Child 0.95/0.97 | Adolescent 0.93/0.95, Child 0.95/0.97 | `scripts/lit_check.csv` (JAACAP 51(2):202-212) | Corrected psychometric validation sensitivity/specificity numbers. |

---

## 2. Structural & Citation Sanitation

1. **Deleted Fabricated & Phantom Citations:**
   - `thabtah2019`: Contained DOI `10.1016/j.ijmedinf.2019.04.020`, belonging to an unrelated UTI study by Tandan et al. Deleted.
   - `zablotsky2020`: Fabricated title/venue in prior draft. Deleted.
   - `addm2025`: Unreleased/unverified CDC MMWR report. Retained official verified `maenner2023`.
   - `onnxruntime2023`: Deleted along with edge benchmark claims per E1.
2. **Author Name & Bracket Verification:**
   - `raj2020`: Corrected authors to *Suman Raj and Sarfaraz Masood* (Procedia CS 167:994-1004).
   - `thabtah2018ijmi`: Corrected authors to *Fadi Thabtah, Firuz Kamalov, Khairan Rajab* (IJMI 117:112-124).
   - `kan2022bnt`: Corrected authors to *Xuan Kan, Daiqing Qi, Hejie Cui, Ying Guo, Carl Yang* (NeurIPS 35:25586-25599).
   - `{Di Martino}`: Wrapped in curly braces in `di2014` and `abraham2017`.
3. **Table I Overhaul:**
   - Strictly 5 neutral columns: `Study`, `Modality`, `Dataset & N`, `Split Protocol`, `Reported Metric`. Audit observations and leakage notes moved to `scripts/lit_check.csv`.
