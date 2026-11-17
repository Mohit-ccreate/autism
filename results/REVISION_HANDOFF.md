# REVISION_HANDOFF.md — Peer Review Revision Pass

**Repository:** `asd-project` | **Branch:** `paper-revision` | **Date:** 2026-11-17  
**Standard:** 100% data-traceable numbers; zero hardcoding; all tests passing (44/44).

---

## 1. Root Causes of the Four Consistency Errors

### A. Tautology Index vs. Demographics AUROC Discrepancy
- **Finding:** Table IV demographics AUROC (0.5801, 0.5180, 0.4473, 0.5527) differed from earlier clean AUROCs (0.5910, 0.4846, 0.4460, 0.5558) that produced Table III TI values (0.818, 1.031, 1.108, 0.888).
- **Cause:** Table IV AUROCs came from a 5-seed, 5-fold CV run in `benchmark.py`, whereas Table III TI values came from an earlier 3-seed run in `tautology_audit.py`.
- **Resolution:** Re-evaluated in **ONE unified run** from **ONE saved prediction set** per cohort across 5 seeds ($5 \times 5 = 25$ evaluations).
- **Instability Flag:** For Child, clean demographic AUROC ($0.4473$) is below chance ($0.50$). Because $\text{TI} = (\text{AUC}_{\text{items}} - \text{AUC}_{\text{clean}}) / (\text{AUC}_{\text{items}} - 0.50)$, the numerator ($0.5527$) exceeds the denominator ($0.5000$), yielding raw $\text{TI} = 1.1054$ (clipped to $1.0000$).

### B. Leaky vs. In-Fold ComBat Paired Difference Discrepancy
- **Finding:** Displayed AUROCs differed by 0.0096 (0.6689 minus 0.6593), but reported paired difference was 0.0156.
- **Cause:** `run_paired_harmonization_tests` passed `meta_df["target"].values` in original dataframe row order, whereas `predictions_map` had predictions sorted site-by-site, causing paired deltas to be calculated with misaligned labels.
- **Resolution:** Re-aligned ground-truth labels to site-concatenated order. Direct subtraction ($0.6689 - 0.6593 = +0.0096$) and paired bootstrap delta match identically: $\Delta = +0.0096$ [95% CI: $-0.0071, 0.0260$], raw $p = 0.2620$, Holm $p = 1.0000$.

### C. Permutation Test p-Value Resolution Limit
- **Finding:** Permutation $p$-value was previously reported as $p = 0.0000$, below sampling resolution.
- **Cause:** Calculated as `mean(null >= obs)` without the standard $+1$ sample correction.
- **Resolution:** Rerun with $B = 10,000$ within-site label permutations using $p = (1 + \text{count}) / (1 + B)$. With 0 exceedances, $p = (1 + 0) / (1 + 10,000) = 0.00010$. The minimum resolution limit $1/(1+B) = 0.00010$ is strictly respected everywhere.

### D. Logistic Regression Accuracy Range Claim
- **Finding:** Text claimed logistic regression achieved 99.4--100% accuracy, omitting Adolescent.
- **Cause:** Adolescent repeated $5 \times 10$ CV accuracy was $95.55\% \pm 5.47\%$ (95% CI: $[84.21\%, 100.00\%]$; standard 5-fold CV: $91.96\%$). Prior text omitted this cohort from the summary range.
- **Resolution:** Reconciled per-cohort accuracy range in text to **95.55% to 100.00%**.

---

## 2. Old Versus New Values for Every Changed Number

| Metric / Parameter | Old Draft | New Revision | 95% Bootstrap CI / p-value | Source / Cause |
|---|---|---|---|---|
| **Adult Clean AUROC (TI)** | 0.5910 | **0.5801** | [0.5581, 0.6017] | Unified 5-seed 5-fold run |
| **Adult TI (Raw / Clipped)** | 0.818 | **0.8398** | [0.7967, 0.8839] | Unified 5-seed 5-fold run |
| **Adult Alt (a) Abs Drop** | Unreported | **0.4199** | [0.3983, 0.4419] | $\text{AUC}_{\text{items}} - \text{AUC}_{\text{clean}}$ |
| **Adult Alt (b) Rel Drop** | Unreported | **1.0174** | Over majority (73.76%) | Relative accuracy drop |
| **Adolescent Clean AUROC** | 0.4846 | **0.5180** | [0.4484, 0.5901] | Unified 5-seed 5-fold run |
| **Adolescent TI (Raw / Clip)**| 1.031 / 1.000| **0.9637** | [0.8621, 1.0692] | Unified 5-seed 5-fold run |
| **Adolescent Alt (a) Abs Drop**| Unreported | **0.4767** | [0.4079, 0.5459] | $\text{AUC}_{\text{items}} - \text{AUC}_{\text{clean}}$ |
| **Child Clean AUROC** | 0.4460 | **0.4473** | [0.4022, 0.4907] | Unified 5-seed 5-fold run |
| **Child TI (Raw / Clip)** | 1.108 / 1.000| **1.1054 / 1.0000** | [1.0441, 1.1682] | Instability: Clean < 0.50 |
| **Child Alt (a) Abs Drop** | Unreported | **0.5526** | [0.5093, 0.5978] | $\text{AUC}_{\text{items}} - \text{AUC}_{\text{clean}}$ |
| **Toddler Clean AUROC** | 0.5558 | **0.5527** | [0.5337, 0.5714] | Unified 5-seed 5-fold run |
| **Toddler TI (Raw / Clip)** | 0.888 | **0.8946** | [0.8576, 0.9322] | Unified 5-seed 5-fold run |
| **Toddler Alt (a) Abs Drop** | Unreported | **0.4473** | [0.4286, 0.4663] | $\text{AUC}_{\text{items}} - \text{AUC}_{\text{clean}}$ |
| **ComBat Paired Diff ($\Delta$)**| +0.0156 | **+0.0096** | [$-0.0071, 0.0260$] | Label alignment resolved |
| **ComBat Holm-adjusted p** | 0.2840 | **1.0000** | Raw $p = 0.2620$ | Recomputed paired test |
| **Permutation p (Within-Site)**| 0.0000 | **0.00010** | $B = 10,000$, count = 0 | $p = (1+\text{count})/(1+B)$ |
| **LR Items Accuracy Range** | 99.4--100% | **95.55--100.00%** | Adolescent 95.55% [84.21, 100] | Cohort inclusion check |
| **Site Decoding Acc. (Confound)**| 64.87% | **73.54%** | [70.61%, 76.46%] | 20 sites, in-fold deconf |
| **Conformal Coverage (Shared)**| 87.00% | **87.00%** | Mean size: 1.595 | Version 1 (Shared 2-way) |
| **Conformal Cov. (Disjoint)**| Unreported | **89.58%** | Mean size: 1.664 | Version 2 (3 Disjoint) |
| **Conformal Cov. (Site-Group)**| Unreported | **89.70%** | Mean size: 1.635 | Version 3 (Site-Grouped) |

---

## 3. ComBat Protocol and Uncertainty Allocation in Plain Language

### ComBat Unseen-Site Protocol (`results/COMBAT_PROTOCOL.md`)
- **Parameter Estimation on Held-Out Sites:** Parametric ComBat estimates location shifts ($\gamma$) and scale shifts ($\delta$) only for training sites. For unseen test sites in LOSO, `parametric_combat_transform` evaluates $\hat{X} = ((X - \hat{\alpha})/\hat{\sigma})\hat{\sigma} + \hat{\alpha} \equiv X$. This is **algebraically the identity transformation**; the held-out test site connectomes remain entirely unmodified.
- **Influence of Held-Out Data:**
  - *(i) None:* Raw connectomes. AUROC = 0.6776 [0.6409, 0.7135].
  - *(ii) Strictly Inductive:* ComBat fit only on training sites; test site unmodified. AUROC = 0.6593 [0.6212, 0.6979]. Zero test data used.
  - *(iii) Transductive:* Test site's own unlabeled feature mean/variance standardize it to training space. AUROC = 0.6689 [0.6315, 0.7063], paired $\Delta = +0.0096$ [$-0.0071, 0.0260$], Holm $p = 1.0000$.
  - *(iv) In-Fold ComBat as Implemented:* Training sites harmonized; test site unmodified. Matches strictly inductive arm ($\Delta = 0.0000$).
  - *(v) Leaky ComBat:* ComBat fit on all pooled data before split. AUROC = 0.6689, paired $\Delta = +0.0096$, Holm $p = 1.0000$.

### Uncertainty Data Allocation Protocol (`results/UNCERTAINTY_PROTOCOL.md`)
- **Version 1 (Shared):** 20% training partition shared between temperature scaling ($T$) and conformal quantile ($\hat{q}$).
- **Version 2 (3 Disjoint Subsets):** Training partition split into Proper-Train (60%), Calibrator (20%), and Conformal-Cal (20%). Disjointness verified by unit test ($\text{Proper} \cap \text{Cal} = \emptyset, \text{Proper} \cap \text{Conf} = \emptyset, \text{Cal} \cap \text{Conf} = \emptyset$). Marginal coverage = **89.58%** (mean set size 1.664).
- **Version 3 (Site-Grouped LOSO):** 3 full sites held out exclusively for conformal calibration; 16 for training. Marginal coverage = **89.70%** (mean set size 1.635).
- **Set-Size Identity:** $\text{mean\_set\_size} = 1.0 \times \text{singleton} + 2.0 \times \text{both} + 0.0 \times \text{empty}$ holds exactly across all versions ($< 10^{-14}$ residual).

---

## 4. Pooled vs. Within-Site Findings & Controlled Probe (`results/per_site_auroc.csv`)

1. **Per-Site LOSO Evaluation:** Evaluated all 8 models across all 20 ABIDE I sites ($N=854$). Zero sites excluded (all 20 contain both classes).
2. **Site-Prevalence Baseline:** Scoring subjects purely with their site's empirical ASD prevalence (zero imaging features) yields AUROC = **0.5902**, proving between-site prevalence differences contribute non-trivial ranking signal.
3. **Within-Site Z-Scoring:** Z-scoring predictions within held-out sites preserves or improves performance (LR: 0.6593 $\to$ **0.6663**; SVM: 0.6765 $\to$ **0.6931**; RF: 0.7003 $\to$ **0.7222**), confirming genuine within-site ranking signals.
4. **Controlled Site Probe (`results/site_probe_results.json`):**
   - Phenotypic variables alone (age, sex, FIQ, motion, diagnosis) decode site with **22.13%** (LR) and **27.17%** (RF) accuracy vs 19.91% majority baseline (NYU, $n=170$).
   - Connectomes after in-fold confound regression (age, sex, FIQ, motion, diagnosis regressed out) decode site with **73.54%** accuracy [95% CI: 70.61%, 76.46%] and balanced accuracy **64.77%** (vs 5.0% chance).
   - This reflects broad **site-associated information** (scanner RF coils, gradient systems, pulse sequences, TR/TE, recruitment criteria) rather than isolated hardware signatures.

---

## 5. De-Duplication Sensitivity Analysis

1. **Identifier Check:** Adult, Adolescent, and Child contain zero participant identifiers. Toddler contains `Case_No` (serial row index $1..1054$). Purged rows are reported strictly as **identical records**, not verified duplicate participants.
2. **Purge Counts on `[A1..A10, Age, Sex, Target]`:**
   - Adult: $704 \to 667$ (37 removed, 5.26%); ASD% $75.7 \to 74.5\%$, Age $27.6 \to 27.9$.
   - Adolescent: $104 \to 97$ (7 removed, 6.73%); ASD% $60.6 \to 58.8\%$, Age $14.1 \to 14.1$.
   - Child: $292 \to 262$ (30 removed, 10.27%); ASD% $48.3 \to 51.5\%$, Age $6.4 \to 6.4$.
   - Toddler: $1054 \to 850$ (204 removed, 19.35%); ASD% $69.1 \to 64.9\%$, Age $27.6 \to 27.5$ mos.
3. **Sensitivity (Raw vs Clean TI):** Adult 0.7813 $\to$ 0.8398; Adolescent 1.0220 $\to$ 0.9637; Child 1.0372 $\to$ 1.1054; Toddler 0.8621 $\to$ 0.8946.

---

## 6. Literature Audit, Worksheet & Prior Comparison

1. **Bibliographic Reference Audit (`references.bib`):** All 28 references verified with complete author lists (all 36 authors for `maenner2023`, all 42 authors for `di2014`, zero "et al." or "others"). Verified DOIs and publisher URLs.
2. **Verification Worksheet (`docs/lit_verification_worksheet.csv`):** Created 23 rows for all papers flagged `needs_human_check = Y` with evidence locations, paraphrases, and blank human override columns. `scripts/recompute_lit_counts.py` recomputes all counts and reports the conservative subset of unflagged full-text papers ($N=10$; 80.0% random split, 60.0% pooled preprocessing, 90.0% neither calibration nor uncertainty).
3. **Prior Work Comparison Table (`results/prior_comparison.csv`):** 10-row comparison linking audited literature to study results. Shows that Heinsfeld random CV (70.0% vs our 70.84%, AUROC 0.7510) drops to 66.04% (AUROC 0.6774) under LOSO, and that reported ~100% questionnaire accuracies collapse when leaking items are removed.

---

## 7. Publication Figure Pack (`figures/paper_pack/`)

All 6 figures generated in vector `.pdf` and 300-DPI `.png` with colorblind-safe palettes:
1. `fig_cc200_heatmaps`: Descriptive mean CC200 diff (ASD minus Control) and mean abs LR weights across 20 LOSO folds.
2. `fig_site_confusion_matrix`: Controlled 20x20 site decoding confusion matrix for deconfounded connectomes (73.54% acc vs 19.91% base rate).
3. `fig_persite_forest`: Forest plot of per-site LOSO AUROC with 95% CIs and sample sizes across all 20 sites.
4. `fig3_loso_vs_random`: Forest plot of Random-CV vs LOSO AUROC per model.
5. `fig_conformal_grouped_bar`: Grouped bar chart of conformal coverage, set sizes, and referral burden.
6. `fig_permutation_histogram`: Within-site label permutation null distribution ($B=10,000$, observed AUROC 0.6593, $p = 0.00010$).
- Updated `figures/paper_pack/MANIFEST.csv` and `FIGURE_CHECK.md`.

---

## 8. What Could Not Be Done and Why

1. **ABIDE II External Validation:** Preprocessed CPAC pipeline CC200 timeseries for ABIDE II are not publicly accessible via open HTTP or unauthenticated S3 buckets (they require authenticated NITRC credentialed downloads). Logged transparently as `SKIPPED` in `results/abide_ii_attempt.json`.
2. **Clinical Verification of Duplicate Participants:** Public repositories (UCI, Kaggle) contain no patient identifiers, timestamps, or medical record numbers. Identical rows were purged algorithmically and described strictly as identical records, not verified duplicate patients.

---

## 9. Verification & Quality Gates

- **Unit Test Suite:** Added 4 new tests in `tests/test_audit_extension.py`. **44 / 44 tests pass** in 5.72s (`pytest tests/ -v`).
- **Truth Ledgers:** Recompiled `results/paper_facts.json`, `results/SUMMARY.md`, `PROGRESS.md`, `CHANGELOG.md`, and `results/RUN_LOG.md`.
- **LaTeX Claims:** Updated `main.tex` lines 212 and 361 to reflect accurate logistic regression accuracy ranges (95.55%--100.00%).
