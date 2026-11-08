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

---

## 3. Paper Revision Branch (`paper-revision`) Extension Ledger

This section records every module, configuration, documentation artifact, and quality gate added during the `paper-revision` branch implementation.

### Added Files & Research Modules

1. **Safety & Audit Foundations:**
   - `AUDIT.md`: Comprehensive audit inventory of all four behavioral cohorts and ABIDE neuroimaging data; catalogs leakage risks, in-fold requirements, and impact on prior publication claims.
   - `PROGRESS.md`: Phase tracking, execution checkpoints, and terminal commands to reproduce all stages.
   - `results/RUN_LOG.md`: Chronological execution ledger recording run timestamps, git commits, seeds, and execution statuses.
   - `results/SUSPICIOUS.md`: Mathematical diagnoses of counter-intuitive findings (e.g., Var A linear model arithmetic reproduction, BNT domain collapse, ComBat decodability artifact).

2. **Module 1 (Behavioral Tautology Audit Extension):**
   - `src/audit/tautology_audit.py`: Implements fast vectorized bootstrap Mann-Whitney U, pairwise Hamming distance near-duplicate analysis, Tautology Index computation ($\text{TI} = 0.818 - 1.108$), and label permutation sanity controls across all four cohorts.
   - `results/tautology_audit.json`: Complete empirical output for Module 1.

3. **Module 3 (Scanner / Site Generalization):**
   - `src/eval/site_analysis.py`: Multi-site LOSO evaluation on ABIDE I CC200 across OHSU, OLIN, PITT, and SDSU ($N=120$); benchmarks harmonization strategies (None, Site Regression, ComBat, and a deliberately wrong **Leaky ComBat** arm); computes pre-ComBat site decodability (73.33%) and motion/age confound partial correlations.
   - `results/site_analysis.json`: Complete empirical output for Module 3.

4. **Module 4 (Uncertainty-Aware Decision Support):**
   - `src/uncertainty/calibration.py`: Probability calibration (ECE, adaptive ECE, Brier, NLL, temperature scaling, isotonic regression); Split Conformal prediction (LAC, APS, Mondrian class-conditional) at $\alpha = 0.10$ (97.76% coverage) and $\alpha = 0.05$; selective prediction risk-coverage curves (AURC = 0.0001); three-way clinical decision policy (screen+, screen-, refer) with 2.24% referral burden.
   - `results/uncertainty_main.json`: Complete empirical output for Module 4.
   - `results/demo_lr_params_demos_only.json`: Trained, honest demographics-only model parameters exported for real-time interactive inference.

5. **Module 5 (Literature Meta-Audit Template):**
   - `docs/lit_audit_template.csv`: Standardized 12-column peer-review audit schema with empty rows for human author evaluation.
   - `scripts/summarize_lit_audit.py`: Automated summary script computing safeguard prevalence, split distributions, and generating summary plots.

6. **Module 6 (Streamlit Multi-Page Web Platform):**
   - `app/Home.py`: Main overview with persistent non-diagnostic medical disclaimer, central thesis testing status, and dataset cards.
   - `app/pages/1_🔍_Tautology_Audit.py`: Interactive rule recovery, feature ablation, single-item baselines, and Tautology Index.
   - `app/pages/2_🏆_Benchmark_Leaderboard.py`: Sortable leaderboard across models, metrics, and protocols with CSV export.
   - `app/pages/3_🧲_Scanner_Generalization.py`: LOSO vs Random CV gap chart, site probe, and harmonization comparison.
   - `app/pages/4_🎯_Uncertainty_Lab.py`: Interactive reliability diagrams, per-site conformal coverage, and risk-coverage slider.
   - `app/pages/5_🖊_Screening_Simulator.py`: Non-leaky demographics-only model with calibrated probabilities and conformal abstention.
   - `app/pages/6_🔬_Reproducibility.py`: Verified configurations, seeds, pipeline schematics, and reproduction commands.
   - `app/pages/7_✅_Audit_Checklist.py`: 15-item ML-for-autism evaluation checklist with category/severity filters and CSV/MD download.
   - `docs/DEMO_SCRIPT.md`: 5-minute timed presentation walkthrough script for faculty review.

7. **Publication Figures & LaTeX Tables:**
   - `src/viz/generate_figures.py`: Generates all 8 required figures in vector PDF and 300-dpi PNG using colorblind-safe palettes:
     - `pipeline_schematic.[pdf/png]`
     - `tautology_ablation.[pdf/png]`
     - `loso_gap.[pdf/png]`
     - `site_probe_confounds.[pdf/png]`
     - `harmonization_comparison.[pdf/png]`
     - `reliability_diagrams.[pdf/png]`
     - `conformal_coverage_per_site.[pdf/png]`
     - `risk_coverage_curves.[pdf/png]`
   - `scripts/export_tables.py`: Exports all main tables in both `.csv` and `.tex` into `results/tables/`:
     - `table_rule_recovery.[csv/tex]`
     - `table_feature_ablation.[csv/tex]`
     - `table_harmonization_comparison.[csv/tex]`
     - `table_calibration.[csv/tex]`
     - `table_conformal_policy.[csv/tex]`
   - `results/paper_facts.json`: Master machine-readable ledger linking every headline number to source JSON files.
   - `results/SUMMARY.md`: Executive research summary, limitations, and table compilation.

8. **Testing, Reproducibility, & Packaging:**
   - `tests/test_audit_extension.py`: 8 comprehensive quality gates (train/test isolation, in-fold scaling, label-shuffle collapse, conformal coverage, schema integrity, figure/table presence, seed reproducibility). Total test suite now **32/32 tests passing**.
   - `Dockerfile`: Standalone container configuration for reproducible cloud/local deployment.
   - `Makefile`: Master automation commands (`make all`, `make pipeline`, `make extended`, `make test`, `make app`).
   - `configs/main.yaml` & `configs/models.yaml`: Centralized experiment configurations and hyperparameter grids.
   - `README.md`: Comprehensive beginner-to-researcher documentation with data lineage, verified SHA-256 checksums, and citation metadata.
   - `requirements.txt`: Converted to UTF-8 with pinned Streamlit and Plotly dependencies.

---

## 4. Correction Pass Ledger (Full ABIDE Scaling, Statistical Honesty & Rigor)

This section details all modifications, new modules, metric diffs, and forensic resolutions completed during the peer-review correction pass.

### A. Scaled ABIDE I to Full Multi-Center Cohort (N=854, 20 Clinical Sites)
- **Files Added:**
  - `scripts/download_all_abide_multithread.py`: 16-worker multi-threaded downloader with exponential retry to bypass urllib socket timeout in Nilearn. Fetched 869/871 QC subjects.
  - `data/abide_full_cache.npz` & `data/abide_full_phenotypic.csv`: Validated CC200 functional connectivity matrices for $N=854$ quality-controlled subjects (398 ASD, 456 Control) across all 20 clinical sites.
  - `data/abide_raw_cache.npz` & `data/abide_raw_phenotypic.csv`: Complete robustness non-QC cohort ($N=989$, 20 sites).
- **Files Modified:**
  - `src/eval/site_analysis.py`: Scaled from 4 sites ($N=120$) to all 20 clinical sites ($N=854$). Implemented subject-level bootstrap, site-cluster bootstrap, within-site permutation tests, and paired bootstrap tests between harmonization arms.
  - `results/site_analysis.json`:
    - In-fold ComBat LOSO AUROC: **0.6593** [Subject 95% CI: $0.6215, 0.6962$; Site-Cluster 95% CI: $0.6075, 0.7045$], Acc = 61.71%. Permutation $p < 0.0001$, Binomial $p = 7.90 \times 10^{-12}$.
    - Random 5-Fold CV AUROC: **0.6728** (Generalization gap: $+0.0135$).
    - Pre-ComBat Site Decodability: **64.87%** (vs 19.91% base rate for NYU majority site).
    - Confound Models: LOSO AUROC = **0.5588**.
    - Non-QC Robustness Arm ($N=989$): LOSO AUROC = **0.6985**.
    - Full 20-site per-site generalization gap table compiled.
- **Metric Diff:**
  - Previous preliminary LOSO AUROC on $N=120$ (4 sites): $0.6098$ [CI: $0.505, 0.708$], permutation $p = 0.2148$, binomial $p = 0.3153$ (indistinguishable from chance).
  - New full-cohort LOSO AUROC on $N=854$ (20 sites): $0.6593$ [CI: $0.6215, 0.6962$], permutation $p < 0.0001$, binomial $p = 7.90 \times 10^{-12}$ (statistically verified signal).

### B. Uncertainty Module Overhaul & Strict Set-Size Decomposition
- **Files Modified:**
  - `src/uncertainty/calibration.py`: Separated into 3 explicit arms: (1) Leak-Free Tabular (demographics only, 4 cohorts), (2) Full ABIDE I LOSO (calibration strictly from training sites), (3) `LEAKY CONTRAST ONLY` (Adult questionnaire items). Enforced set-size identity verification.
  - `results/uncertainty_main.json`: Output structured by evaluation arm with full calibration (uncalibrated, Platt, Isotonic, Temperature), split conformal (LAC, APS, Mondrian), selective prediction, and triage policy.
- **Set-Size Identity Verified:**
  $$\text{mean\_set\_size} = 1.0 \times \text{fraction\_singleton} + 2.0 \times \text{fraction\_both} + 0.0 \times \text{fraction\_empty}$$
  Verified to $< 10^{-6}$ across all conformal methods and arms.
- **Metric Diff:**
  - ABIDE I LOSO LAC ($\alpha=0.10$): Marginal coverage = **87.00%**, Mean set size = **1.595**, Empty = **0.0%**, Singleton = **40.5%**, Both = **59.5%**, Triage burden = **59.5%**.
  - Adult Demographics LAC ($\alpha=0.10$): Marginal coverage = **89.96%**, Mean set size = **1.553**, Empty = **0.0%**, Singleton = **44.7%**, Both = **55.3%**, Triage burden = **55.3%**.
  - Per-site coverage under LOSO reported honestly: 4 sites over-covered ($>95\%$), 11 under-covered ($<85\%$), 5 well-calibrated ($85\%-95\%$).

### C. Statistical Honesty & Forensic Audits
- **Files Added:**
  - `src/audit/dedup_and_threshold_audit.py`: Exhaustive deduplication accounting and empirical threshold sensitivity sweep.
  - `results/dedup_and_threshold_audit.json`: Logged exact deduplication accounting and threshold sweeps for thresholds 5, 6, 7, 8 across all cohorts.
- **Files Modified:**
  - `src/audit/tautology_audit.py`: Output both raw $\text{TI}_{\text{raw}}$ and clipped $\text{TI}_{\text{clipped}} \in [0, 1]$, explaining mathematical mechanism of $\text{TI} > 1.0$.
  - `results/SUSPICIOUS.md`: Added forensic diagnoses for (1) Threshold mismatch ($\tau=7$ vs $\tau=6$), (2) Tautology Index $> 1.0$, (3) Preliminary $N=120$ CI vs permutation reconciliation, (4) Conformal set-size identity resolution, and (5) Nilearn S3 timeout root cause.
- **Findings:**
  - UCI Adult, Adolescent, and Child binary targets match $\tau=7$ with 100.00% concordance (vs 83.5%–89.7% at published $\tau=6$). Toddler Q-CHAT-10 matches published $\tau=4$ with 100.00% concordance.
  - Paired bootstrap test between Leaky ComBat and In-Fold ComBat: $\Delta = +0.0156$ [95% CI: $-0.0013, 0.0328$], $p_{\text{holm}} = 0.2840$. Because 95% CI includes zero, verdict is **NOT SUPPORTED** and reported as "no detectable difference at this sample size (N=854, 20 sites)".

### D. Missing Components & Leaderboard
- **Files Added:**
  - `src/models/autoencoder.py`: Deep Autoencoder with LayerNorm and drop_last in PyTorch for robust representation learning on fMRI.
  - `src/eval/benchmark.py`: Complete Module 2 benchmark leaderboard across 8 neuroimaging models and 6 tabular models.
  - `results/benchmark_leaderboard.json`: Full benchmark metrics, CIs, and generalization gaps.
  - `scripts/attempt_abide_ii.py` & `results/abide_ii_attempt.json`: Verified ABIDE II preprocessed CC200 files are not on public S3 (`HTTP 404`); institutional access required for raw scans. Marked as **SKIPPED**.
- **Results:**
  - ABIDE I Leaderboard (LOSO AUROC / Random CV AUROC): Random Forest (0.7003 / 0.6998), LightGBM (0.6992 / 0.7019), Extra Trees (0.6867 / 0.6959), Heinsfeld Autoencoder (0.6820 / 0.6685), Linear SVM (0.6765 / 0.6888), Small MLP (0.6671 / 0.6724), L2 Logistic Regression (0.6593 / 0.6728), Tangent Space Logistic (0.6092).

### E. Claim Language & Scientific Verdicts
- **Files Modified:**
  - `scripts/compile_paper_facts_and_summary.py`: Replaced monolithic binary verdicts with per-component scientific verdicts.
  - `results/paper_facts.json` & `results/SUMMARY.md`:
    - Thesis 1 (Behavioral Tautology): `SUPPORTED`
    - Thesis 2 (Scanner Generalization Gap): `SUPPORTED`
    - Thesis 3 (Leaky ComBat Inflation): `NOT SUPPORTED`
    - Thesis 4 (Uncertainty-Aware Decision Support): `PARTIALLY SUPPORTED`
  - `results/tables/`: Generated 10 publication tables in both `.csv` and `.tex`. Prior runs archived in `results/legacy/pre_correction_pass/`.

### F. Quality Gates & Web Application
- **Files Modified:**
  - `tests/test_audit_extension.py`: Expanded to 13 test functions covering set-size identity, synthetic conformal exchangeability, in-fold ComBat isolation, JSON schemas, ABIDE $N=854$ dataset integrity, and paired bootstrap identity. **37/37 tests passing (100%)**.
  - `app/pages/2_🏆_Benchmark_Leaderboard.py`: Updated to load dynamically from `results/benchmark_leaderboard.json`.
  - `app/pages/3_🧲_Scanner_Generalization.py`: Updated to display full $N=854$ ABIDE results across all 20 clinical sites, statistical reconciliation, and paired bootstrap test.
  - `app/pages/4_🎯_Uncertainty_Lab.py`: Updated to support multi-arm evaluation, set-size decomposition, per-site coverage breakdown under LOSO, and 3-way triage policy.

---

## 5. Literature Meta-Audit (N=33 Published Papers Ledger)

- **Date:** 2026-11-07
- **Deliverables Added/Modified:**
  - `docs/lit_audit_template.csv`: Standardized 18-column corpus auditing 33 real, published papers (16 Tabular AQ-10/Q-CHAT-10, 17 ABIDE I/II) from 2015 to 2026 with verified open-access URLs/DOIs.
  - `results/lit_search_log.md`: Query logs, database sources (PMC, PubMed, arXiv, medRxiv, IEEE Xplore, HAL, ETASR), and inclusion/exclusion criteria.
  - `scripts/summarize_lit_audit.py`: Updated aggregator computing safeguard prevalence with explicit denominators and rendering 4-panel publication visualization.
  - `results/lit_audit_summary.json`: Granular counts, denominators, and rates across overall, tabular, and neuroimaging subsets.
  - `figures/lit_meta_audit_summary.png` & `figures/lit_meta_audit_summary.pdf`: Publication-grade visualization of methodological pitfalls vs safeguards.
  - `docs/lit_audit_for_human_review.md`: Complete inspection document providing direct URLs, evidence level, and verbatim methods justifications.
  - `tests/test_lit_meta_audit.py`: 3 automated tests verifying corpus size (30-40), schema conformity, evidence level flags, human review triggers, and summary consistency (test suite total: 40/40 passing).

---

## 6. Blind Re-Verification, Paper Number Cross-Check & Figure Pack

- **Date:** 2026-11-07
- **Deliverables Added/Modified:**
  - `docs/lit_audit_template_v1.csv`: Preserved initial baseline version of the literature audit.
  - `results/lit_reverify_blind.csv`: Re-extracted fields across all 33 papers independently from source text before comparison.
  - `results/lit_reverify_comparison.csv`: Field-by-field agreement evaluation (363 comparison rows). 100% agreement on leaking features, calibration, uncertainty, external validation, and evidence level.
  - `docs/lit_audit_template.csv`: Reconciled categorical fields based on source passages; corrected URL prefixes (PMID typos) for Sollis 2025, Shrivastava 2024, Tartarisco 2021, and Katlariwala 2026. Flagged all changed or uncertain rows as `needs_human_check = Y` (23 of 33, 69.7%).
  - `scripts/summarize_lit_audit.py`: Updated to compute exact union of calibration or uncertainty (2 of 33, 6.1%) and complement (31 of 33, 93.9%), plus min/median/max reported accuracy per stratum with named papers.
  - `results/paper_number_check.csv`: Meticulous cross-check of 50 empirical claims against `results/*.json`. Verified 44 exact matches and flagged 6 exact mismatches (Child recovered accuracy 100.0% vs draft 99.98%; LightGBM accuracy 64.29% vs 64.64%; Autoencoder accuracy 62.65% vs 62.88%; MLP accuracy 61.83% vs 62.41%; Tangent Space accuracy 59.72% vs 58.78%; Conformal ABIDE LOSO 95% coverage 92.39% vs 94.03%).
  - `figures/paper_pack/`: Consolidated all 9 publication figures as vector `.pdf` and 300-DPI `.png` files with canonical naming `fig1` through `fig9`. Verified typography, explicit units, legibility at 8.5 cm and 17.5 cm, and Wong (2011) colorblind accessibility.
  - `figures/paper_pack/MANIFEST.csv`: Index of figures, paper numbers, target sections, factual captions, and source result files.
  - `figures/paper_pack/FIGURE_CHECK.md`: Detailed quality and formatting audit report.
  - `results/PAPER_HANDOFF.md`: Self-contained handoff document (1,432 words) summarizing literature findings, re-verification rates, number check mismatches, and per-site LOSO distributions.

- **Core Empirical Findings:**
  - Behavioral Tautology (Label Leakage): 15 of 16 (93.8%) tabular papers contain direct label-leaking questionnaire items, producing near-perfect accuracy (96%-100%) that collapses under leak-free controls. 0 of 17 (0.0%) neuroimaging papers contain label leakage.
  - Evaluation Protocols: 13 of 13 (100.0%) tabular and 10 of 17 (58.8%) neuroimaging papers use random cross-validation. Only 4 of 17 (23.5%) neuroimaging papers evaluate leave-one-site-out (LOSO) generalization.
  - Preprocessing Isolation: 12 of 27 (44.4%) evaluated papers fit feature selection, PCA, or imputation across the entire dataset prior to folding.
  - Calibration & Uncertainty: Only 1 of 33 (3.0%) papers reports probability calibration, and only 2 of 33 (6.1%) report predictive uncertainty. 31 of 33 (93.9%) provide uncalibrated, point-prediction outputs with no abstention capability.
  - External Validation: Only 5 of 33 (15.2%) test on independent external cohorts.

---

## 7. NeuroTriage ML Application Deployment

- **Date:** 2026-11-08
- **Deliverables Added/Modified:**
  - `webapp/neurotriage.html`: Ultra-modern dark obsidian application featuring interactive 3D Three.js connectome lattice (200 Craddock ROIs), live behavioral screener with real-time tautology audit toggle, connectome triage with conformal prediction set displays, 4 Chart.js benchmark graphs, and a 14-item methodological compliance checklist.
  - `webapp/neurotriage.css`: Obsidian theme system (`#0A0E17`), glassmorphic panels, electric cyan accents (`#00F2FE`), violet glow, and emerald/rose/amber medical status indicators.
  - `webapp/neurotriage.js`: Core client-side engine supporting 3D WebGL particle lattice with mouse inertia, real-time AQ-10 psychometric scoring, feature ablation simulator, conformal sets $\{0\}, \{1\}, \{0, 1\}$ at $\alpha = 0.10$, and dynamic integrity score gauge.
  - `webapp/index.html`: Added quick-launch navigation link to NeuroTriage ML in the header bar.
- **Verification:** All 40 pytest tests pass, `node --check webapp/neurotriage.js` verified syntax with zero errors, clinical non-diagnostic language rules verified, and HTTP 200 response confirmed at `http://localhost:8000/neurotriage.html`.



