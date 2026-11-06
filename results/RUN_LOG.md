# RUN_LOG.md — Execution Log for ASD Screening Paper Revision

**Branch:** paper-revision  
**Hardware:** CPU (Windows, standard laptop)  
**Python:** 3.11 (venv)  

---

## Run Log Entries

### 2026-11-05T18:12:00+05:30 — Phase -1: Safety Backup
- **Status:** SUCCESS
- **Git commit (pre-revision):** 8cb7dfc
- **Branch created:** paper-revision
- **Note:** OneDrive provides cloud backup; filesystem sibling copy not performed (parent directory is OneDrive).

### 2026-11-05T18:20:00+05:30 — Phase 0: Audit
- **Status:** SUCCESS
- **Output:** AUDIT.md created
- **Key findings:**
  - L1: Label is 100% deterministic function of item sum (confirmed)
  - L2: Random CV leaks site information (confirmed and labeled)
  - P1: Age imputation before split (minor risk, documented)
  - P4: Only 4/17 ABIDE sites used (gap, documented)

### [PENDING] Module 1 Extension: Tautology Index
- **Status:** PENDING
- **Script:** `src/audit/tautology_audit.py` (to be created)

### [PENDING] Module 2: Extended Benchmark
- **Status:** PENDING
- **Script:** `src/eval/benchmark.py` (to be created)

### [PENDING] Module 3: Scanner Generalization
- **Status:** PENDING
- **Script:** `src/eval/site_analysis.py` (to be created)

### [PENDING] Module 4: Uncertainty
- **Status:** PENDING
- **Script:** `src/uncertainty/calibration.py`, `conformal.py`, `selective.py` (to be created)

### [PENDING] Module 6: Streamlit App
- **Status:** PENDING
- **Script:** `app/Home.py` (to be created)

---

## Failed/Skipped Items

| Item | Status | Reason | Action |
|------|--------|--------|--------|
| ABIDE II download | SKIPPED | Access requires institutional account; URL verification needed | Documented in AUDIT.md; external validation marked as SKIPPED |
| Filesystem backup to sibling | SKIPPED | OneDrive parent; would double storage | Using OneDrive cloud sync + git |
| Saliency4ASD (Optional C) | SKIPPED | Not readily available; listed as optional | Logged here |
[2026-11-06T05:33:35.561741+00:00] ======================================================================
[2026-11-06T05:33:35.565978+00:00]   EXTENDED PIPELINE START
[2026-11-06T05:33:35.646760+00:00]   Git: 8cb7dfcbdb5f
[2026-11-06T05:33:35.648206+00:00] ======================================================================
[2026-11-06T05:41:49.553511+00:00] ======================================================================
[2026-11-06T05:41:49.554204+00:00]   EXTENDED PIPELINE START
[2026-11-06T05:41:49.631523+00:00]   Git: 8cb7dfcbdb5f
[2026-11-06T05:41:49.632402+00:00] ======================================================================
[2026-11-06T05:46:51.447464+00:00] [OK] Module 1: Tautology Audit Extension completed in 295.7s
[2026-11-06T05:46:57.055287+00:00] [OK] Module 3: Site/Scanner Generalization Analysis completed in 5.6s
[2026-11-06T05:46:57.205522+00:00] [OK] Module 4: Uncertainty Analysis (Calibration + Conformal) completed in 0.1s
[2026-11-06T05:47:04.634429+00:00] [OK] Figure Generation completed in 7.4s
[2026-11-06T05:47:04.635026+00:00] 
Compiling paper_facts.json...
[2026-11-06T05:47:04.742406+00:00]   paper_facts.json written to C:\Users\mohit\OneDrive\Desktop\asd-project\results\paper_facts.json
[2026-11-06T05:47:04.747978+00:00] Building SUMMARY.md...
[2026-11-06T05:47:04.750136+00:00]   SUMMARY.md written to C:\Users\mohit\OneDrive\Desktop\asd-project\results\SUMMARY.md
[2026-11-06T05:47:04.751446+00:00] 
======================================================================
[2026-11-06T05:47:04.753370+00:00]   EXTENDED PIPELINE COMPLETE in 315.2s
[2026-11-06T05:47:04.754650+00:00]   Steps done:    ['Module 1: Tautology Audit', 'Module 3: Site Analysis', 'Module 4: Uncertainty', 'Figure Generation', 'paper_facts.json', 'SUMMARY.md']
[2026-11-06T05:47:04.755903+00:00]   Steps failed:  []
[2026-11-06T05:47:04.757154+00:00]   Steps skipped: []
[2026-11-06T05:47:04.757723+00:00] ======================================================================


---

## Correction Pass Execution Log (Phase A–F)

### 2026-11-06T16:30:00+05:30 — Archiving Prior State
- **Status:** SUCCESS
- **Action:** Archived pre-correction results, tables, and `SUMMARY.md` into `results/legacy/pre_correction_pass/` per Global Rule 2.

### 2026-11-06T16:35:00+05:30 — Deduplication & Threshold Sensitivity Audit (Parts C.4 & C.5)
- **Status:** SUCCESS
- **Script:** `src/audit/dedup_and_threshold_audit.py` -> `results/dedup_and_threshold_audit.json`
- **Audit Findings:**
  - **Adult ($704 \to 667$):** 37 exact duplicates purged (5.26%). Items AUROC: 1.000 vs 1.000; Demographics AUROC: 0.6085 (raw) vs 0.5578 (clean). Best empirical cutoff: $\tau=7$ (100.0% concordance). Allison et al. published cutoff $\tau=6$ (89.66% concordance, 69 discordant cases).
  - **Adolescent ($104 \to 97$):** 7 exact duplicates purged (6.73%). Items AUROC: 0.9977 vs 0.9956; Demographics AUROC: 0.5277 (raw) vs 0.5634 (clean). Best empirical cutoff: $\tau=7$ (100.0% concordance). Allison et al. published cutoff $\tau=6$ (83.51% concordance, 16 discordant cases).
  - **Child ($292 \to 262$):** 30 exact duplicates purged (10.27%). Items AUROC: 1.000 vs 0.9998; Demographics AUROC: 0.5143 (raw) vs 0.4507 (clean). Best empirical cutoff: $\tau=7$ (100.0% concordance). Allison et al. published cutoff $\tau=6$ (85.50% concordance, 38 discordant cases).
  - **Toddler ($1,054 \to 850$):** 204 exact duplicates purged (19.35%). Items AUROC: 1.000 vs 1.000; Demographics AUROC: 0.5583 (raw) vs 0.5464 (clean). Best empirical cutoff: $\tau=4$ (100.0% concordance). Published Q-CHAT-10 cutoff: $\tau=4$ (100.0% concordance, 0 discordant cases).

### 2026-11-06T16:40:00+05:30 — Explanation: Why ABIDE Previously Used N=120 from 4 Sites
- **Root Cause Analysis:**
  1. `nilearn.datasets.fetch_abide_pcp` uses single-threaded sequential HTTP requests with a hardcoded socket read timeout of 15.1 seconds.
  2. Downloading 871 subjects sequentially over Amazon S3 inevitably encounters transient network latency spikes, triggering `HTTPSConnectionPool Read timed out (read timeout=15.1)` and terminating execution.
  3. In initial development, an explicit filter `n_subjects=120` across 4 sites (`OHSU`, `OLIN`, `PITT`, `SDSU`) was implemented as a shortcut to prevent the sequential fetcher from stalling.
- **Correction Applied:**
  - Built a resilient multithreaded downloader with exponential backoff and 16 worker threads (`scripts/download_all_abide_multithread.py`).
  - Successfully downloaded 869 of 871 QC subjects (only 2 timed out: `UM_1_0050327`, `UM_1_0050329`).
  - Successfully processed and cached **N=854 quality-controlled subjects across all 20 clinical sites** with valid correlation matrices into `data/abide_full_cache.npz`.
  - Balanced diagnosis: 456 Controls (53.4%), 398 ASD (46.6%).
  - Additionally downloaded and cached the **non-QC cohort (N=989 subjects, 20 sites)** into `data/abide_raw_cache.npz` as a robustness check.

### 2026-11-06T16:48:00+05:30 — ABIDE II External Validation Attempt (Part D.2)
- **Status:** SKIPPED (Documented)
- **Script:** `scripts/attempt_abide_ii.py` -> `results/abide_ii_attempt.json`
- **Result:**
  - NITRC composite phenotypic CSV accessible (`HTTP 200`).
  - Preprocessed CPAC CC200 time-series are not hosted on public Amazon S3 (`HTTP 404`).
  - NITRC raw fMRI requires institutional authenticated access and manual pipeline preprocessing.
  - Concluded and marked **SKIPPED** per Rule 2 and Section D.2.

### 2026-11-06T17:00:00+05:30 — Full ABIDE I Site & Scanner Generalization (Part A & C)
- **Status:** SUCCESS
- **Script:** `src/eval/site_analysis.py` -> `results/site_analysis.json`
- **Key Findings:**
  - Scaled to **N=854 subjects across all 20 clinical sites**.
  - **In-Fold ComBat LOSO:** AUROC = **0.6593** [Subject 95% CI: $0.6215, 0.6962$; Site Cluster 95% CI: $0.6075, 0.7045$], Acc = 61.71%. Permutation $p < 0.0001$, Binomial $p = 7.90 \times 10^{-12}$.
  - **Harmonization Comparison:** None (0.6776), Site Regression (0.6679), In-Fold ComBat (0.6593), Leaky ComBat (0.6689).
  - **Paired Bootstrap Test (Leaky vs In-Fold):** $\Delta = +0.0156$ [95% CI: $-0.0013, 0.0328$], $p_{\text{holm}} = 0.2840$. **Because 95% CI includes zero, reported strictly as: "No detectable difference at this sample size (N=854, 20 sites)".**
  - **Site Probe:** Connectomes predict physical scanner ID with **64.87% accuracy** (vs 19.91% majority base rate for NYU), confirming strong scanner hardware artifacts.
  - **Confound Models:** Motion, age, sex, and FIQ achieve LOSO AUROC = 0.5588.
  - **Robustness Arm:** Non-QC cohort ($N=989$, 20 sites) achieves LOSO ComBat AUROC = 0.6985 [0.6640, 0.7315], Acc = 65.52%.

### 2026-11-06T17:15:00+05:30 — Module 2 Benchmark Leaderboard (Part D.1)
- **Status:** SUCCESS
- **Script:** `src/eval/benchmark.py` -> `results/benchmark_leaderboard.json`
- **Architectures Evaluated:** L2 Logistic Regression, Linear SVM, Random Forest, Extra Trees, LightGBM, Small MLP, Heinsfeld Autoencoder (`src/models/autoencoder.py`), Tangent-Space Logistic.
- **Tabular Demographics Benchmark (Leak-Free):**
  - Adult ($N=667$): LR AUROC 0.5801, Linear SVM 0.5715, RF 0.5621, ET 0.5626, LGBM 0.5592, MLP 0.5779.
  - Adolescent ($N=97$): LR AUROC 0.5180, Linear SVM 0.4468, RF 0.5869, ET 0.5614, LGBM 0.5693, MLP 0.5555.
  - Child ($N=262$): LR AUROC 0.4473, Linear SVM 0.4700, RF 0.4362, ET 0.4400, LGBM 0.4145, MLP 0.4470.
  - Toddler ($N=850$): LR AUROC 0.5527, Linear SVM 0.5458, RF 0.5528, ET 0.5570, LGBM 0.5519, MLP 0.5496.
- **Full ABIDE I Leaderboard (N=854, 20 Sites):**
  - Random Forest: LOSO AUROC **0.7003**, Random CV 0.6998 (Gap: -0.0005)
  - LightGBM: LOSO AUROC 0.6992, Random CV 0.7019 (Gap: +0.0027)
  - Extra Trees: LOSO AUROC 0.6867, Random CV 0.6959 (Gap: +0.0092)
  - Heinsfeld Autoencoder: LOSO AUROC 0.6820, Random CV 0.6685 (Gap: -0.0135)
  - Linear SVM: LOSO AUROC 0.6765, Random CV 0.6888 (Gap: +0.0123)
  - Small MLP: LOSO AUROC 0.6671, Random CV 0.6724 (Gap: +0.0053)
  - Logistic Regression (L2): LOSO AUROC 0.6593, Random CV 0.6728 (Gap: +0.0135)
  - Tangent Space Logistic: LOSO AUROC 0.6092.

### 2026-11-06T17:40:00+05:30 — Module 4 Uncertainty-Aware Decision Support (Part B)
- **Status:** SUCCESS
- **Script:** `src/uncertainty/calibration.py` -> `results/uncertainty_main.json`
- **Evaluated Settings:**
  - Arm 1: Leak-Free Tabular (Demographics Only) across all 4 cohorts (Adult, Adolescent, Child, Toddler).
  - Arm 2: Full ABIDE I ($N=854$, 20 sites) under LOSO, with calibration drawn strictly from training sites only.
  - Arm 3: Labeled `LEAKY CONTRAST ONLY` (Adult questionnaire items).
- **Conformal Set-Size Identity Verification:**
  - Rigorously verified: $\text{mean set size} = 1.0 \times \text{fraction}_{\text{singleton}} + 2.0 \times \text{fraction}_{\text{both}} + 0.0 \times \text{fraction}_{\text{empty}}$ (error $< 10^{-6}$).
  - ABIDE LOSO Conformal LAC ($\alpha=0.10$): Marginal coverage = **87.00%**, Mean set size = **1.595**, Empty = **0.0%**, Singleton = **40.5%**, Both = **59.5%**.
  - Adult Tabular Conformal LAC ($\alpha=0.10$): Marginal coverage = **89.96%**, Mean set size = **1.553**, Empty = **0.0%**, Singleton = **44.7%**, Both = **55.3%**.
- **Per-Site Coverage Reporting:**
  - 4 sites over-covered ($>95\%$), 11 under-covered ($<85\%$), 5 well-calibrated ($85\%-95\%$).
  - Over-coverage explicitly documented as an artifact of conservative calibration quantiles under scanner domain shift.

### 2026-11-06T17:50:00+05:30 — Visualizations, Tables & Master Facts
- **Status:** SUCCESS
- **Figures:** All 8 figures regenerated in vector PDF and 300-dpi PNG (`figures/*.pdf`, `figures/*.png`) via `src/viz/generate_figures.py`.
- **Tables:** All 9 benchmark tables exported in both `.csv` and `.tex` (`results/tables/`) via `scripts/export_tables.py`.
- **Ledgers:** `results/paper_facts.json` and `results/SUMMARY.md` compiled via `scripts/compile_paper_facts_and_summary.py`.
