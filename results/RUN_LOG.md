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

### 2026-11-07T10:45:00+05:30 — Literature Meta-Audit (N=33 Published Papers)
- **Status:** SUCCESS
- **Script:** `scripts/summarize_lit_audit.py`
- **Output Artifacts:**
  - `docs/lit_audit_template.csv`: 33 authenticated, published peer-reviewed papers spanning 2015-2026 across 18 columns.
  - `docs/lit_audit_for_human_review.md`: Granular evidence ledger with clickable URLs, evidence levels, and textual excerpts for quick verification.
  - `results/lit_search_log.md`: Detailed query log across PMC, PubMed, arXiv, medRxiv, IEEE Xplore, HAL, and ETASR with inclusion/exclusion criteria.
  - `results/lit_audit_summary.json`: Strict statistical aggregations computed over non-unclear cells with explicit denominators.
  - `figures/lit_meta_audit_summary.png` & `figures/lit_meta_audit_summary.pdf`: Multi-panel publication figure displaying methodological safeguard prevalence.
  - `tests/test_lit_meta_audit.py`: 3 automated quality gates verifying schema, valid URLs, non-empty fields, and count integrity (40/40 passing overall).

### 2026-11-07T12:25:00+05:30 — Blind Re-Verification, Number Cross-Check & Paper Figure Pack
- **Status:** SUCCESS
- **Scripts:**
  - `scripts/reverify_literature_audit.py` -> `results/lit_reverify_blind.csv`, `results/lit_reverify_comparison.csv`
  - `scripts/apply_reconciled_lit_audit.py` -> `docs/lit_audit_template.csv` (backup `docs/lit_audit_template_v1.csv` preserved)
  - `scripts/generate_paper_number_check.py` -> `results/paper_number_check.csv` (50 items audited, 6 mismatches flagged)
  - `scripts/build_paper_pack.py` -> `figures/paper_pack/` (`fig1` through `fig9` in `.pdf` and 300-DPI `.png`, `FIGURE_CHECK.md`, `MANIFEST.csv`)
  - `scripts/summarize_lit_audit.py` -> regenerated `results/lit_audit_summary.json` with union calculations & accuracy distributions
- **Final Output File:** `results/PAPER_HANDOFF.md` (Self-contained, 1,432 words).

- **Key Empirical Summary:**
  - Corpus: 16 Tabular (AQ-10 / Q-CHAT-10), 17 Neuroimaging (ABIDE I/II). 31 full_text (93.9%), 2 abstract_only (6.1%), 3 flagged needs_human_check (9.1%).
  - Behavioral Tautology / Label Leakage: Tabular = 15 of 16 (93.8%); Neuroimaging = 0 of 17 (0.0%).
  - Data Splitting: Tabular Random CV = 13 of 13 (100.0%); Neuroimaging Random CV = 10 of 17 (58.8%), Site-wise (LOSO) = 4 of 17 (23.5%), Subject-wise = 3 of 17 (17.6%).
  - In-Fold Preprocessing: Tabular = 4 of 11 (36.4%); Neuroimaging = 11 of 16 (68.8%).
  - Calibration: 1 of 33 (3.0%).
### 2026-11-08T16:20:00+05:30 — Anti-Gravity Dark Glassmorphism UI/UX Redesign
- **Status:** SUCCESS
- **Files Modified:** `webapp/index.html`, `webapp/style.css`, `CHANGELOG.md`, `PROGRESS.md`
- **Enhancements Implemented:**
  1. Horizontal Scrollbar Removal: Sub-tabs styled as floating pill container with `backdrop-filter: blur(12px)`, rounded-full pills, subtle border highlights, and native scrollbar suppressed via `scrollbar-width: none` and `::-webkit-scrollbar { display: none }`.
  2. Header Density Reorganization: Secondary accessibility buttons moved into a top-right floating pill toolbar, giving title `NEURO-SCREEN AI` and status badges breathing room.
  3. Anti-Gravity Floating Depth Cards: `.glass-card` styled with `radial-gradient(135% 135% at 50% 0%, rgba(30, 41, 59, 0.45) 0%, rgba(15, 23, 42, 0.6) 100%)`, 1px subtle glass borders, elevation shadows (`box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7)`), and smooth hover lift (`transform: translateY(-3px)`).
  4. Age Cohort Redesign: Floating segmented tiles with active cyan glow outline (`box-shadow: 0 0 20px rgba(56, 189, 248, 0.25)`).
  5. Question Presentation & Tactile Answers: Crisp prominent question text (`1.25rem`, `#F8FAFC`, `font-weight: 600`), wide tactile glass answer cards with emerald hover/active glow for Agree (`#btn-ans-agree`) and indigo/slate glow for Disagree (`#btn-ans-disagree`).
  6. 3D Neural Connectome Stage: Holographic viewport cyber-glow (`radial-gradient(circle at center, rgba(56, 189, 248, 0.12) 0%, rgba(10, 15, 30, 0.8) 75%)`, `border: 1px solid rgba(56, 189, 248, 0.25)`), and animated pulsing live indicator badge (`● CC200 Atlas Live`).
  7. Refined Clinical Safety Banner: Restyled `.clinical-disclaimer-banner` into a low-profile amber glass callout.
  8. Sensory-Friendly Invariants: Disabled hover transform and animations under `body.sensory-friendly`.
- **Quality Gates:** 40/40 pytest tests pass, `node --check webapp/app.js` returns 0 syntax errors, web server returns HTTP 200 OK.

### 2026-11-09T10:45:00+05:30 — NEURO-SCREEN AI Client UI Refactor & Results Routing
- **Status:** SUCCESS
- **Files Modified:** `webapp/index.html`, `webapp/style.css`, `webapp/app.js`, `CHANGELOG.md`, `results/RUN_LOG.md`
- **Fixes Implemented:**
  1. Sub-Navigation Bar Overflow Fix: `.nav-tabs-container` updated to responsive flexbox with wrapping (`display: flex; flex-wrap: wrap; gap: 0.5rem; overflow-x: hidden; max-width: 100%;`) with native scrollbar suppression.
  2. Strengths Profile Grid & Sticky Preview: Replaced `.grid-2col` with `.profile-grid` (`display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.5rem; align-items: start;`) and made the "About Me" preview card sticky (`position: sticky; top: 1.5rem; align-self: start;`).
  3. Gating & Results Generation Flow:
     - Implemented `switchToTab(targetTabId)` ensuring synchronous updating of tab buttons and active panels.
     - Completion of question 10 or clicking `#btn-calculate`/`#submit-screening` triggers `AppState.hasCompletedAssessment = true;` and routes to `tab-results`.
     - `#results-empty-state` is hidden and `#results-calculated-container` is displayed with guarded DOM updates.
     - Added aliases `window.calculateAndDisplayResults` and `window.calculateAndRenderResults`.
  4. Dynamic Strengths & Support Profile:
     - Bound event listeners to all 12 checkboxes (`#chk-sound`, `#chk-light`, `#chk-texture`, `#chk-crowds`, `#chk-written`, `#chk-direct`, `#chk-delay`, `#chk-routine`, `#chk-detail`, `#chk-focus`, `#chk-honesty`, `#chk-creative`) and perspective buttons.
     - Updated `renderAboutMeCard()` to include tactile sensitivity (`#chk-texture`) and creative problem-solving (`#chk-creative`), dynamically tailored to Adult, Teen, and Child audiences.
     - Enhanced `#btn-copy-profile-text` and `#btn-download-profile-pdf` with robust fallbacks.
  5. Accessibility: Added high-contrast (`body.high-contrast`) and dyslexia font (`body.dyslexia-font`) styling across `.profile-grid`, `.nav-tabs-container`, and `#about-me-card-content`.
- **Quality Gates:** 40/40 pytest tests pass (6.23s), `node --check webapp/app.js` confirmed 0 errors, web server returns HTTP 200 OK.

### 2026-11-09T13:00:00+05:30 — NEURO-SCREEN AI Scoring, Explainability & UI Clipping Refactor
- **Status:** SUCCESS
- **Files Modified:** `webapp/index.html`, `webapp/style.css`, `webapp/styles.css`, `webapp/app.js`, `CHANGELOG.md`, `PROGRESS.md`, `results/RUN_LOG.md`
- **Fixes Implemented:**
  1. Removed Hard Blockers: Unlocked on-the-fly scoring and explainability calculation without requiring 100% questionnaire completion.
  2. Fallback Synthesis: Missing items resolved from active traits in Strengths/Sensory Reflection Profile or normative baseline defaults.
  3. Direct Routing to Results: Navigating directly to `tab-results` or clicking `Generate Assessment` immediately computes traits, reveals `#results-calculated-container`, hides `#results-empty-state`, and mounts metrics.
  4. Feature Attribution Component: Implemented dynamic horizontal breakdown bars for Sensory Sensitivity, Social Communication, and Detail / Focus Fixation.
  5. Console Integrity: Emits required verification string `[NeuroScreen] Results successfully generated and mounted`.
  6. Sub-Nav Bar Wrapping & Overflow: Flexbox wrapping with `max-width: 100%` and cross-browser scrollbar suppression applied to `.nav-tabs-container`.
  7. Grid Unclipping: Set `.grid-2col`, `.profile-section`, and `.profile-grid` to responsive auto-fit columns (`repeat(auto-fit, minmax(320px, 1fr))`) and sticky profile preview.
- **Quality Gates:** 40/40 pytest tests pass (33.77s), `node --check webapp/app.js` confirmed 0 syntax errors, `http://localhost:8000/index.html` confirmed HTTP 200 OK.

### 2026-11-09T17:30:00+05:30 — Unified Multi-Modal Pipeline, Connectome Ingestion & Architecture Refactor
- **Status:** SUCCESS
- **Files Modified:** `webapp/index.html`, `webapp/style.css`, `webapp/styles.css`, `webapp/app.js`, `CHANGELOG.md`, `PROGRESS.md`, `results/RUN_LOG.md`
- **Architectural Upgrades:**
  1. Multi-Modal Stepper: Systematic 3-step triage workflow (Modality selection -> Data ingestion -> Tri-envelope results) with visual active state synchronization.
  2. Production Connectome Ingestion: Multi-format dropzone (`.csv`, `.tsv`, `.json`, `.npy`, `.txt`), 3 live matrix integrity tags (200x200 symmetric, $[-1, 1]$ bounds, NaN/Inf check), 1-click presets (NYU, Stanford, Utah), 6 model family cards, and Canvas mini-heatmap preview with hover tooltip.
  3. Global Event Delegation: Eliminated inline `onclick` handlers and pointer-event click traps on button children with `e.target.closest(...)`.
  4. Reactive AppState & Multi-Modal Fusion: Centralized store `window.NeuroAppState` coupled with split-conformal inference engine computing coverage sets $C_{0.10}(x)$ and nonconformity metrics.
  5. Feature Attribution: Extended trait impact bars with Default Mode Network hyper-coupling and Frontoparietal Control Network segregation indicators.
  6. Accessibility & Doctor Export: Dyslexia font, high contrast, text scaling (100%/115%/130%), sensory mode, and clean print stylesheet (`@media print`).
- **Quality Gates:** 40/40 pytest tests pass (9.62s), `node --check webapp/app.js` confirmed 0 syntax errors, web server returns HTTP 200 OK.







### 2026-11-17T06:30:00+05:30 — Peer Review Revision Pass: Root Cause Audits, Protocols & Figure Pack
- **Branch:** `paper-revision`
- **Status:** SUCCESS
- **Execution Summary:**
  1. Module 1 (A.1, A.4, C):
     - Identified participant identifiers: Adult, Adolescent, Child contain zero clinical IDs. Toddler contains `Case_No` (1..1054 serial row index). Purged rows are identical records, not verified duplicate participants.
     - Dedup counts: Adult $704 \to 667$ (37 removed, 5.26%), Adolescent $104 \to 97$ (7 removed, 6.73%), Child $292 \to 262$ (30 removed, 10.27%), Toddler $1054 \to 850$ (204 removed, 19.35%).
     - Unified one-run evaluation on clean data:
       * Adult: Clean AUROC = 0.5801, Items AUROC = 0.9999, Items Acc = 99.55% [99.31, 99.76], Majority = 73.76%, TI Raw = 0.8398 [0.797, 0.884], Clipped = 0.8398, Alt (a) Abs Drop = 0.4199, Alt (b) Rel Acc Drop = 1.0174.
       * Adolescent: Clean AUROC = 0.5180, Items AUROC = 0.9946, Items Acc = 91.96% [89.48, 94.23] (Repeated $5\times 10$ Acc: 95.55% [84.21, 100.00]), Majority = 57.73%, TI Raw = 0.9637 [0.862, 1.069], Clipped = 0.9637, Alt (a) Abs Drop = 0.4767, Alt (b) Rel Acc Drop = 1.1145.
       * Child: Clean AUROC = 0.4473, Items AUROC = 1.0000, Items Acc = 99.54% [99.16, 99.85], Majority = 54.58%, TI Raw = 1.1054 [1.044, 1.168], Clipped = 1.0000, Alt (a) Abs Drop = 0.5526, Alt (b) Rel Acc Drop = 1.0424. Instability identified: clean AUROC (0.4473) is below chance (0.50), causing numerator to exceed denominator.
       * Toddler: Clean AUROC = 0.5527, Items AUROC = 1.0000, Items Acc = 100.00% [100.00, 100.00], Majority = 72.71%, TI Raw = 0.8946 [0.858, 0.932], Clipped = 0.8946, Alt (a) Abs Drop = 0.4473, Alt (b) Rel Acc Drop = 1.0009.
     - Without deduplication (raw): Adult TI raw 0.7813; Adolescent TI raw 1.0220; Child TI raw 1.0372; Toddler TI raw 0.8621.
  2. Module 2 (A.2, A.3, D):
     - Discrepancy between displayed difference (0.0096) and reported paired difference (0.0156) resolved: `meta_df["target"].values` in original dataframe row order was passed to `run_paired_harmonization_tests` while `predictions_map` had predictions sorted site-by-site. Once aligned, direct subtraction and paired bootstrap delta match identically: $\Delta = +0.0096$ [$-0.0071, 0.0260$], Holm $p = 1.0000$.
     - Within-site permutation test rerun with $B = 10,000$: $p = (1 + 0) / (1 + 10,000) = 0.00010$. Finite resolution limit $1/(1+B)$ respected.
     - ComBat unseen-site protocol documented in `results/COMBAT_PROTOCOL.md` across 5 arms (none, strictly inductive, transductive, in-fold as implemented, leaky).
  3. Module 3 (B):
     - Evaluated all 8 models under LOSO across all 20 sites. Excluded sites: `[]` (all 20 sites contain both classes). Saved `results/per_site_auroc.csv` (160 rows) and `results/per_site_summary.json`.
     - Site-prevalence oracle baseline: AUROC = 0.5902 (using zero imaging features).
     - Within-site z-scored ranking preserves ranking performance (e.g. LR: 0.6593 to 0.6663; Linear SVM: 0.6765 to 0.6931; RF: 0.7003 to 0.7222).
  4. Module 4 (E):
     - Documented uncertainty allocation protocol in `results/UNCERTAINTY_PROTOCOL.md`.
     - Version 1 (Shared 2-way): LAC 90% Cov = 87.00%, Set Size = 1.595.
     - Version 2 (Three strictly DISJOINT subsets): LAC 90% Cov = 89.58%, Set Size = 1.664.
     - Version 3 (Site-Grouped LOSO): LAC 90% Cov = 89.70%, Set Size = 1.635.
     - Set-size decomposition identity holds with residual $< 10^{-14}$.
  5. Module 5 (F):
     - Controlled site probe: Phenotypic alone (age, sex, FIQ, motion, diagnosis) decodes site with 22.13% (LR) / 27.17% (RF) vs 19.91% majority baseline.
     - Connectomes after in-fold confound regression decode site with 73.54% accuracy (balanced acc 64.77%). Described as site-associated information, not hardware signatures alone.
  6. Module 6 (H):
     - All 28 references in `references.bib` verified with complete author lists (all 36 authors for `maenner2023`, all 42 authors for `di2014`, zero 'others' or 'et al.'). Verified DOIs/URLs. Zero unverified guesses.
  7. Module 7 (I & J):
     - Built `docs/lit_verification_worksheet.csv` (23 rows flagged `needs_human_check`).
     - Added `scripts/recompute_lit_counts.py` with human override support and conservative subset reporting ($n=10$ unflagged full-text papers).
     - Built `results/prior_comparison.csv` (10 rows comparing audited literature to study results).
  8. Module 8 (G):
     - Generated 6 publication figures in `figures/paper_pack/` (PDF + 300-DPI PNG, colorblind-safe) and updated `MANIFEST.csv` and `FIGURE_CHECK.md`.
  9. Quality Gates (K):
     - Added 4 new unit tests in `tests/test_audit_extension.py`.
     - 44/44 pytest tests pass in 5.72s.
     - Recompiled `results/paper_facts.json`, `results/SUMMARY.md`, and all tables in `results/tables/`.
