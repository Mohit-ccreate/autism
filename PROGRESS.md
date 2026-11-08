# PROGRESS.md — Phase Tracking for ASD Screening Paper Revision

**Branch:** `paper-revision`  
**Last updated:** 2026-11-07T12:30:00+05:30  
**Status:** All Correction Pass Phases, Blind Re-Verification, Number Cross-Checks & Figure Pack Complete (40/40 Pytest Quality Gates Passing)

---

## Phase Status Overview

| Phase | Description | Status | Last Updated |
|---|---|---|---|
| **Phase -1** | Safety backup & git branch `paper-revision` | ✅ DONE | 2026-11-05 |
| **Phase 0** | Methodological audit (`AUDIT.md`) | ✅ DONE | 2026-11-05 |
| **Correction Pass A** | Scale ABIDE I to full dataset ($N=854$ QC, 20 sites; $N=989$ non-QC) | ✅ DONE | 2026-11-06 |
| **Correction Pass B** | Fix Uncertainty Module (leak-free + ABIDE LOSO; set-size identity) | ✅ DONE | 2026-11-06 |
| **Correction Pass C** | Statistical Honesty ($N=120$ vs $N=854$ reconciliation; paired bootstrap; raw/clipped TI; dedup accounting; threshold sweep) | ✅ DONE | 2026-11-06 |
| **Correction Pass D** | Missing components (8-model leaderboard; Heinsfeld AE; ABIDE II skip; 8-page Streamlit app; 8 figures & 10 tables) | ✅ DONE | 2026-11-06 |
| **Correction Pass E** | Claim Language (nuanced verdicts: SUPPORTED, PARTIALLY SUPPORTED, NOT SUPPORTED) | ✅ DONE | 2026-11-06 |
| **Correction Pass F** | Quality Gates (40 passing automated tests; clean reproducibility) | ✅ DONE | 2026-11-07 |
| **Correction Pass G** | Literature Meta-Audit ($N=33$ published papers, 18-col audit, summary JSON/figures) | ✅ DONE | 2026-11-07 |
| **Correction Pass H** | Blind Re-Verification, Number Cross-Check ($50$ claims, $6$ mismatches), Figure Pack (`fig1`-`fig9`), & Paper Handoff | ✅ DONE | 2026-11-07 |
| **Correction Pass I** | NeuroTriage ML Ultra-Modern Web Application (3D Connectome, Screener, Conformal Triage, Benchmark Charts, 14-Item Checklist) | ✅ DONE | 2026-11-08 |
| **Correction Pass J** | "Anti-Gravity" Dark Glassmorphism UI/UX Redesign (Killed scrollbar, floating pills, elevation cards, CC200 live badge) | ✅ DONE | 2026-11-08 |

---

## Detailed Correction Pass Deliverables & Findings

### Part A: Full ABIDE I Scaling ($N=854$, 20 Sites) ✅ DONE
- **Root Cause of Prior $N=120$ Limit:** Nilearn's `fetch_abide_pcp` uses sequential urllib with a rigid 15.1-second socket timeout, failing under standard Amazon S3 latency. The prior author put `n_subjects=120` from 4 sites as a shortcut.
- **Solution:** Implemented `scripts/download_all_abide_multithread.py` with 16 parallel workers and exponential backoff retry. Acquired 869/871 QC subjects and built validated caches `data/abide_full_cache.npz` ($N=854$, 20 sites) and `data/abide_raw_cache.npz` ($N=989$, 20 sites).
- **LOSO vs Random CV Benchmark:**
  - In-fold ComBat L2 Logistic Regression LOSO AUROC: **0.6593** [Subject 95% CI: $0.6215, 0.6962$; Site Cluster 95% CI: $0.6075, 0.7045$], Acc = 61.71%. Permutation $p < 0.0001$, Binomial $p = 7.90 \times 10^{-12}$.
  - Random 5-Fold CV AUROC: **0.6728** (Generalization gap: $+0.0135$).
- **Harmonization Comparison:**
  - None: AUROC 0.6776 [0.6409, 0.7135]
  - Site Regression: AUROC 0.6679 [0.6309, 0.7041]
  - In-Fold ComBat: AUROC 0.6593 [0.6215, 0.6962]
  - Leaky ComBat (leaked across folds): AUROC 0.6689 [0.6322, 0.7051]
  - **Paired Bootstrap Difference (Leaky minus In-Fold):** $\Delta = +0.0156$ [95% CI: $-0.0013, 0.0328$], $p_{\text{raw}} = 0.0710$, $p_{\text{holm}} = 0.2840$. **Because the 95% CI includes zero, verdict is strictly reported as: "No detectable difference at this sample size (N=854, 20 sites)".**
- **Site Probe:** Functional connectomes decode physical scanner ID with **64.87% accuracy** (vs 19.91% chance for NYU majority site), proving scanner hardware fingerprints persist.
- **Confound & Robustness Models:** Motion, age, sex, and FIQ achieve LOSO AUROC = 0.5588. The non-QC cohort ($N=989$, 20 sites) achieves LOSO AUROC = 0.6985.

### Part B: Uncertainty Module Overhaul ✅ DONE
- **Leaky Contrast Labeling:** The questionnaire item model was explicitly labeled `LEAKY CONTRAST ONLY` and retained solely as a methodological baseline.
- **Leak-Free Tabular Models (Demographics Only):**
  - Adult Demographics LAC ($\alpha=0.10$): Marginal coverage = **89.96%**, Mean set size = **1.553**, Empty = **0.0%**, Singleton = **44.7%**, Both = **55.3%**, Triage burden = **55.3%**.
- **ABIDE I LOSO Setting (Calibration from Training Sites Only):**
  - ABIDE LOSO LAC ($\alpha=0.10$): Marginal coverage = **87.00%**, Mean set size = **1.595**, Empty = **0.0%**, Singleton = **40.5%**, Both = **59.5%**, Triage burden = **59.5%**.
- **Set-Size Identity Enforced & Verified:**
  $$\text{mean\_set\_size} = 1.0 \times \text{fraction\_singleton} + 2.0 \times \text{fraction\_both} + 0.0 \times \text{fraction\_empty}$$
  Verified to $< 10^{-6}$ across all conformal methods and arms.
- **Per-Site Coverage Reporting:** Under LOSO, 4 sites over-cover ($>95\%$), 11 under-cover ($<85\%$), and 5 are well-calibrated ($85\%-95\%$). Over-coverage is explicitly reported as a consequence of scanner domain shift.
- **Scientific Safeguard:** Manuscript and UI explicitly state that abstention does **not** eliminate errors.

### Part C: Statistical Honesty ✅ DONE
- **Reconciliation of Preliminary $N=120$ Discrepancy:**
  - Subject-level bootstrap CI $[0.505, 0.708]$ ignored site clustering.
  - Within-site permutation test ($p=0.2148$) and exact binomial test ($p=0.3153$) respected site structure.
  - **Consistent conclusion:** On $N=120$, classification was statistically indistinguishable from random guessing. Scaled to $N=854$, signal is statistically confirmed ($p < 0.0001$).
- **Tautology Index (>1.0 Resolution):**
  - Reported both raw $\text{TI}_{\text{raw}}$ and clipped $\text{TI}_{\text{clipped}} \in [0, 1]$.
  - Child raw $\text{TI} = 1.108$ and Adolescent raw $\text{TI} = 1.031$ occur because clean demographic models perform below chance ($< 0.50$), making the numerator larger than the denominator.
- **Deduplication Accounting:**
  - Exact duplicates purged: Adult ($704 \to 667$, 37 purged), Adolescent ($104 \to 97$, 7 purged), Child ($292 \to 262$, 30 purged), Toddler ($1054 \to 850$, 204 purged). Full ablation reported with and without deduplication.
- **Threshold Sweeps:**
  - Adult, Adolescent, and Child UCI binary targets match $\tau=7$ with 100.00% concordance (vs 83.5%–89.7% at published $\tau=6$). Toddler Q-CHAT-10 matches published $\tau=4$ with 100.00% concordance. Full sensitivity exported across thresholds 5, 6, 7, 8.

### Part D: Complete Missing Components ✅ DONE
- **Module 2 Benchmark Leaderboard:** Evaluated 8 neuroimaging models (Random Forest, LightGBM, Extra Trees, Heinsfeld Autoencoder, Linear SVM, Small MLP, L2 Logistic Regression, Tangent Space Logistic) and 6 tabular models across 4 cohorts.
- **ABIDE II External Validation:** Attempted in `scripts/attempt_abide_ii.py`. Confirmed NITRC preprocessed CPAC CC200 files are not on public S3 (`HTTP 404`); institutional credentials required for raw DICOMs. Correctly marked as **SKIPPED**.
- **Streamlit Platform:** All 8 pages updated to load dynamically from `results/`.
- **Figures & Tables:** All 8 publication figures regenerated (`figures/*.pdf`, `figures/*.png`). All 10 publication tables exported (`results/tables/*.csv`, `results/tables/*.tex`). Prior runs archived in `results/legacy/pre_correction_pass/`.

### Part E: Claim Language ✅ DONE
- Replaced monolithic binary verdicts with per-component scientific verdicts in `results/paper_facts.json` and `results/SUMMARY.md`:
  - **Thesis 1 (Behavioral Tautology):** `SUPPORTED` (100% rule concordance, collapse on demographics to $\sim 0.50$).
  - **Thesis 2 (Scanner Generalization Gap):** `SUPPORTED` (Site decodability = 64.9% vs 19.9% base rate).
  - **Thesis 3 (Leaky ComBat Inflation):** `NOT SUPPORTED` (Paired bootstrap $\Delta = +0.0156$ [$-0.0013, 0.0328$], $p = 0.2840$; no detectable difference at $N=854$).
  - **Thesis 4 (Uncertainty Decision Support):** `PARTIALLY SUPPORTED` (Conformal guarantees hold marginally and on exchangeable data, but scanner shift induces per-site coverage variability, and abstention does not eliminate errors).

### Part F: Quality Gates ✅ DONE
- Automated test suite expanded to **40 tests** covering set-size identity, synthetic conformal exchangeability, in-fold ComBat isolation, schema validation, ABIDE $N=854$ dataset integrity, paired bootstrap identity, and literature meta-audit schema/summary validation. **40/40 passing (100%)**.

### Part G: Literature Meta-Audit ($N=33$ Papers) ✅ DONE
- **Corpus Coverage:** Audited 33 real, published papers spanning 2015 to 2026 across 18 columns in `docs/lit_audit_template.csv`. Every row includes an authenticated open-access URL or DOI.
  - Subsets: 16 Tabular (UCI AQ-10, Q-CHAT-10) and 17 Neuroimaging (ABIDE I/II).
  - Evidence levels: 31 full_text (93.9%), 2 abstract_only (6.1%). 3 rows flagged `needs_human_check = Y` (9.1%).
- **Key Empirical Prevalences:**
  - **Behavioral Tautology (Label Leakage):** 15 of 16 (93.8%) tabular papers include the questionnaire item responses that deterministically define the diagnostic target, reporting inflated accuracies of 96%–100%. 0 of 17 (0.0%) neuroimaging papers contain label leakage.
  - **Data Splitting Protocol:** 13 of 13 (100.0%) tabular and 10 of 17 (58.8%) neuroimaging papers rely on random cross-validation. Only 4 of 17 (23.5%) neuroimaging papers evaluate leave-one-site-out (LOSO) generalization.
  - **In-Fold Preprocessing:** 12 of 27 (44.4%) evaluated papers fit feature selection, scaling, or imputation on the pooled dataset outside cross-validation folds.
  - **Calibration & Uncertainty:** Only 1 of 33 (3.0%) papers reports calibration error; only 2 of 33 (6.1%) report predictive uncertainty. 31 of 33 (93.9%) provide uncalibrated, point-prediction outputs with no abstention capability.
  - **External Validation:** 5 of 33 (15.2%) evaluate on independent external cohorts.
- **Artifacts:**
  - Verified CSV: `docs/lit_audit_template.csv`
  - Review ledger with verbatim methods quotes: `docs/lit_audit_for_human_review.md`
  - Search log: `results/lit_search_log.md`
  - Aggregation engine: `scripts/summarize_lit_audit.py`
  - Structured summary: `results/lit_audit_summary.json`
  - Publication figure: `figures/lit_meta_audit_summary.png` & `figures/lit_meta_audit_summary.pdf`

### Part H: Blind Re-Verification, Number Cross-Check & Paper Pack ✅ DONE
- **Blind Independent Re-Verification:**
  - `results/lit_reverify_blind.csv`: Re-extracted fields across all 33 papers from full text / publisher pages prior to comparing against v1 template.
  - `results/lit_reverify_comparison.csv`: Computed exact agreement rates per field across 363 comparison cells (100% agreement on leaking features, calibration, uncertainty, external validation, evidence level).
  - Preserved original template as `docs/lit_audit_template_v1.csv`; updated `docs/lit_audit_template.csv` with reconciled categorical fields and corrected URL typos. Flagged all changed or uncertain rows as `needs_human_check = Y` (23 of 33, 69.7%).
  - Computed exact union: Calibration OR Uncertainty = **2 of 33 (6.1%)**; Neither = **31 of 33 (93.9%)**.
  - Accuracy benchmarks: Questionnaire items Min 96.80% (Thabtah 2022), Median 99.05%, Max 100.00% (Hossain 2021); ABIDE Random CV Min 69.50% (Ktena 2018), Median 70.35%, Max 76.80% (Bi 2018); ABIDE LOSO Min 60.00% (Nielsen 2013), Median 65.00%, Max 67.50% (Kan 2022).
- **Paper Number Cross-Check:**
  - `results/paper_number_check.csv`: Evaluated 50 claims against `results/*.json`. 44 matched exactly; 6 mismatches flagged (Child recovered cutoff accuracy 100.0% vs draft 99.98%; LightGBM accuracy 64.29% vs 64.64%; Autoencoder accuracy 62.65% vs 62.88%; MLP accuracy 61.83% vs 62.41%; Tangent Space accuracy 59.72% vs 58.78%; Conformal ABIDE LOSO 95% coverage 92.39% vs 94.03%).
- **Publication Figure Pack:**
  - `figures/paper_pack/`: 9 figures in vector `.pdf` and 300-DPI `.png` named `fig1` through `fig9`. Verified typography, single/double column legibility, explicit units, open whitespace legend placement, and Wong (2011) colorblind accessibility.
  - `figures/paper_pack/MANIFEST.csv` & `figures/paper_pack/FIGURE_CHECK.md`.
- **Master Handoff Ledger:**
  - `results/PAPER_HANDOFF.md`: Self-contained 1,432-word ledger detailing all findings, reconciliation decisions, mismatches, and per-site distributions.

---

## Commands to Reproduce Everything from Scratch

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Download and validate full ABIDE I dataset (N=854 QC subjects across 20 sites)
python scripts/download_all_abide_multithread.py

# 3. Run deduplication audit and empirical threshold sensitivity sweep
python src/audit/dedup_and_threshold_audit.py

# 4. Run full ABIDE I site analysis (LOSO across all 20 sites, paired bootstrap, site probe)
python src/eval/site_analysis.py

# 5. Run full benchmark leaderboard (8 neuroimaging + 6 tabular models)
python src/eval/benchmark.py

# 6. Run uncertainty module (calibration, conformal LAC/APS/Mondrian, selective prediction)
python src/uncertainty/calibration.py

# 7. Generate publication vector figures and LaTeX/CSV tables
python src/viz/generate_figures.py
python scripts/export_tables.py
python scripts/compile_paper_facts_and_summary.py

# 8. Run literature meta-audit aggregation and generate figures
python scripts/summarize_lit_audit.py

# 9. Run blind re-verification, paper number cross-check, and assemble figure pack
python scripts/reverify_literature_audit.py
python scripts/generate_paper_number_check.py
python scripts/build_paper_pack.py

# 10. Run full automated verification suite (40 tests)
python -m pytest tests/ -v

# 11. Launch Streamlit interactive research platform
streamlit run app/Home.py

# 12. Launch ultra-modern NeuroTriage ML webapp (http://localhost:8000/neurotriage.html)
python -m http.server 8000 --directory webapp
```



