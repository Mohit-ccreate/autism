# PROGRESS.md — Phase Tracking for ASD Screening Paper Revision

**Branch:** `paper-revision`  
**Last updated:** 2026-11-06T12:40:00+05:30  
**Git commit:** `8cb7dfcbdb5f` -> `paper-revision` branch  

---

## Phase Status Overview

| Phase | Description | Status | Last Updated |
|---|---|---|---|
| **Phase -1** | Safety backup & git branch `paper-revision` | ✅ DONE | 2026-11-05 |
| **Phase 0** | Audit (`AUDIT.md`) | ✅ DONE | 2026-11-05 |
| **Data** | Dataset validation & SHA-256 verification | ✅ DONE | 2026-11-06 |
| **Module 1** | Behavioral tautology audit | ✅ DONE | 2026-11-06 |
| **Module 2** | Leakage-free benchmark | ✅ DONE | 2026-11-06 |
| **Module 3** | Scanner/site generalization (ABIDE I LOSO) | ✅ DONE | 2026-11-06 |
| **Module 4** | Uncertainty-aware decision support (calibration & conformal) | ✅ DONE | 2026-11-06 |
| **Module 5** | Literature meta-audit template (`docs/lit_audit_template.csv`) | ✅ DONE | 2026-11-06 |
| **Module 6** | Streamlit multi-page web app (`app/Home.py`) | ✅ DONE | 2026-11-06 |
| **Figures & Tables**| All 8 publication figures (PDF+PNG) & 5 LaTeX/CSV tables | ✅ DONE | 2026-11-06 |
| **Tests** | Full automated pytest verification suite (32/32 passing) | ✅ DONE | 2026-11-06 |
| **Docs & Artifacts** | `paper_facts.json`, `SUMMARY.md`, `SUSPICIOUS.md`, `README.md` | ✅ DONE | 2026-11-06 |

---

## Detailed Phase Status & Findings

### Phase -1: Safety Backup ✅ DONE
- [x] Committed snapshot before revision to `main` branch (commit `8cb7dfc`).
- [x] Created and checked out revision branch `paper-revision`.
- [x] Filesystem backup verified; logged in `results/RUN_LOG.md`.

### Phase 0: Audit ✅ DONE
- [x] Comprehensive methodological audit documented in `AUDIT.md`.
- [x] Identified 2 confirmed leakage modes (L1: deterministic threshold; L2: scanner hardware cross-fold leakage).
- [x] Documented impact on original publication claims.

### Data Validation & Integrity ✅ DONE
- [x] UCI Adult AQ-10 (N=667 clean): SHA-256 `657021cddb337358640470cea758b72087bc4519e1234f7bc59c2b0160a0060d`
- [x] UCI Adolescent AQ-10 (N=97 clean): SHA-256 `4753e2ce2c791c32181ad9413c55b6556828b9578a22d166d0cef7caaf93c89d`
- [x] UCI Child AQ-10 (N=262 clean): SHA-256 `aff0928863f6c930d4102bda422fb46b4a337356ef431ddfacb3bd07925f2aa2`
- [x] Toddler Q-CHAT-10 (N=850 clean): SHA-256 `818d604290cdae767f40a1bfcc2ca5634f1ba9ed1c31bfd41aa4b450943d973f`
- [x] ABIDE I CC200 Cache (N=120, 4 sites): SHA-256 `33e743f746e0fe883f7401eb0f7b2639bc3f85af88c997e786332d7f3ef1dfbc`
- [x] ABIDE II external validation marked SKIPPED (access restricted; logged).

### Module 1: Behavioral Tautology Audit ✅ DONE
- [x] **Rule-Recovery Test:** 100.0% concordance across all 4 cohorts (Adult $\tau=7$, Adolescent $\tau=7$, Child $\tau=7$, Toddler $\tau=4$; 0 discordant cases).
- [x] **Nested Feature-Set Ablation:** Var A (Items), Var B (+Demographics), Var C (Demographics only), Var D (+Total Score).
- [x] **Tautology Index:** $\text{TI} = 0.818$ (Adult), $1.031$ (Adolescent), $1.108$ (Child), $0.889$ (Toddler).
- [x] **Near-Duplicate Analysis:** Exact and near-duplicates quantified via pairwise Hamming distance.
- [x] **Label-Shuffle Sanity Control:** AUROC collapses to chance ($0.4909 - 0.5000$).

### Module 2 & 3: Scanner / Site Generalization ✅ DONE
- [x] **Honest LOSO Benchmark:** Logistic Regression + fold-internal ComBat achieves 55.00% accuracy, AUROC 0.6098 (not significant vs chance, binomial $p = 0.3153$).
- [x] **Random CV Leakage Gap:** Random 5-fold CV inflates LR to 58.33% and deep BNT to 68.33% (which collapses to 45.83% under honest LOSO).
- [x] **Harmonization Comparison:** Evaluated None (0.615), Site Regression (0.598), Fold-Internal ComBat (0.605), and deliberately wrong **Leaky ComBat** (0.627).
- [x] **Site Probe:** Pre-ComBat site predictability is 73.3% (vs 25.0% chance).
- [x] **Confound Controls:** Motion $|r| = 0.1399$ ($< 0.15$), Age $|r| = 0.1839$ ($< 0.25$).

### Module 4: Uncertainty-Aware Decision Support ✅ DONE
- [x] **Probability Calibration:** Uncalibrated ECE (0.0594) reduced to 0.0182 with temperature scaling ($T=0.82$).
- [x] **Split Conformal Prediction:** 97.76% empirical marginal coverage at $\alpha = 0.10$ (nominal 90%) with mean set size 0.978.
- [x] **Selective Prediction:** Risk-coverage trade-off curves computed (AURC = 0.0001 on calibrated screener).
- [x] **Clinical Triage Policy:** Three-way decision policy (screen+, screen-, refer) with 2.24% triage referral burden.

### Module 5: Literature Meta-Audit Template ✅ DONE
- [x] `docs/lit_audit_template.csv` created with required schema (empty rows for human authoring).
- [x] `scripts/summarize_lit_audit.py` created to summarize counts, percentages, and generate figures.

### Module 6: Web App & Faculty Demonstration ✅ DONE
- [x] Streamlit multi-page platform (`app/Home.py` + 7 sub-pages in `app/pages/`).
- [x] Persistent medical disclaimer: *"RESEARCH PROTOTYPE — NOT A DIAGNOSTIC DEVICE"*.
- [x] Presentation walkthrough script (`docs/DEMO_SCRIPT.md`).

### Quality Gates & Reproducibility ✅ DONE
- [x] All 8 publication figures generated in vector PDF + 300-dpi PNG (`figures/`).
- [x] All 5 main benchmark tables exported in both `.csv` and `.tex` (`results/tables/`).
- [x] Master facts JSON (`results/paper_facts.json`) and summary (`results/SUMMARY.md`).
- [x] Suspicious result diagnostics (`results/SUSPICIOUS.md`).
- [x] Full automated test suite passes: **32/32 tests passing in <2 seconds**.
- [x] `Dockerfile` and `Makefile` for one-command containerized reproduction.

---

## Commands to Reproduce Everything

```powershell
# 1. Activate environment
.\venv\Scripts\Activate.ps1

# 2. Run original baseline pipeline (Phases 1-5)
python run_all.py

# 3. Run extended audit pipeline (Modules 1-6)
python run_extended.py

# 4. Export LaTeX and CSV tables
python scripts/export_tables.py

# 5. Run complete test suite (32 tests)
python -m pytest tests/ -v

# 6. Launch Streamlit research platform demo
streamlit run app/Home.py
```
