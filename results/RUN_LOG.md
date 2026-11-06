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
