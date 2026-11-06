# AUDIT.md — Methodological Audit of Existing ASD Screening ML Pipeline

**Audit Date:** 2026-11-05  
**Auditor:** Automated systematic review (paper-revision branch)  
**Git commit audited:** 8cb7dfc (pre-revision snapshot)  
**Branch:** paper-revision  

---

## 1. DATASETS USED

### A. Behavioral / Tabular Screening Data

| Dataset | File | N (raw) | N (after dedup) | ASD rate | Download source |
|---------|------|---------|-----------------|----------|-----------------|
| UCI Adult AQ-10 | `data/uci_adult.csv` | 704 | 667 | ~32% | UCI ML Repo #426 |
| UCI Adolescent AQ-10 | `data/uci_adolescent.csv` | 104 | 97 | ~84% | UCI ML Repo #420 |
| UCI Child AQ-10 | `data/uci_child.csv` | 292 | 262 | ~73% | UCI ML Repo #419 |
| Toddler Q-CHAT-10 | `data/toddler_qchat10.csv` | 1,054 | 850 | ~65% | Kaggle/Thabtah |

**Version note:** Files already downloaded; no SHA256 hashes recorded in prior pipeline.  
**Action required:** Compute and store SHA256 for each raw file (see Phase 0 fix).

### B. Neuroimaging (ABIDE I)

| Dataset | Cache file | N loaded | Sites | Download method |
|---------|-----------|----------|-------|-----------------|
| ABIDE I CC200 | `data/abide_cache.npz` (N/A — uses .1D files) | 120 subjects | 4 sites (OHSU, OLIN, PITT, SDSU) | nilearn `fetch_abide_pcp` or manual .1D |
| ABIDE I full | `data/abide/ABIDE_pcp/cpac/filt_global/*.1D` | 120 | 4 | Manual subset |

**Gap identified:** Only 4 of 17 ABIDE-I sites used. Full dataset has ~871 quality-controlled subjects across 17 sites. The existing pipeline uses a convenience subset (N=120). This limits generalizability claims.

---

## 2. PREPROCESSING PIPELINE REVIEW

### `src/data_preprocessing.py`

| Step | Line(s) | Description | Leakage Risk | Verdict |
|------|---------|-------------|-------------|---------|
| Load CSV | 28-29 | Reads raw data | None | ✅ OK |
| Rename columns | 32-38 | Standardizes names | None | ✅ OK |
| `leakage_score` column kept | 41-44 | Sum of A1-A10 or `result` | **HIGH** if used in training | Correctly handled — excluded from Var A and B |
| Target encoding | 47 | YES→1, NO→0 | None | ✅ OK |
| Item binarization | 50-51 | Clip 0-1 | None | ✅ OK |
| Age imputation | 54-58 | Median of training set? | **MEDIUM** | ⚠️ Imputation applied to full data before split |
| Sex/jaundice/family binary | 61-63 | String→0/1 | None | ✅ OK |
| Deduplication | 68-71 | Drop exact item+age+sex+target duplicates | None | ✅ OK |
| Train/test split | 163-168 | Stratified 80/20, seed=42 | None | ✅ OK |

**Issue 1 (Medium severity):** Age imputation at line 54-58 uses `df['age'].median()` on the FULL dataframe before `train_test_split`. If test rows have missing age, the imputed value is informed by test data. Fix: impute inside the split (compute median only on train, apply to both).

### `src/train_abide.py`

| Step | Line(s) | Description | Leakage Risk | Verdict |
|------|---------|-------------|-------------|---------|
| Load .1D files | 134-204 | Pearson correlation, Fisher-z | None | ✅ OK |
| ComBat fit | 247 | `parametric_combat_fit` on train fold only | **None — correctly inside LOSO fold** | ✅ OK |
| ComBat transform | 248 | Applied to test fold | None | ✅ OK |
| Feature selection | 251-253 | ANOVA, `fit_transform` on train only | None | ✅ OK |
| Scaler | 255-257 | StandardScaler fit on train only | None | ✅ OK |
| Random CV (5-fold) | Separate eval | Splits ignore site boundaries | **HIGH** — site leakage documented | ⚠️ Correctly labeled as "Inflated" |

**Issue 2 (Design):** N=120 from 4 sites is a very small subset. Existing paper acknowledges this but it limits statistical power. Full dataset fetch should be attempted.

### `src/train_questionnaire.py`

| Step | Description | Risk | Verdict |
|------|-------------|------|---------|
| Feature variants A/B | A=10 items only, B=10 items + demographics | Correct — leakage score excluded | ✅ OK |
| Optuna tuning | Hyperparameter search | Need to verify inner CV used | ⚠️ Check that tuning uses only training data |
| Calibration | Platt/isotonic | Fit on training fold | Need to verify |

**Issue 3 (Medium):** Need to verify that Optuna hyperparameter tuning in `train_questionnaire.py` uses only training-fold data and not the test set.

---

## 3. MODELS AND EVALUATION

### Tabular Models (Questionnaire)

| Model | Variants | CV Protocol | Seeds | Inner CV |
|-------|----------|-------------|-------|---------|
| Logistic Regression | Var A, B, Leakage | 5×10 Repeated StratKFold | seed=42 | None (fixed C) |
| LightGBM, XGBoost, CatBoost | Var A, B | 5×10 Repeated StratKFold | seed=42 | None (fixed params) |

**Issue 4 (Medium):** Only seed 42 used. Specification requires seeds 0-9 for reproducibility assessment. Single seed inflates confidence in reported means.

### Neuroimaging Models (ABIDE)

| Model | Protocol | Sites | Features |
|-------|----------|-------|---------|
| L2 LR | LOSO | 4 | Top-250 ANOVA, ComBat |
| Linear SVM | LOSO | 4 | Top-250 ANOVA, ComBat |
| RF | LOSO | 4 | Top-250 ANOVA, ComBat |
| LightGBM | LOSO | 4 | Top-250 ANOVA, ComBat |
| BNT, ViT, CNN, GAT | LOSO | 4 | Full connectivity |
| L2 LR | Random 5-fold (leaky) | 4 | Top-250 ANOVA, ComBat |

---

## 4. LEAKAGE RISKS — COMPREHENSIVE INVENTORY

### CONFIRMED LEAKAGE (documented, used as finding)

| ID | Location | Line(s) | Type | Severity | Status |
|----|----------|---------|------|----------|--------|
| L1 | `data_preprocessing.py` | 41-44 | Label-derived feature (`result`/`leakage_score` = sum of A1-A10) | **CRITICAL** | ✅ Documented as key finding; excluded from Var A and B |
| L2 | `train_abide.py` | Random 5-fold block | Site boundary violation | **HIGH** | ✅ Documented as "Inflated" arm |

### POTENTIAL LEAKAGE (to investigate)

| ID | Location | Line(s) | Type | Severity | Fix Plan |
|----|----------|---------|------|----------|----------|
| P1 | `data_preprocessing.py` | 54-58 | Age median imputation on full data before split | MEDIUM | Compute median on train only; apply to test |
| P2 | `train_questionnaire.py` | Optuna | Hyperparameter selection; need to verify test-fold isolation | MEDIUM | Confirm Optuna uses inner CV folds only |
| P3 | `forensic_evaluation.py` | Conformal block | Calibration set must be from training data only | MEDIUM | Verify q_hat fit on calibration split from training |
| P4 | Current ABIDE pipeline | ComBat | Only 4/17 sites — selection bias may inflate cross-site generalizability claims | MEDIUM | Use full 17-site dataset |
| P5 | `research_lab.py` | Deep models | BNT/ViT trained with what seed? Overfitting check? | LOW | Verify seeds and add permutation test |

### NOT LEAKAGE (but requires documentation)

| ID | Location | Description | Note |
|----|----------|-------------|------|
| N1 | `data_preprocessing.py` | Deduplication before split | Records removed before split, so no information from test bleeds into train via dedup |
| N2 | `train_abide.py` | ComBat fit on training sites only | Correct implementation |

---

## 5. RANDOM SEEDS AND REPRODUCIBILITY

| Component | Current seed | Required seeds | Gap |
|-----------|-------------|----------------|-----|
| Data split | 42 | 0-9 | Missing 0,1,...,9 except 42 |
| Questionnaire CV | 42 | 0-9 | Missing |
| ABIDE LOSO | 42 | 0-9 | Missing |
| Permutation tests | 42 (after v4 fix) | 0-9 | Partially fixed |
| Deep models | 42 | 0-9 | Missing |

---

## 6. METRICS AND STATISTICAL RIGOR

### What exists
- AUROC, accuracy, balanced accuracy, sensitivity, specificity, F1, MCC, Brier, ECE
- Wilson and Clopper-Pearson CIs for ABIDE
- Bootstrap CIs for questionnaire (5×10 CV)
- Exact binomial test for ABIDE LOSO (p=0.3153)
- Within-site permutation test (B=1,000, p=0.2148)

### What's missing per new specification
- Subject-level bootstrap (2,000 resamples) for ABIDE AUROC CIs
- Site-level cluster bootstrap for site analyses
- DeLong test or paired bootstrap for AUROC differences
- McNemar paired test (some exists for distillation)
- Wilcoxon signed-rank across sites/seeds
- Holm-Bonferroni correction for multiple comparisons
- AUPRC (average precision) for ABIDE
- MCC for ABIDE
- Tautology Index with bootstrap CIs (new metric)

---

## 7. CALIBRATION AND UNCERTAINTY

### What exists
- ECE (10-bin) in CV results
- Brier score
- Conformal prediction (LAC, single split) on adult questionnaire data
- 100 random splits for conformal coverage assessment

### What's missing
- Calibration before/after temperature scaling, isotonic regression
- Reliability diagrams (publication-quality)
- Split conformal on ABIDE (not just questionnaire)
- Mondrian (class-conditional) conformal
- Coverage under site shift (calibrate on training sites, test on held-out)
- Selective prediction / risk-coverage curves
- AURC (Area Under Risk-Coverage curve)

---

## 8. SITE/SCANNER GENERALIZATION

### What exists
- Random CV vs LOSO comparison
- ComBat harmonization (training-fold-only vs deliberate leaky arm missing)
- Site probe: pre/post-ComBat site decodability (pre=73.33%, post=DELETED due to artifact)
- Motion/age confound correlations (r=-0.14, r=-0.18)

### What's missing
- Deliberate leaky-ComBat arm (ComBat fit on all data before CV) — explicitly required to quantify inflation
- Leave-site-group-out (LSGO) beyond just LOSO
- Site as a predictor of diagnosis (base-rate probe)
- Robustness: AAL, HO atlases; filt_noglobal preprocessing
- Full 17-site ABIDE analysis (currently only 4)
- External validation ABIDE-I → ABIDE-II (at least attempt)

---

## 9. WEB APPLICATION

### What exists
- Browser-based PWA (`webapp/index.html`) with 15 tabs
- Client-side inference with ONNX/parameter vectors
- 3D brain visualization
- Ethical disclaimers

### What's missing per new specification
- Streamlit multipage app (`app/`) — required for faculty demo
- Pages: Home, Tautology Explorer, Benchmark Leaderboard, Scanner Generalization, Uncertainty Lab, Screening Simulator, Reproducibility, Audit Checklist
- `results/paper_facts.json` consumed by app
- Dockerfile for deployment

---

## 10. DOCUMENTATION AND PAPER SUPPORT

### What exists
- `main.tex` — IEEE conference paper draft (6 pages)
- `reports/` — `paper_draft.md`, `model_card.md`, etc.
- `AUDIT_REPORT.md` — 14 verified claims

### What's missing per new specification
- `AUDIT.md` (this file — new)
- `PROGRESS.md` (new)
- `results/RUN_LOG.md` (new)
- `results/SUSPICIOUS.md` (new, if needed)
- `results/SUMMARY.md` (new)
- `results/paper_facts.json` (new)
- `results/tables/*.tex` and `*.csv` (new)
- `docs/lit_audit_template.csv` (new)
- `docs/DEMO_SCRIPT.md` (new)
- `configs/*.yaml` (new)
- `Dockerfile` (new)
- `Makefile` (new)
- Publication-quality figures in `figures/` (PDF + PNG) — partial

---

## 11. IMPACT ON ORIGINAL SUBMISSION

### Claims NOT affected by audit
1. **Questionnaire tautology (100% concordance):** Confirmed correct. The label is deterministically `sum(A1..A10) >= 7` (Adult/Adolescent/Child) or `>= 4` (Toddler). This is the central finding and is robustly established.
2. **ABIDE LOSO LR accuracy = 55.0%:** Correctly computed with ComBat inside folds.
3. **BNT LOSO accuracy = 45.83%:** Correctly computed; below-chance result correctly attributed to p>>n regime.
4. **Permutation test p=0.2148:** Correctly computed with seed=42.
5. **Scanner neutralization drop (94.2% → 41.7%):** Correctly computed.

### Claims AFFECTED or requiring additional nuance
1. **"4 sites, N=120" framing:** Technically correct but incomplete. Full ABIDE-I has 17 sites, N=871. The existing claims hold but should be explicitly framed as "4-site pilot" rather than "ABIDE-I benchmark."
2. **Age imputation (Issue P1):** Minor impact — most subjects have valid age. The direction of bias is unknown but likely negligible.
3. **Single seed (Issue 4):** Results may show slightly different values across seeds. The existing values are valid but uncertainty in those values is uncharacterized.
4. **Missing deliberate leaky-ComBat arm:** Cannot currently quantify the full magnitude of ComBat-leakage inflation. This is a gap, not a falsified claim.

### Claims DELETED in v4 (correctly handled)
- Post-ComBat 0.0% site decodability (artifact — correctly removed)
- Edge hardware latency (unverified — correctly removed)
- Abraham et al. inflation claim (corrected per lit verification)

---

## 12. NEXT STEPS (Fix Plan)

| Priority | Issue | Action | Module |
|----------|-------|--------|--------|
| HIGH | Missing PROGRESS.md, RUN_LOG.md | Create immediately | docs |
| HIGH | paper_facts.json missing | Create after all runs | results |
| HIGH | Streamlit app missing | Build app/ | Module 6 |
| HIGH | Calibration + conformal on ABIDE | Build src/uncertainty/ | Module 4 |
| HIGH | Full ABIDE site analysis | Extend train_abide to 17 sites | Module 3 |
| MEDIUM | AUDIT.md formal document | ✅ This file | Phase 0 |
| MEDIUM | Leaky-ComBat arm | Add deliberately-leaky arm to ABIDE analysis | Module 3 |
| MEDIUM | Site base-rate probe | Add site→diagnosis predictor | Module 3 |
| MEDIUM | Tautology Index metric | Compute with bootstrap CIs | Module 1 |
| MEDIUM | Multiple seeds (0-9) | Update configs and scripts | Global |
| LOW | Age imputation before split | Fix in data_preprocessing.py | Phase 0 |
| LOW | SHA256 of raw data files | Compute and store | Phase 0 |
| LOW | Literature meta-audit template | Create CSV template | Module 5 |
