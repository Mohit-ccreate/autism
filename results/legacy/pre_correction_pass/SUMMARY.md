# SUMMARY.md — Key Results and Findings

> **MEDICAL DISCLAIMER:** This is a research prototype. NOT a diagnostic device.

**Generated:** 2026-11-06T05:47:04.636436+00:00  
**Git commit:** 8cb7dfcbdb5f  

---

## 1. Datasets

| Dataset | N (raw) | N (after dedup) | ASD rate | Sites |
|---------|---------|-----------------|----------|-------|
| UCI Adult AQ-10 | 704 | 667 | ~32% | N/A |
| UCI Adolescent AQ-10 | 104 | 97 | ~84% | N/A |
| UCI Child AQ-10 | 292 | 262 | ~73% | N/A |
| Toddler Q-CHAT-10 | 1,054 | 850 | ~65% | N/A |
| ABIDE I CC200 (4-site) | — | 120 | ~50% | OHSU, OLIN, PITT, SDSU |

---

## 2. Key Findings

### Finding 1: Questionnaire Label Tautology

The ASD label in all 4 UCI/Kaggle datasets is a **deterministic step function** of the item sum:
- Adult/Adolescent/Child: `sum(A1..A10) >= 7` → ASD (100.0% concordance, 0 discordant rows)
- Toddler: `sum(A1..A10) >= 4` → ASD (100.0% concordance, 0 discordant rows)

Any ML model trained with the total score or all 10 items will approach 100% by arithmetic, not learning.

### Finding 2: Generalization Gap (ABIDE LOSO vs Random CV)

| Protocol | LR Accuracy | LR AUROC |
|----------|------------|---------|
| Random 5-fold (leaky) | 58.3% | 0.6424 |
| LOSO (honest) | 55.0% | 0.6098 |
| BNT LOSO | 45.8% | 0.4883 |

LOSO accuracy is statistically indistinguishable from chance (binomial p=0.315).

### Finding 3: Scanner Site Signatures

Site identity is predictable from connectivity features (pre-ComBat: ~73% accuracy).
Random CV inflates results by ~3.3pp (AUROC basis) due to scanner hardware leakage.

### Finding 4: Uncertainty Quantification

- Conformal coverage (LAC, α=0.10): 97.8%
- Review burden (refer-for-expert): 2.2%

---

## 3. Limitations

1. ABIDE analysis uses only 4/17 sites (N=120). Full-site validation is a next step.
2. ABIDE II external validation not performed (access restricted; SKIPPED).
3. Single primary seed (42) used for original results; seeds 0-9 added in revision.
4. Age imputation performed on full data before split (minor risk; see AUDIT.md P1).

## 4. Skipped Items

| Item | Reason |
|------|--------|
| ABIDE II external validation | Requires institutional access |
| Saliency4ASD eye-tracking | Not readily available (marked optional) |
| Full 17-site ABIDE LOSO | Nilearn cache download; attempted if network available |

## 5. Figure List

| Figure | Description | Path |
|--------|-------------|------|
| fig_pipeline | End-to-end pipeline schematic | figures/pipeline_schematic.pdf |
| fig_tautology | Tautology ablation across feature sets | figures/tautology_ablation.pdf |
| fig_loso_gap | Random-CV vs LOSO accuracy gap | figures/loso_gap.pdf |
| fig_site_probe | Site decodability and confounds | figures/site_probe.pdf |
| fig_harmonization | Harmonization comparison incl. leaky arm | figures/harmonization.pdf |
| fig_reliability | Reliability diagrams before/after calibration | figures/reliability_*.pdf |
| fig_conformal | Conformal coverage per site | figures/conformal_coverage.pdf |
| fig_risk_coverage | Risk-coverage curves | figures/risk_coverage_*.pdf |
