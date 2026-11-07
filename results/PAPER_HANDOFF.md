# PAPER_HANDOFF.md — Forensic Literature Audit & Number Cross-Check Ledger

**Branch:** `paper-revision`  
**Date:** 2026-11-07  
**Scope:** Literature Meta-Audit Blind Re-Verification, Experimental Number Cross-Check, and Publication Figure Pack Verification.  

---

## 1. Corrected Literature Summary

We conducted a systematic empirical meta-audit of **33 real, published, peer-reviewed papers** (spanning 2013–2026) applying machine learning to autism screening across tabular questionnaires (UCI AQ-10, Q-CHAT-10; $N=16$) and neuroimaging (ABIDE I/II; $N=17$).

### Key Methodological Prevalences
- **Corpus:** $31\text{ of }33\text{ (93.9\%)}$ open-access full text; $2\text{ of }33\text{ (6.1\%)}$ publisher abstract (`abstract_only`).
- **Behavioral Tautology (Label Leakage):** $15\text{ of }33\text{ (45.5\%)}$ overall. In tabular studies, **$15\text{ of }16\text{ (93.8\%)}$** include the questionnaire items defining the diagnostic target. In neuroimaging, **$0\text{ of }17\text{ (0.0\%)}$** exhibit label leakage.
- **Data Splitting Protocols ($N=31$, 2 unclear):** Random CV in $24\text{ of }31\text{ (77.4\%)}$, Site-wise (LOSO) in $4\text{ of }31\text{ (12.9\%)}$, Subject-wise in $3\text{ of }31\text{ (9.7\%)}$. Tabular models exclusively use Random CV ($14\text{ of }14\text{, 100.0\%}$). In ABIDE neuroimaging, $10\text{ of }17\text{ (58.8\%)}$ pool across centers with random CV; only $4\text{ of }17\text{ (23.5\%)}$ evaluate Leave-One-Site-Out (LOSO) generalization.
- **Preprocessing Isolation ($N=31$, 2 unclear):** In-fold in $16\text{ of }31\text{ (51.6\%)}$; leaked/pooled across dataset in $15\text{ of }31\text{ (48.4\%)}$. Tabular: $10\text{ of }14\text{ (71.4\%)}$ leaked preprocessing. Neuroimaging: $12\text{ of }17\text{ (70.6\%)}$ properly isolated preprocessing.
- **Harmonization (ABIDE):** $14\text{ of }17\text{ (82.4\%)}$ none; $3\text{ of }17\text{ (17.6\%)}$ site regression or distance learning. Zero published papers used ComBat on full data or in-fold ComBat.
- **Calibration & Uncertainty Integration:**
  - Probability calibration reported: **$1\text{ of }33\text{ (3.0\%)}$** (Singh et al., 2026).
  - Predictive uncertainty reported: **$2\text{ of }33\text{ (6.1\%)}$** (Singh et al., 2026; Huang & Chung, 2020).
  - **Union (Calibration OR Uncertainty):** **$2\text{ of }33\text{ (6.1\%)}$**.
  - **Complement (Neither Reported):** **$31\text{ of }33\text{ (93.9\%)}$** report uncalibrated point predictions without clinical abstention.
- **External Validation:** Evaluated on external cohorts in **$5\text{ of }33\text{ (15.2\%)}$** (1 tabular [Sollis 2025]; 4 neuroimaging [Abraham 2017, Plitt 2015, Traut 2022, Kan 2022]).

### Accuracy Benchmarks Across Methodological Strata
1. **Questionnaire Models with Label-Defining Items ($N=15$):**
   - **Minimum:** 96.80% — Thabtah et al. (2022) (*Health and Technology*)
   - **Median:** 99.05%
   - **Maximum:** 100.00% — Hossain et al. (2021) (*Health Information Science and Systems*)
2. **ABIDE Models with Random/Pooled CV ($N=10$):**
   - **Minimum:** 69.50% — Ktena et al. (2018) (*NeuroImage*)
   - **Median:** 70.35%
   - **Maximum:** 76.80% — Bi et al. (2018) (*Frontiers in Genetics*)
3. **ABIDE Models with Site-Wise / LOSO CV ($N=4$):**
   - **Minimum:** 60.00% — Nielsen et al. (2013) (*Frontiers in Human Neuroscience*)
   - **Median:** 65.00%
   - **Maximum:** 67.50% — Kan et al. (2022) (*NeurIPS*)

---

## 2. Blind Re-Verification Agreement Rates & Audit List

An independent blind re-extraction was conducted across all 33 papers prior to comparing against the initial draft schema:

| Extracted Field | Agreements / Total | Agreement Rate |
|---|---|---|
| **Leaking Features** | 33 of 33 | **100.0%** |
| **Calibration Reported** | 33 of 33 | **100.0%** |
| **Uncertainty Reported** | 33 of 33 | **100.0%** |
| **External Validation** | 33 of 33 | **100.0%** |
| **Evidence Level** | 33 of 33 | **100.0%** |
| **Harmonization** | 30 of 33 | **90.9%** |
| **Preprocessing Fit In-Fold** | 25 of 33 | **75.8%** |
| **Split Type** | 24 of 33 | **72.7%** |
| **Dataset Description** | 10 of 33 | **30.3%** *(text phrasing)* |
| **Sample Size (N)** | 14 of 33 | **42.4%** *(sub-cohort breakdown)* |
| **Reported Accuracy String** | 2 of 33 | **6.1%** *(verbatim quote differences)* |

### Papers Flagged for Human Review (`needs_human_check = Y`)
In accordance with Rule 7, all 23 papers whose categorical fields were modified during reconciliation or remain based on abstracts/partial text are flagged:
- **Abstract Only (2):** Vakadkar et al. (2021), Hadi et al. (2026).
- **URL Fixes (4):** Sollis et al. (2025, PMC12594979), Shrivastava et al. (2024, PMC10917726), Tartarisco et al. (2021, PMC8004748), Katlariwala et al. (2026, 10.1101/2026.11.01.26367003).
- **Categorical Reconciliations (17):** Thabtah (2022), Abu-Errub (2025), Hasan (2023), Nielsen (2013), Li (2018), Sherkatghanad (2020), Dvornek (2017), Kan (2022), Hossain (2021), Thabtah (2020), Heinsfeld (2018), Bi (2018), Shao (2023), Huang (2020), Abraham (2017), Plitt (2015), Ktena (2018).

---

## 3. Paper Number Cross-Check Mismatches Only

Across 50 empirical claims evaluated against `results/*.json` in `results/paper_number_check.csv`, 44 matched exactly. The 6 mismatches are:

1. **Child Empirical Recovered Cutoff Accuracy:**
   - *Paper/Prompt Value:* 7 (99.98%)
   - *Results Value:* 7 (100.00%)
   - *Source File:* `results/dedup_and_threshold_audit.json`
   - *Diagnosis:* In the deduplicated Child cohort ($N=262$), all 262 instances match $\tau=7$ with 0 discordant cases (100.00%). 99.98% was an artifact from the legacy pre-deduplication dataset.
2. **LightGBM Leaderboard Accuracy:**
   - *Paper/Prompt Value:* 64.64% (LOSO AUROC 0.6992 [0.6631, 0.7336], Rand CV 0.7019)
   - *Results Value:* 64.29% (549 of 854 subjects; balanced accuracy 63.81%)
   - *Source File:* `results/benchmark_leaderboard.json`
   - *Diagnosis:* AUROCs and CIs match exactly; draft accuracy was slightly misquoted by +0.35 percentage points.
3. **Autoencoder Leaderboard Accuracy:**
   - *Paper/Prompt Value:* 62.88% (LOSO AUROC 0.6820 [0.6453, 0.7175], Rand CV 0.6685)
   - *Results Value:* 62.65% (535 of 854 subjects; balanced accuracy 62.53%)
   - *Source File:* `results/benchmark_leaderboard.json`
   - *Diagnosis:* AUROCs and CIs match exactly; accuracy discrepancy is 0.23 percentage points.
4. **Small MLP Leaderboard Accuracy:**
   - *Paper/Prompt Value:* 62.41% (LOSO AUROC 0.6671 [0.6288, 0.7042], Rand CV 0.6724)
   - *Results Value:* 61.83% (528 of 854 subjects; balanced accuracy 61.63%)
   - *Source File:* `results/benchmark_leaderboard.json`
   - *Diagnosis:* AUROCs and CIs match exactly; draft accuracy was misquoted by +0.58 percentage points.
5. **Tangent Space Logistic Leaderboard Accuracy:**
   - *Paper/Prompt Value:* 58.78% (LOSO AUROC 0.6092 [0.5707, 0.6481])
   - *Results Value:* 59.72% (510 of 854 subjects; balanced accuracy 59.50%)
   - *Source File:* `results/benchmark_leaderboard.json`
   - *Diagnosis:* AUROC matches exactly; actual accuracy is 59.72% (+0.94 percentage points higher than draft).
6. **Conformal ABIDE LOSO 95% Coverage & Set Size:**
   - *Paper/Prompt Value:* 94.03% coverage, 1.815 mean size, 0/18.5/81.5 decomposition, 81.5% referral, 74.05% decided acc
   - *Results Value:* 92.39% coverage, 1.733 mean size, 0/26.7/73.3 decomposition, 73.3% referral, 73.68% decided acc
   - *Source File:* `results/uncertainty_main.json`, `results/tables/table_conformal_policy.csv`
   - *Diagnosis:* Prompt draft numbers reflected an intermediate test fold. The true full-dataset ($N=854$) LAC at $\alpha=0.05$ achieves 92.39% marginal coverage and 1.733 mean set size.

---

## 4. Per-Site LOSO Summary

Evaluating Leave-One-Site-Out across all 20 clinical sites ($N=854$) reveals marked inter-site variance:

- **Per-Site LOSO AUROC:**
  - Minimum: **0.3654** (OHSU, $n=25$)
  - Median: **0.6464**
  - Mean: **0.6616**
  - Maximum: **0.9375** (CMU, $n=8$)
  - Range: $[0.3654, 0.9375]$
- **Per-Site LOSO Accuracy:**
  - Minimum: **44.00%** (OHSU, $n=25$)
  - Median: **61.76%**
  - Mean: **61.69%**
  - Maximum: **77.61%** (USM, $n=67$)
  - Range: $[44.00\%, 77.61\%]$
- **Pooled Full-Dataset Benchmark:** LOSO AUROC = **0.6593**, Accuracy = **61.71%** (527 of 854 correct).

**Consistency Finding:** Pooled performance closely aligns with the per-site median ($0.6593\text{ vs }0.6464\text{ AUROC}$; $61.71\%\text{ vs }61.76\%\text{ accuracy}$). However, individual site performance fluctuates drastically from near-failure ($0.3654$ at OHSU) to high decodability ($0.7994$ at USM, $0.9375$ at CMU), proving that scanner hardware and protocol variations create strong non-exchangeability across centers.

---

## 5. Figure Issues Audited and Resolved

All 9 publication figures were compiled into `figures/paper_pack/` (vector `.pdf` and 300-DPI `.png`) and verified in `FIGURE_CHECK.md` and `MANIFEST.csv`:
- `fig1_pipeline`: Forensic audit and benchmark workflow schematic.
- `fig2_tautology_ablation`: Feature ablation demonstrating collapse to chance on demographics.
- `fig3_loso_vs_random`: Leaderboard scanner generalization gap across 7 model families.
- `fig4_site_probe_confounds`: Multi-class site decodability (64.87% vs 19.91% base rate) and confound correlations.
- `fig5_harmonization`: Harmonization comparison showing no detectable inflation from leaky ComBat ($\Delta=+0.0156$, $p_{\text{holm}}=0.2840$).
- `fig6_reliability`: Temperature scaling calibration curves for Adult Tabular and ABIDE LOSO.
- `fig7_conformal_per_site`: Conformal coverage variations under LOSO across all 20 clinical sites.
- `fig8_risk_coverage`: Risk-coverage selective prediction and triage policy curves.
- `fig9_literature_audit`: 4-panel literature meta-audit summary of 33 published papers.

**Audited Design Standards:**
1. **Typography:** Standardized 10 pt labels; legible at single-column (8.5 cm) and double-column (17.5 cm) widths.
2. **Units & Labels:** Explicit units on all axes (AUROC, Prevalence %, Accuracy %, Bins).
3. **Color Accessibility:** Wong (2011) colorblind-safe hex values (`#0072B2`, `#E69F00`, `#009E73`, `#D55E00`, `#999999`).
4. **Data Parity:** Every bar, point, and confidence interval matches `results/*.json`.

---

## 6. Execution Constraints & Unresolved Tasks

1. **ABIDE II CC200 External Validation:** Standardized CPAC CC200 preprocessed time-series are unavailable on public AWS S3 (`s3://fcp-indi/data/Projects/ABIDE2/` returns `HTTP 404`). Raw DICOM preprocessing requires manual container pipelines or authenticated portal access. Correctly cataloged as **SKIPPED** in `results/paper_facts.json`.
2. **Paywalled Literature Papers:** Vakadkar et al. (2021) and Hadi et al. (2026) were evaluated via publisher abstracts due to paywalls. All findings are conservatively labeled `abstract_only` and marked for human verification (`needs_human_check = Y`).
3. **Full Automated Quality Gates:** All **40 of 40 automated tests** in `pytest` are passing. Clean reproducibility is maintained on branch `paper-revision`.
