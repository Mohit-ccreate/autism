# SUSPICIOUS.md — Suspicious Result Diagnostics and Forensic Resolutions

This register documents every empirical observation where metric values initially appeared unexpectedly high, below chance, or contradictory, along with the mathematical/forensic diagnosis and experimental resolution.

---

## 1. Tabular Variant A AUROC ~ 1.0000 Without Total Score Column

- **Symptom:** In `src/audit/tautology_audit.py` and `src/forensic_evaluation.py`, Logistic Regression and Stacking Ensembles achieve AUROC > 0.999 on Adult, Child, and Toddler cohorts even after dropping the explicit `result` / `Qchat-10-Score` columns.
- **Initial Suspicion:** Did the data preprocessing fail to remove the total score? Did test labels leak into the feature matrix?
- **Forensic Diagnosis:**
  1. We inspected feature column names: only `A1` through `A10` were present ($10$ binary features).
  2. We inspected model weights: Logistic Regression learned near-uniform positive coefficients ($w_i \approx 0.85 - 1.10$) across all items, with bias $b \approx -7.0 \times \bar{w}$ (for adult threshold $\tau=7$) and $b \approx -4.0 \times \bar{w}$ (for toddler threshold $\tau=4$).
  3. **Mathematical Mechanism:** The Allison et al. diagnostic rule is an unweighted sum:
     $$\hat{y} = \mathbb{I}\left(\sum_{i=1}^{10} A_i \ge \tau\right)$$
     Because $A_i \in \{0, 1\}$, any linear classifier can compute $f(\mathbf{x}) = \mathbf{1}^T \mathbf{x} - \tau$, perfectly recovering the exact step function without needing the pre-summed score column.
- **Resolution:**
  - This is **not a bug in preprocessing**; it is the fundamental finding of the **Behavioral Tautology Thesis**.
  - Standard machine learning applied to all 10 items is mathematically degenerate: it merely learns addition.
  - To test genuine behavioral/demographic prediction, we designed and benchmarked **Variant C (Demographics Only: Age, Sex, Jaundice, Family ASD)**. Under Variant C, AUROC immediately collapses to **0.4460 (Child), 0.4846 (Adolescent), 0.5558 (Toddler), and 0.5910 (Adult)**, proving that the non-tautological signal in these screening datasets is weak.

---

## 2. ABIDE-I Brain Network Transformer (BNT) Random CV (68.3%) vs. LOSO (45.8%)

- **Symptom:** BNT achieves 68.33% accuracy (ROC-AUC 0.7120) under 5-Fold Stratified Random Cross-Validation, but collapses to 45.83% accuracy (ROC-AUC 0.4883, below chance) under Leave-One-Site-Out (LOSO) Cross-Validation.
- **Initial Suspicion:** Was the LOSO fold code broken, or was BNT under-trained?
- **Forensic Diagnosis:**
  1. Site probe classifiers (`src/eval/site_analysis.py`) proved that functional connectivity matrices carry high scanner hardware signatures (site decodability = 73.3% vs. 25.0% chance).
  2. Under Random CV, subjects from the same imaging site appear in both train and test partitions.
  3. With $N=120$ subjects and $p = 19,900$ connectivity edges, high-capacity attention architectures easily memorize the site-specific hardware noise covariance.
  4. In LOSO CV, test subjects originate from an entirely unseen physical scanner. The hardware signatures act as adversarial domain shift, causing performance to drop below random guessing.
- **Resolution:**
  - Confirms the **Site Generalization Thesis**: Random cross-validation on multi-center neuroimaging data yields severely inflated and clinically invalid performance estimates.
  - Regularized linear baselines (L2 Logistic Regression with fold-internal ComBat) generalize significantly better (55.00% LOSO accuracy) than overparameterized deep models.

---

## 3. ComBat Harmonization Decodability Artifact

- **Symptom:** In early pipeline iterations, post-ComBat site predictability was reported as exactly 0.00%.
- **Diagnosis:** Fitting ComBat on the pooled dataset (including test samples) prior to evaluation induces negative test covariance across sample batches, creating an artificial zero-accuracy artifact.
- **Resolution:**
  - Fold-internal ComBat fitting was strictly enforced (`parametric_combat_fit` inside training folds only).
  - The artifactual sentence was purged from the manuscript per Rule A1, and ComBat fit on full data is now explicitly evaluated and labeled as the **LEAKY arm** to quantify inflation.

---

## 4. Screening Cutoff Mismatch: UCI Label Binarization ($\tau=7$) vs. Allison et al. 2012 ($\tau=6$)

- **Symptom:** In `src/audit/dedup_and_threshold_audit.py`, testing the published screening cutoff of $\ge 6$ (Allison et al., 2012) against the dataset binary target `Class/ASD` yielded 10.3%–16.5% label discordance across Adult, Adolescent, and Child cohorts.
- **Initial Suspicion:** Was the rule extraction code buggy, or was the AQ-10 item reverse-scoring inverted?
- **Forensic Diagnosis:**
  1. We performed an empirical threshold sweep $\tau \in [1, 10]$ across all cohorts.
  2. For Adult ($N=667$), Adolescent ($N=97$), and Child ($N=262$), a threshold of $\tau=7$ yields **100.00% concordance** with the binary label:
     - Adult: $\tau=6$ accuracy is 89.66% (69 discordant cases); $\tau=7$ accuracy is 100.00% (0 discordant cases).
     - Adolescent: $\tau=6$ accuracy is 83.51% (16 discordant cases); $\tau=7$ accuracy is 100.00% (0 discordant cases).
     - Child: $\tau=6$ accuracy is 85.50% (38 discordant cases); $\tau=7$ accuracy is 100.00% (0 discordant cases).
  3. For Toddler ($N=850$), the empirical cutoff $\tau=4$ matches published Q-CHAT-10 literature with **100.00% concordance** (0 discordant cases).
  4. The UCI dataset creator (F. Thabtah) applied a strict cutoff $>6$ (equivalent to integer score $\ge 7$) rather than the standard clinical cutoff $\ge 6$ when generating the binary target `Class/ASD` for the three older cohorts.
- **Resolution:**
  - Full threshold sensitivity tables were exported to `results/tables/table_threshold_sensitivity.{csv,tex}` and logged in `results/dedup_and_threshold_audit.json`.
  - The paper explicitly notes this 1-point cutoff shift in the dataset preparation history.

---

## 5. Tautology Index Exceeding 1.0 on Child and Adolescent Cohorts

- **Symptom:** In `src/audit/tautology_audit.py`, the raw Tautology Index for the Child cohort was computed as $1.108$ and for the Adolescent cohort as $1.031$, exceeding the theoretical maximum of $1.0$.
- **Initial Suspicion:** Did the formula have an inverted denominator or sign error?
- **Forensic Diagnosis:**
  1. The Tautology Index formula is defined as:
     $$\text{TI} = \frac{\text{AUROC}_{\text{items}} - \text{AUROC}_{\text{clean}}}{\text{AUROC}_{\text{items}} - 0.50}$$
  2. When the non-leaking demographic model performs below chance ($\text{AUROC}_{\text{clean}} < 0.50$), the numerator $(\text{AUROC}_{\text{items}} - \text{AUROC}_{\text{clean}})$ is strictly larger than the denominator $(\text{AUROC}_{\text{items}} - 0.50)$.
  3. On the Child cohort, clean demographic AUROC is $0.4460$ (below chance), yielding:
     $$\text{TI}_{\text{raw}} = \frac{0.9998 - 0.4460}{0.9998 - 0.5000} = \frac{0.5538}{0.4998} = 1.108$$
  4. On the Adolescent cohort ($N=97$), clean demographic AUROC is $0.4846$, yielding $\text{TI}_{\text{raw}} = 1.031$.
- **Resolution:**
  - We report both **raw** ($\text{TI}_{\text{raw}}$) and **clipped** ($\text{TI}_{\text{clipped}} \in [0, 1]$) metrics in all tables and JSON outputs.
  - An explanatory note is provided explaining that $\text{TI} > 1.0$ is an expected mathematical consequence whenever demographic features alone possess zero or slightly negative predictive value relative to random guessing on small samples.

---

## 6. Preliminary N=120 Statistical Discrepancy: Bootstrap CI [0.505, 0.708] vs. Permutation p=0.2148 and Binomial p=0.3153

- **Symptom:** On the initial $N=120$ 4-site subset, the subject-level bootstrap 95% CI $[0.505, 0.708]$ excluded $0.50$, suggesting statistical significance, while the within-site permutation test ($p=0.2148$) and exact binomial test ($p=0.3153$) were clearly non-significant.
- **Initial Suspicion:** Was the bootstrap CI or the permutation test incorrectly coded?
- **Forensic Diagnosis:**
  1. **Subject-level bootstrap** resampled individuals with replacement uniformly across the entire pool of 120 subjects, ignoring the site-cluster structure. This artificially reduced variance and yielded an anti-conservative confidence interval.
  2. **Within-site permutation test** permuted labels strictly within each imaging site, preserving scanner-specific class imbalances and exchangeability within scanner blocks.
  3. **Exact two-tailed binomial test** evaluated overall accuracy ($66/120 = 55.00\%$) against null probability $p_0 = 0.50$ ($p = 0.3153$).
  4. Both rigorous tests demonstrated that the preliminary $N=120$ performance was **statistically indistinguishable from random guessing**.
- **Resolution:**
  - When scaled to the complete ABIDE-I dataset ($N=854$, 20 sites), both the subject-level bootstrap CI $[0.6215, 0.6962]$ and the cluster-level site bootstrap CI $[0.6075, 0.7045]$ strictly exclude $0.50$.
  - The within-site permutation test confirms $p < 0.0001$ and exact binomial test confirms $p = 7.90 \times 10^{-12}$ ($527/854$ correct = 61.71%).
  - The manuscript explicitly reconciles the $N=120$ artifact by contrasting subject-level vs. site-cluster variance.

---

## 7. Conformal Prediction Set-Size Reporting Inconsistency

- **Symptom:** An earlier draft summary contained the phrase "mean set size 0.978 with zero empty sets", which is mathematically impossible for binary prediction sets since non-empty sets have size $\ge 1$.
- **Initial Suspicion:** Was the conformal code returning set sizes $< 1.0$ while failing to flag empty sets?
- **Forensic Diagnosis:**
  1. We inspected the raw logs of the conformal calibration run on Adult data.
  2. The fraction of empty sets was actually $0.0224$ ($3/134$ cases were empty sets $\emptyset$).
  3. The fraction of singletons was $0.9776$ ($131/134$ cases), and both-label sets was $0.0000$.
  4. The true expected set size was $0.0 \times 0.0224 + 1.0 \times 0.9776 + 2.0 \times 0.0000 = 0.9776 \approx 0.978$.
  5. The draft text summary had informally claimed "zero empty sets" without checking the breakdown.
- **Resolution:**
  - Enforced the mathematical set-size identity in code and added unit tests:
    $$\text{mean\_set\_size} = 1.0 \times \text{fraction\_singleton} + 2.0 \times \text{fraction\_both} + 0.0 \times \text{fraction\_empty}$$
  - Full ABIDE LOSO Conformal LAC results: Coverage = $87.00\%$, Mean Set Size = $1.595$, Empty = $0.0\%$, Singleton = $40.5\%$, Both = $59.5\%$, Triage burden = $59.5\%$.
  - Tabular Demographics Conformal LAC results: Coverage = $89.96\%$, Mean Set Size = $1.553$, Empty = $0.0\%$, Singleton = $44.7\%$, Both = $55.3\%$.

---

## 8. Root Cause of Previous ABIDE Sample Size Limitation (N=120)

- **Symptom:** The initial repository implementation evaluated only $N=120$ subjects across 4 sites (OHSU, OLIN, PITT, SDSU) rather than the complete ABIDE-I cohort.
- **Forensic Diagnosis:**
  1. `nilearn.datasets.fetch_abide_pcp` issues sequential, single-threaded HTTP GET requests to Amazon S3.
  2. The default socket read timeout in urllib is $15.1$ seconds.
  3. Due to normal S3 API latency and network jitter, downloading hundreds of 1D files sequentially consistently raised `ReadTimeoutError` after 100–150 subjects.
  4. Rather than diagnosing the socket timeout, the previous researcher added a hardcoded argument `n_subjects=120` from 4 sites as an ad-hoc workaround.
- **Resolution:**
  - Implemented `scripts/download_all_abide_multithread.py` with 16 parallel worker threads, connection pooling, and exponential backoff retry logic.
  - Successfully fetched 869 of 871 quality-checked subjects and validated $N=854$ complete connectomes across all 20 clinical sites, alongside $N=989$ subjects for the non-QC robustness cohort.

