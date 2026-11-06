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
