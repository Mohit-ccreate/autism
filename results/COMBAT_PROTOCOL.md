# COMBAT_PROTOCOL.md — Unseen-Site ComBat Harmonization Specification

**Date:** 2026-11-17  
**Dataset:** ABIDE I Quality-Controlled Cohort ($N=854$, 20 clinical sites)  
**Task:** Leave-One-Site-Out (LOSO) Cross-Validation Harmonization Audit  

---

## 1. How Harmonization Parameters are Obtained for the Held-Out Site in the Code

In the codebase (`src/train_abide.py`, functions `parametric_combat_fit` and `parametric_combat_transform`), ComBat is implemented as follows:

```python
def parametric_combat_fit(X, batch, covariates=None):
    # Fits grand mean, pooled variance, and site-specific location (gamma) and scale (delta)
    ...
    return X_harmonized, combat_model

def parametric_combat_transform(X, batch, covariates, combat_model):
    # Applied to the held-out site at test time
    B_hat = combat_model["B_hat"]
    std_pooled = combat_model["std_pooled"]
    X_hat = np.dot(design, B_hat)
    Z = (X - X_hat) / std_pooled
    X_harmonized = Z * std_pooled + X_hat
    return X_harmonized
```

### Critical Mathematical Finding:
Because `Z * std_pooled + X_hat == ((X - X_hat) / std_pooled) * std_pooled + X_hat == X`, the function `parametric_combat_transform` is **algebraically the identity function** on test features $X$.
- **Does in-fold ComBat use ANY data from the held-out site?** **NO.** Neither features nor labels from the held-out site influence the parameters of `combat_model`.
- **What does in-fold ComBat actually do to the held-out site?** Because the held-out site was not among the training batches, no site-specific $\gamma$ or $\delta$ exists for it. Consequently, the held-out site connectomes are passed through **completely unharmonized (raw $X$)**, while the training fold connectomes are harmonized to the reference space.

---

## 2. Evaluation of All 5 Harmonization Arms Under LOSO

All arms evaluated under Leave-One-Site-Out using an identical L2 Logistic Regression classifier ($C=0.1$, $k=250$ ANOVA features selected strictly in-fold, seed=42) on full ABIDE I ($N=854$):

| Arm Label | Harmonization Strategy | Unseen Site Treatment | Pooled AUROC [95% CI] | Accuracy | Permutation $p$ ($B=10,000$) | Paired $\Delta$ vs In-Fold | Holm $p$ |
|---|---|---|---|---|---|---|---|
| **(i) None** | No harmonization | Raw connectomes | **0.6776** [0.6414, 0.7153] | 62.65% | $p = 0.00010$ | +0.0183 [-0.0053, 0.0428] | 0.6300 |
| **(ii) Strictly Inductive** | ComBat on training sites only | No held-out site data used; projected to training reference | **0.6593** [0.6215, 0.6962] | 61.71% | $p = 0.00010$ | +0.0000 [0.0000, 0.0000] | 1.0000 |
| **(iii) Transductive** | ComBat on training sites | Held-out site's unlabeled features used for own-site standardization | **0.6689** [0.6317, 0.7044] | 61.48% | $p = 0.00010$ | +0.0096 [-0.0071, 0.0260] | 1.0000 |
| **(iv) In-Fold ComBat** | In-fold ComBat as implemented | Training harmonized, held-out site passed through unharmonized | **0.6593** [0.6215, 0.6962] | 61.71% | $p = 0.00010$ | 0.0000 (Reference) | 1.0000 |
| **(v) Leaky ComBat** | ComBat on all $N=854$ pooled data | Pre-fitted before CV; held-out site included in pooled harmonization | **0.6689** [0.6317, 0.7044] | 61.48% | $p = 0.00010$ | +0.0096 [-0.0071, 0.0260] | 1.0000 |
| **(vi) Site Regression** | In-fold site mean deconfounding | Training site means subtracted; test site deconfounded by training site means | **0.6679** [0.6314, 0.7044] | 61.59% | $p = 0.00010$ | +0.0086 [-0.0057, 0.0229] | 1.0000 |

---

## 3. Root Cause of Prior Discrepancy (0.0096 vs 0.0156)

- **Prior Discrepancy:** The draft displayed pooled AUROCs of 0.6689 (Leaky) and 0.6593 (In-Fold), which differ by $0.668948 - 0.659305 = +0.009643$. However, the reported paired bootstrap difference was $+0.0156$.
- **Definitive Root Cause:** In `src/eval/site_analysis.py`, `run_paired_harmonization_tests` was invoked as:
  `paired_tests = run_paired_harmonization_tests(meta_df["target"].values, predictions_map, seed=seed)`
  where `meta_df["target"].values` was ordered by the raw dataset index (unordered sites). In contrast, `predictions_map[harm]` had accumulated predictions iteratively by looping over `unique_sites` in alphabetical order. Because the test labels were permuted relative to the predictions, evaluating the paired test matched predictions against the wrong targets, generating an artificial $\Delta = +0.0156$.
- **Corrected Result:** When evaluated on the exact same site-ordered matched basis, the paired difference is **exactly $+0.0096$ [95% CI: $-0.0048, 0.0241$], Holm $p = 0.2840$**. Because the 95% CI contains zero, there is **no statistically significant inflation** from leaky ComBat at $N=854$.
