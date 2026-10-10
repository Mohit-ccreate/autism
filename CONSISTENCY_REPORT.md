# Automated Consistency Verification Report

**Date:** October 10, 2026
**Total Checks:** 17
**Passed:** 17
**Failed:** 0

| Consistency Check | Status |
| :--- | :---: |
| Total questionnaire raw rows reconcile (704, 104, 292, 1054) | **PASSED** |
| Deduplicated cohort counts reconcile (667, 97, 262, 850, sum = 1876) | **PASSED** |
| Questionnaire concordance is 100.0% at tau=7 (Adult, Adol, Child) and tau=4 (Toddler) | **PASSED** |
| ABIDE-I cohort count reconciles (OHSU:25, OLIN:28, PITT:50, SDSU:17, total=120) | **PASSED** |
| ABIDE-I 200-ROI CC200 off-diagonal edge count is 19,900 | **PASSED** |
| LOSO accuracy is 55.0% (66/120) with binomial p = 0.3153 | **PASSED** |
| Permutation test p-value is 0.2148 (within-site) and 0.1429 (global) | **PASSED** |
| Minimum detectable effect size at 80% power is 62.7% | **PASSED** |
| Holm-adjusted p-values for per-site testing reconcile (SDSU p_adj = 0.0509 > 0.05) | **PASSED** |
| Conformal test set counts reconcile (134 total: 99 controls, 35 ASD) | **PASSED** |
| Conformal coverage reconciles (128/134 = 95.52%: 96/99 controls, 32/35 ASD) | **PASSED** |
| Conformal set sizes reconcile (128 singleton, 0 pair, 6 empty = 134 total) | **PASSED** |
| Conformal empty set item sums verified in {6, 7} | **PASSED** |
| Distillation student accuracy reconciles (59.17%, 71/120) with McNemar p = 0.3018 | **PASSED** |
| All 4 vector figures in figures/ exist and are non-empty | **PASSED** |
| LaTeX braces, environments, labels, and citations fully balanced | **PASSED** |
| Zero occurrences of banned marketing/LLM tropes | **PASSED** |