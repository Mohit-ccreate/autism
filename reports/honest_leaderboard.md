# Honest Model Leaderboard

**Generated:** 2026-09-30 01:00:24  
**Evaluation Standard:** 100% Real Evaluated Numbers from Untouched Test Partitions and Leave-One-Site-Out Cross-Validation.

---

## 1. Questionnaire Screening Models (Untouched 20% Test Sets)

| Cohort     | Input Features       | Best Architecture   | Accuracy                | Sensitivity   | Specificity   | ROC-AUC              |   Brier Score | Conformal Uncertainty   |
|:-----------|:---------------------|:--------------------|:------------------------|:--------------|:--------------|:---------------------|--------------:|:------------------------|
| Adult      | Items Only           | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0105 | 5.2%                    |
| Adult      | Items + Demographics | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0108 | 6.7%                    |
| Adult      | Items + Demographics | StackingEnsemble    | 93.28% [88.1%-97.0%]    | 82.86%        | 96.97%        | 0.9739 [0.946-0.994] |        0.0578 | 3.7%                    |
| Adult      | Items + Demographics | LogisticRegression  | 73.88% [66.4%-81.3%]    | 0.00%         | 100.00%       | 0.6866 [0.581-0.796] |        0.185  | 52.2%                   |
| Adolescent | Items Only           | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0383 | 0.0%                    |
| Adolescent | Items + Demographics | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0415 | 5.0%                    |
| Adolescent | Items + Demographics | LogisticRegression  | 90.00% [75.0%-100.0%]   | 83.33%        | 100.00%       | 0.9896 [0.939-1.000] |        0.0997 | 50.0%                   |
| Adolescent | Items + Demographics | XGBoost             | 65.00% [45.0%-85.0%]    | 91.67%        | 25.00%        | 0.6198 [0.353-0.870] |        0.2326 | 95.0%                   |
| Child      | Items Only           | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0362 | 18.9%                   |
| Child      | Items + Demographics | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0375 | 22.6%                   |
| Child      | Items + Demographics | LogisticRegression  | 79.25% [67.9%-90.6%]    | 70.83%        | 86.21%        | 0.8958 [0.802-0.970] |        0.1336 | 18.9%                   |
| Child      | Items + Demographics | CatBoost            | 54.72% [41.5%-67.9%]    | 0.00%         | 100.00%       | 0.5381 [0.372-0.697] |        0.2489 | 96.2%                   |
| Toddler    | Items Only           | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0091 | 3.5%                    |
| Toddler    | Items + Demographics | LogisticRegression  | 100.00% [100.0%-100.0%] | 100.00%       | 100.00%       | 1.0000 [1.000-1.000] |        0.0085 | 5.9%                    |
| Toddler    | Items + Demographics | StackingEnsemble    | 88.24% [83.5%-92.9%]    | 91.94%        | 78.26%        | 0.9352 [0.896-0.970] |        0.0883 | 2.4%                    |
| Toddler    | Items + Demographics | CatBoost            | 72.94% [66.5%-79.4%]    | 100.00%       | 0.00%         | 0.6268 [0.525-0.717] |        0.1931 | 41.2%                   |
| Unified    | Items Only           | StackingEnsemble    | 85.64% [82.2%-89.1%]    | 84.54%        | 86.81%        | 0.9384 [0.916-0.959] |        0.1088 | 4.8%                    |
| Unified    | Items + Demographics | CatBoost            | 97.87% [96.3%-99.2%]    | 98.45%        | 97.25%        | 0.9991 [0.998-1.000] |        0.0149 | 19.4%                   |
| Unified    | Items + Demographics | CatBoost            | 76.86% [72.6%-81.1%]    | 77.32%        | 76.37%        | 0.8526 [0.815-0.888] |        0.1566 | 31.9%                   |
| Unified    | Items + Demographics | CatBoost            | 69.41% [64.4%-73.9%]    | 74.23%        | 64.29%        | 0.7484 [0.692-0.798] |        0.204  | 51.9%                   |

---

## 2. ABIDE I Brain Functional Connectivity (Leave-One-Site-Out CV)

**Overall Mean LOSO Accuracy:** **55.83%** | **Mean LOSO ROC-AUC:** **0.6073**

| Held-Out Scanner Site   |   Test Subjects | ASD / Control   | LR Accuracy   |   LR ROC-AUC | Linear SVM Accuracy   | Random Forest Accuracy   |
|:------------------------|----------------:|:----------------|:--------------|-------------:|:----------------------|:-------------------------|
| OHSU                    |              25 | 12 / 13         | 48.00%        |       0.5513 | 48.00%                | 48.00%                   |
| OLIN                    |              28 | 14 / 14         | 60.71%        |       0.5816 | 64.29%                | 64.29%                   |
| PITT                    |              50 | 24 / 26         | 50.00%        |       0.5833 | 48.00%                | 48.00%                   |
| SDSU                    |              17 | 8 / 9           | 76.47%        |       0.8472 | 76.47%                | 82.35%                   |

> [!NOTE]
> In contrast, the **5-Fold Random Split Baseline** achieved an inflated accuracy of **58.33%** and ROC-AUC of **0.6424**.  
> This discrepancy (2.5% artificial gain) demonstrates why random splitting across multi-site fMRI datasets produces ungeneralizable findings.
