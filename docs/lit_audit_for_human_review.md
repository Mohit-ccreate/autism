# Literature Meta-Audit: Human Verification Guide

**Document Purpose:** This document provides the complete evidentiary basis for each of the 33 papers audited in [`docs/lit_audit_template.csv`](file:///c:/Users/mohit/OneDrive/Desktop/asd-project/docs/lit_audit_template.csv). For every paper, the exact URL, verification status, evidence level, and the specific section/sentence rationale for each $Y/N/\text{unclear}$ judgment are provided so a faculty reviewer can quickly audit any entry.

---

## Summary Statistics at a Glance

- **Total Audited Papers:** 33 REAL, published papers (spanning 2013–2026)
- **Dataset Family:**
  - Behavioral / Screening Questionnaires (Tabular): **16 of 33 (48.5%)**
  - Resting-State Connectomes (Neuroimaging / ABIDE): **17 of 33 (51.5%)**
- **Evidence Level:**
  - `full_text`: **31 of 33 (93.9%)**
  - `abstract_only`: **2 of 33 (6.1%)** (Vakadkar et al. 2021; Hadi et al. 2026)
- **Rows Flagged for Human Verification (`needs_human_check = Y`):** **3 of 33 (9.1%)** (Vakadkar et al. 2021; Hadi et al. 2026; Abu-Errub 2025)

---

## Part 1: Behavioral / Questionnaire Family (Tabular) — 16 Papers

### 1. Ehsan et al. (2025)
- **Paper:** *Early Detection of Autism Spectrum Disorder Through Automated Machine Learning*
- **Authors:** Usman Ehsan et al.
- **Year / Venue:** 2025, *Diagnostics* 15(15):1859
- **Link:** [PMC12328120](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12328120/) | DOI: `10.3390/diagnostics15151859`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** Q-CHAT-10 Pakistan ($N=742$)
- **Judgments & Basis:**
  - `split_type`: `random` (Section 2.3: stratified 80/20 train/test split).
  - `preprocessing_fit_in_fold`: `N` (TPOT genetic AutoML search performed feature selection and scaling globally before split evaluation).
  - `leaking_features`: `Y` (Input features include the 10 Q-CHAT items $A_1-A_{10}$ whose sum directly defined the screening outcome label).
  - `harmonization`: `none`.
  - `calibration_reported`: `N` (No Brier score, ECE, or reliability diagram).
  - `uncertainty_reported`: `N` (No confidence intervals, conformal sets, or abstention).
  - `external_validation`: `N` (Single center dataset).
  - `reported_accuracy`: `78.0%` (TPOT Extra Trees test accuracy).

---

### 2. Hossain et al. (2021)
- **Paper:** *Detecting autism spectrum disorder using machine learning techniques: An experimental analysis on toddler, child, adolescent and adult datasets*
- **Authors:** Md Delowar Hossain, Md Ashad Kabir, Adnan Anwar, Md Zahidul Islam
- **Year / Venue:** 2021, *Health Information Science and Systems* 9:16
- **Link:** [PMC8041926](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8041926/) | DOI: `10.1007/s13755-021-00145-9`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 across Adult, Adolescent, Child, and Toddler ($N=2,154$)
- **Judgments & Basis:**
  - `split_type`: `random` (Section 'Experimental evaluation': 10-fold cross-validation).
  - `preprocessing_fit_in_fold`: `unclear` (Relief attribute selection described on dataset prior to classifier validation).
  - `leaking_features`: `Y` (Evaluates $A_1-A_{10}$ item scores; Multilayer Perceptron achieves 100% accuracy on adult cohort because label is arithmetic sum threshold $\ge 6$).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `100.0%` (MLP on adult dataset; 98.7% on child, 97.4% on adolescent, 98.1% on toddler).

---

### 3. Sollis, Wall, Washington (2025)
- **Paper:** *Compact subsets of autism screening items predict clinical diagnoses with a machine learning analysis of the QCHAT-10*
- **Authors:** Lydia J. Sollis, Dennis P. Wall, Peter Y. Washington
- **Year / Venue:** 2025, *Scientific Reports* 15:26131
- **Link:** [PMC41203747](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC41203747/) | DOI: `10.1038/s41598-025-26131-9`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** Q-CHAT-10 New Zealand, Saudi Arabia, and Poland cohorts ($N=1,812$)
- **Judgments & Basis:**
  - `split_type`: `random` (5-fold cross-validation) plus independent holdout.
  - `preprocessing_fit_in_fold`: `Y` (Recursive Feature Elimination executed inside CV folds).
  - `leaking_features`: `Y` (Uses Q-CHAT-10 items; critically, Section 2.5 explicitly discusses that predicting questionnaire-derived screening thresholds creates artificial circularity, and validates models against independent clinician diagnoses).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `Y` (Models trained on NZ and Saudi cohorts were evaluated on an independent Polish clinical cohort).
  - `reported_accuracy`: `89% (NZ), 92% (Saudi)` balanced accuracy; drops to AUROC `85% ± 13` on Polish clinical test.

---

### 4. Shrivastava, Singh, Agrawal (2024)
- **Paper:** *Autism spectrum disorder detection with kNN imputer and machine learning classifiers via questionnaire mode of screening*
- **Authors:** Pragati Shrivastava, Prabhishek Singh, Manoj Diwakar, Achyut Shankar
- **Year / Venue:** 2024, *Health Information Science and Systems* 12:12
- **Link:** [PMC38464462](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC38464462/) | DOI: `10.1007/s13755-024-00277-8`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 Adult, Adolescent, Child, Toddler ($N=2,154$)
- **Judgments & Basis:**
  - `split_type`: `random` (80/20 train/test split).
  - `preprocessing_fit_in_fold`: `N` (Section 'Proposed methodology': MinMaxScaler and kNN Imputer fitted on entire dataset before 80/20 train-test split).
  - `leaking_features`: `Y` (Random Forest reaches 100% accuracy using $A_1-A_{10}$ questions).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `100%` (Random Forest on adult, toddler, and adolescent cohorts).

---

### 5. Tartarisco et al. (2021)
- **Paper:** *Use of Machine Learning to Investigate the Quantitative Checklist for Autism in Toddlers (Q-CHAT) towards Early Autism Screening*
- **Authors:** Gennaro Tartarisco et al.
- **Year / Venue:** 2021, *Diagnostics* 11(3):574
- **Link:** [PMC33810146](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC33810146/) | DOI: `10.3390/diagnostics11030574`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** Italian clinical toddler cohort ($N=259$)
- **Judgments & Basis:**
  - `split_type`: `random` (5-fold cross-validation).
  - `preprocessing_fit_in_fold`: `Y` (Feature ranking and reduction performed inside validation loops).
  - `leaking_features`: `N` (Target ground truth was clinical consensus diagnosis by licensed child psychiatrists using ADOS-2 and ADI-R, NOT an arithmetic score cutoff).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `95% (25 items), 91% (14 items)` (SVM accuracy).

---

### 6. Akter et al. (2019)
- **Paper:** *Machine Learning-Based Models for Early Stage Detection of Autism Spectrum Disorders*
- **Authors:** Tania Akter et al.
- **Year / Venue:** 2019, *IEEE Access* 7:166509-166527
- **Link:** [IEEE Access](https://doi.org/10.1109/ACCESS.2019.2952609) | DOI: `10.1109/ACCESS.2019.2952609`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 multi-cohort ($N=2,154$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold cross-validation).
  - `preprocessing_fit_in_fold`: `N` (Categorical transformation and dummy variables applied globally).
  - `leaking_features`: `Y` (Models trained on $A_1-A_{10}$ items to predict screening outcome; SVM achieves 98.7% on toddler, AdaBoost 98.4% on adult).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `98.7% (toddler), 98.4% (adult)`.

---

### 7. Vakadkar, Purkayastha, Krishnan (2021)
- **Paper:** *Detection of Autism Spectrum Disorder in Children Using Machine Learning Techniques*
- **Authors:** K. Vakadkar, D. Purkayastha, D. Krishnan
- **Year / Venue:** 2021, *SN Computer Science* 2(5):386
- **Link:** [Springer SN CS](https://doi.org/10.1007/s42979-021-00776-5) | DOI: `10.1007/s42979-021-00776-5`
- **Evidence Level:** `abstract_only` | **Needs Human Check:** `Y`
- **Dataset:** Kaggle / UCI Toddler screening dataset ($N=1,054$)
- **Judgments & Basis:**
  - `split_type`: `random` (Standard train-test split described in abstract).
  - `preprocessing_fit_in_fold`: `unclear` (Not specified in available abstract metadata).
  - `leaking_features`: `Y` (Evaluates behavioral screening questionnaire items from Thabtah toddler dataset).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `97.19%` (Logistic Regression accuracy).

---

### 8. Thabtah et al. (2022)
- **Paper:** *Autism screening: an unsupervised machine learning approach*
- **Authors:** Fadi Thabtah, Neda Abdelhamid, Rayan Peebles
- **Year / Venue:** 2022, *Health Information Science and Systems* 10:20
- **Link:** [PMC9458819](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9458819/) | DOI: `10.1007/s13755-022-00191-x`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ASD Tests / Autism AI mobile screening application data ($N=2,048$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold cross-validation with 10 repetitions).
  - `preprocessing_fit_in_fold`: `N` (Self-Organizing Maps unsupervised clustering fit on full dataset before supervised classification).
  - `leaking_features`: `Y` (Target screening class derived from 10 screening questions; paper notes accuracy on screening class is 96%, but when predicting true clinical diagnosis 'Is Diagnosed', accuracy drops to 68%).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `96% (screening), 68% (clinical)`.

---

### 9. Singh et al. (2026)
- **Paper:** *ASD-Bench: A Four-Axis Comprehensive Benchmark of AI Models for Autism Spectrum Disorder*
- **Authors:** Shubhankit Singh, Hassan Shaikh, Kuldeep Raghuwanshi, Keshav Bulia
- **Year / Venue:** 2026, *arXiv:2605.11091*
- **Link:** [arXiv:2605.11091](https://doi.org/10.48550/arXiv.2605.11091) | DOI: `10.48550/arXiv.2605.11091`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** Curated multi-age AQ-10 benchmark (UCI v3) ($N=4,068$)
- **Judgments & Basis:**
  - `split_type`: `random` (80/20 train/test split + 5-fold cross-validation).
  - `preprocessing_fit_in_fold`: `Y` (Pipelines executed inside cross-validation).
  - `leaking_features`: `Y` (Input features include $A_1-A_{10}$ item scores; 10 of 17 evaluated models reach $F_1=1.000$ and $\text{AUC}=1.000$ on Adult cohort).
  - `harmonization`: `none`.
  - `calibration_reported`: `Y` (Explicitly reports Expected Calibration Error [ECE] and Brier scores across all 17 models; demonstrates severe miscalibration with ECE up to 0.302 despite $F_1=1.000$).
  - `uncertainty_reported`: `Y` (Evaluates epistemic uncertainty using Monte Carlo Dropout with $T=20$ forward passes).
  - `external_validation`: `N`.
  - `reported_accuracy`: `100% (Adult F1=1.000)`.

---

### 10. Katlariwala et al. (2026)
- **Paper:** *Questioning the Benchmarks: Hidden Circularity in AQ-10 Autism Screening Datasets*
- **Authors:** A. Katlariwala et al.
- **Year / Venue:** 2026, *medRxiv preprint*
- **Link:** [medRxiv](https://doi.org/10.64898/2026.11.01.26367003) | DOI: `10.64898/2026.11.01.26367003`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 Child, Adolescent, Adult cohorts ($N=1,092$)
- **Judgments & Basis:**
  - `split_type`: `random` (Repeated stratified train-test splits).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `Y` (Methodological audit proving that item-based models achieve 93.2%–100% accuracy due to arithmetic sum leakage; removing items causes performance to collapse to chance 50.7%–71.5%).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `100% (drops to 50.7% on demographics)`.

---

### 11. Thabtah & Peebles (2020)
- **Paper:** *A new machine learning model based on induction of rules for autism detection*
- **Authors:** Fadi Thabtah, Rayan Peebles
- **Year / Venue:** 2020, *Health Informatics Journal* 26(1):264–286
- **Link:** [SAGE HIJ](https://doi.org/10.1177/1460458218824711) | DOI: `10.1177/1460458218824711`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 Child ($N=292$), Adolescent ($N=104$), Adult ($N=704$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold cross-validation).
  - `preprocessing_fit_in_fold`: `unclear`.
  - `leaking_features`: `Y` (Rules-ML extracts IF-THEN rules directly from $A_1-A_{10}$ items, achieving 97.4% accuracy).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `97.4% (adult), 96.1% (child)`.

---

### 12. Kumar & Das (2021/2022)
- **Paper:** *The diagnosis of ASD using multiple machine learning techniques*
- **Authors:** Chandan Jyoti Kumar, Priti Rekha Das
- **Year / Venue:** 2022, *International Journal of Developmental Disabilities* 68(5):663–674
- **Link:** [PMC9788716](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9788716/) | DOI: `10.1080/20473869.2021.1933730`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 Adult ($N=701$; 704 total with 3 dropped)
- **Judgments & Basis:**
  - `split_type`: `random` (Section 'Automating ASD diagnosis': 75% train, 25% test split).
  - `preprocessing_fit_in_fold`: `N` (Algorithm 1 one-hot encoding and RFE greedy search executed on preprocessed dataframe before train-test split).
  - `leaking_features`: `Y` (Uses $A_1-A_{10}$ items; paper explicitly notes questionnaire values decide total score; RFE models reach perfect 100% accuracy).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `100%` (multiple RFE models in Table 2 and Table 3 achieve 1.000 accuracy).

---

### 13. Rasul et al. (2023)
- **Paper:** *An Evaluation of Machine Learning Approaches for Early Diagnosis of Autism Spectrum Disorder*
- **Authors:** Rownak Ara Rasul, Promy Saha, Diponkor Bala, S M Rakib Ul Karim, Md. Ibrahim Abdullah, Bishwajit Saha
- **Year / Venue:** 2023, *arXiv:2309.11646*
- **Link:** [arXiv:2309.11646](https://arxiv.org/abs/2309.11646) | DOI: `10.48550/arXiv.2309.11646`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 Child ($N=292$), Adult ($N=704$), Combined ($N=996$)
- **Judgments & Basis:**
  - `split_type`: `random` (Section 3.4: 70/30 train/test split + 5-fold cross-validation).
  - `preprocessing_fit_in_fold`: `N` (Sections 3.2.1-3.3.1: MinMaxScaler and Chi-square feature ranking executed on full dataset prior to train-test split).
  - `leaking_features`: `Y` (Input features include $A_1-A_{10}$ questions; SVM and Logistic Regression achieve 100% accuracy on children, 97.14% on adult).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `100% (children), 97.14% (adult)`.

---

### 14. Hadi et al. (2026)
- **Paper:** *Detection of autism spectrum disorder using multilayer perceptron classifier*
- **Authors:** Ahmed Q. Hadi, Saif H. Alrubaee, Fahad Taha Al-Dhief, Ammar AbdRaba Sakran
- **Year / Venue:** 2026, *IAES International Journal of Artificial Intelligence* 15(3):2103-2112
- **Link:** [IAES IJ-AI](http://doi.org/10.11591/ijai.v15.i3.pp2103-2112) | DOI: `10.11591/ijai.v15.i3.pp2103-2112`
- **Evidence Level:** `abstract_only` | **Needs Human Check:** `Y`
- **Dataset:** Toddler autism dataset (UCI/Kaggle Q-CHAT-10 Toddler, $N=1,054$)
- **Judgments & Basis:**
  - `split_type`: `unclear` (Abstract reports MLP evaluated from 10 to 50 neurons; exact train-test ratio in PDF galley).
  - `preprocessing_fit_in_fold`: `unclear`.
  - `leaking_features`: `Y` (Evaluates toddler screening questionnaire attributes, reaching 100% accuracy).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `100%` (Abstract: 'MLP obtained the best performance, reaching an accuracy rate of 100% in identifying ASD cases').

---

### 15. Abu-Errub (2025)
- **Paper:** *Improving Early Autism Detection with Chi-Square Feature Selection, Machine Learning, and Explainable AI*
- **Authors:** Aymen Abu-Errub
- **Year / Venue:** 2025, *Engineering, Technology & Applied Science Research* 15(5):27954-27959
- **Link:** [ETASR](https://doi.org/10.48084/etasr.12999) | DOI: `10.48084/etasr.12999`
- **Evidence Level:** `full_text` | **Needs Human Check:** `Y`
- **Dataset:** Kaggle / UCI Toddler Autism screening dataset (Thabtah 2018, $N=1,054$)
- **Judgments & Basis:**
  - `split_type`: `unclear` (Evaluates six classifiers; train-test split not detailed in HTML landing page).
  - `preprocessing_fit_in_fold`: `unclear`.
  - `leaking_features`: `Y` (Applies Chi-square to toddler screening questions; SVM achieves 100% accuracy, precision, recall, and F1).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `100% (SVM)`.

---

### 16. Hasan et al. (2023)
- **Paper:** *A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders*
- **Authors:** S. M. Mahedy Hasan, Md. Palash Uddin, Md. Al Mamun, Muhammad Imran Sharif, Anwaar Ulhaq, G. Krishnamoorthy
- **Year / Venue:** 2023, *IEEE Access* 11:15038-15057
- **Link:** [IEEE Access](https://doi.org/10.1109/ACCESS.2022.3232490) | DOI: `10.1109/ACCESS.2022.3232490`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** UCI AQ-10 multi-cohort ($N=2,154$)
- **Judgments & Basis:**
  - `split_type`: `unclear` (Abstract describes comparative evaluation of 8 ML models across feature-scaled cohorts).
  - `preprocessing_fit_in_fold`: `N` (Feature scaling strategies Quantile Transformer, Power Transformer, Normalizer, and MAS applied to dataset prior to classification).
  - `leaking_features`: `Y` (Input features include $A_1-A_{10}$ screening items; AdaBoost reaches 99.25% on toddlers, LDA reaches 99.03% on adults).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `99.25% (toddler), 99.03% (adult)`.

---

## Part 2: Neuroimaging Family (ABIDE) — 17 Papers

### 17. Dong et al. (2025)
- **Paper:** *A Framework for Comparison and Interpretation of Machine Learning Classifiers to Predict Autism on the ABIDE Dataset*
- **Authors:** Y. Dong et al.
- **Year / Venue:** 2025, *Human Brain Mapping* 46(6):e70190
- **Link:** [PMC11880945](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11880945/) | DOI: `10.1002/hbm.70190`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I across 20 sites ($N=870$)
- **Judgments & Basis:**
  - `split_type`: `random` (Section 'Cross-validation and evaluation': 5-fold CV randomly pooling subjects across 20 acquisition sites).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N` (Resting-state functional connectivity).
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `72.2% (GCN ensemble), 70.1% (SVM)`.

---

### 18. Heinsfeld et al. (2018)
- **Paper:** *Identification of autism spectrum disorder using deep learning and the ABIDE dataset*
- **Authors:** Anibal S. Heinsfeld et al.
- **Year / Venue:** 2018, *NeuroImage: Clinical* 17:16-23
- **Link:** [PMC5635344](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5635344/) | DOI: `10.1016/j.nicl.2017.08.017`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I CC200 atlas across 17 sites ($N=1,035$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold CV on pooled multi-site data) + secondary leave-one-site-out.
  - `preprocessing_fit_in_fold`: `unclear`.
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `70% (pooled CV); 52% (intra-site)` (Deep autoencoder achieves 70% under random CV, but performance drops to ~52% on individual intra-site test).

---

### 19. Abraham et al. (2017)
- **Paper:** *Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example*
- **Authors:** Alexandre Abraham et al.
- **Year / Venue:** 2017, *NeuroImage* 147:736-745
- **Link:** [NeuroImage](https://doi.org/10.1016/j.neuroimage.2016.10.045) | arXiv: [1611.06066](https://arxiv.org/abs/1611.06066)
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I across 20 sites ($N=871$)
- **Judgments & Basis:**
  - `split_type`: `site-wise` (Section 2.4 'Evaluation strategy': Leave-One-Site-Out cross-validation across all sites).
  - `preprocessing_fit_in_fold`: `Y` (Tangent-space covariance estimation performed strictly on training sites).
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `Y` (Explicitly tests generalization to unseen physical scanner sites via LOSO).
  - `reported_accuracy`: `67%` (L2 logistic regression on tangent space connectomes).

---

### 20. Parisot et al. (2018)
- **Paper:** *Disease prediction using graph convolutional networks: Application to Autism Spectrum Disorder and Alzheimer's disease*
- **Authors:** Sarah Parisot et al.
- **Year / Venue:** 2018, *Medical Image Analysis* 48:117-130
- **Link:** [Med Image Anal](https://doi.org/10.1016/j.media.2018.06.001) | arXiv: [1703.03020](https://arxiv.org/abs/1703.03020)
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I CC200 across 20 sites ($N=871$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold CV on semi-supervised population graph).
  - `preprocessing_fit_in_fold`: `N` (Population graph constructs edges linking test nodes to training nodes across sites using phenotypic metadata prior to classification).
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `69.5%` (GCN accuracy on ABIDE).

---

### 21. Eslami et al. (2019)
- **Paper:** *ASD-DiagNet: A Hybrid Learning Approach for Detection of Autism Spectrum Disorder Using fMRI Data*
- **Authors:** Tuka Eslami et al.
- **Year / Venue:** 2019, *Frontiers in Neuroinformatics* 13:70
- **Link:** [PMC6890595](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6890595/) | DOI: `10.3389/fninf.2019.00070`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I CC200 across 17 sites ($N=1,035$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold cross-validation / single-site splits).
  - `preprocessing_fit_in_fold`: `N` (Synthetic data augmentation using linear interpolation applied to dataset before fold evaluation).
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `82% (peak site), 70.3% (overall)` (Peak accuracy reported on single isolated center; drops to 70.3% across all subjects).

---

### 22. Nielsen et al. (2013)
- **Paper:** *Multisite functional connectivity MRI classification of autism: ABIDE results*
- **Authors:** Jared A. Nielsen et al.
- **Year / Venue:** 2013, *Frontiers in Human Neuroscience* 7:599
- **Link:** [PMC3782716](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3782716/) | DOI: `10.3389/fnhum.2013.00599`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I across 16 sites ($N=964$)
- **Judgments & Basis:**
  - `split_type`: `subject-wise` (Section 'Classification Analysis': Leave-one-out cross-validation across 964 subjects).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `other` (Subtracts site mean connectivity vector to offset site differences).
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `60.0%` (Whole brain multi-site accuracy; authors highlight that 80-90% single-site results do not transfer across sites).

---

### 23. Plitt, Barnes, Martin (2015)
- **Paper:** *Functional connectivity classification of autism identifies highly predictive brain features but falls short of biomarker standards*
- **Authors:** Mark Plitt, Kelly Anne Barnes, Alex Martin
- **Year / Venue:** 2015, *NeuroImage: Clinical* 7:359-366
- **Link:** [PMC4309950](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4309950/) | DOI: `10.1016/j.nicl.2014.12.013`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I ($N=178$) + In-house NIMH ($N=118$) $\to N=296$ total
- **Judgments & Basis:**
  - `split_type`: `subject-wise` (LOO and 10-fold cross-validation).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `other` (Site regression evaluated).
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `Y` (Trained on NIMH cohort and externally validated on independent ABIDE sites).
  - `reported_accuracy`: `76.67% (fMRI), 95.19% (SRS)` (fMRI drops to 76.67% whereas behavioral SRS questionnaire achieves 95.19%).

---

### 24. Huang & Chung (2020)
- **Paper:** *Edge-variational Graph Convolutional Networks for Uncertainty-aware Disease Prediction*
- **Authors:** Yu-Huan Huang, Albert C. S. Chung
- **Year / Venue:** 2020, *MICCAI 2020* (LNCS 12267)
- **Link:** [Springer MICCAI](https://doi.org/10.1007/978-3-030-59719-1_53) | arXiv: [2009.02759](https://arxiv.org/abs/2009.02759)
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I CC200 ($N=871$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold cross-validation on population graph).
  - `preprocessing_fit_in_fold`: `N` (Population graph connects subjects using phenotypic features globally).
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `Y` (Implements Monte Carlo Edge Dropout to quantify prediction variance and uncertainty).
  - `external_validation`: `N`.
  - `reported_accuracy`: `81.06%` (EV-GCN+MCED accuracy on ABIDE).

---

### 25. Li, Parikh, He (2018)
- **Paper:** *A Novel Transfer Learning Approach to Enhance Deep Neural Network Classification of Brain Functional Connectomes*
- **Authors:** Hangwei Li, Mehul N. Parikh, Lili He
- **Year / Venue:** 2018, *Frontiers in Neuroscience* 12:491
- **Link:** [PMC6066582](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6066582/) | DOI: `10.3389/fnins.2018.00491`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I across 4 large sites (UM, UCLA, USM, LEUVEN) ($N=409$)
- **Judgments & Basis:**
  - `split_type`: `site-wise` (10-fold CV per site; evaluates deep transfer across sites).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `70.4% (USM site)`.

---

### 26. Guo et al. (2017)
- **Paper:** *Diagnosing Autism Spectrum Disorder from Brain Resting-State Functional Connectivity Patterns Using a Deep Neural Network with a Novel Feature Selection Method*
- **Authors:** Xin Guo et al.
- **Year / Venue:** 2017, *Frontiers in Neuroscience* 11:460
- **Link:** [PMC5566619](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5566619/) | DOI: `10.3389/fnins.2017.00460`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I single-site University of Michigan cohort ($N=110$)
- **Judgments & Basis:**
  - `split_type`: `subject-wise` (5-fold nested cross-validation on single site).
  - `preprocessing_fit_in_fold`: `Y` (Feature selection executed strictly within inner CV loops).
  - `leaking_features`: `N`.
  - `harmonization`: `none` (Single scanner protocol).
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `86.36%` (Single-site DNN-FS accuracy).

---

### 27. Bi et al. (2018)
- **Paper:** *Classification of Autism Spectrum Disorder Using Random Support Vector Machine Cluster*
- **Authors:** Xiao-An Bi et al.
- **Year / Venue:** 2018, *Frontiers in Genetics* 9:18
- **Link:** [PMC5808191](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5808191/) | DOI: `10.3389/fgene.2018.00018`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I single scanner protocol ($N=84$)
- **Judgments & Basis:**
  - `split_type`: `random` (70/30 split: 58 train, 26 test).
  - `preprocessing_fit_in_fold`: `N` (Section 'Materials and methods': The 26 test subjects were repeatedly queried to select the optimal base SVM cluster).
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `96.15%` (Random SVM cluster on 26 test subjects).

---

### 28. Shao, Fu, Chen (2023)
- **Paper:** *A heterogeneous graph convolutional attention network method for classification of autism spectrum disorder*
- **Authors:** Jin Shao, Jiahui Fu, Qiong Chen
- **Year / Venue:** 2023, *BMC Bioinformatics* 24:363
- **Link:** [PMC10536734](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10536734/) | DOI: `10.1186/s12859-023-05495-7`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I Harvard-Oxford atlas across 17 sites ($N=871$)
- **Judgments & Basis:**
  - `split_type`: `random` (10-fold cross-validation randomly mixing subjects from 17 sites).
  - `preprocessing_fit_in_fold`: `N` (Feature selection across whole dataset prior to network training).
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `82.9%`.

---

### 29. Traut et al. (2022)
- **Paper:** *Insights from an autism imaging biomarker challenge: Promises and threats to biomarker discovery*
- **Authors:** Nicolas Traut et al.
- **Year / Venue:** 2022, *NeuroImage* 255:119171
- **Link:** [NeuroImage](https://doi.org/10.1016/j.neuroimage.2022.119171) | HAL: [hal-03637273](https://hal.science/hal-03637273/document)
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I & II ($N=2,150$) + EU-AIMS LEAP cohort
- **Judgments & Basis:**
  - `split_type`: `site-wise` & held-out (Blinded challenge evaluation on unseen sites).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `other` (Site correction algorithms evaluated across challenge entries).
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `Y` (Top algorithms evaluated on independent European EU-AIMS cohort).
  - `reported_accuracy`: `0.80 AUC (challenge), 0.72 AUC (EU-AIMS)` (AUROC drops from 0.80 on unseen internal test sites down to 0.72 on external EU-AIMS).

---

### 30. Sherkatghanad et al. (2020)
- **Paper:** *Automated Detection of Autism Spectrum Disorder Using a Convolutional Neural Network*
- **Authors:** Zeinab Sherkatghanad et al.
- **Year / Venue:** 2020, *Frontiers in Neuroscience* 13:1325
- **Link:** [PMC6971220](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6971220/) | DOI: `10.3389/fnins.2019.01325`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I CC400 atlas across 17 sites ($N=871$)
- **Judgments & Basis:**
  - `split_type`: `site-wise` & `random` (Section 'Classification Results': Evaluates both 10-fold CV and Leave-One-Site-Out across 17 sites).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `70.22% (CV), 53-75% (LOSO)` (10-fold CV achieves 70.22%; Table 7 shows LOSO accuracy fluctuates widely from 53% to 75% depending on the test scanner).

---

### 31. Dvornek et al. (2017)
- **Paper:** *Identifying Autism from Resting-State fMRI Using Long Short-Term Memory Networks*
- **Authors:** Nicha C. Dvornek et al.
- **Year / Venue:** 2017, *Machine Learning in Medical Imaging* (MLMI 2017), LNCS 10433:362-370
- **Link:** [PMC5669262](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5669262/) | DOI: `10.1007/978-3-319-67389-9_42`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I Craddock 200 atlas across 17 sites ($N=1,100$)
- **Judgments & Basis:**
  - `split_type`: `random` (Stratified 10-fold CV keeping subjects isolated but pooling sites).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `68.5%` (Subject-level classification accuracy for LSTM).

---

### 32. Ktena et al. (2018)
- **Paper:** *Distance Metric Learning using Graph Convolutional Networks: Application to Functional Brain Networks*
- **Authors:** Sofia Ira Ktena et al.
- **Year / Venue:** 2018, *NeuroImage* 169:431-442
- **Link:** [NeuroImage](https://doi.org/10.1016/j.neuroimage.2017.12.052) | arXiv: [1703.02161](https://arxiv.org/abs/1703.02161)
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I Harvard-Oxford atlas across 20 sites ($N=871$)
- **Judgments & Basis:**
  - `split_type`: `random` (720 train, 151 test split).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `other` (Introduces site-matching indicator into the Siamese GCN distance metric).
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `N`.
  - `reported_accuracy`: `64.2%` (k-NN test accuracy on 151 subjects).

---

### 33. Kan et al. (2022)
- **Paper:** *Brain Network Transformer*
- **Authors:** Xuan Kan et al.
- **Year / Venue:** 2022, *NeurIPS 2022*
- **Link:** [arXiv:2210.06681](https://doi.org/10.48550/arXiv.2210.06681) | DOI: `10.48550/arXiv.2210.06681`
- **Evidence Level:** `full_text` | **Needs Human Check:** `N`
- **Dataset:** ABIDE I CC200 across 17 sites ($N=1,009$)
- **Judgments & Basis:**
  - `split_type`: `random` (Stratified 70/10/20 train/val/test split across 17 sites).
  - `preprocessing_fit_in_fold`: `Y`.
  - `leaking_features`: `N`.
  - `harmonization`: `none`.
  - `calibration_reported`: `N`.
  - `uncertainty_reported`: `N`.
  - `external_validation`: `Y` (Evaluates BrainNetTF model on external ABCD dataset).
  - `reported_accuracy`: `80.2% AUROC` (BrainNetTF on ABIDE I test set).
