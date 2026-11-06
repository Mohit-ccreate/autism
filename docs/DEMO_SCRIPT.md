# 5-Minute Faculty Presentation Walkthrough Script

**Project:** *A Methodological Audit and Reproducible Benchmark of Machine Learning for Autism Screening: Behavioral Tautologies, Scanner Generalization, and Uncertainty-Aware Decision Support.*  
**Target Audience:** Faculty Reviewer / Program Committee / Clinical Informatics Panel  
**Application Command:** `streamlit run app/Home.py`  
**Time Limit:** 5 Minutes  

---

## Walkthrough Timeline

### Minute 0:00 – 0:45 | Overview, Central Thesis & Ethical Disclaimer
- **Screen:** `Home.py` (Main Overview Page)
- **Action:**
  1. Point out the persistent banner at the top of every screen:  
     > *"RESEARCH PROTOTYPE — NOT A DIAGNOSTIC TOOL."*
  2. State the central thesis clearly:  
     *"Published ML screening papers often report 95–100% accuracy on behavioral data and 70–85% on ABIDE neuroimaging. We audited whether these numbers reflect genuine clinical generalizability or three methodological artifacts: (1) behavioral feature leakage, (2) scanner site hardware leakage, and (3) uncalibrated, abstention-free predictions."*
  3. Highlight the 3 metric cards on the Home page:
     - 100% label concordance on questionnaire sums.
     - 55.0% honest LOSO accuracy on ABIDE-I fMRI (indistinguishable from chance, binomial $p = 0.315$).
     - Conformal abstention identifying ambiguous cases near screening decision boundaries.

---

### Minute 0:45 – 1:45 | Module 1: Behavioral Tautology Audit
- **Screen:** Sidebar -> `1_🔍_Tautology_Audit`
- **Action:**
  1. Select dataset dropdown: Toggle between **Adult (AQ-10)** and **Toddler (Q-CHAT-10)**.
  2. Show **Rule Recovery Test**:
     - Adult/Child/Adolescent: $\sum_{i=1}^{10} A_i \ge 7 \iff \text{ASD}$ (Concordance: 100.0%, 0 discordant cases).
     - Toddler: $\sum_{i=1}^{10} A_i \ge 4 \iff \text{ASD}$ (Concordance: 100.0%, 0 discordant cases).
  3. Show the **Feature Ablation Plot**:
     - Point out that models with total score (Var D) achieve 100% trivial accuracy.
     - Even without the total score column, the 10 binary items linearly sum to the threshold (AUROC > 0.99).
     - When using only demographics (Var C: age, sex, jaundice, family history), AUROC collapses to 0.44–0.59.
  4. Point out the **Tautology Index** ($\text{TI} = 0.82 - 1.10$), proving that predictive accuracy in literature is an artifact of the arithmetic definition.

---

### Minute 1:45 – 2:45 | Module 3: Scanner Generalization & The Leakage Gap
- **Screen:** Sidebar -> `3_🧲_Scanner_Generalization`
- **Action:**
  1. Show **Random-CV vs. LOSO Bar Chart**:
     - Logistic Regression: 58.3% (Random CV) drops to 55.0% (LOSO).
     - Brain Network Transformer (BNT): 68.3% (Random CV) collapses to 45.8% (LOSO).
     - Explain the mechanism: Random CV allows scanner hardware fingerprints to leak across train/test splits.
  2. Show the **Site Probe**:
     - Pre-ComBat site decodability is 73.3% (chance = 25%), proving that functional connectivity matrices carry distinct scanner hardware signatures.
  3. Show the **Harmonization Arm Comparison**:
     - Compare: No harmonization (0.615 AUROC), Fold-internal ComBat (0.605 AUROC), and **Leaky ComBat** (0.627 AUROC).
     - Emphasize that fitting ComBat across the entire cohort before cross-validation artificially inflates performance and must be treated as a methodological defect.

---

### Minute 2:45 – 3:45 | Module 4: Uncertainty Lab & Selective Prediction
- **Screen:** Sidebar -> `4_🎯_Uncertainty_Lab`
- **Action:**
  1. Navigate to **Reliability Diagrams tab**:
     - Show uncalibrated Logistic Regression (ECE = 0.0594) vs. temperature-scaled calibration (ECE = 0.0182).
     - Explain why raw softmax outputs overstate certainty in clinical screening.
  2. Navigate to **Conformal Coverage tab**:
     - Explain Split Conformal prediction ($1 - \alpha = 90\%$ nominal coverage):
     - Marginal coverage achieved: 97.8% with average set size 0.98.
     - Show that non-singleton or empty prediction sets isolate boundary scores where clinical ambiguity exists.
  3. Navigate to **Risk-Coverage tab**:
     - Move the **Coverage Target slider** from 100% to 80%.
     - Demonstrate how abstaining on the 20% most uncertain cases drives empirical risk down.
     - Show the **Three-Way Policy**: Screen-positive / Screen-negative / Refer-for-expert-review (2.2% triage referral burden).

---

### Minute 3:45 – 4:30 | Live Screening Simulator Demo
- **Screen:** Sidebar -> `5_🖊_Screening_Simulator`
- **Action:**
  1. Note the prominent medical disclaimer and zero-logging policy.
  2. Adjust sliders for a hypothetical scenario (Age = 25, Sex = Male, Jaundice = No, Family History = Yes).
  3. Click **"Show Model Output (Demonstration Only)"**.
  4. Point out the honest output:
     - Calibrated probability (~35–45%) reflecting population demographics.
     - Conformal prediction set and clinical decision recommendation.
     - Contrast this with a total-score screener that creates a false illusion of diagnostic certainty.

---

### Minute 4:30 – 5:00 | Reproducibility & Audit Checklist
- **Screen:** Sidebar -> `7_✅_Audit_Checklist`
- **Action:**
  1. Display the 15-point ML-for-Autism Evaluation Checklist.
  2. Filter by "Critical" severity (Subject-wise splits, leaking features, in-fold harmonization, clinical framing).
  3. Conclude:
     *"Every number shown in this demo and paper comes directly from code that actually ran and is traceable to `results/paper_facts.json`. The entire pipeline can be verified with `python run_all.py` and `python run_extended.py`."*
