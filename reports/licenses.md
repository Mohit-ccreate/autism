# Dataset, Instrument, and Library Licenses & Usage Terms

**Last Verified:** September 2026  
**Auditor:** Autonomous Clinical-AI Systems Engineer

---

## 1. Clinical Screening Instruments

### Quantitative Checklist for Autism in Toddlers (Q-CHAT-10)
- **Authors:** Carrie Allison, Simon Baron-Cohen, Sally Wheelwright, Tony Charman, et al. (Autism Research Centre, University of Cambridge).
- **Publication Reference:** Allison, C., Baron-Cohen, S., et al. (2012). *The Q-CHAT (Quantitative CHecklist for Autism in Toddlers): a normally distributed measure of autistic traits.* J Autism Dev Disord 42(1), 86–97.
- **License / Terms of Use:** Developed and published by the Autism Research Centre (ARC), University of Cambridge. Available freely for personal, educational, and clinical research use with attribution.
- **Usage Restrictions:** The questionnaire items must not be altered. It is an initial screening instrument only, not a diagnostic confirmation. Commercial reselling of the test instrument itself without institutional licensing from Cambridge Enterprise is prohibited.
- **Compliance in this Project:** Used strictly for non-commercial educational and screening research. Clear attribution to ARC Cambridge is maintained throughout the UI and documentation.

### Autism Spectrum Quotient 10-Item Version (AQ-10) (Adult, Adolescent, Child)
- **Authors:** Simon Baron-Cohen, Sally Wheelwright, Carrie Allison, et al. (Autism Research Centre, University of Cambridge). Recommended by the UK National Institute for Health and Care Excellence (NICE clinical guideline CG142).
- **Publication Reference:** Allison, C., Auyeung, B., & Baron-Cohen, S. (2012). *Toward brief "red flags" for autism screening: the short Autism Spectrum Quotient and the short Quantitative Checklist for Autism in Toddlers in 1,000 cases and 3,000 controls.* J Am Acad Child Adolesc Psychiatry, 51(2), 202-212.
- **License / Terms of Use:** Freely accessible for clinical and academic research under Autism Research Centre open access terms.
- **Usage Restrictions:** Must include the caveat that scoring above threshold indicates traits warranting specialist assessment, not clinical autism diagnosis.
- **Compliance in this Project:** Full attribution to ARC and NICE CG142 provided.

---

## 2. Benchmark Datasets

### UCI Machine Learning Repository Autism Datasets
- **Donated By:** Dr. Fadi Fayez Thabtah (2017).
- **Datasets Included:**
  - Adult (ID 426, DOI: 10.24432/C5F019)
  - Adolescent (ID 420, DOI: 10.24432/C5V89T)
  - Child (ID 419, DOI: 10.24432/C5659W)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Terms:** Anyone is free to share, copy, and adapt the material for any purpose, even commercially, provided appropriate credit is given and license link is supplied.
- **Compliance:** Full citation to Dr. Fadi Thabtah and UCI repository provided.

### Toddler Autism Dataset (Q-CHAT-10 Kaggle)
- **Author:** Dr. Fadi Fayez Thabtah (2018).
- **License:** CC BY-NC-SA 4.0 (Non-Commercial, Attribution, Share-Alike).
- **Terms:** Educational and open research permitted. Non-commercial derivative models must share under equivalent research license.
- **Compliance:** This repository and all trained artifacts are published under MIT / CC BY-NC-SA 4.0 for non-commercial research.

### ABIDE I (Autism Brain Imaging Data Exchange I)
- **Consortium:** International Neuroimaging Data-sharing Initiative (INDI) / 1000 Functional Connectomes Project.
- **Publication Reference:** Di Martino, A., Yan, C. G., Li, Q., Denio, E., Castellanos, F. X., Alaerts, K., ... & Milham, M. P. (2014). *The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism.* Molecular psychiatry, 19(6), 659-667.
- **License / Usage Agreement:** Open Access 1000 Functional Connectomes Data Use Agreement.
- **Terms:** 
  1. Data is de-identified in compliance with HIPAA privacy rules.
  2. Users must not attempt to re-identify any individual research participant.
  3. Redistribution must maintain original citations and attribution.
  4. Non-commercial scientific research use only.
- **Compliance:** All fMRI processing adheres strictly to de-identified ROI time-series. No facial or structural MRI surfaces containing identifiable features are stored or transmitted.

---

## 3. Open Source Software Dependencies
All open-source Python and web libraries used in this project are governed by permissive licenses:
- `scikit-learn`: BSD-3-Clause
- `xgboost`: Apache-2.0
- `lightgbm`: MIT
- `catboost`: Apache-2.0
- `optuna`: MIT
- `shap`: MIT
- `nilearn`: BSD-3-Clause
- `neuroHarmonize` / `neuroCombat`: MIT
- `three.js`: MIT
- `Chart.js`: MIT
- `GSAP`: Standard GreenSock Public License / Open-source web use
- `onnxruntime-web`: MIT
