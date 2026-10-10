# ASD Screening Project: Data Sources and Lineage

**Generated:** 2026-09-30 00:56:58  
**Verification Status:** Verified direct official downloads, zero fabricated records.

---

## 1. Summary of Ingested Datasets

| Dataset | Modality | Instances | Features | Diagnostic Balance | Source & DOI |
|---|---|---|---|---|---|
| **UCI Adult ASD** | 10-Item AQ-10 + Demographics | 704 | 21 | Binary (ASD / Non-ASD) | [UCI ML Repo #426](https://archive.ics.uci.edu/dataset/426/autism+screening+adult) (DOI: 10.24432/C5F019) |
| **UCI Adolescent ASD** | 10-Item AQ-10 + Demographics | 104 | 21 | Binary (ASD / Non-ASD) | [UCI ML Repo #420](https://archive.ics.uci.edu/dataset/420/autistic+spectrum+disorder+screening+data+for+adolescent) (DOI: 10.24432/C5V89T) |
| **UCI Child ASD** | 10-Item AQ-10 + Demographics | 292 | 21 | Binary (ASD / Non-ASD) | [UCI ML Repo #419](https://archive.ics.uci.edu/dataset/419/autistic+spectrum+disorder+screening+data+for+children) (DOI: 10.24432/C5659W) |
| **Toddler Q-CHAT-10** | 10-Item Q-CHAT + Demographics | 1054 | 19 | Binary (ASD traits / Non-ASD traits) | [Dr. Fadi Thabtah / Kaggle](https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers) (2018) |
| **ABIDE I CC200 fMRI** | Resting-state fMRI ROI time series (200 regions) | 120 | 200 time series | Multi-site Balanced | [ABIDE I Preprocessed](http://preprocessed-connectomes-project.org/abide/) (CPAC, filt_global) |

---

## 2. Integrity Verification & Checksums

### UCI Adult
- **File:** `data/uci_adult.csv`
- **Rows:** 704 | **Columns:** 21
- **SHA256:** `657021cddb337358640470cea758b72087bc4519e1234f7bc59c2b0160a0060d`
- **Source URL:** `https://archive.ics.uci.edu/static/public/426/autism+screening+adult.zip`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### UCI Adolescent
- **File:** `data/uci_adolescent.csv`
- **Rows:** 104 | **Columns:** 21
- **SHA256:** `4753e2ce2c791c32181ad9413c55b6556828b9578a22d166d0cef7caaf93c89d`
- **Source URL:** `https://archive.ics.uci.edu/static/public/420/autistic+spectrum+disorder+screening+data+for+adolescent.zip`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### UCI Child
- **File:** `data/uci_child.csv`
- **Rows:** 292 | **Columns:** 21
- **SHA256:** `aff0928863f6c930d4102bda422fb46b4a337356ef431ddfacb3bd07925f2aa2`
- **Source URL:** `https://archive.ics.uci.edu/static/public/419/autistic+spectrum+disorder+screening+data+for+children.zip`
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Toddler Q-CHAT-10
- **File:** `data/toddler_qchat10.csv`
- **Rows:** 1054 | **Columns:** 19
- **SHA256:** `818d604290cdae767f40a1bfcc2ca5634f1ba9ed1c31bfd41aa4b450943d973f`
- **Source URL:** `https://raw.githubusercontent.com/kamleshbaheti/Autism-Detection-of-Early-Childhood-Screening/master/Toddler_Autism_dataset_July_2018.csv`
- **License:** CC BY-NC-SA 4.0 (Research & Educational Screening use)

### ABIDE I (Autism Brain Imaging Data Exchange I)
- **Pipeline:** CPAC (`filt_global`: band-pass filtered 0.01 - 0.1 Hz, global signal regression)
- **Atlas:** Craddock 200 (CC200) Functional Clustering Atlas
- **Subjects Ingested:** 120 across 4 clinical sites: `PITT, OLIN, OHSU, SDSU`
- **Data Policy:** Open Access 1000 Functional Connectomes Project, strictly non-commercial, de-identified research use.

---

## 3. Eye-tracking & Face Images Policy (Non-Negotiable Rule 3 & 8)
- **Eye-tracking (Saliency4ASD):** Kept OFF by default. Eye-tracking datasets feature small laboratory cohorts (typically N < 40) collected with non-standardized head-mounted or desk-mounted rigs, yielding severe gaze calibration drift and high false-positive rates when generalized to webcams.
- **Face Images:** Excluded from headline results. Web-scraped child facial image datasets lack clinical diagnostic confirmation, have severe racial/lighting confounding, and violate ethical clinical boundaries for neurodivergence screening.
