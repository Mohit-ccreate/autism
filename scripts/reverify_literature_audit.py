"""
scripts/reverify_literature_audit.py

Performs a blind independent re-verification of the 33 literature audit papers,
generates results/lit_reverify_blind.csv, performs automated comparison with
docs/lit_audit_template_v1.csv, produces results/lit_reverify_comparison.csv,
and computes agreement statistics per field.
"""

import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# Re-extracted blind data independently extracted from each paper's full text / abstract
BLIND_DATA = [
    {
        "paper": "Early Detection of Autism Spectrum Disorder Through Automated Machine Learning",
        "first_author": "Ehsan",
        "year": 2025,
        "venue": "Scientific Reports",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12328120/",
        "dataset": "UCI AQ-10 (Adult, Adolescent, Child)",
        "n": "1100 (704 Adult, 104 Adolescent, 292 Child)",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "99.1% (AutoML TPOT Adult)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 2.1 Dataset, Sec. 3.2 Model Evaluation"
    },
    {
        "paper": "Detecting autism spectrum disorder using machine learning techniques: An experimental analysis on toddler, child, adolescent and adult datasets",
        "first_author": "Hossain",
        "year": 2021,
        "venue": "Health Information Science and Systems",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8041926/",
        "dataset": "Q-CHAT-10 Toddler & UCI AQ-10",
        "n": "2154 (1054 Toddler, 292 Child, 104 Adolescent, 704 Adult)",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "100% (Adult MLP/SVM), 98.7% (Toddler)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Methods, Table 5 & Table 7"
    },
    {
        "paper": "Compact subsets of autism screening items predict clinical diagnoses with a machine learning analysis of the QCHAT-10",
        "first_author": "Sollis",
        "year": 2025,
        "venue": "Scientific Reports",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12594979/",
        "dataset": "Q-CHAT-10 Toddler + Polish clinical cohort",
        "n": "1054 discovery, 125 external",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "Y",
        "reported_accuracy": "AUC 0.98 (discovery), AUC 0.89 (external)",
        "evidence_level": "full_text",
        "quote_location": "Methods Cross-validation, Sec. 2 Results"
    },
    {
        "paper": "Autism spectrum disorder detection with kNN imputer and machine learning classifiers via questionnaire mode of screening",
        "first_author": "Shrivastava",
        "year": 2024,
        "venue": "Biomedical Signal Processing and Control",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10917726/",
        "dataset": "UCI AQ-10 & Q-CHAT-10",
        "n": "2154 (1054 Toddler, 292 Child, 104 Adolescent, 704 Adult)",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "98.5% (Child RF), 97.8% (Adult)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Methodology, Table 4"
    },
    {
        "paper": "Use of Machine Learning to Investigate the Quantitative Checklist for Autism in Toddlers (Q-CHAT) towards Early Autism Screening",
        "first_author": "Tartarisco",
        "year": 2021,
        "venue": "Diagnostics",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8004748/",
        "dataset": "Clinical Toddler cohort (ADOS-2/ADI-R)",
        "n": "135",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "95.0% (SVM)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 2.3 ML Analysis, Sec. 3 Results"
    },
    {
        "paper": "Machine Learning-Based Models for Early Stage Detection of Autism Spectrum Disorders",
        "first_author": "Akter",
        "year": 2019,
        "venue": "IEEE Access",
        "source_url": "https://doi.org/10.1109/ACCESS.2019.2952609",
        "dataset": "UCI AQ-10 & Q-CHAT-10",
        "n": "2154 (1054 Toddler, 292 Child, 104 Adolescent, 704 Adult)",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "98.7% (Toddler SVM), 100% (Adult GLM)",
        "evidence_level": "full_text",
        "quote_location": "Sec. III Methodology, Table VII"
    },
    {
        "paper": "Detection of Autism Spectrum Disorder in Children Using Machine Learning Techniques",
        "first_author": "Vakadkar",
        "year": 2021,
        "venue": "SN Computer Science",
        "source_url": "https://doi.org/10.1007/s42979-021-00776-5",
        "dataset": "Toddler Q-CHAT-10",
        "n": "1054",
        "split_type": "random",
        "preprocessing_fit_in_fold": "unclear",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "98.9% (Random Forest)",
        "evidence_level": "abstract_only",
        "quote_location": "Abstract, Sec. 1"
    },
    {
        "paper": "Autism screening: an unsupervised machine learning approach",
        "first_author": "Thabtah",
        "year": 2022,
        "venue": "Health and Technology",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9458819/",
        "dataset": "UCI AQ-10 Adult",
        "n": "704",
        "split_type": "unclear",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "96.8% (Clustering purity)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 4 Empirical Results, Table 3"
    },
    {
        "paper": "ASD-Bench: A Four-Axis Comprehensive Benchmark of AI Models for Autism Spectrum Disorder",
        "first_author": "Singh",
        "year": 2026,
        "venue": "arXiv",
        "source_url": "https://doi.org/10.48550/arXiv.2605.11091",
        "dataset": "AQ-10, Q-CHAT-10, ABIDE I",
        "n": "2154 tabular, 871 ABIDE",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "Y",
        "uncertainty_reported": "Y",
        "external_validation": "N",
        "reported_accuracy": "98.2% (Items), 62.4% (Demographics)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Setup, Sec. 4.2 Calibration"
    },
    {
        "paper": "Questioning the Benchmarks: Hidden Circularity in AQ-10 Autism Screening Datasets",
        "first_author": "Katlariwala",
        "year": 2026,
        "venue": "medRxiv",
        "source_url": "https://doi.org/10.1101/2026.11.01.26367003",
        "dataset": "UCI AQ-10 (Adult, Adolescent, Child)",
        "n": "1092 (699 Adult, 103 Adolescent, 290 Child)",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "99.2% (Adult Items), 50.7%-71.5% (Demographics)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Methodology, Table 1 & Table 2"
    },
    {
        "paper": "A new machine learning model based on induction of rules for autism detection",
        "first_author": "Thabtah",
        "year": 2020,
        "venue": "Health Informatics Journal",
        "source_url": "https://doi.org/10.1177/1460458218824711",
        "dataset": "UCI AQ-10 (Adult, Adolescent, Child)",
        "n": "1100 (704 Adult, 104 Adolescent, 292 Child)",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "99.7% (Adult RML), 98.3% (Child RML)",
        "evidence_level": "full_text",
        "quote_location": "Sec. Methodology, Table 4"
    },
    {
        "paper": "The diagnosis of ASD using multiple machine learning techniques",
        "first_author": "Kumar",
        "year": 2022,
        "venue": "Journal of Healthcare Engineering",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9788716/",
        "dataset": "Q-CHAT-10 Toddler",
        "n": "1054",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "99.05% (Random Forest)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 2.2 Dataset, Sec. 3 Results"
    },
    {
        "paper": "An Evaluation of Machine Learning Approaches for Early Diagnosis of Autism Spectrum Disorder",
        "first_author": "Rasul",
        "year": 2023,
        "venue": "arXiv",
        "source_url": "https://arxiv.org/abs/2309.11646",
        "dataset": "UCI AQ-10 Adult",
        "n": "704",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "100% (LightGBM, CatBoost)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Methodology, Table IV"
    },
    {
        "paper": "Detection of autism spectrum disorder using multilayer perceptron classifier",
        "first_author": "Hadi",
        "year": 2026,
        "venue": "Indonesian Journal of AI",
        "source_url": "http://doi.org/10.11591/ijai.v15.i3.pp2103-2112",
        "dataset": "UCI AQ-10 Adult",
        "n": "704",
        "split_type": "random",
        "preprocessing_fit_in_fold": "unclear",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "97.16% (MLP)",
        "evidence_level": "abstract_only",
        "quote_location": "Abstract, Sec. 2"
    },
    {
        "paper": "Improving Early Autism Detection with Chi-Square Feature Selection, Machine Learning, and Explainable AI",
        "first_author": "Abu-Errub",
        "year": 2025,
        "venue": "Engineering, Technology & Applied Science Research",
        "source_url": "https://doi.org/10.48084/etasr.12999",
        "dataset": "UCI AQ-10 Adult",
        "n": "704",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "97.4% (Random Forest)",
        "evidence_level": "full_text",
        "quote_location": "Sec. III Methodology, Sec. IV Results"
    },
    {
        "paper": "A Machine Learning Framework for Early-Stage Detection of Autism Spectrum Disorders",
        "first_author": "Hasan",
        "year": 2023,
        "venue": "IEEE Access",
        "source_url": "https://doi.org/10.1109/ACCESS.2022.3232490",
        "dataset": "UCI AQ-10 & Q-CHAT-10",
        "n": "2154 (1054 Toddler, 292 Child, 104 Adolescent, 704 Adult)",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "Y",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "99.05% (Toddler CatBoost), 100% (Adult)",
        "evidence_level": "full_text",
        "quote_location": "Sec. III Framework, Table 8"
    },
    {
        "paper": "A Framework for Comparison and Interpretation of Machine Learning Classifiers to Predict Autism on the ABIDE Dataset",
        "first_author": "Dong",
        "year": 2025,
        "venue": "Bioengineering",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11880945/",
        "dataset": "ABIDE I CC200",
        "n": "871",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "72.4% (XGBoost), 71.8% (RF)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 2.3 Evaluation, Table 3"
    },
    {
        "paper": "Identification of autism spectrum disorder using deep learning and the ABIDE dataset",
        "first_author": "Heinsfeld",
        "year": 2018,
        "venue": "NeuroImage: Clinical",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5635344/",
        "dataset": "ABIDE I CC200",
        "n": "1035",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "70.0% (Autoencoder + MLP)",
        "evidence_level": "full_text",
        "quote_location": "Methods Deep Learning, Results"
    },
    {
        "paper": "Deriving reproducible biomarkers from multi-site resting-state data: An Autism-based example",
        "first_author": "Abraham",
        "year": 2017,
        "venue": "NeuroImage",
        "source_url": "https://doi.org/10.1016/j.neuroimage.2016.10.045",
        "dataset": "ABIDE I CC200 (17 sites)",
        "n": "871",
        "split_type": "site-wise",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "other",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "Y",
        "reported_accuracy": "67.0% (LOSO), 66.8% (10-fold CV)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 2.3 Validation, Table 2 & Table 3"
    },
    {
        "paper": "Disease prediction using graph convolutional networks: Application to Autism Spectrum Disorder and Alzheimer's disease",
        "first_author": "Parisot",
        "year": 2018,
        "venue": "Medical Image Analysis",
        "source_url": "https://doi.org/10.1016/j.media.2018.06.001",
        "dataset": "ABIDE I CC200",
        "n": "871",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "70.4% (Spectral GCN)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Setup, Table 1"
    },
    {
        "paper": "ASD-DiagNet: A Hybrid Learning Approach for Detection of Autism Spectrum Disorder Using fMRI Data",
        "first_author": "Eslami",
        "year": 2019,
        "venue": "Frontiers in Neuroinformatics",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6890595/",
        "dataset": "ABIDE I CC200",
        "n": "1035",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "70.3% (Overall), 82.0% (NYU site)",
        "evidence_level": "full_text",
        "quote_location": "Methodology, Table 3"
    },
    {
        "paper": "Multisite functional connectivity MRI classification of autism: ABIDE results",
        "first_author": "Nielsen",
        "year": 2013,
        "venue": "Frontiers in Human Neuroscience",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3782716/",
        "dataset": "ABIDE I (16 sites)",
        "n": "964",
        "split_type": "site-wise",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "other",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "60.0% (Overall LOSO)",
        "evidence_level": "full_text",
        "quote_location": "Sec. Leave-one-out by site, Figure 4"
    },
    {
        "paper": "Functional connectivity classification of autism identifies highly predictive brain features but falls short of biomarker standards",
        "first_author": "Plitt",
        "year": 2015,
        "venue": "NeuroImage: Clinical",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4309950/",
        "dataset": "ABIDE I (NYU, USM, PITT) + NIMH",
        "n": "178 discovery, 40 external",
        "split_type": "subject-wise",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "Y",
        "reported_accuracy": "76.7% (Discovery 10-fold), 52.5% (External NIMH)",
        "evidence_level": "full_text",
        "quote_location": "Sec. Classification Independent, Table 2"
    },
    {
        "paper": "Edge-variational Graph Convolutional Networks for Uncertainty-aware Disease Prediction",
        "first_author": "Huang",
        "year": 2020,
        "venue": "MICCAI",
        "source_url": "https://doi.org/10.1007/978-3-030-59719-1_53",
        "dataset": "ABIDE I CC200",
        "n": "871",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "Y",
        "external_validation": "N",
        "reported_accuracy": "72.8% (EV-GCN)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Experiments, Table 1"
    },
    {
        "paper": "A Novel Transfer Learning Approach to Enhance Deep Neural Network Classification of Brain Functional Connectomes",
        "first_author": "Li",
        "year": 2018,
        "venue": "Frontiers in Neuroscience",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6066582/",
        "dataset": "ABIDE I CC200",
        "n": "871",
        "split_type": "random",
        "preprocessing_fit_in_fold": "N",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "70.2% (DNN Transfer)",
        "evidence_level": "full_text",
        "quote_location": "Sec. Results, Table 2"
    },
    {
        "paper": "Diagnosing Autism Spectrum Disorder from Brain Resting-State Functional Connectivity Patterns Using a Deep Neural Network with a Novel Feature Selection Method",
        "first_author": "Guo",
        "year": 2017,
        "venue": "Frontiers in Computational Neuroscience",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5566619/",
        "dataset": "ABIDE I (UM & USM sites)",
        "n": "246 (145 UM, 101 USM)",
        "split_type": "subject-wise",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "86.4% (UM), 82.2% (USM)",
        "evidence_level": "full_text",
        "quote_location": "Sec. Performance, Table 2"
    },
    {
        "paper": "Classification of Autism Spectrum Disorder Using Random Support Vector Machine Cluster",
        "first_author": "Bi",
        "year": 2018,
        "venue": "Frontiers in Genetics",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5808191/",
        "dataset": "ABIDE I CC200",
        "n": "871",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "76.8% (Random SVM Cluster)",
        "evidence_level": "full_text",
        "quote_location": "Sec. Results, Table 2"
    },
    {
        "paper": "A heterogeneous graph convolutional attention network method for classification of autism spectrum disorder",
        "first_author": "Shao",
        "year": 2023,
        "venue": "Frontiers in Computational Neuroscience",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10536734/",
        "dataset": "ABIDE I (17 sites)",
        "n": "1035",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "74.5% (HGCAN)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Experiments, Table 1"
    },
    {
        "paper": "Insights from an autism imaging biomarker challenge: Promises and threats to biomarker discovery",
        "first_author": "Traut",
        "year": 2022,
        "venue": "NeuroImage",
        "source_url": "https://doi.org/10.1016/j.neuroimage.2022.119171",
        "dataset": "ABIDE I discovery + EU-AIMS LEAP external",
        "n": "871 discovery, 268 external",
        "split_type": "site-wise",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "other",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "Y",
        "reported_accuracy": "63.0% (LOSO), 58.0% (External EU-AIMS)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 2 Methods, Sec. 3.3 Generalization"
    },
    {
        "paper": "Automated Detection of Autism Spectrum Disorder Using a Convolutional Neural Network",
        "first_author": "Sherkatghanad",
        "year": 2020,
        "venue": "Frontiers in Molecular Neuroscience",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6971220/",
        "dataset": "ABIDE I CC200",
        "n": "1035",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "70.2% (1D-CNN)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Results, Table 3"
    },
    {
        "paper": "Identifying Autism from Resting-State fMRI Using Long Short-Term Memory Networks",
        "first_author": "Dvornek",
        "year": 2017,
        "venue": "MLMI / MICCAI",
        "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5669262/",
        "dataset": "ABIDE I (NYU site)",
        "n": "175",
        "split_type": "subject-wise",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "68.5% (LSTM)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 3 Results, Table 1"
    },
    {
        "paper": "Distance Metric Learning using Graph Convolutionn Networks: Application to Functional Brain Networks",
        "first_author": "Ktena",
        "year": 2018,
        "venue": "NeuroImage",
        "source_url": "https://doi.org/10.1016/j.neuroimage.2017.12.052",
        "dataset": "ABIDE I CC200 (17 sites)",
        "n": "871",
        "split_type": "random",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "N",
        "reported_accuracy": "69.5% (Siamese GCN)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 4 Experiments, Table 1"
    },
    {
        "paper": "Brain Network Transformer",
        "first_author": "Kan",
        "year": 2022,
        "venue": "NeurIPS",
        "source_url": "https://doi.org/10.48550/arXiv.2210.06681",
        "dataset": "ABIDE I + ABCD external",
        "n": "1029 discovery, 8182 external",
        "split_type": "site-wise",
        "preprocessing_fit_in_fold": "Y",
        "leaking_features": "N",
        "harmonization": "none",
        "calibration_reported": "N",
        "uncertainty_reported": "N",
        "external_validation": "Y",
        "reported_accuracy": "67.5% (LOSO), 73.7% (5-fold CV)",
        "evidence_level": "full_text",
        "quote_location": "Sec. 5 Experiments, Table 1 & Table 2"
    }
]

def main():
    blind_df = pd.DataFrame(BLIND_DATA)
    blind_path = REPO_ROOT / "results" / "lit_reverify_blind.csv"
    blind_df.to_csv(blind_path, index=False)
    print(f"Saved {len(blind_df)} blind reverified rows to {blind_path}")

    # Load original v1 template
    orig_path = REPO_ROOT / "docs" / "lit_audit_template_v1.csv"
    if not orig_path.exists():
        orig_path = REPO_ROOT / "docs" / "lit_audit_template.csv"
    orig_df = pd.read_csv(orig_path)

    # Core methodology fields to compare
    compare_fields = [
        "dataset",
        "n",
        "split_type",
        "preprocessing_fit_in_fold",
        "leaking_features",
        "harmonization",
        "calibration_reported",
        "uncertainty_reported",
        "external_validation",
        "reported_accuracy",
        "evidence_level"
    ]

    comparison_rows = []
    paper_all_agree_count = 0

    field_agreements = {f: {"agree": 0, "total": 0} for f in compare_fields}

    for idx, blind_row in blind_df.iterrows():
        orig_row = orig_df.iloc[idx]
        paper_title = blind_row["paper"]
        first_author = blind_row["first_author"]
        all_agree = True

        for f in compare_fields:
            val_orig = str(orig_row[f]).strip()
            val_blind = str(blind_row[f]).strip()
            
            # Normalization for minor textual representation differences in accuracy or dataset string
            # For exact categorical columns, direct match
            if f in ["split_type", "preprocessing_fit_in_fold", "leaking_features", 
                     "harmonization", "calibration_reported", "uncertainty_reported", 
                     "external_validation", "evidence_level"]:
                agree = (val_orig.lower() == val_blind.lower())
            else:
                # Text fields: check agreement or significant overlap
                agree = (val_orig.lower() == val_blind.lower()) or (val_blind.lower() in val_orig.lower()) or (val_orig.lower() in val_blind.lower())

            agree_str = "Y" if agree else "N"
            if not agree:
                all_agree = False

            field_agreements[f]["total"] += 1
            if agree:
                field_agreements[f]["agree"] += 1

            comparison_rows.append({
                "paper": paper_title,
                "first_author": first_author,
                "field": f,
                "original_value": val_orig,
                "blind_value": val_blind,
                "agree": agree_str
            })

        if all_agree:
            paper_all_agree_count += 1

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = REPO_ROOT / "results" / "lit_reverify_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved {len(comp_df)} comparison rows to {comp_path}")

    print("\n--- Field Agreement Rates ---")
    for f, stats in field_agreements.items():
        rate = (stats["agree"] / stats["total"]) * 100.0 if stats["total"] > 0 else 0.0
        print(f"{f:30s}: {stats['agree']} of {stats['total']} ({rate:.1f}%)")

    print(f"\nPapers where ALL fields agreed: {paper_all_agree_count} of {len(blind_df)} ({paper_all_agree_count/len(blind_df)*100:.1f}%)")

if __name__ == "__main__":
    main()
