"""
Create:
1. docs/lit_verification_worksheet.csv (23 rows flagged needs_human_check)
2. scripts/recompute_lit_counts.py (recomputes counts with denominators, supports human overrides, conservative subset)
3. results/prior_comparison.csv (prior audit rows with full-text evidence vs matched-protocol study results)
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"
RESULTS_DIR = BASE_DIR / "results"
SCRIPTS_DIR = BASE_DIR / "scripts"

DOCS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


def build_lit_verification_worksheet():
    template_path = DOCS_DIR / "lit_audit_template.csv"
    df = pd.read_csv(template_path)
    
    # 23 flagged papers
    flagged = df[df["needs_human_check"] == "Y"].copy()
    
    flagged_data = [
        {
            "paper": "Hossain et al. (2021) - A Machine Learning-Based Approach for Autism Spectrum Disorder Detection in Children",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8041926/",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Sec. 3 Materials and Methods",
            "short paraphrase of the passage": "Imputation of missing values and categorical one-hot encoding performed on full dataset prior to train/test partitioning.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Sollis et al. (2025) - Development and validation of a machine learning-based screening tool for autism",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12594979/",
            "field": "source_url",
            "value_in_audit": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12594979/",
            "location_of_evidence": "Full text / PMC record",
            "short paraphrase of the passage": "PMCID corrected to PMC12594979. Discovery n=1054 5-fold CV repeated 10x; external validation on Polish cohort n=125.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Shrivastava et al. (2024) - Machine learning models for early detection of autism spectrum disorder",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10917726/",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Sec. 3.2 Methodology",
            "short paraphrase of the passage": "PMCID corrected to PMC10917726. 10-fold CV on UCI AQ-10; kNN imputation fit globally across entire dataset outside cross-validation.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Tartarisco et al. (2021) - A Machine Learning Approach for Autism Spectrum Disorder Screening in Toddlers",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8004748/",
            "field": "source_url",
            "value_in_audit": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8004748/",
            "location_of_evidence": "Methods: Participants and Clinical Protocol",
            "short paraphrase of the passage": "PMCID corrected to PMC8004748. Clinical toddler cohort n=135 evaluated against expert clinical consensus ADOS-2, not tautological item sum.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Vakadkar et al. (2021) - Detecting Autism Spectrum Disorder in Toddlers Using Machine Learning",
            "url": "https://doi.org/10.1007/s42979-021-00776-5",
            "field": "evidence_level",
            "value_in_audit": "abstract_only",
            "location_of_evidence": "Abstract / Publisher Record",
            "short paraphrase of the passage": "Paywalled article in SN Computer Science; audited strictly from abstract record reporting toddler ASD classification.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Thabtah et al. (2022) - Unsupervised Machine Learning for Autism Spectrum Disorder Screening",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9458819/",
            "field": "split_type",
            "value_in_audit": "unclear",
            "location_of_evidence": "Sec. 4 Data Analysis",
            "short paraphrase of the passage": "Unsupervised clustering (k-means, MCA) and association rule discovery evaluated against diagnostic labels without a supervised train/test fold split.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Katlariwala et al. (2026) - Benchmarking behavioral autism screening: audit of circularity and generalization",
            "url": "https://doi.org/10.1101/2026.11.01.26367003",
            "field": "source_url",
            "value_in_audit": "https://doi.org/10.1101/2026.11.01.26367003",
            "location_of_evidence": "Methods",
            "short paraphrase of the passage": "medRxiv DOI prefix corrected. Repeated 5-fold CV explicitly auditing circularity across AQ-10 item vs demographic models.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Thabtah & Peebles (2020) - A new machine learning model based on rules for autism screening",
            "url": "https://doi.org/10.1177/1460458218824711",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Methods: Machine Learning Process",
            "short paraphrase of the passage": "RML rule induction and continuous feature discretization applied across full dataset prior to 10-fold cross-validation.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Hadi et al. (2026) - Machine learning approach for early autism detection in toddlers",
            "url": "http://doi.org/10.11591/ijai.v15.i3.pp2103-2112",
            "field": "evidence_level",
            "value_in_audit": "abstract_only",
            "location_of_evidence": "Abstract / Publisher Record",
            "short paraphrase of the passage": "Paywalled article in IAES IJ-AI; audited strictly from abstract record reporting toddler ASD screening with ML classifiers.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Abu-Errub (2025) - Early Autism Detection Using Machine Learning Classifiers",
            "url": "https://doi.org/10.48084/etasr.12999",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Sec. III-IV Methodology",
            "short paraphrase of the passage": "Random 80:20 split; Chi-square feature importance ranking and selection evaluated on full dataset prior to classifier training.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Hasan et al. (2023) - Machine Learning Based Autism Spectrum Disorder Detection",
            "url": "https://doi.org/10.1109/ACCESS.2022.3232490",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Sec. III Proposed Framework",
            "short paraphrase of the passage": "10-fold CV on UCI AQ-10; Mutual Information feature selection and SMOTE balancing applied on pooled dataset prior to fold training.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Heinsfeld et al. (2018) - Identification of Autism Spectrum Disorder Using Deep Learning and the ABIDE Dataset",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5635344/",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Methods: Deep Learning",
            "short paraphrase of the passage": "Deep autoencoder unsupervised pretraining executed on all 1,035 subjects simultaneously prior to 10-fold random CV splitting.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Abraham et al. (2017) - Deriving Reproducible Biomarkers from Multi-Site Resting-State fMRI",
            "url": "https://doi.org/10.1016/j.neuroimage.2016.10.045",
            "field": "split_type",
            "value_in_audit": "site-wise",
            "location_of_evidence": "Sec. 2.3 & 3 Validation",
            "short paraphrase of the passage": "Leave-center-out cross-validation across 17 sites; site covariate regression applied inside cross-validation folds.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Nielsen et al. (2013) - Multisite functional connectivity MRI classification of autism",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3782716/",
            "field": "split_type",
            "value_in_audit": "site-wise",
            "location_of_evidence": "Sec. Leave-one-out cross-validation by site",
            "short paraphrase of the passage": "Evaluated strictly Leave-One-Site-Out (LOSO) across all 16 clinical imaging sites in ABIDE I.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Plitt et al. (2015) - Functional Connectivity Predicts Clinical Severity in Autism Spectrum Disorder",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4309950/",
            "field": "harmonization",
            "value_in_audit": "none",
            "location_of_evidence": "Methods: Subject Selection",
            "short paraphrase of the passage": "Sites selected with matching scan protocols but no algorithmic site harmonization model (neither ComBat nor site regression) applied.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Huang & Chung (2020) - Disease Prediction Using Graph Convolutional Networks with Population Graph",
            "url": "https://doi.org/10.1007/978-3-030-59719-1_53",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Sec. 3 Methods",
            "short paraphrase of the passage": "Population graph constructed using phenotypic similarity across all subjects simultaneously prior to fold evaluation.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Li et al. (2018) - Deep Learning for ASD Classification with Multi-Site fMRI",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6066582/",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "N",
            "location_of_evidence": "Sec. Results",
            "short paraphrase of the passage": "Pooled 10-fold CV on ABIDE I; autoencoder transfer learning weights pretrained on entire dataset.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Bi et al. (2018) - Random Support Vector Machine Cluster for ASD Prediction",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5808191/",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "Y",
            "location_of_evidence": "Sec. Methods",
            "short paraphrase of the passage": "Recursive feature elimination and ensemble clustering fit inside cross-validation training folds.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Shao et al. (2023) - Heterogeneous Graph Attention Network for ASD Classification",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10536734/",
            "field": "preprocessing_fit_in_fold",
            "value_in_audit": "Y",
            "location_of_evidence": "Sec. 3 Model",
            "short paraphrase of the passage": "Heterogeneous graph attention network trained with 10-fold CV with test edges masked.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Sherkatghanad et al. (2020) - Automated Detection of Autism Spectrum Disorder Using Deep Learning",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6971220/",
            "field": "split_type",
            "value_in_audit": "random",
            "location_of_evidence": "Sec. 3 Results",
            "short paraphrase of the passage": "10-fold cross-validation pooled randomly across all 1,035 ABIDE subjects without site grouping.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Dvornek et al. (2017) - Identifying Autism from Resting-State fMRI Using Long Short-Term Memory Networks",
            "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5669262/",
            "field": "split_type",
            "value_in_audit": "subject-wise",
            "location_of_evidence": "Sec. 3 Experiments",
            "short paraphrase of the passage": "Evaluated on single clinical site (NYU, n=175) using 10-fold subject cross-validation.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Ktena et al. (2018) - Metric Learning with Spectral Graph Convolutions on Brain Networks",
            "url": "https://doi.org/10.1016/j.neuroimage.2017.12.052",
            "field": "harmonization",
            "value_in_audit": "none",
            "location_of_evidence": "Sec. 4 Experiments",
            "short paraphrase of the passage": "Siamese graph CNN distance metric learning without site covariate regression or ComBat.",
            "human_verified": "",
            "human_value": ""
        },
        {
            "paper": "Kan et al. (2022) - Brain Network Transformer",
            "url": "https://doi.org/10.48550/arXiv.2210.06681",
            "field": "split_type",
            "value_in_audit": "site-wise",
            "location_of_evidence": "Sec. 5 & Tables 1-2",
            "short paraphrase of the passage": "Evaluated both 5-fold CV and Leave-One-Site-Out (LOSO) across ABIDE sites; external validation on ABCD.",
            "human_verified": "",
            "human_value": ""
        }
    ]

    df_worksheet = pd.DataFrame(flagged_data)
    out_csv = DOCS_DIR / "lit_verification_worksheet.csv"
    df_worksheet.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with {len(df_worksheet)} rows.")


def build_recompute_script():
    script_content = '''"""
Recomputes all literature audit counts (with denominators) from docs/lit_audit_template.csv
and optionally docs/lit_verification_worksheet.csv (incorporating human overrides when present).
Also reports counts restricted to the conservative subset: full-text papers NOT flagged for review (n=10).
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "docs"

def recompute_lit_counts(use_human_overrides: bool = True):
    template_path = DOCS_DIR / "lit_audit_template.csv"
    worksheet_path = DOCS_DIR / "lit_verification_worksheet.csv"
    
    df = pd.read_csv(template_path)
    
    if use_human_overrides and worksheet_path.exists():
        ws = pd.read_csv(worksheet_path)
        overrides = ws[(ws["human_verified"].astype(str).str.upper() == "Y") & (ws["human_value"].notna()) & (ws["human_value"].str.strip() != "")]
        for _, r in overrides.iterrows():
            paper_str = str(r["paper"])
            field = str(r["field"])
            val = str(r["human_value"])
            # Match author
            for idx, d_row in df.iterrows():
                if d_row["first_author"].lower() in paper_str.lower():
                    df.at[idx, field] = val
                    df.at[idx, "needs_human_check"] = "N"

    n_total = len(df)
    n_tabular = int((df["dataset"].str.contains("AQ-10|Q-CHAT", case=False, na=False)).sum())
    n_neuro = n_total - n_tabular
    
    ev_full = int((df["evidence_level"] == "full_text").sum())
    ev_abs = int((df["evidence_level"] == "abstract_only").sum())
    
    flagged = int((df["needs_human_check"].astype(str).str.upper() == "Y").sum())
    
    # Split type
    eval_split = df[df["split_type"].isin(["random", "site-wise", "subject-wise"])]
    n_split_eval = len(eval_split)
    n_random = int((eval_split["split_type"] == "random").sum())
    n_sitewise = int((eval_split["split_type"] == "site-wise").sum())
    n_subjwise = int((eval_split["split_type"] == "subject-wise").sum())
    n_unclear_split = n_total - n_split_eval

    # Preprocessing in fold
    eval_prep = df[df["preprocessing_fit_in_fold"].isin(["Y", "N"])]
    n_prep_eval = len(eval_prep)
    n_prep_yes = int((eval_prep["preprocessing_fit_in_fold"] == "Y").sum())
    n_prep_no = int((eval_prep["preprocessing_fit_in_fold"] == "N").sum())
    n_prep_unclear = n_total - n_prep_eval

    # Leakage
    n_leak_feat = int((df["leaking_features"].astype(str).str.upper() == "Y").sum())
    n_leak_eval = int(df["leaking_features"].isin(["Y", "N"]).sum())
    
    # Harmonization in Neuroimaging
    df_neuro = df[~df["dataset"].str.contains("AQ-10|Q-CHAT", case=False, na=False)]
    n_neuro_tot = len(df_neuro)
    n_harm_none = int((df_neuro["harmonization"] == "none").sum())
    n_harm_combat = int((df_neuro["harmonization"] == "combat").sum())
    n_harm_other = int((df_neuro["harmonization"] == "other").sum())

    # Calibration & Uncertainty
    n_cal = int((df["calibration_reported"].astype(str).str.upper() == "Y").sum())
    n_unc = int((df["uncertainty_reported"].astype(str).str.upper() == "Y").sum())
    n_neither = int(((df["calibration_reported"].astype(str).str.upper() != "Y") & (df["uncertainty_reported"].astype(str).str.upper() != "Y")).sum())

    print("=" * 70)
    print("  LITERATURE META-AUDIT: RECOMPUTED COUNTS & PROPORTIONS")
    print("=" * 70)
    print(f"Total Corpus: {n_total} papers ({n_tabular} Tabular, {n_neuro} Neuroimaging)")
    print(f"Evidence Level: Full-Text = {ev_full}/{n_total} ({ev_full/n_total*100:.1f}%), Abstract-Only = {ev_abs}/{n_total} ({ev_abs/n_total*100:.1f}%)")
    print(f"Flagged for Human Review: {flagged}/{n_total} ({flagged/n_total*100:.1f}%)\n")

    print("OVERALL METHODOLOGICAL PROFILE:")
    print(f"  Split Protocol (evaluated n={n_split_eval}, unclear={n_unclear_split}):")
    print(f"    - Random Split:       {n_random}/{n_split_eval} ({n_random/n_split_eval*100:.1f}%)")
    print(f"    - Site-Wise (LOSO):   {n_sitewise}/{n_split_eval} ({n_sitewise/n_split_eval*100:.1f}%)")
    print(f"    - Subject-Wise:       {n_subjwise}/{n_split_eval} ({n_subjwise/n_split_eval*100:.1f}%)")
    print(f"  Preprocessing Inside Folds (evaluated n={n_prep_eval}, unclear={n_prep_unclear}):")
    print(f"    - Fit Inside Folds:   {n_prep_yes}/{n_prep_eval} ({n_prep_yes/n_prep_eval*100:.1f}%)")
    print(f"    - Fit on Pooled Data: {n_prep_no}/{n_prep_eval} ({n_prep_no/n_prep_eval*100:.1f}%)")
    print(f"  Neuroimaging Harmonization (n={n_neuro_tot}):")
    print(f"    - None (Raw):         {n_harm_none}/{n_neuro_tot} ({n_harm_none/n_neuro_tot*100:.1f}%)")
    print(f"    - ComBat:             {n_harm_combat}/{n_neuro_tot} ({n_harm_combat/n_neuro_tot*100:.1f}%)")
    print(f"    - Other:              {n_harm_other}/{n_neuro_tot} ({n_harm_other/n_neuro_tot*100:.1f}%)")
    print(f"  Calibration & Uncertainty Neglect:")
    print(f"    - Calibration Reported:   {n_cal}/{n_total} ({n_cal/n_total*100:.1f}%)")
    print(f"    - Uncertainty Reported:   {n_unc}/{n_total} ({n_unc/n_total*100:.1f}%)")
    print(f"    - Neither Cal nor Unc:    {n_neither}/{n_total} ({n_neither/n_total*100:.1f}%)\n")

    # CONSERVATIVE SUBSET: Full-Text Papers NOT Flagged for Review
    cons = df[(df["evidence_level"] == "full_text") & (df["needs_human_check"].astype(str).str.upper() == "N")]
    n_c = len(cons)
    print("=" * 70)
    print(f"  CONSERVATIVE SUBSET: FULL-TEXT PAPERS NOT FLAGGED FOR REVIEW (N = {n_c})")
    print("=" * 70)
    c_tab = int((cons["dataset"].str.contains("AQ-10|Q-CHAT", case=False, na=False)).sum())
    c_neu = n_c - c_tab
    print(f"Composition: {c_tab} Tabular, {c_neu} Neuroimaging")
    
    c_split_eval = cons[cons["split_type"].isin(["random", "site-wise", "subject-wise"])]
    n_c_split = len(c_split_eval)
    c_rand = int((c_split_eval["split_type"] == "random").sum())
    c_site = int((c_split_eval["split_type"] == "site-wise").sum())
    c_subj = int((c_split_eval["split_type"] == "subject-wise").sum())
    print(f"  Split Protocol (n={n_c_split}):")
    print(f"    - Random Split:       {c_rand}/{n_c_split} ({c_rand/n_c_split*100:.1f}%)")
    print(f"    - Site-Wise (LOSO):   {c_site}/{n_c_split} ({c_site/n_c_split*100:.1f}%)")
    print(f"    - Subject-Wise:       {c_subj}/{n_c_split} ({c_subj/n_c_split*100:.1f}%)")

    c_prep_eval = cons[cons["preprocessing_fit_in_fold"].isin(["Y", "N"])]
    n_c_prep = len(c_prep_eval)
    c_prep_yes = int((c_prep_eval["preprocessing_fit_in_fold"] == "Y").sum())
    c_prep_no = int((c_prep_eval["preprocessing_fit_in_fold"] == "N").sum())
    print(f"  Preprocessing Inside Folds (n={n_c_prep}):")
    print(f"    - Fit Inside Folds:   {c_prep_yes}/{n_c_prep} ({c_prep_yes/n_c_prep*100:.1f}%)")
    print(f"    - Fit on Pooled Data: {c_prep_no}/{n_c_prep} ({c_prep_no/n_c_prep*100:.1f}%)")

    c_neuro = cons[~cons["dataset"].str.contains("AQ-10|Q-CHAT", case=False, na=False)]
    c_harm_none = int((c_neuro["harmonization"] == "none").sum())
    print(f"  Neuroimaging Harmonization (n={len(c_neuro)}):")
    print(f"    - None (Raw):         {c_harm_none}/{len(c_neuro)} ({c_harm_none/len(c_neuro)*100:.1f}%)")

    c_cal = int((cons["calibration_reported"].astype(str).str.upper() == "Y").sum())
    c_unc = int((cons["uncertainty_reported"].astype(str).str.upper() == "Y").sum())
    c_neither = int(((cons["calibration_reported"].astype(str).str.upper() != "Y") & (cons["uncertainty_reported"].astype(str).str.upper() != "Y")).sum())
    print(f"  Calibration & Uncertainty Neglect (n={n_c}):")
    print(f"    - Calibration Reported:   {c_cal}/{n_c} ({c_cal/n_c*100:.1f}%)")
    print(f"    - Uncertainty Reported:   {c_unc}/{n_c} ({c_unc/n_c*100:.1f}%)")
    print(f"    - Neither Cal nor Unc:    {c_neither}/{n_c} ({c_neither/n_c*100:.1f}%)\n")

    return {
        "n_total": n_total,
        "n_conservative": n_c,
        "overall": {
            "random_split_rate": f"{n_random}/{n_split_eval}",
            "prep_in_fold_rate": f"{n_prep_yes}/{n_prep_eval}",
            "neither_cal_unc_rate": f"{n_neither}/{n_total}"
        },
        "conservative": {
            "random_split_rate": f"{c_rand}/{n_c_split}",
            "prep_in_fold_rate": f"{c_prep_yes}/{n_c_prep}",
            "neither_cal_unc_rate": f"{c_neither}/{n_c}"
        }
    }

if __name__ == "__main__":
    recompute_lit_counts(use_human_overrides=True)
'''
    out_script = SCRIPTS_DIR / "recompute_lit_counts.py"
    with open(out_script, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"[OK] Wrote {out_script}")


def build_prior_comparison_csv():
    # Load prior comparison rows from audited full-text literature
    rows = [
        {
            "paper": "Heinsfeld et al. (2018)",
            "year": 2018,
            "dataset": "ABIDE I (CC200)",
            "N": 1035,
            "split_protocol": "random",
            "metric": "Accuracy",
            "reported_value": "70.0%",
            "study_matched_metric": "Random-CV Accuracy (Heinsfeld AE)",
            "study_matched_value": "70.84% (AUROC 0.7510)",
            "like_for_like": "Partial",
            "pipeline_difference_notes": "Heinsfeld pretrained autoencoder on full N=1035 pooled prior to CV split; our study evaluates N=854 with strictly in-fold training."
        },
        {
            "paper": "Heinsfeld et al. (2018)",
            "year": 2018,
            "dataset": "ABIDE I (CC200)",
            "N": 1035,
            "split_protocol": "site-wise (LOSO)",
            "metric": "Accuracy",
            "reported_value": "Not Reported (Random Only)",
            "study_matched_metric": "LOSO Accuracy (Heinsfeld AE)",
            "study_matched_value": "66.04% (AUROC 0.6774)",
            "like_for_like": "Novel Comparison",
            "pipeline_difference_notes": "Original paper only evaluated random 10-fold CV; our matched LOSO evaluation reveals drop to 66.04%."
        },
        {
            "paper": "Abraham et al. (2017)",
            "year": 2017,
            "dataset": "ABIDE I (Multi-atlas)",
            "N": 871,
            "split_protocol": "site-wise (LOSO)",
            "metric": "Accuracy",
            "reported_value": "66.8%",
            "study_matched_metric": "LOSO Accuracy (Logistic Regression)",
            "study_matched_value": "65.69% (AUROC 0.6593)",
            "like_for_like": "Yes",
            "pipeline_difference_notes": "Abraham evaluated leave-center-out with site covariate regression; our LR LOSO yields 65.69% accuracy."
        },
        {
            "paper": "Parisot et al. (2018)",
            "year": 2018,
            "dataset": "ABIDE I (CC200)",
            "N": 871,
            "split_protocol": "random",
            "metric": "Accuracy",
            "reported_value": "69.5%",
            "study_matched_metric": "Random-CV Accuracy (Logistic Regression / SVM)",
            "study_matched_value": "68.85% / 70.14% (AUROC 0.7420 / 0.7580)",
            "like_for_like": "Partial",
            "pipeline_difference_notes": "Parisot used population GCN with phenotypic acquisition edges; our study compares matched random-CV on CC200 connectomes."
        },
        {
            "paper": "Eslami et al. (2019)",
            "year": 2019,
            "dataset": "ABIDE I (CC200)",
            "N": 1035,
            "split_protocol": "random",
            "metric": "Accuracy",
            "reported_value": "70.3%",
            "study_matched_metric": "Random-CV Accuracy (Random Forest / LightGBM)",
            "study_matched_value": "71.43% / 70.26% (AUROC 0.7710 / 0.7602)",
            "like_for_like": "Partial",
            "pipeline_difference_notes": "Eslami evaluated ASD-DiagNet with data augmentation; our matched random-CV baseline yields 71.43% on clean data."
        },
        {
            "paper": "Kan et al. (2022)",
            "year": 2022,
            "dataset": "ABIDE I (CC200)",
            "N": 1035,
            "split_protocol": "site-wise (LOSO)",
            "metric": "AUROC",
            "reported_value": "0.6870",
            "study_matched_metric": "LOSO AUROC (Random Forest)",
            "study_matched_value": "0.7003 [0.6582, 0.7381]",
            "like_for_like": "Yes",
            "pipeline_difference_notes": "Kan evaluated Brain Network Transformer under LOSO; our top tabular/classical model (Random Forest) achieves 0.7003."
        },
        {
            "paper": "Nielsen et al. (2013)",
            "year": 2013,
            "dataset": "ABIDE I (16 sites)",
            "N": 964,
            "split_protocol": "site-wise (LOSO)",
            "metric": "Accuracy",
            "reported_value": "60.0%",
            "study_matched_metric": "LOSO Accuracy (Tangent Space LR)",
            "study_matched_value": "60.42% (AUROC 0.6092)",
            "like_for_like": "Yes",
            "pipeline_difference_notes": "Nielsen evaluated leave-one-site-out classification on functional connectivity across 16 ABIDE sites."
        },
        {
            "paper": "Akter et al. (2019)",
            "year": 2019,
            "dataset": "UCI AQ-10 Adult",
            "N": 704,
            "split_protocol": "random",
            "metric": "Accuracy",
            "reported_value": "98.7%",
            "study_matched_metric": "Items-only Accuracy (Adult)",
            "study_matched_value": "99.55% [99.31%, 99.76%]",
            "like_for_like": "No (Tautological)",
            "pipeline_difference_notes": "Akter trained classifiers on items A1-A10 that deterministically sum to define the target screening label (Tautology Index = 0.8398); clean demographics-only accuracy drops to 73.76% majority baseline."
        },
        {
            "paper": "Hossain et al. (2021)",
            "year": 2021,
            "dataset": "UCI AQ-10 Child",
            "N": 292,
            "split_protocol": "random",
            "metric": "Accuracy",
            "reported_value": "100.0%",
            "study_matched_metric": "Items-only Accuracy (Child)",
            "study_matched_value": "99.54% [99.16%, 99.85%]",
            "like_for_like": "No (Tautological)",
            "pipeline_difference_notes": "Hossain reported 100% accuracy using AQ-10 items with global imputation; our matched item evaluation replicates near 100% (99.54%), but clean demographics-only AUROC is 0.4473 (Tautology Index = 1.1054 raw, clipped 1.0000)."
        },
        {
            "paper": "Ehsan et al. (2025)",
            "year": 2025,
            "dataset": "Q-CHAT-10 Toddler",
            "N": 1054,
            "split_protocol": "random",
            "metric": "Accuracy",
            "reported_value": "99.0%",
            "study_matched_metric": "Items-only Accuracy (Toddler)",
            "study_matched_value": "100.00% [100.00%, 100.00%]",
            "like_for_like": "No (Tautological)",
            "pipeline_difference_notes": "Ehsan evaluated toddler screening items that deterministically define diagnosis (cutoff >= 3); clean demographic AUROC drops to 0.5527."
        }
    ]

    df_prior = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "prior_comparison.csv"
    df_prior.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv} with {len(df_prior)} prior comparison rows.")


def main():
    print("Building literature verification worksheet...")
    build_lit_verification_worksheet()
    print("Building recompute script...")
    build_recompute_script()
    print("Building prior comparison table...")
    build_prior_comparison_csv()


if __name__ == "__main__":
    main()
