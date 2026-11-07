import pandas as pd
from pathlib import Path

REPO_ROOT = Path("c:/Users/mohit/OneDrive/Desktop/asd-project")

def update_template():
    df = pd.read_csv(REPO_ROOT / "docs" / "lit_audit_template_v1.csv")

    # Define corrections per first_author
    corrections = {
        "Sollis": {
            "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12594979/",
            "needs_human_check": "Y",
            "notes": "PMCID corrected (PMC12594979). Discovery n=1054 5-fold CV repeated 10x, external validation on Polish cohort n=125."
        },
        "Shrivastava": {
            "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10917726/",
            "needs_human_check": "Y",
            "notes": "PMCID corrected (PMC10917726). 10-fold CV on UCI AQ-10 cohorts; kNN imputation fit globally outside folds."
        },
        "Tartarisco": {
            "source_url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8004748/",
            "needs_human_check": "Y",
            "notes": "PMCID corrected (PMC8004748). Clinical toddler cohort n=135 against clinical consensus ADOS-2 (not tautological item threshold)."
        },
        "Katlariwala": {
            "source_url": "https://doi.org/10.1101/2026.11.01.26367003",
            "needs_human_check": "Y",
            "notes": "medRxiv DOI prefix corrected. 5-fold repeated stratified CV auditing circularity across AQ-10 item vs demographic models."
        },
        "Thabtah": [
            # There are two Thabtah papers: 2022 and 2020
        ],
        "Hossain": {
            "preprocessing_fit_in_fold": "N",
            "needs_human_check": "Y",
            "notes": "Sec. 3 Materials and Methods: imputation and one-hot encoding performed on full dataset prior to train/test partitioning."
        },
        "Abu-Errub": {
            "split_type": "random",
            "preprocessing_fit_in_fold": "N",
            "needs_human_check": "Y",
            "notes": "Sec. III-IV: Random 80:20 split; Chi-square feature selection evaluated on full dataset prior to model training."
        },
        "Hasan": {
            "split_type": "random",
            "needs_human_check": "Y",
            "notes": "Sec. III: 10-fold cross-validation on UCI AQ-10; Mutual Information feature selection and SMOTE applied on whole dataset."
        },
        "Heinsfeld": {
            "preprocessing_fit_in_fold": "N",
            "needs_human_check": "Y",
            "notes": "Methods Deep Learning: deep autoencoder unsupervised pretraining executed on all 1,035 subjects prior to fold splitting."
        },
        "Abraham": {
            "harmonization": "other",
            "needs_human_check": "Y",
            "notes": "Sec. 2.3 & 3: Leave-center-out cross-validation across 17 sites; site covariate regression applied inside cross-validation."
        },
        "Nielsen": {
            "split_type": "site-wise",
            "needs_human_check": "Y",
            "notes": "Sec. Leave-one-out cross-validation by site: evaluated strictly Leave-One-Site-Out (LOSO) across all 16 clinical sites."
        },
        "Plitt": {
            "harmonization": "none",
            "needs_human_check": "Y",
            "notes": "Methods: Selected sites with matching scan protocols but no algorithmic site harmonization model (neither ComBat nor site regression) applied."
        },
        "Huang": {
            "preprocessing_fit_in_fold": "N",
            "needs_human_check": "Y",
            "notes": "Sec. 3: Population graph constructed using phenotypic data across all subjects simultaneously."
        },
        "Li": {
            "split_type": "random",
            "preprocessing_fit_in_fold": "N",
            "needs_human_check": "Y",
            "notes": "Sec. Results: Pooled 10-fold CV on ABIDE I; autoencoder transfer learning weights pretrained on entire dataset."
        },
        "Bi": {
            "preprocessing_fit_in_fold": "Y",
            "needs_human_check": "Y",
            "notes": "Sec. Methods: Recursive feature elimination and ensemble clustering fit inside cross-validation folds."
        },
        "Shao": {
            "preprocessing_fit_in_fold": "Y",
            "needs_human_check": "Y",
            "notes": "Sec. 3: Heterogeneous graph attention network trained with 10-fold CV with test edges masked."
        },
        "Sherkatghanad": {
            "split_type": "random",
            "needs_human_check": "Y",
            "notes": "Sec. 3 Results: 10-fold cross-validation pooled across all 1,035 ABIDE subjects."
        },
        "Dvornek": {
            "split_type": "subject-wise",
            "needs_human_check": "Y",
            "notes": "Sec. 3: Evaluated on single clinical site (NYU, n=175) using 10-fold subject cross-validation."
        },
        "Ktena": {
            "harmonization": "none",
            "needs_human_check": "Y",
            "notes": "Sec. 4: Siamese graph CNN distance metric learning without site covariate regression or ComBat."
        },
        "Kan": {
            "split_type": "site-wise",
            "needs_human_check": "Y",
            "notes": "Sec. 5 & Tables 1-2: Evaluated both 5-fold CV and Leave-One-Site-Out (LOSO) across ABIDE sites; external validation on ABCD."
        }
    }

    for idx, row in df.iterrows():
        author = row["first_author"]
        year = row["year"]

        # Handle Thabtah two papers specifically
        if author == "Thabtah" and year == 2022:
            df.at[idx, "split_type"] = "unclear"
            df.at[idx, "needs_human_check"] = "Y"
            df.at[idx, "notes"] = "Sec. 4: Unsupervised clustering (k-means, MCA) evaluated against diagnostic label; no supervised fold partition."
        elif author == "Thabtah" and year == 2020:
            df.at[idx, "preprocessing_fit_in_fold"] = "N"
            df.at[idx, "needs_human_check"] = "Y"
            df.at[idx, "notes"] = "Methods: RML rule induction and discretization executed on full dataset prior to 10-fold CV."
        elif author in corrections:
            corr = corrections[author]
            for col, val in corr.items():
                df.at[idx, col] = val

    out_path = REPO_ROOT / "docs" / "lit_audit_template.csv"
    df.to_csv(out_path, index=False)
    print(f"Successfully updated {len(df)} rows in {out_path}")

if __name__ == "__main__":
    update_template()
