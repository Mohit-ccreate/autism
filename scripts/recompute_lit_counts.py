"""
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
        overrides = ws[(ws["human_verified"].astype(str).str.upper() == "Y") & (ws["human_value"].notna()) & (ws["human_value"].astype(str).str.strip() != "")]
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
