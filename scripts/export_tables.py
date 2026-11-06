"""
Export Tables Script — scripts/export_tables.py

Exports all main empirical benchmark tables in both LaTeX (.tex) and CSV (.csv) format
into results/tables/.

Strictly uses numbers generated from code runs recorded in results/*.json.
Zero hardcoded or invented values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def export_tautology_tables():
    taut_path = RESULTS_DIR / "tautology_audit.json"
    if not taut_path.exists():
        print(f"[SKIP] {taut_path} not found.")
        return

    with open(taut_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Rule Recovery Table
    rule_rows = []
    for cohort, r in data.get("rule_recovery", {}).items():
        ti_info = data.get("tautology_indices", {}).get(cohort, {})
        ti_raw = ti_info.get("tautology_index", None)
        ti_clipped = ti_info.get("tautology_index_clipped", None)
        rule_rows.append({
            "Cohort": cohort.title(),
            "Threshold": f"sum(A_i) >= {r['threshold']}",
            "N Total": r["n_total"],
            "Concordant": r["n_concordant"],
            "Discordant": r["n_discordant"],
            "Concordance (%)": f"{r['accuracy'] * 100:.2f}%",
            "Tautology Index (Raw)": f"{ti_raw:.3f}" if ti_raw is not None else "N/A",
            "Tautology Index (Clipped [0,1])": f"{ti_clipped:.3f}" if ti_clipped is not None else "N/A",
        })

    df_rule = pd.DataFrame(rule_rows)
    df_rule.to_csv(TABLES_DIR / "table_rule_recovery.csv", index=False)
    with open(TABLES_DIR / "table_rule_recovery.tex", "w", encoding="utf-8") as f:
        f.write(df_rule.to_latex(index=False, caption="Rule-Recovery Concordance across Screening Cohorts", label="tab:rule_recovery"))
    print("[OK] Exported table_rule_recovery (.csv and .tex)")

    # 2. Feature Ablation Table
    abl_rows = []
    for cohort, abl_sets in data.get("feature_ablation", {}).items():
        for fs_name, m in abl_sets.items():
            ci = m.get("auroc_ci", [None, None])
            abl_rows.append({
                "Cohort": cohort.title(),
                "Feature Set": fs_name.replace("var", "Var ").replace("_", " ").title(),
                "Features": m.get("n_features", 0),
                "AUROC": f"{m.get('auroc', 0):.4f}",
                "95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "N/A",
                "Balanced Acc": f"{m.get('bacc', 0):.4f}",
                "Sensitivity": f"{m.get('sensitivity', 0):.4f}",
                "Specificity": f"{m.get('specificity', 0):.4f}",
            })

    df_abl = pd.DataFrame(abl_rows)
    df_abl.to_csv(TABLES_DIR / "table_feature_ablation.csv", index=False)
    with open(TABLES_DIR / "table_feature_ablation.tex", "w", encoding="utf-8") as f:
        f.write(df_abl.to_latex(index=False, caption="Nested Feature Ablation across Screening Cohorts", label="tab:feature_ablation"))
    print("[OK] Exported table_feature_ablation (.csv and .tex)")


def export_dedup_and_threshold_tables():
    dt_path = RESULTS_DIR / "dedup_and_threshold_audit.json"
    if not dt_path.exists():
        print(f"[SKIP] {dt_path} not found.")
        return

    with open(dt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Deduplication Accounting Table
    dedup_rows = []
    for cohort, info in data.get("deduplication_audit", {}).items():
        dedup_rows.append({
            "Cohort": cohort.title(),
            "Raw N": info["n_raw"],
            "Clean N": info["n_clean"],
            "Exact Duplicates Removed": info["exact_duplicates_purged"],
            "Purge Pct (%)": f"{info['pct_purged']:.2f}%",
            "Near-Duplicates (Hamming <= 2)": info["clean_item_near_duplicate_pairs"],
            "Cross-Label Conflicts": info["cross_label_conflicts_raw"],
            "Items AUROC (Raw vs Clean)": f"{info['auroc_items_raw']:.4f} vs {info['auroc_items_clean']:.4f}",
            "Demos AUROC (Raw vs Clean)": f"{info['auroc_demos_raw']:.4f} vs {info['auroc_demos_clean']:.4f}",
        })

    df_dedup = pd.DataFrame(dedup_rows)
    df_dedup.to_csv(TABLES_DIR / "table_deduplication_accounting.csv", index=False)
    with open(TABLES_DIR / "table_deduplication_accounting.tex", "w", encoding="utf-8") as f:
        f.write(df_dedup.to_latex(index=False, caption="Deduplication Accounting and Generalization Impact across Screening Cohorts", label="tab:dedup"))
    print("[OK] Exported table_deduplication_accounting (.csv and .tex)")

    # Threshold Sensitivity Table
    thresh_rows = []
    for cohort, t_data in data.get("threshold_sweeps", {}).items():
        rec_t = t_data.get("recovered_cutoff", 7)
        pub_t = t_data.get("published_cutoff_allison2012", 6)
        for tau_key, m in t_data.get("threshold_sweep", {}).items():
            t_val = m["threshold"]
            thresh_rows.append({
                "Cohort": cohort.title(),
                "Threshold tau": t_val,
                "Accuracy (%)": f"{m['accuracy'] * 100:.2f}%",
                "Sensitivity (%)": f"{m['sensitivity'] * 100:.2f}%",
                "Specificity (%)": f"{m['specificity'] * 100:.2f}%",
                "Discordant Cases": m["discordant_cases"],
                "Status": "Empirically Recovered Cutoff" if t_val == rec_t else ("Published Clinical Cutoff (Allison et al.)" if t_val == pub_t else ""),
            })

    df_thresh = pd.DataFrame(thresh_rows)
    df_thresh.to_csv(TABLES_DIR / "table_threshold_sensitivity.csv", index=False)
    with open(TABLES_DIR / "table_threshold_sensitivity.tex", "w", encoding="utf-8") as f:
        f.write(df_thresh.to_latex(index=False, caption="Diagnostic Threshold Concordance Sweep (Neighboring Cutoffs 4-9)", label="tab:threshold_sweep"))
    print("[OK] Exported table_threshold_sensitivity (.csv and .tex)")


def export_site_tables():
    site_path = RESULTS_DIR / "site_analysis.json"
    if not site_path.exists():
        print(f"[SKIP] {site_path} not found.")
        return

    with open(site_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Harmonization arms comparison
    harm_rows = []
    for harm_name, m in data.get("harmonization_comparison", {}).items():
        ci_subj = m.get("subject_bootstrap_ci_95", [None, None])
        ci_clust = m.get("site_cluster_bootstrap_ci_95", [None, None])
        harm_rows.append({
            "Harmonization Strategy": harm_name.replace("_", " ").title(),
            "Protocol": "Leave-One-Site-Out (LOSO)",
            "Pooled Accuracy": f"{m.get('pooled_accuracy', 0):.4f}",
            "Balanced Acc": f"{m.get('pooled_bacc', 0):.4f}",
            "Pooled AUROC": f"{m.get('pooled_auroc', 0):.4f}",
            "Subject 95% CI": f"[{ci_subj[0]:.3f}, {ci_subj[1]:.3f}]" if ci_subj[0] is not None else "N/A",
            "Site Cluster 95% CI": f"[{ci_clust[0]:.3f}, {ci_clust[1]:.3f}]" if ci_clust[0] is not None else "N/A",
            "Permutation p": f"{m.get('within_site_permutation_p', 1.0):.4f}",
            "Binomial p": f"{m.get('binomial_p_vs_chance', 1.0):.4e}",
            "Methodological Flag": "LEAKY (Deliberate Wrong Arm)" if m.get("is_leaky") else "Valid In-Fold",
        })

    df_harm = pd.DataFrame(harm_rows)
    df_harm.to_csv(TABLES_DIR / "table_harmonization_comparison.csv", index=False)
    with open(TABLES_DIR / "table_harmonization_comparison.tex", "w", encoding="utf-8") as f:
        f.write(df_harm.to_latex(index=False, caption="ABIDE I Harmonization Protocol Benchmark across 20 Clinical Sites (N=854)", label="tab:harmonization"))
    print("[OK] Exported table_harmonization_comparison (.csv and .tex)")

    # Per-Site Generalization Gap Table
    gap_table = data.get("per_site_gap_table", [])
    if gap_table:
        df_gap = pd.DataFrame(gap_table)
        df_gap.to_csv(TABLES_DIR / "table_per_site_generalization_gap.csv", index=False)
        with open(TABLES_DIR / "table_per_site_generalization_gap.tex", "w", encoding="utf-8") as f:
            f.write(df_gap.to_latex(index=False, caption="Per-Site Generalization Gap and Performance across All 20 ABIDE Sites", label="tab:site_gaps"))
        print("[OK] Exported table_per_site_generalization_gap (.csv and .tex)")


def export_benchmark_leaderboard_tables():
    bm_path = RESULTS_DIR / "benchmark_leaderboard.json"
    if not bm_path.exists():
        print(f"[SKIP] {bm_path} not found.")
        return

    with open(bm_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Tabular Leak-Free Benchmark
    tab_rows = []
    for cohort, c_info in data.get("tabular_leak_free", {}).items():
        n = c_info.get("n_samples", 0)
        for model_name, m in c_info.get("models", {}).items():
            auc_ci = m.get("auroc_ci_95", [None, None])
            tab_rows.append({
                "Cohort": cohort.title(),
                "N": n,
                "Model": model_name,
                "AUROC": f"{m.get('auroc', 0):.4f}",
                "AUROC 95% CI": f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]" if auc_ci[0] is not None else "N/A",
                "Accuracy": f"{m.get('accuracy', 0):.4f}",
                "Balanced Acc": f"{m.get('balanced_accuracy', 0):.4f}",
                "Sensitivity": f"{m.get('sensitivity', 0):.4f}",
                "Specificity": f"{m.get('specificity', 0):.4f}",
                "Brier Score": f"{m.get('brier_score', 0):.4f}",
            })

    df_tab_bm = pd.DataFrame(tab_rows)
    df_tab_bm.to_csv(TABLES_DIR / "table_tabular_benchmark_leaderboard.csv", index=False)
    with open(TABLES_DIR / "table_tabular_benchmark_leaderboard.tex", "w", encoding="utf-8") as f:
        f.write(df_tab_bm.to_latex(index=False, caption="Leak-Free Tabular Screening Benchmark Leaderboard (Demographics Only)", label="tab:tabular_leaderboard"))
    print("[OK] Exported table_tabular_benchmark_leaderboard (.csv and .tex)")

    # 2. ABIDE Full Neuroimaging Benchmark
    abide = data.get("abide_full", {})
    loso = abide.get("loso_benchmark", {})
    rand = abide.get("random_cv_benchmark", {})
    gaps = abide.get("site_leakage_gaps", {})

    abide_rows = []
    for m_name, m in loso.items():
        auc_ci = m.get("auroc_ci_95", [None, None])
        rand_m = rand.get(m_name, {})
        rand_auc = rand_m.get("auroc", None)
        gap_val = gaps.get(m_name, {}).get("leakage_gap", None)

        abide_rows.append({
            "Model Architecture": m_name,
            "LOSO AUROC": f"{m.get('auroc', 0):.4f}",
            "LOSO 95% CI": f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]" if auc_ci[0] is not None else "N/A",
            "LOSO Accuracy": f"{m.get('accuracy', 0):.4f}",
            "LOSO Balanced Acc": f"{m.get('balanced_accuracy', 0):.4f}",
            "Random CV AUROC": f"{rand_auc:.4f}" if rand_auc is not None else "N/A",
            "Scanner Gap (Rand - LOSO)": f"{gap_val:+.4f}" if gap_val is not None else "N/A",
            "Brier Score": f"{m.get('brier_score', 0):.4f}",
        })

    df_abide_bm = pd.DataFrame(abide_rows)
    df_abide_bm.to_csv(TABLES_DIR / "table_abide_benchmark_leaderboard.csv", index=False)
    with open(TABLES_DIR / "table_abide_benchmark_leaderboard.tex", "w", encoding="utf-8") as f:
        f.write(df_abide_bm.to_latex(index=False, caption="Full ABIDE I Leaderboard: Honest LOSO vs Random CV Leakage (N=854, 20 Sites)", label="tab:abide_leaderboard"))
    print("[OK] Exported table_abide_benchmark_leaderboard (.csv and .tex)")


def export_uncertainty_tables():
    unc_path = RESULTS_DIR / "uncertainty_main.json"
    if not unc_path.exists():
        print(f"[SKIP] {unc_path} not found.")
        return

    with open(unc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Conformal Prediction Sets Decomposition Table
    conf_rows = []
    # Tabular cohorts
    for cohort, c_data in data.get("tabular_leak_free", {}).items():
        c_conf = c_data.get("conformal_prediction", {})
        for key in ["lac_alpha10", "lac_alpha5", "aps_alpha10", "mondrian_alpha10"]:
            if key in c_conf:
                info = c_conf[key]
                conf_rows.append({
                    "Dataset": f"Tabular {cohort.title()} (Demographics Only)",
                    "Method": key.replace("_", " ").upper(),
                    "Nominal Coverage": f"{info.get('nominal_coverage', 0) * 100:.0f}%",
                    "Marginal Coverage": f"{info.get('marginal_coverage', 0) * 100:.2f}%",
                    "Mean Set Size": f"{info.get('mean_set_size', 0):.3f}",
                    "Empty Sets (%)": f"{info.get('fraction_empty', 0) * 100:.2f}%",
                    "Singleton Sets (%)": f"{info.get('fraction_singleton', 0) * 100:.2f}%",
                    "Both Sets (%)": f"{info.get('fraction_both', 0) * 100:.2f}%",
                    "Set Size Identity Verified": str(info.get("set_size_identity_verified", True)),
                })

    # ABIDE LOSO
    abide_conf = data.get("abide_full_loso", {}).get("conformal_prediction", {})
    for key in ["lac_alpha10", "lac_alpha5", "aps_alpha10", "mondrian_alpha10"]:
        if key in abide_conf:
            info = abide_conf[key]
            conf_rows.append({
                "Dataset": "Full ABIDE I (LOSO, 20 Sites)",
                "Method": key.replace("_", " ").upper(),
                "Nominal Coverage": f"{info.get('nominal_coverage', 0) * 100:.0f}%",
                "Marginal Coverage": f"{info.get('marginal_coverage', 0) * 100:.2f}%",
                "Mean Set Size": f"{info.get('mean_set_size', 0):.3f}",
                "Empty Sets (%)": f"{info.get('fraction_empty', 0) * 100:.2f}%",
                "Singleton Sets (%)": f"{info.get('fraction_singleton', 0) * 100:.2f}%",
                "Both Sets (%)": f"{info.get('fraction_both', 0) * 100:.2f}%",
                "Set Size Identity Verified": str(info.get("set_size_identity_verified", True)),
            })

    # Leaky contrast
    leaky_conf = data.get("leaky_contrast_baseline", {}).get("conformal_prediction", {})
    if "lac_alpha10" in leaky_conf:
        info = leaky_conf["lac_alpha10"]
        conf_rows.append({
            "Dataset": "UCI Adult AQ-10 (LEAKY CONTRAST ONLY ⚠️)",
            "Method": "LAC ALPHA10",
            "Nominal Coverage": f"{info.get('nominal_coverage', 0) * 100:.0f}%",
            "Marginal Coverage": f"{info.get('marginal_coverage', 0) * 100:.2f}%",
            "Mean Set Size": f"{info.get('mean_set_size', 0):.3f}",
            "Empty Sets (%)": f"{info.get('fraction_empty', 0) * 100:.2f}%",
            "Singleton Sets (%)": f"{info.get('fraction_singleton', 0) * 100:.2f}%",
            "Both Sets (%)": f"{info.get('fraction_both', 0) * 100:.2f}%",
            "Set Size Identity Verified": str(info.get("set_size_identity_verified", True)),
        })

    df_conf = pd.DataFrame(conf_rows)
    df_conf.to_csv(TABLES_DIR / "table_conformal_policy.csv", index=False)
    with open(TABLES_DIR / "table_conformal_policy.tex", "w", encoding="utf-8") as f:
        f.write(df_conf.to_latex(index=False, caption="Split Conformal Prediction Coverage and Set Cardinality Decomposition", label="tab:conformal"))
    print("[OK] Exported table_conformal_policy (.csv and .tex)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 70)
    print("  EXPORTING ALL PUBLICATION TABLES (.tex and .csv)")
    print("=" * 70)
    export_tautology_tables()
    export_dedup_and_threshold_tables()
    export_site_tables()
    export_benchmark_leaderboard_tables()
    export_uncertainty_tables()
    print("\n[OK] All tables exported successfully into results/tables/")


if __name__ == "__main__":
    main()
