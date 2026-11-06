"""
Export Tables Script — scripts/export_tables.py

Exports all main empirical benchmark tables in both LaTeX (.tex) and CSV (.csv) format
into results/tables/.

Strictly uses numbers generated from code runs recorded in results/*.json.
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
        ti = data.get("tautology_indices", {}).get(cohort, {}).get("tautology_index", None)
        rule_rows.append({
            "Cohort": cohort.title(),
            "Threshold": f"sum(A_i) >= {r['threshold']}",
            "N Total": r["n_total"],
            "Concordant": r["n_concordant"],
            "Discordant": r["n_discordant"],
            "Concordance (%)": f"{r['accuracy'] * 100:.2f}%",
            "Tautology Index": f"{ti:.3f}" if ti is not None else "N/A",
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


def export_site_tables():
    site_path = RESULTS_DIR / "site_analysis.json"
    if not site_path.exists():
        print(f"[SKIP] {site_path} not found.")
        return

    with open(site_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    harm_rows = []
    for harm_name, m in data.get("harmonization_comparison", {}).items():
        ci = m.get("auroc_ci_95", [None, None])
        harm_rows.append({
            "Harmonization Strategy": harm_name.replace("_", " ").title(),
            "Protocol": "Leave-One-Site-Out (LOSO)",
            "Pooled Accuracy": f"{m.get('pooled_accuracy', 0):.4f}",
            "Pooled AUROC": f"{m.get('pooled_auroc', 0):.4f}",
            "95% Bootstrap CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "N/A",
            "Methodological Flag": "LEAKY (Deliberate Wrong Arm)" if m.get("is_leaky") else "Valid In-Fold",
        })

    df_harm = pd.DataFrame(harm_rows)
    df_harm.to_csv(TABLES_DIR / "table_harmonization_comparison.csv", index=False)
    with open(TABLES_DIR / "table_harmonization_comparison.tex", "w", encoding="utf-8") as f:
        f.write(df_harm.to_latex(index=False, caption="ABIDE I Harmonization Protocol Benchmark", label="tab:harmonization"))
    print("[OK] Exported table_harmonization_comparison (.csv and .tex)")


def export_uncertainty_tables():
    unc_path = RESULTS_DIR / "uncertainty_main.json"
    if not unc_path.exists():
        print(f"[SKIP] {unc_path} not found.")
        return

    with open(unc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Calibration Metrics
    cal_rows = []
    cal = data.get("calibration", {})
    if "uncalibrated" in cal:
        u = cal["uncalibrated"]
        cal_rows.append({
            "Calibration Method": "Uncalibrated Baseline (Softmax)",
            "ECE (15 bins)": f"{u.get('ece_15bin', 0):.4f}",
            "Adaptive ECE": f"{u.get('ece_adaptive', 0):.4f}",
            "Brier Score": f"{u.get('brier', 0):.4f}",
            "NLL": f"{u.get('nll', 0):.4f}",
        })
    for m_name in ["temperature_scaling", "isotonic_regression", "platt_scaling"]:
        if m_name in cal:
            m = cal[m_name].get("metrics", {})
            cal_rows.append({
                "Calibration Method": m_name.replace("_", " ").title(),
                "ECE (15 bins)": f"{m.get('ece_15bin', 0):.4f}",
                "Adaptive ECE": f"{m.get('ece_adaptive', 0):.4f}",
                "Brier Score": f"{m.get('brier', 0):.4f}",
                "NLL": f"{m.get('nll', 0):.4f}",
            })

    df_cal = pd.DataFrame(cal_rows)
    df_cal.to_csv(TABLES_DIR / "table_calibration.csv", index=False)
    with open(TABLES_DIR / "table_calibration.tex", "w", encoding="utf-8") as f:
        f.write(df_cal.to_latex(index=False, caption="Probability Calibration Error Before and After Recalibration", label="tab:calibration"))
    print("[OK] Exported table_calibration (.csv and .tex)")

    # Conformal Decision Policy
    conf_rows = []
    for alpha in [0.05, 0.10]:
        for score in ["lac", "aps"]:
            k = f"conformal_{score}_alpha{int(alpha*100)}"
            if k in data:
                c = data[k]
                conf_rows.append({
                    "Conformal Score": score.upper(),
                    "Nominal Coverage": f"{(1 - alpha) * 100:.0f}%",
                    "Empirical Coverage": f"{c.get('marginal_coverage', 0) * 100:.2f}%",
                    "Mean Set Size": f"{c.get('mean_set_size', 0):.3f}",
                    "Empty Sets (%)": f"{c.get('fraction_empty', 0) * 100:.2f}%",
                    "Singleton Sets (%)": f"{c.get('fraction_singleton', 0) * 100:.2f}%",
                })

    df_conf = pd.DataFrame(conf_rows)
    df_conf.to_csv(TABLES_DIR / "table_conformal_policy.csv", index=False)
    with open(TABLES_DIR / "table_conformal_policy.tex", "w", encoding="utf-8") as f:
        f.write(df_conf.to_latex(index=False, caption="Split Conformal Prediction Coverage and Set Cardinality", label="tab:conformal"))
    print("[OK] Exported table_conformal_policy (.csv and .tex)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 70)
    print("  EXPORTING PUBLICATION TABLES (.tex and .csv)")
    print("=" * 70)
    export_tautology_tables()
    export_site_tables()
    export_uncertainty_tables()
    print("[OK] All tables exported to results/tables/")


if __name__ == "__main__":
    main()
