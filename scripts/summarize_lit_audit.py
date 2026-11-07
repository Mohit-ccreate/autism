"""
Literature Meta-Audit Summary Script — scripts/summarize_lit_audit.py

Reads docs/lit_audit_template.csv, computes:
- Counts and proportions per field strictly over rows where the value is not "unclear"
- Explicit number of unclear rows per field
- Rigorous formatting of all proportions with denominators (e.g., "12 of 20 (60.0%)"), never bare percentages
- Stratified breakdowns: Overall, Tabular (AQ-10/Q-CHAT-10), and Neuroimaging (ABIDE)
- Exports results/lit_audit_summary.json
- Generates publication figure figures/lit_meta_audit_summary.png (and .pdf).

Usage:
    python scripts/summarize_lit_audit.py [--input docs/lit_audit_template.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize literature meta-audit CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DOCS_DIR / "lit_audit_template.csv",
        help="Path to filled lit_audit_template.csv",
    )
    return parser.parse_args()


def classify_family(row) -> str:
    ds = str(row.get("dataset", "")).upper()
    paper = str(row.get("paper", "")).upper()
    if "ABIDE" in ds or "ABIDE" in paper or "FMRI" in ds:
        return "Neuroimaging"
    return "Tabular"


def format_rate(k: int, n: int) -> str:
    if n == 0:
        return f"0 of 0 (N/A)"
    pct = (k / n) * 100.0
    return f"{k} of {n} ({pct:.1f}%)"


def analyze_binary_field(series: pd.Series, positive_val: str = "Y") -> dict:
    clean_series = series.astype(str).str.strip()
    is_unclear = clean_series.str.lower() == "unclear"
    unclear_count = int(is_unclear.sum())
    
    known_series = clean_series[~is_unclear]
    total_known = len(known_series)
    pos_count = int((known_series.str.upper() == positive_val.upper()).sum())
    neg_count = total_known - pos_count
    
    return {
        "positive_count": pos_count,
        "negative_count": neg_count,
        "total_evaluated": total_known,
        "unclear_count": unclear_count,
        "formatted_rate": format_rate(pos_count, total_known),
        "percentage": round((pos_count / total_known * 100.0), 1) if total_known > 0 else 0.0,
    }


def analyze_categorical_field(series: pd.Series) -> dict:
    clean_series = series.astype(str).str.strip()
    is_unclear = clean_series.str.lower() == "unclear"
    unclear_count = int(is_unclear.sum())
    
    known_series = clean_series[~is_unclear]
    total_known = len(known_series)
    
    counts = known_series.value_counts().to_dict()
    formatted = {}
    percentages = {}
    for cat, cnt in counts.items():
        formatted[cat] = format_rate(cnt, total_known)
        percentages[cat] = round((cnt / total_known * 100.0), 1) if total_known > 0 else 0.0
        
    return {
        "counts": counts,
        "total_evaluated": total_known,
        "unclear_count": unclear_count,
        "formatted": formatted,
        "percentages": percentages,
    }


def summarize_audit(csv_path: Path):
    if not csv_path.exists():
        print(f"[ERROR] CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"[ERROR] CSV {csv_path} is empty.")
        return

    df["family"] = df.apply(classify_family, axis=1)
    n_total = len(df)
    n_tabular = int((df["family"] == "Tabular").sum())
    n_neuro = int((df["family"] == "Neuroimaging").sum())

    ev_counts = df["evidence_level"].value_counts().to_dict() if "evidence_level" in df.columns else {}
    needs_check_count = int((df["needs_human_check"].astype(str).str.upper() == "Y").sum()) if "needs_human_check" in df.columns else 0

    print("=" * 78)
    print("        COMPREHENSIVE LITERATURE META-AUDIT STATISTICAL SUMMARY")
    print("=" * 78)
    print(f"Total Papers Audited: {n_total}")
    print(f"  - Behavioral / Questionnaire (Tabular): {n_tabular} of {n_total} ({(n_tabular/n_total)*100:.1f}%)")
    print(f"  - Resting-State Connectomes (Neuroimaging): {n_neuro} of {n_total} ({(n_neuro/n_total)*100:.1f}%)")
    print("\nEvidence Level Distribution:")
    for ev, cnt in ev_counts.items():
        print(f"  - {ev}: {format_rate(cnt, n_total)}")
    print(f"Papers Marked for Human Verification (needs_human_check = Y): {format_rate(needs_check_count, n_total)}")

    summary_dict = {
        "meta": {
            "total_papers": n_total,
            "tabular_papers": n_tabular,
            "neuroimaging_papers": n_neuro,
            "evidence_levels": {k: {"count": v, "formatted": format_rate(v, n_total)} for k, v in ev_counts.items()},
            "needs_human_check": {"count": needs_check_count, "formatted": format_rate(needs_check_count, n_total)},
        },
        "overall": {},
        "tabular": {},
        "neuroimaging": {},
    }

    eval_fields = [
        ("split_type", "categorical"),
        ("preprocessing_fit_in_fold", "binary"),
        ("leaking_features", "binary"),
        ("harmonization", "categorical"),
        ("calibration_reported", "binary"),
        ("uncertainty_reported", "binary"),
        ("external_validation", "binary"),
    ]

    print("\n" + "-" * 78)
    print("1. OVERALL COHORT RESULTS (Across All 33 Papers)")
    print("-" * 78)
    for field, ftype in eval_fields:
        if field not in df.columns:
            continue
        if ftype == "binary":
            res = analyze_binary_field(df[field])
            summary_dict["overall"][field] = res
            print(f"• {field.replace('_', ' ').title()}:")
            print(f"    Rate: {res['formatted_rate']} (Positive = {res['positive_count']}, Negative = {res['negative_count']})")
            print(f"    Unclear: {res['unclear_count']} papers")
        else:
            res = analyze_categorical_field(df[field])
            summary_dict["overall"][field] = res
            print(f"• {field.replace('_', ' ').title()}:")
            for cat, fmt in res["formatted"].items():
                print(f"    - {cat}: {fmt}")
            print(f"    Unclear: {res['unclear_count']} papers")

    print("\n" + "-" * 78)
    print("2. TABULAR (AQ-10 / Q-CHAT-10) SPECIFIC FINDINGS")
    print("-" * 78)
    df_tab = df[df["family"] == "Tabular"]
    for field, ftype in eval_fields:
        if field not in df_tab.columns:
            continue
        if ftype == "binary":
            res = analyze_binary_field(df_tab[field])
            summary_dict["tabular"][field] = res
            print(f"• {field.replace('_', ' ').title()}: {res['formatted_rate']} (unclear: {res['unclear_count']})")
        else:
            res = analyze_categorical_field(df_tab[field])
            summary_dict["tabular"][field] = res
            print(f"• {field.replace('_', ' ').title()}:")
            for cat, fmt in res["formatted"].items():
                print(f"    - {cat}: {fmt}")
            print(f"    Unclear: {res['unclear_count']}")

    print("\n" + "-" * 78)
    print("3. NEUROIMAGING (ABIDE) SPECIFIC FINDINGS")
    print("-" * 78)
    df_neuro = df[df["family"] == "Neuroimaging"]
    for field, ftype in eval_fields:
        if field not in df_neuro.columns:
            continue
        if ftype == "binary":
            res = analyze_binary_field(df_neuro[field])
            summary_dict["neuroimaging"][field] = res
            print(f"• {field.replace('_', ' ').title()}: {res['formatted_rate']} (unclear: {res['unclear_count']})")
        else:
            res = analyze_categorical_field(df_neuro[field])
            summary_dict["neuroimaging"][field] = res
            print(f"• {field.replace('_', ' ').title()}:")
            for cat, fmt in res["formatted"].items():
                print(f"    - {cat}: {fmt}")
            print(f"    Unclear: {res['unclear_count']}")

    # Export structured JSON
    out_json = RESULTS_DIR / "lit_audit_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"\n[OK] Structured summary saved to: {out_json}")

    # Generate Publication Figure with exact denominators
    generate_figure(df, summary_dict)


def generate_figure(df: pd.DataFrame, summary: dict):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    plt.subplots_adjust(hspace=0.38, wspace=0.28)

    palette = {
        "Tabular": "#D55E00",      # Vermillion
        "Neuroimaging": "#0072B2", # Deep Blue
        "Neutral": "#56B4E9",      # Sky Blue
        "Alert": "#CC79A7",        # Reddish Purple
        "Safe": "#009E73",         # Bluish Green
    }

    # ----------------------------------------------------
    # Panel A: Split Protocol by Dataset Family
    # ----------------------------------------------------
    ax = axes[0, 0]
    tab_splits = summary["tabular"]["split_type"]["counts"]
    neuro_splits = summary["neuroimaging"]["split_type"]["counts"]
    tab_tot = summary["tabular"]["split_type"]["total_evaluated"]
    neuro_tot = summary["neuroimaging"]["split_type"]["total_evaluated"]

    categories = ["random", "site-wise", "subject-wise"]
    x = np.arange(len(categories))
    width = 0.35

    tab_vals = [tab_splits.get(c, 0) for c in categories]
    neuro_vals = [neuro_splits.get(c, 0) for c in categories]

    b1 = ax.bar(x - width/2, tab_vals, width, label=f"Tabular (N={tab_tot})", color=palette["Tabular"], alpha=0.88)
    b2 = ax.bar(x + width/2, neuro_vals, width, label=f"ABIDE Neuroimaging (N={neuro_tot})", color=palette["Neuroimaging"], alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(["Random CV /\nHoldout Split", "Site-Wise /\nLOSO Split", "Subject-Wise\nLOO / Nested"], fontsize=9)
    ax.set_ylabel("Paper Count", fontsize=10)
    ax.set_title("A: Evaluation Split Protocols (Excluding Unclear)", fontsize=11, fontweight="bold")
    ax.legend(frameon=True, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, val, tot in zip(b1, tab_vals, [tab_tot]*len(tab_vals)):
        if val > 0:
            ax.annotate(f"{val} of {tot}\n({val/tot*100:.0f}%)",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="semibold")

    for bar, val, tot in zip(b2, neuro_vals, [neuro_tot]*len(neuro_vals)):
        if val > 0:
            ax.annotate(f"{val} of {tot}\n({val/tot*100:.0f}%)",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7.5, fontweight="semibold")

    # ----------------------------------------------------
    # Panel B: Leaking Features Prevalence
    # ----------------------------------------------------
    ax = axes[0, 1]
    tab_leak = summary["tabular"]["leaking_features"]
    neuro_leak = summary["neuroimaging"]["leaking_features"]
    overall_leak = summary["overall"]["leaking_features"]

    groups = ["Tabular\n(AQ-10/Q-CHAT)", "ABIDE\nNeuroimaging", "Overall\nCorpus"]
    pos_rates = [
        tab_leak["positive_count"],
        neuro_leak["positive_count"],
        overall_leak["positive_count"],
    ]
    totals = [
        tab_leak["total_evaluated"],
        neuro_leak["total_evaluated"],
        overall_leak["total_evaluated"],
    ]
    percentages = [p/t * 100 for p, t in zip(pos_rates, totals)]
    colors = [palette["Tabular"], palette["Neuroimaging"], "#7F7F7F"]

    bars = ax.bar(groups, percentages, color=colors, width=0.5, alpha=0.88)
    ax.set_ylabel("Prevalence (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("B: Presence of Label-Leaking Features", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, k, n, pct in zip(bars, pos_rates, totals, percentages):
        ax.annotate(f"{k} of {n}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # ----------------------------------------------------
    # Panel C: Methodological Rigor Safeguards
    # ----------------------------------------------------
    ax = axes[1, 0]
    preproc = summary["overall"]["preprocessing_fit_in_fold"]
    ext_val = summary["overall"]["external_validation"]

    measures = ["Preprocessing\nFit In-Fold", "External\nValidation"]
    pos_m = [preproc["positive_count"], ext_val["positive_count"]]
    tot_m = [preproc["total_evaluated"], ext_val["total_evaluated"]]
    pct_m = [k/n * 100 for k, n in zip(pos_m, tot_m)]

    bars = ax.bar(measures, pct_m, color=[palette["Safe"], palette["Neutral"]], width=0.45, alpha=0.88)
    ax.set_ylabel("Adherence Rate (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("C: Sound Preprocessing & External Validation", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, k, n, pct in zip(bars, pos_m, tot_m, pct_m):
        ax.annotate(f"{k} of {n}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # ----------------------------------------------------
    # Panel D: Calibration & Uncertainty Reporting Rates
    # ----------------------------------------------------
    ax = axes[1, 1]
    calib = summary["overall"]["calibration_reported"]
    uncert = summary["overall"]["uncertainty_reported"]

    items = ["Probabilistic\nCalibration Reported", "Uncertainty / Conformal\nPrediction Reported"]
    pos_u = [calib["positive_count"], uncert["positive_count"]]
    tot_u = [calib["total_evaluated"], uncert["total_evaluated"]]
    pct_u = [k/n * 100 for k, n in zip(pos_u, tot_u)]

    bars = ax.bar(items, pct_u, color=[palette["Alert"], "#E69F00"], width=0.45, alpha=0.88)
    ax.set_ylabel("Reporting Rate (%)", fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_title("D: Neglect of Uncertainty & Calibration", fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    for bar, k, n, pct in zip(bars, pos_u, tot_u, pct_u):
        ax.annotate(f"{k} of {n}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.suptitle("Literature Meta-Audit (N=33 Published Papers, 2013-2026)\nEmpirical Verification of Methodological Vulnerabilities and Neglected Safeguards",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_png = FIGURES_DIR / "lit_meta_audit_summary.png"
    out_pdf = FIGURES_DIR / "lit_meta_audit_summary.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Summary figure saved to: {out_png} and {out_pdf}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    summarize_audit(args.input)
