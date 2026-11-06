"""
Literature Meta-Audit Summary Script — scripts/summarize_lit_audit.py

Reads a completed docs/lit_audit_template.csv, computes:
- Counts and proportions for split protocols (random vs site-wise)
- Leakage risk prevalence (leaking features, preprocessing outside folds, harmonization on full data)
- Reporting rates for calibration, uncertainty quantification, and external validation
- Generates summary statistics and a summary visualization (figures/lit_meta_audit_summary.png).

Usage:
    python scripts/summarize_lit_audit.py [--input docs/lit_audit_template.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize literature meta-audit CSV.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DOCS_DIR / "lit_audit_template.csv",
        help="Path to the filled lit_audit_template.csv",
    )
    return parser.parse_args()


def summarize_audit(csv_path: Path):
    if not csv_path.exists():
        print(f"[ERROR] Template CSV not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Check if empty (only header or whitespace)
    if len(df) == 0:
        print(f"[INFO] '{csv_path}' has 0 populated rows. The template is ready for human author input.")
        print("Required schema columns:")
        for col in df.columns:
            print(f"  - {col}")
        return

    print("=" * 70)
    print("  LITERATURE META-AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total papers audited: {len(df)}")

    # Split type breakdown
    if "split_type" in df.columns:
        print("\nSplit Protocol Distribution:")
        print(df["split_type"].value_counts(dropna=False))

    # Preprocessing in-fold
    if "preprocessing_fit_in_fold" in df.columns:
        print("\nPreprocessing Fit Inside Folds (Y/N/Unclear):")
        print(df["preprocessing_fit_in_fold"].value_counts(dropna=False))

    # Leaking features
    if "leaking_features" in df.columns:
        print("\nPresence of Leaking Features (Y/N):")
        print(df["leaking_features"].value_counts(dropna=False))

    # Calibration & Uncertainty
    if "calibration_reported" in df.columns:
        print("\nCalibration Reported (Y/N):")
        print(df["calibration_reported"].value_counts(dropna=False))

    if "uncertainty_reported" in df.columns:
        print("\nUncertainty Quantification Reported (Y/N):")
        print(df["uncertainty_reported"].value_counts(dropna=False))

    # Generate Summary Figure if rows >= 3
    if len(df) >= 3:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 1. Split Protocol
        if "split_type" in df.columns:
            df["split_type"].value_counts().plot(kind="bar", ax=axes[0], color="#0072B2")
            axes[0].set_title("Split Protocol")
            axes[0].set_ylabel("Paper Count")
        
        # 2. Methodological Rigor Checks
        rigor_cols = ["preprocessing_fit_in_fold", "leaking_features", "external_validation"]
        present_cols = [c for c in rigor_cols if c in df.columns]
        if present_cols:
            y_counts = [int((df[c].astype(str).str.upper() == "Y").sum()) for c in present_cols]
            axes[1].bar(present_cols, y_counts, color="#009E73")
            axes[1].set_title("Key Safeguards Present (Y)")
            axes[1].set_xticklabels([c.replace("_", "\n") for c in present_cols], rotation=15, fontsize=8)

        # 3. Calibration & Uncertainty
        reporting_cols = ["calibration_reported", "uncertainty_reported"]
        present_rep = [c for c in reporting_cols if c in df.columns]
        if present_rep:
            rep_counts = [int((df[c].astype(str).str.upper() == "Y").sum()) for c in present_rep]
            axes[2].bar(present_rep, rep_counts, color="#E69F00")
            axes[2].set_title("Uncertainty Reporting (Y)")
            axes[2].set_xticklabels([c.replace("_", "\n") for c in present_rep], rotation=15, fontsize=8)

        plt.suptitle("Literature Meta-Audit: Methodological Safeguard Prevalence", fontsize=11, y=1.03)
        plt.tight_layout()
        out_fig = FIGURES_DIR / "lit_meta_audit_summary.png"
        fig.savefig(out_fig, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[OK] Summary visualization saved to {out_fig}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    summarize_audit(args.input)
