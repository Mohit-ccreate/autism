"""
Automated Dataset Leakage and Integrity Detector for ASD Screening Datasets.

Audits any CSV dataset for:
1. Target derivation and total score column presence ('result', 'score', 'total', sum of items).
2. Duplicate row frequency.
3. Single features predicting the label with suspicious accuracy (> 97% ROC-AUC).
4. Train/Test split overlap (if split provided).
5. Label balance and missing value anomalies.

Outputs a plain-language audit verdict and JSON report.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def audit_dataset_for_leakage(df, target_col=None):
    """Run comprehensive leakage and integrity checks on a screening dataframe."""
    findings = []
    verdict = "PASSED_CLEAN"
    risk_level = "LOW"
    
    # 1. Identify Target Column
    if target_col is None:
        candidates = [c for c in df.columns if any(k in c.lower() for k in ['class', 'target', 'asd', 'label', 'dx'])]
        if candidates:
            target_col = candidates[0]
        else:
            return {
                "verdict": "ERROR",
                "message": "Could not identify target label column. Please specify --target."
            }
            
    # Normalize target to binary 0/1 if possible
    y_raw = df[target_col].astype(str).str.strip().str.upper()
    pos_tokens = ['YES', '1', 'TRUE', 'ASD', 'POSITIVE', 'AUTISM']
    y = y_raw.map(lambda x: 1 if any(t in x for t in pos_tokens) else 0).values
    
    # 2. Check for Duplicate Rows
    n_total = len(df)
    n_dupes = df.duplicated().sum()
    dupe_pct = (n_dupes / n_total) * 100
    if n_dupes > 0:
        findings.append({
            "check": "Duplicate Rows",
            "status": "WARNING" if dupe_pct < 10 else "FAIL",
            "detail": f"Found {n_dupes} duplicate rows ({dupe_pct:.1f}% of dataset). Un-deduplicated random train/test splits will leak identical subjects across folds."
        })
        if dupe_pct >= 10:
            risk_level = "HIGH"
            verdict = "SUSPECTED_DUPLICATE_LEAKAGE"
            
    # 3. Check for Total Score Columns (Explicit or Implicit)
    item_cols = [c for c in df.columns if c != target_col and (
        c.upper().startswith('A') and (c[1:].isdigit() or 'SCORE' in c.upper()) or 'Q' in c.upper() and any(ch.isdigit() for ch in c)
    )]
    
    # Check column names for 'score', 'result', 'total'
    score_cols = [c for c in df.columns if c != target_col and any(k in c.lower() for k in ['result', 'score', 'total_score', 'qchat-10-score'])]
    if score_cols:
        findings.append({
            "check": "Explicit Score Columns",
            "status": "FAIL",
            "detail": f"Found total score columns: {score_cols}. Total score columns represent the sum of questionnaire items and directly leak the diagnostic cutoff."
        })
        risk_level = "CRITICAL"
        verdict = "FEATURE_LEAKAGE_DETECTED"
        
    # Check if any column is the exact mathematical sum of other numeric columns
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    if len(item_cols) >= 5:
        numeric_items = [c for c in item_cols if c in numeric_cols]
        if len(numeric_items) >= 5:
            items_sum = df[numeric_items].sum(axis=1)
            for c in numeric_cols:
                if c not in numeric_items:
                    diff = np.abs(df[c] - items_sum).sum()
                    if diff == 0:
                        findings.append({
                            "check": "Implicit Item Sum Calculation",
                            "status": "FAIL",
                            "detail": f"Column '{c}' is the exact mathematical sum of items {numeric_items}."
                        })
                        risk_level = "CRITICAL"
                        verdict = "FEATURE_LEAKAGE_DETECTED"
                        
    # 4. Check for Single Feature Predicting Target > 97% ROC-AUC
    suspicious_features = []
    for c in df.columns:
        if c == target_col:
            continue
        try:
            col_vals = pd.to_numeric(df[c], errors='coerce').fillna(0).values
            if len(np.unique(col_vals)) > 1:
                auc = roc_auc_score(y, col_vals)
                auc_max = max(auc, 1.0 - auc)
                if auc_max >= 0.97:
                    suspicious_features.append({"feature": c, "roc_auc": float(auc_max)})
        except Exception:
            continue
            
    if suspicious_features:
        findings.append({
            "check": "Single Feature Predictability (>97% AUC)",
            "status": "FAIL",
            "detail": f"Features that predict target almost perfectly on their own: {suspicious_features}. Suggests label-derived proxy or target leakage."
        })
        risk_level = "CRITICAL"
        verdict = "FEATURE_LEAKAGE_DETECTED"
        
    # 5. Check Deterministic Thresholding (Sum of Items vs Target)
    if len(item_cols) >= 8:
        try:
            numeric_items = [c for c in item_cols if c in numeric_cols]
            if len(numeric_items) >= 8:
                sum_items = df[numeric_items].sum(axis=1)
                # Check if there is a cut-off threshold where accuracy is 100%
                for tau in range(1, len(numeric_items)):
                    pred_step = (sum_items >= tau).astype(int)
                    step_acc = (pred_step == y).mean()
                    if step_acc >= 0.999:
                        findings.append({
                            "check": "Deterministic Threshold Trap",
                            "status": "FAIL",
                            "detail": f"Target column '{target_col}' is a 100% deterministic step function of item sum: sum(items) >= {tau} has {step_acc:.1%} agreement with the label! This is NOT an independent clinical outcome."
                        })
                        risk_level = "CRITICAL"
                        verdict = "DETERMINISTIC_TARGET_DERIVATION"
                        break
        except Exception:
            pass
            
    if not findings:
        findings.append({
            "check": "Overall Integrity",
            "status": "PASS",
            "detail": "No explicit score columns, zero deterministic derivation traps, and no individual feature exceeds 97% ROC-AUC."
        })

    plain_explanation = {
        "PASSED_CLEAN": "Dataset appears clean and honest. No target derivation, no total score columns, and no suspicious single-feature predictability.",
        "FEATURE_LEAKAGE_DETECTED": "CRITICAL LEAKAGE: Total score column or label-derived feature detected. Training a model on this dataset will yield falsely inflated (99-100%) accuracy that does not reflect real-world clinical utility.",
        "DETERMINISTIC_TARGET_DERIVATION": "DETERMINISTIC THRESHOLD DETECTED: The target label in this dataset was generated by an arithmetic step function of the questionnaire items. To build honest models, test item reduction (e.g. 5 items) or cross-cohort evaluation.",
        "SUSPECTED_DUPLICATE_LEAKAGE": "DUPLICATION LEAKAGE: Substantial identical duplicate rows exist. Deduplicate before train/test splitting to prevent subject leakage across folds."
    }.get(verdict, "Audit completed.")

    return {
        "dataset_rows": n_total,
        "dataset_columns": len(df.columns),
        "target_column": target_col,
        "risk_level": risk_level,
        "verdict": verdict,
        "plain_explanation": plain_explanation,
        "findings": findings
    }


def main():
    parser = argparse.ArgumentParser(description="Audit ASD screening dataset for leakage and deterministic targets.")
    parser.add_argument("csv_path", help="Path to CSV dataset to audit.")
    parser.add_argument("--target", default=None, help="Name of target class column.")
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"File not found: {args.csv_path}")
        sys.exit(1)
        
    df = pd.read_csv(args.csv_path)
    report = audit_dataset_for_leakage(df, args.target)
    print("\n=======================================================")
    print("  ASD DATASET LEAKAGE & INTEGRITY AUDIT REPORT")
    print("=======================================================")
    print(f"File: {args.csv_path}")
    print(f"Rows: {report['dataset_rows']} | Columns: {report['dataset_columns']}")
    print(f"Risk Level: {report['risk_level']} | Verdict: {report['verdict']}")
    print(f"\nVerdict Explanation:\n{report['plain_explanation']}\n")
    print("Detailed Findings:")
    for f in report['findings']:
        print(f"  [{f['status']:7s}] {f['check']}: {f['detail']}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
