"""
Model Monitoring & Covariate Drift Detection Module for ASD Screening.

Calculates Population Stability Index (PSI) and answer distribution drift
between new evaluation batches and the reference training cohort.
"""

import os
import json
import numpy as np
import pandas as pd


def compute_psi(expected, actual, num_bins=5, epsilon=1e-4):
    """
    Compute Population Stability Index (PSI) between reference and new distributions.
    expected: baseline training feature array
    actual: new evaluation batch feature array
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    
    # Handle binary items directly
    if len(np.unique(expected)) <= 2:
        bins = [-0.5, 0.5, 1.5]
    else:
        bins = np.percentile(expected, np.linspace(0, 100, num_bins + 1))
        bins[0] -= 1e-4
        bins[-1] += 1e-4
        bins = np.unique(bins)
        
    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual, bins=bins)
    
    exp_pct = (exp_counts / len(expected)) + epsilon
    act_pct = (act_counts / len(actual)) + epsilon
    
    # Normalize to sum to 1
    exp_pct /= exp_pct.sum()
    act_pct /= act_pct.sum()
    
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi_val)


def audit_batch_drift(batch_df, cohort_name='adult'):
    """
    Compare uploaded batch CSV against baseline training distribution.
    Returns per-item drift statistics, PSI values, and overall alert level.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ref_path = os.path.join(base_dir, "data", "processed", f"{cohort_name}_train.csv")
    
    if not os.path.exists(ref_path):
        return {"error": f"Reference dataset for cohort '{cohort_name}' not found."}
        
    ref_df = pd.read_csv(ref_path)
    item_cols = [f'A{i}' for i in range(1, 11)]
    
    drift_report = {
        "cohort": cohort_name,
        "reference_subjects": len(ref_df),
        "batch_subjects": len(batch_df),
        "item_drift": {},
        "overall_psi": 0.0,
        "drift_status": "STABLE",
        "action_required": "None. Batch distribution matches baseline training distribution."
    }
    
    psi_list = []
    
    for item in item_cols:
        if item in batch_df.columns:
            ref_vals = pd.to_numeric(ref_df[item], errors='coerce').fillna(0).values
            act_vals = pd.to_numeric(batch_df[item], errors='coerce').fillna(0).values
            
            psi = compute_psi(ref_vals, act_vals)
            psi_list.append(psi)
            
            ref_pos_rate = float(np.mean(ref_vals == 1))
            act_pos_rate = float(np.mean(act_vals == 1))
            
            status = "STABLE" if psi < 0.10 else ("MODERATE_DRIFT" if psi < 0.25 else "SIGNIFICANT_DRIFT")
            
            drift_report["item_drift"][item] = {
                "psi": round(psi, 4),
                "baseline_pos_rate": round(ref_pos_rate, 3),
                "batch_pos_rate": round(act_pos_rate, 3),
                "status": status
            }
            
    if psi_list:
        mean_psi = float(np.mean(psi_list))
        max_psi = float(np.max(psi_list))
        drift_report["overall_psi"] = round(mean_psi, 4)
        drift_report["max_item_psi"] = round(max_psi, 4)
        
        if max_psi >= 0.25 or mean_psi >= 0.15:
            drift_report["drift_status"] = "ALERT_SIGNIFICANT_DRIFT"
            drift_report["action_required"] = "High distribution shift detected. Clinicians should exercise caution as questionnaire response patterns deviate from the training cohort."
        elif max_psi >= 0.10:
            drift_report["drift_status"] = "WARNING_MODERATE_DRIFT"
            drift_report["action_required"] = "Slight response pattern shift observed. Monitor model confidence intervals closely."
            
    return drift_report
