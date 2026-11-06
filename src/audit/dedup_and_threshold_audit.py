"""
Deduplication & Threshold Sensitivity Audit — src/audit/dedup_and_threshold_audit.py

Addresses Parts C.4 and C.5:
1. Exact row-level deduplication accounting:
   - Reasons for duplicate rows (identical questionnaire response patterns and demographics)
   - Exact duplicates count
   - Near-duplicates count (Hamming distance <= 2 on 10 binary items)
   - Cross-label conflicts (same responses, opposite diagnostic label)
2. Performance comparison WITH deduplication vs. WITHOUT deduplication
   (showing how duplicates inflate test set memorization)
3. Threshold sensitivity sweep (tau in {4, 5, 6, 7, 8, 9}):
   - Compares empirically recovered cutoffs (Adult tau=7, Ado tau=7, Child tau=7, Toddler tau=4)
   - Evaluates performance at the clinical screening cutoff from Allison et al. (2012) (tau=6)
   - Documents the exact discrepancy between published instructions and dataset labels.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ITEM_COLS = [f"A{i}" for i in range(1, 11)]
DEMO_COLS = ["age", "sex", "jaundice", "family_asd"]

RAW_FILES = {
    "adult": DATA_DIR / "uci_adult.csv",
    "adolescent": DATA_DIR / "uci_adolescent.csv",
    "child": DATA_DIR / "uci_child.csv",
    "toddler": DATA_DIR / "toddler_qchat10.csv",
}


def fast_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n1 = int(np.sum(y_true == 1))
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(y_prob)
    rank_sum = np.sum(ranks[y_true == 1])
    u = rank_sum - (n1 * (n1 + 1)) / 2.0
    return float(u / (n1 * n0))


def load_raw_cohort(cohort: str) -> pd.DataFrame:
    raw_path = RAW_FILES[cohort]
    df = pd.read_csv(raw_path)

    if cohort == "toddler":
        col_map = {
            "Sex": "sex", "Jaundice": "jaundice", "Family_mem_with_ASD": "family_asd",
            "Class/ASD Traits ": "target", "Class/ASD": "target", "Qchat-10-Score": "leakage_score"
        }
        df = df.rename(columns=col_map)
        for i in range(1, 11):
            if f"A{i}" in df.columns:
                df[f"A{i}"] = pd.to_numeric(df[f"A{i}"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        if "Age_Mons" in df.columns:
            df["age"] = df["Age_Mons"].astype(float) / 12.0
        else:
            df["age"] = 2.0
        df["sex"] = df["sex"].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ["m", "male", "1"] else 0)
        df["jaundice"] = df["jaundice"].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ["yes", "y", "1"] else 0)
        df["family_asd"] = df["family_asd"].astype(str).str.strip().str.lower().map(lambda x: 1 if x in ["yes", "y", "1"] else 0)
        df["target"] = df["target"].astype(str).str.strip().str.upper().map(lambda x: 1 if x in ["YES", "1", "TRUE"] else 0)
        if "leakage_score" not in df.columns:
            df["leakage_score"] = df[[f"A{i}" for i in range(1, 11)]].sum(axis=1)
        else:
            df["leakage_score"] = pd.to_numeric(df["leakage_score"], errors="coerce")
    else:
        col_map = {
            "A1_Score": "A1", "A2_Score": "A2", "A3_Score": "A3", "A4_Score": "A4", "A5_Score": "A5",
            "A6_Score": "A6", "A7_Score": "A7", "A8_Score": "A8", "A9_Score": "A9", "A10_Score": "A10",
            "gender": "sex", "jundice": "jaundice", "austim": "family_asd",
            "Class/ASD": "target", "Class/ASD Traits ": "target"
        }
        df = df.rename(columns=col_map)
        for i in range(1, 11):
            if f"A{i}" in df.columns:
                df[f"A{i}"] = pd.to_numeric(df[f"A{i}"], errors="coerce").fillna(0).astype(int).clip(0, 1)
        if "result" in df.columns:
            df["leakage_score"] = pd.to_numeric(df["result"], errors="coerce")
        else:
            df["leakage_score"] = df[[f"A{i}" for i in range(1, 11)]].sum(axis=1)
        df["target"] = df["target"].astype(str).str.strip().str.upper().map(lambda x: 1 if x in ["YES", "1", "TRUE"] else 0)
        
        # Demographics
        if "age" in df.columns:
            age_numeric = pd.to_numeric(df["age"], errors="coerce")
            default_age = 25.0 if cohort == "adult" else (14.0 if cohort == "adolescent" else 7.0)
            df["age"] = age_numeric.fillna(age_numeric.median() if age_numeric.notna().any() else default_age)
        if "sex" in df.columns:
            df["sex"] = df["sex"].astype(str).str.strip().str.lower().map({"m": 1, "male": 1, "f": 0, "female": 0}).fillna(0).astype(int)
        if "jaundice" in df.columns:
            df["jaundice"] = df["jaundice"].astype(str).str.strip().str.lower().map({"yes": 1, "1": 1, "no": 0, "0": 0}).fillna(0).astype(int)
        if "family_asd" in df.columns:
            df["family_asd"] = df["family_asd"].astype(str).str.strip().str.lower().map({"yes": 1, "1": 1, "no": 0, "0": 0}).fillna(0).astype(int)

    return df


def audit_deduplication(cohort: str) -> Dict:
    raw_df = load_raw_cohort(cohort)
    n_raw = len(raw_df)

    # Match columns used for deduplication
    match_cols = ITEM_COLS + [c for c in ["age", "sex", "target"] if c in raw_df.columns]

    # Exact duplicates
    exact_dup_mask = raw_df.duplicated(subset=match_cols, keep="first")
    n_exact_dups = int(exact_dup_mask.sum())
    clean_df = raw_df[~exact_dup_mask].copy().reset_index(drop=True)
    n_clean = len(clean_df)

    # Near duplicates on item vectors
    items_raw = raw_df[ITEM_COLS].values
    items_clean = clean_df[ITEM_COLS].values

    # Pairwise Hamming on clean data
    dists_clean = pdist(items_clean, metric="cityblock")
    near_clean_count = int(np.sum((dists_clean > 0) & (dists_clean <= 2)))
    exact_pattern_clean = int(np.sum(dists_clean == 0))

    # Cross-label conflicts in raw
    grouped_raw = raw_df.groupby(ITEM_COLS)["target"].nunique()
    cross_label_raw = int((grouped_raw > 1).sum())

    grouped_clean = clean_df.groupby(ITEM_COLS)["target"].nunique()
    cross_label_clean = int((grouped_clean > 1).sum())

    # Evaluate AUROC with vs without deduplication (Var A: items only)
    def eval_df(df_in: pd.DataFrame, seed: int = 42):
        X = df_in[ITEM_COLS].values.astype(float)
        y = df_in["target"].values.astype(int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs, trues = [], []
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=500, random_state=seed))])
        for tr_i, te_i in skf.split(X, y):
            pipe.fit(X[tr_i], y[tr_i])
            probs.extend(pipe.predict_proba(X[te_i])[:, 1].tolist())
            trues.extend(y[te_i].tolist())
        return fast_auc(np.array(trues), np.array(probs))

    # Also evaluate Demographics-only (Var C) with vs without deduplication
    def eval_demos(df_in: pd.DataFrame, seed: int = 42):
        d_cols = [c for c in DEMO_COLS if c in df_in.columns]
        X = df_in[d_cols].values.astype(float)
        y = df_in["target"].values.astype(int)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        probs, trues = [], []
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, max_iter=500, random_state=seed))])
        for tr_i, te_i in skf.split(X, y):
            pipe.fit(X[tr_i], y[tr_i])
            probs.extend(pipe.predict_proba(X[te_i])[:, 1].tolist())
            trues.extend(y[te_i].tolist())
        return fast_auc(np.array(trues), np.array(probs))

    auc_raw_items = eval_df(raw_df)
    auc_clean_items = eval_df(clean_df)
    auc_raw_demos = eval_demos(raw_df)
    auc_clean_demos = eval_demos(clean_df)

    return {
        "cohort": cohort,
        "n_raw": n_raw,
        "n_removed": n_exact_dups,
        "n_clean": n_clean,
        "exact_duplicates_purged": n_exact_dups,
        "pct_purged": round(n_exact_dups / n_raw * 100, 2),
        "clean_item_near_duplicate_pairs": near_clean_count,
        "clean_identical_item_patterns_retained": exact_pattern_clean,
        "cross_label_conflicts_raw": cross_label_raw,
        "cross_label_conflicts_clean": cross_label_clean,
        "auroc_items_raw": round(auc_raw_items, 4),
        "auroc_items_clean": round(auc_clean_items, 4),
        "auroc_demos_raw": round(auc_raw_demos, 4),
        "auroc_demos_clean": round(auc_clean_demos, 4),
    }


def audit_threshold_sweeps(cohort: str) -> Dict:
    """Sweep cutoffs tau in {4, 5, 6, 7, 8, 9} against true dataset label."""
    clean_df = load_raw_cohort(cohort)
    match_cols = ITEM_COLS + [c for c in ["age", "sex", "target"] if c in clean_df.columns]
    clean_df = clean_df.drop_duplicates(subset=match_cols).reset_index(drop=True)

    item_sum = clean_df[ITEM_COLS].sum(axis=1)
    y_true = clean_df["target"].values
    n = len(clean_df)

    sweep_results = {}
    best_tau = None
    best_acc = -1.0

    for tau in [4, 5, 6, 7, 8, 9]:
        pred = (item_sum >= tau).astype(int)
        acc = float(np.mean(pred == y_true))
        sens = float(np.sum((pred == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1)))
        spec = float(np.sum((pred == 0) & (y_true == 0)) / max(1, np.sum(y_true == 0)))
        discordant = int(np.sum(pred != y_true))

        sweep_results[f"tau_{tau}"] = {
            "threshold": tau,
            "accuracy": round(acc, 4),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "discordant_cases": discordant,
        }

        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    # Published cutoff from Allison et al. 2012 is tau=6 for AQ-10
    published_tau = 6 if cohort != "toddler" else 4
    mismatch_with_published = (best_tau != published_tau)

    return {
        "cohort": cohort,
        "n_eval": n,
        "recovered_cutoff": best_tau,
        "recovered_accuracy": round(best_acc, 4),
        "published_cutoff_allison2012": published_tau,
        "accuracy_at_published_cutoff": sweep_results[f"tau_{published_tau}"]["accuracy"],
        "cutoff_mismatch_flag": mismatch_with_published,
        "note": (
            f"Published AQ-10 cutoff is score >= {published_tau}. "
            f"Empirical dataset labels match score >= {best_tau} with {best_acc*100:.2f}% concordance."
        ),
        "threshold_sweep": sweep_results,
    }


def run_full_dedup_and_threshold_audit() -> Dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("  DEDUPLICATION & THRESHOLD SENSITIVITY AUDIT")
    print("=" * 70)

    output = {
        "deduplication_audit": {},
        "threshold_sweeps": {},
    }

    cohorts = ["adult", "adolescent", "child", "toddler"]

    for c in cohorts:
        print(f"\n--- Cohort: {c.upper()} ---")
        d_res = audit_deduplication(c)
        output["deduplication_audit"][c] = d_res
        print(f"  Deduplication: {d_res['n_raw']} -> {d_res['n_clean']} (purged {d_res['n_removed']} exact duplicates, {d_res['pct_purged']}%)")
        print(f"  AUROC items (Raw vs Clean): {d_res['auroc_items_raw']} vs {d_res['auroc_items_clean']}")
        print(f"  AUROC demos (Raw vs Clean): {d_res['auroc_demos_raw']} vs {d_res['auroc_demos_clean']}")

        t_res = audit_threshold_sweeps(c)
        output["threshold_sweeps"][c] = t_res
        print(f"  Threshold Sweep: Recovered tau={t_res['recovered_cutoff']} (Acc={t_res['recovered_accuracy']:.4f})")
        print(f"  Published tau={t_res['published_cutoff_allison2012']} (Acc={t_res['accuracy_at_published_cutoff']:.4f})")
        if t_res["cutoff_mismatch_flag"]:
            print(f"  [DISCREPANCY] Dataset cutoff (tau={t_res['recovered_cutoff']}) differs from published clinical instruction (tau={t_res['published_cutoff_allison2012']})!")

    out_path = RESULTS_DIR / "dedup_and_threshold_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Results saved to {out_path}")
    return output


if __name__ == "__main__":
    run_full_dedup_and_threshold_audit()
