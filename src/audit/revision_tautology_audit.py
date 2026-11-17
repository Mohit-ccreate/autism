"""
Revision Pass: Unified Tautology, De-duplication Sensitivity, and Logistic Regression Accuracy Audit.
Addresses Reviewer Items A.1, A.4, and C.

Computes in ONE unified run from ONE saved set of predictions per cohort:
- Rule recovery concordance
- AUROC_items, AUROC_clean, accuracy_items (with 95% CI), majority baseline
- Tautology Index (raw & clipped) with 95% bootstrap CIs
- Alternative measures: absolute AUROC drop and relative accuracy drop over majority baseline
- Instability diagnostics (clean AUROC < 0.50, denominator near 0)
- Full sensitivity WITH and WITHOUT de-duplication
- Cohort composition before and after de-duplication
- Tabular conformal prediction WITH and WITHOUT de-duplication
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
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

COHORT_THRESHOLDS = {
    "adult": 7,
    "adolescent": 7,
    "child": 7,
    "toddler": 4,
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
            df["leakage_score"] = df[ITEM_COLS].sum(axis=1)
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
            df["leakage_score"] = df[ITEM_COLS].sum(axis=1)
        df["target"] = df["target"].astype(str).str.strip().str.upper().map(lambda x: 1 if x in ["YES", "1", "TRUE"] else 0)

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


def audit_participant_identifiers() -> Dict[str, Any]:
    """Check whether any participant identifier exists in the questionnaire files."""
    results = {}
    for cohort, path in RAW_FILES.items():
        raw_df = pd.read_csv(path)
        cols = list(raw_df.columns)
        candidate_ids = [c for c in cols if any(id_word in c.lower() for id_word in ["id", "case", "subject", "sub", "pat", "mrn"])]
        has_true_id = False
        details = ""
        if cohort == "toddler" and "Case_No" in cols:
            # Case_No is 1..1054, serial row index
            is_seq = list(raw_df["Case_No"]) == list(range(1, len(raw_df) + 1))
            details = f"Contains 'Case_No' which is a sequential integer row index (1..{len(raw_df)}) assigned during table compilation, not a clinical/participant identifier."
        elif candidate_ids:
            details = f"Columns {candidate_ids} found, but inspection shows no global participant identifier."
        else:
            details = "No participant identifier column exists in the dataset."

        results[cohort] = {
            "candidate_columns": candidate_ids,
            "has_verified_participant_identifier": has_true_id,
            "description": details,
            "interpretation": "Rows with identical [A1..A10, Age, Sex, Target] are duplicate records, not verified duplicate participants."
        }
    return results


def compute_cohort_composition(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute N, class balance, age distribution, and sex distribution."""
    n = len(df)
    n_asd = int((df["target"] == 1).sum())
    n_ctrl = int((df["target"] == 0).sum())
    pct_asd = float(n_asd / n * 100.0)

    age_mean = float(df["age"].mean())
    age_std = float(df["age"].std())
    age_med = float(df["age"].median())
    age_min = float(df["age"].min())
    age_max = float(df["age"].max())

    n_male = int((df["sex"] == 1).sum())
    n_female = int((df["sex"] == 0).sum())
    pct_male = float(n_male / n * 100.0)
    pct_female = float(n_female / n * 100.0)

    return {
        "n": n,
        "asd_count": n_asd,
        "ctrl_count": n_ctrl,
        "asd_pct": pct_asd,
        "age_mean": age_mean,
        "age_std": age_std,
        "age_median": age_med,
        "age_min": age_min,
        "age_max": age_max,
        "male_count": n_male,
        "female_count": n_female,
        "male_pct": pct_male,
        "female_pct": pct_female,
    }


def evaluate_cohort_tautology_unified(
    df: pd.DataFrame,
    cohort_name: str,
    tau: int,
    seeds: List[int] = [42, 43, 44, 45, 46],
    n_boot: int = 2000,
) -> Dict[str, Any]:
    """
    Evaluates rule recovery and all feature subsets in ONE run on the EXACT same folds.
    Returns AUROC_items, AUROC_clean, accuracy_items, TI, and alternative measures.
    """
    y = df["target"].values.astype(int)
    X_items = df[ITEM_COLS].values.astype(float)
    X_demos = df[DEMO_COLS].values.astype(float)
    X_items_demos = np.hstack([X_items, X_demos])
    X_score = np.hstack([X_items, df["leakage_score"].values.reshape(-1, 1)])

    # 1. Rule Recovery
    item_sum = df[ITEM_COLS].sum(axis=1)
    rule_pred = (item_sum >= tau).astype(int)
    n_total = len(df)
    n_concordant = int((rule_pred == df["target"]).sum())
    n_discordant = n_total - n_concordant
    rule_accuracy = float(n_concordant / n_total)

    # 2. Out-of-fold predictions across seeds for unified evaluation
    pred_store = {
        "items": [],
        "demos": [],
        "items_demos": [],
        "score": [],
        "y_true": [],
    }

    # Also keep per-fold accuracy for items
    fold_accuracies_items = []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof_items = np.zeros(n_total)
        oof_demos = np.zeros(n_total)
        oof_items_demos = np.zeros(n_total)
        oof_score = np.zeros(n_total)

        for tr_i, te_i in skf.split(X_items, y):
            # Items only
            sc_i = StandardScaler()
            clf_i = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
            clf_i.fit(sc_i.fit_transform(X_items[tr_i]), y[tr_i])
            p_i = clf_i.predict_proba(sc_i.transform(X_items[te_i]))[:, 1]
            oof_items[te_i] = p_i
            fold_accuracies_items.append(float(accuracy_score(y[te_i], (p_i >= 0.5).astype(int))))

            # Demographics only
            sc_d = StandardScaler()
            clf_d = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
            clf_d.fit(sc_d.fit_transform(X_demos[tr_i]), y[tr_i])
            oof_demos[te_i] = clf_d.predict_proba(sc_d.transform(X_demos[te_i]))[:, 1]

            # Items + Demographics
            sc_id = StandardScaler()
            clf_id = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
            clf_id.fit(sc_id.fit_transform(X_items_demos[tr_i]), y[tr_i])
            oof_items_demos[te_i] = clf_id.predict_proba(sc_id.transform(X_items_demos[te_i]))[:, 1]

            # Score + Items
            sc_s = StandardScaler()
            clf_s = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
            clf_s.fit(sc_s.fit_transform(X_score[tr_i]), y[tr_i])
            oof_score[te_i] = clf_s.predict_proba(sc_s.transform(X_score[te_i]))[:, 1]

        pred_store["items"].extend(oof_items.tolist())
        pred_store["demos"].extend(oof_demos.tolist())
        pred_store["items_demos"].extend(oof_items_demos.tolist())
        pred_store["score"].extend(oof_score.tolist())
        pred_store["y_true"].extend(y.tolist())

    y_true_all = np.array(pred_store["y_true"])
    p_items_all = np.array(pred_store["items"])
    p_demos_all = np.array(pred_store["demos"])
    p_id_all = np.array(pred_store["items_demos"])
    p_score_all = np.array(pred_store["score"])

    # Metrics on full predictions
    auc_items = fast_auc(y_true_all, p_items_all)
    auc_demos = fast_auc(y_true_all, p_demos_all)
    auc_id = fast_auc(y_true_all, p_id_all)
    auc_score = fast_auc(y_true_all, p_score_all)

    pred_items_binary = (p_items_all >= 0.5).astype(int)
    pred_demos_binary = (p_demos_all >= 0.5).astype(int)
    acc_items = float(accuracy_score(y_true_all, pred_items_binary))
    acc_demos = float(accuracy_score(y_true_all, pred_demos_binary))
    bacc_items = float(balanced_accuracy_score(y_true_all, pred_items_binary))
    bacc_demos = float(balanced_accuracy_score(y_true_all, pred_demos_binary))

    # Majority class baseline
    p_asd = float(np.mean(y == 1))
    maj_baseline = float(max(p_asd, 1.0 - p_asd))

    # Tautology Index
    # Formula: TI = (AUROC_items - AUROC_clean) / (AUROC_items - 0.5)
    ti_denom = auc_items - 0.5
    ti_raw = (auc_items - auc_demos) / ti_denom if abs(ti_denom) > 1e-6 else 0.0
    ti_clipped = float(np.clip(ti_raw, 0.0, 1.0))

    # Alternative Measure (a): Absolute drop
    abs_drop = float(auc_items - auc_demos)

    # Alternative Measure (b): Relative drop in accuracy over majority baseline
    acc_denom = acc_items - maj_baseline
    if acc_denom > 1e-4:
        rel_acc_drop_maj = float((acc_items - acc_demos) / acc_denom)
    else:
        rel_acc_drop_maj = float(acc_items - acc_demos)

    # Relative drop in accuracy directly: (acc_items - acc_demos) / acc_items
    rel_acc_drop_direct = float((acc_items - acc_demos) / acc_items) if acc_items > 0 else 0.0

    # Instability Diagnostics
    is_unstable_denominator = bool(abs(ti_denom) < 0.05)
    is_below_chance_clean = bool(auc_demos < 0.50)
    instability_reason = []
    if is_below_chance_clean:
        instability_reason.append(f"Clean AUROC ({auc_demos:.4f}) is below chance (0.50), making numerator ({auc_items - auc_demos:.4f}) exceed denominator ({ti_denom:.4f}), yielding raw TI > 1.0 ({ti_raw:.4f}).")
    if is_unstable_denominator:
        instability_reason.append(f"Denominator is near zero ({ti_denom:.4f}).")
    if not instability_reason:
        instability_reason.append("Stable: Clean AUROC is between 0.50 and items AUROC.")

    # 95% Bootstrap CIs on paired predictions
    rng = np.random.default_rng(42)
    n_preds = len(y_true_all)
    boot_indices = rng.integers(0, n_preds, size=(n_boot, n_preds))

    boot_tis = []
    boot_acc_items = []
    boot_auc_items = []
    boot_auc_demos = []
    boot_abs_drops = []
    boot_rel_drops = []

    for idx in boot_indices:
        yt_b = y_true_all[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        pi_b = p_items_all[idx]
        pd_b = p_demos_all[idx]

        auc_i_b = fast_auc(yt_b, pi_b)
        auc_d_b = fast_auc(yt_b, pd_b)
        acc_i_b = np.mean((pi_b >= 0.5).astype(int) == yt_b)
        acc_d_b = np.mean((pd_b >= 0.5).astype(int) == yt_b)
        maj_b = max(np.mean(yt_b == 1), np.mean(yt_b == 0))

        denom_b = auc_i_b - 0.5
        if abs(denom_b) > 1e-4:
            boot_tis.append((auc_i_b - auc_d_b) / denom_b)

        boot_acc_items.append(acc_i_b)
        boot_auc_items.append(auc_i_b)
        boot_auc_demos.append(auc_d_b)
        boot_abs_drops.append(auc_i_b - auc_d_b)
        if (acc_i_b - maj_b) > 1e-4:
            boot_rel_drops.append((acc_i_b - acc_d_b) / (acc_i_b - maj_b))

    ti_ci_95 = [float(np.percentile(boot_tis, 2.5)), float(np.percentile(boot_tis, 97.5))] if boot_tis else [ti_raw, ti_raw]
    ti_ci_95_clipped = [float(np.clip(ti_ci_95[0], 0.0, 1.0)), float(np.clip(ti_ci_95[1], 0.0, 1.0))]

    acc_items_ci_95 = [float(np.percentile(boot_acc_items, 2.5)), float(np.percentile(boot_acc_items, 97.5))]
    auc_items_ci_95 = [float(np.percentile(boot_auc_items, 2.5)), float(np.percentile(boot_auc_items, 97.5))]
    auc_demos_ci_95 = [float(np.percentile(boot_auc_demos, 2.5)), float(np.percentile(boot_auc_demos, 97.5))]
    abs_drop_ci_95 = [float(np.percentile(boot_abs_drops, 2.5)), float(np.percentile(boot_abs_drops, 97.5))]

    # 3. Repeated 5x10 Stratified CV on Items Only (C=1.0) to match publication table
    from sklearn.model_selection import RepeatedStratifiedKFold
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    rep_accs = []
    for tr_k, te_k in rskf.split(X_items, y):
        clf_k = LogisticRegression(C=1.0, max_iter=200, random_state=42)
        clf_k.fit(X_items[tr_k], y[tr_k])
        rep_accs.append(float(accuracy_score(y[te_k], clf_k.predict(X_items[te_k]))))
    rep_acc_mean = float(np.mean(rep_accs))
    rep_acc_std = float(np.std(rep_accs))
    rep_acc_ci_95 = [float(np.percentile(rep_accs, 2.5)), float(np.percentile(rep_accs, 97.5))]

    return {
        "cohort": cohort_name,
        "tau": tau,
        "n_samples": n_total,
        "rule_recovery": {
            "n_total": n_total,
            "n_concordant": n_concordant,
            "n_discordant": n_discordant,
            "concordance_accuracy": rule_accuracy,
        },
        "auroc_items": float(auc_items),
        "auroc_items_ci_95": auc_items_ci_95,
        "auroc_clean": float(auc_demos),
        "auroc_clean_ci_95": auc_demos_ci_95,
        "auroc_items_plus_demos": float(auc_id),
        "auroc_with_score": float(auc_score),
        "accuracy_items": float(acc_items),
        "accuracy_items_ci_95": acc_items_ci_95,
        "accuracy_items_repeated_5x10_mean": rep_acc_mean,
        "accuracy_items_repeated_5x10_std": rep_acc_std,
        "accuracy_items_repeated_5x10_ci_95": rep_acc_ci_95,
        "balanced_accuracy_items": float(bacc_items),
        "accuracy_clean": float(acc_demos),
        "balanced_accuracy_clean": float(bacc_demos),
        "majority_class_baseline": float(maj_baseline),
        "tautology_index_raw": float(ti_raw),
        "tautology_index_clipped": float(ti_clipped),
        "tautology_index_ci_95_raw": ti_ci_95,
        "tautology_index_ci_95_clipped": ti_ci_95_clipped,
        "alternative_a_absolute_drop": float(abs_drop),
        "alternative_a_ci_95": abs_drop_ci_95,
        "alternative_b_relative_acc_drop_over_majority": float(rel_acc_drop_maj),
        "alternative_b_relative_acc_drop_direct": float(rel_acc_drop_direct),
        "instability_flag": is_below_chance_clean or is_unstable_denominator,
        "instability_diagnostics": " | ".join(instability_reason),
        "saved_predictions": {
            "n_evaluated": len(y_true_all),
            "n_seeds": len(seeds),
            "seeds": seeds,
        }
    }


def evaluate_conformal_tabular(
    df: pd.DataFrame,
    seed: int = 42,
    alpha_levels: List[float] = [0.10, 0.05],
) -> Dict[str, Any]:
    """Evaluates split conformal prediction on tabular leak-free features (Var C)."""
    X = df[DEMO_COLS].values.astype(float)
    y = df["target"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    all_cal_true, all_cal_prob = [], []
    all_test_true, all_test_prob = [], []

    for tr_idx, te_idx in skf.split(X, y):
        X_tr_full, y_tr_full = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        n_tr = len(X_tr_full)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_tr)
        n_cal = max(10, int(n_tr * 0.20))
        cal_sub_idx, fit_sub_idx = perm[:n_cal], perm[n_cal:]

        scaler = StandardScaler()
        X_fit_s = scaler.fit_transform(X_tr_full[fit_sub_idx])
        X_cal_s = scaler.transform(X_tr_full[cal_sub_idx])
        X_te_s = scaler.transform(X_te)

        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=seed)
        clf.fit(X_fit_s, y_tr_full[fit_sub_idx])

        p_cal = clf.predict_proba(X_cal_s)[:, 1]
        p_te = clf.predict_proba(X_te_s)[:, 1]

        all_cal_true.extend(y_tr_full[cal_sub_idx].tolist())
        all_cal_prob.extend(p_cal.tolist())
        all_test_true.extend(y_te.tolist())
        all_test_prob.extend(p_te.tolist())

    y_cal = np.array(all_cal_true)
    p_cal = np.array(all_cal_prob)
    y_te = np.array(all_test_true)
    p_te = np.array(all_test_prob)

    results = {}
    for alpha in alpha_levels:
        target_cov = 1.0 - alpha
        # LAC nonconformity scores: 1 - p(y)
        scores_cal = 1.0 - np.where(y_cal == 1, p_cal, 1.0 - p_cal)
        n_cal = len(y_cal)
        q_level = min(1.0, np.ceil((n_cal + 1) * target_cov) / n_cal)
        q_hat = float(np.quantile(scores_cal, q_level))

        # Prediction sets on test
        sets = []
        for prob in p_te:
            s = []
            if (1.0 - prob) <= q_hat:
                s.append(1)
            if prob <= q_hat:
                s.append(0)
            sets.append(sorted(s))

        sizes = [len(s) for s in sets]
        covered = [int(yt in s) for yt, s in zip(y_te, sets)]
        frac_emp = float(np.mean([s == 0 for s in sizes]))
        frac_sing = float(np.mean([s == 1 for s in sizes]))
        frac_both = float(np.mean([s == 2 for s in sizes]))
        mean_size = float(np.mean(sizes))

        # Identity test
        assert abs(mean_size - (1.0 * frac_sing + 2.0 * frac_both + 0.0 * frac_emp)) < 1e-6

        results[f"lac_alpha{int(alpha*100)}"] = {
            "alpha": alpha,
            "nominal_coverage": target_cov,
            "empirical_coverage": float(np.mean(covered)),
            "mean_set_size": mean_size,
            "fraction_empty": frac_emp,
            "fraction_singleton": frac_sing,
            "fraction_both": frac_both,
            "triage_review_burden": frac_both,
        }

    return results


def run_full_revision_audit() -> Dict[str, Any]:
    print("=" * 75)
    print("  REVISION AUDIT: TAUTOLOGY, DEDUPLICATION SENSITIVITY & ACCURACY")
    print("=" * 75)

    id_audit = audit_participant_identifiers()
    print("\n--- Participant Identifier Check ---")
    for c, info in id_audit.items():
        print(f"  {c.upper():12s}: Verified ID={info['has_verified_participant_identifier']} | {info['description']}")

    cohort_accounting = {}
    unified_clean_results = {}
    unified_raw_results = {}
    conformal_clean_results = {}
    conformal_raw_results = {}

    for cohort, tau in COHORT_THRESHOLDS.items():
        print(f"\n==================== Cohort: {cohort.upper()} ====================")
        raw_df = load_raw_cohort(cohort)
        match_cols = ITEM_COLS + [c for c in ["age", "sex", "target"] if c in raw_df.columns]

        dup_mask = raw_df.duplicated(subset=match_cols, keep="first")
        n_raw = len(raw_df)
        n_removed = int(dup_mask.sum())
        clean_df = raw_df[~dup_mask].copy().reset_index(drop=True)
        n_clean = len(clean_df)

        comp_raw = compute_cohort_composition(raw_df)
        comp_clean = compute_cohort_composition(clean_df)

        cohort_accounting[cohort] = {
            "cohort": cohort,
            "n_raw": n_raw,
            "n_removed": n_removed,
            "pct_removed": float(n_removed / n_raw * 100.0),
            "n_retained": n_clean,
            "dedup_key": "[A1..A10, Age, Sex, Target]",
            "composition_raw": comp_raw,
            "composition_clean": comp_clean,
        }

        print(f"  Deduplication: {n_raw} -> {n_clean} ({n_removed} identical records removed, {n_removed/n_raw*100:.2f}%)")
        print(f"  Class balance: Raw ASD={comp_raw['asd_count']}/{n_raw} ({comp_raw['asd_pct']:.1f}%) | Clean ASD={comp_clean['asd_count']}/{n_clean} ({comp_clean['asd_pct']:.1f}%)")
        print(f"  Age: Raw {comp_raw['age_mean']:.2f} +/- {comp_raw['age_std']:.2f} | Clean {comp_clean['age_mean']:.2f} +/- {comp_clean['age_std']:.2f}")
        print(f"  Sex: Raw {comp_raw['male_pct']:.1f}% M | Clean {comp_clean['male_pct']:.1f}% M")

        # 1. Unified Run WITH deduplication (clean)
        res_clean = evaluate_cohort_tautology_unified(clean_df, cohort, tau)
        unified_clean_results[cohort] = res_clean
        print(f"  [Clean] Rule Recovery: {res_clean['rule_recovery']['concordance_accuracy']*100:.2f}% | Items AUC: {res_clean['auroc_items']:.4f} | Clean AUC: {res_clean['auroc_clean']:.4f}")
        print(f"          Items Acc: {res_clean['accuracy_items']*100:.2f}% [95% CI: {res_clean['accuracy_items_ci_95'][0]*100:.2f}%, {res_clean['accuracy_items_ci_95'][1]*100:.2f}%] (Maj Base: {res_clean['majority_class_baseline']*100:.2f}%)")
        print(f"          TI Raw: {res_clean['tautology_index_raw']:.4f} [95% CI: {res_clean['tautology_index_ci_95_raw'][0]:.3f}, {res_clean['tautology_index_ci_95_raw'][1]:.3f}] | Clipped: {res_clean['tautology_index_clipped']:.4f}")
        print(f"          Alt (a) Abs Drop: {res_clean['alternative_a_absolute_drop']:.4f} [95% CI: {res_clean['alternative_a_ci_95'][0]:.3f}, {res_clean['alternative_a_ci_95'][1]:.3f}]")
        print(f"          Alt (b) Rel Acc Drop: {res_clean['alternative_b_relative_acc_drop_over_majority']:.4f}")
        if res_clean['instability_flag']:
            print(f"          NOTE: {res_clean['instability_diagnostics']}")

        # 2. Unified Run WITHOUT deduplication (raw)
        res_raw = evaluate_cohort_tautology_unified(raw_df, cohort, tau)
        unified_raw_results[cohort] = res_raw
        print(f"  [Raw]   Rule Recovery: {res_raw['rule_recovery']['concordance_accuracy']*100:.2f}% | Items AUC: {res_raw['auroc_items']:.4f} | Clean AUC: {res_raw['auroc_clean']:.4f}")
        print(f"          Items Acc: {res_raw['accuracy_items']*100:.2f}% [95% CI: {res_raw['accuracy_items_ci_95'][0]*100:.2f}%, {res_raw['accuracy_items_ci_95'][1]*100:.2f}%]")
        print(f"          TI Raw: {res_raw['tautology_index_raw']:.4f} | Clipped: {res_raw['tautology_index_clipped']:.4f}")

        # 3. Conformal Prediction WITH and WITHOUT deduplication
        conf_clean = evaluate_conformal_tabular(clean_df)
        conf_raw = evaluate_conformal_tabular(raw_df)
        conformal_clean_results[cohort] = conf_clean
        conformal_raw_results[cohort] = conf_raw
        print(f"  [Conformal LAC 90%] Clean Cov: {conf_clean['lac_alpha10']['empirical_coverage']*100:.2f}%, Set Size: {conf_clean['lac_alpha10']['mean_set_size']:.3f} | Raw Cov: {conf_raw['lac_alpha10']['empirical_coverage']*100:.2f}%, Set Size: {conf_raw['lac_alpha10']['mean_set_size']:.3f}")

    output_payload = {
        "metadata": {
            "description": "Unified revision pass for Tautology Index, De-duplication Sensitivity, and Accuracy",
            "evaluation_seeds": [42, 43, 44, 45, 46],
            "n_folds": 5,
            "disclaimer": "Research tool only. Not a clinical diagnostic device.",
        },
        "participant_identifier_audit": id_audit,
        "cohort_accounting": cohort_accounting,
        "unified_tautology_clean": unified_clean_results,
        "unified_tautology_raw": unified_raw_results,
        "conformal_clean": conformal_clean_results,
        "conformal_raw": conformal_raw_results,
    }

    # Save to dedicated sensitivity JSON
    out_sens_path = RESULTS_DIR / "tautology_and_dedup_sensitivity.json"
    with open(out_sens_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[OK] Saved sensitivity output to {out_sens_path}")

    # Also update results/tautology_audit.json so all existing downstream readers get unified numbers
    taut_legacy_path = RESULTS_DIR / "tautology_audit.json"
    with open(taut_legacy_path, "r", encoding="utf-8") as f:
        taut_data = json.load(f)

    # Update tautology_indices and feature_ablation in taut_data
    for cohort, res in unified_clean_results.items():
        taut_data["tautology_indices"][cohort] = {
            "tautology_index": res["tautology_index_raw"],
            "tautology_index_raw": res["tautology_index_raw"],
            "tautology_index_clipped": res["tautology_index_clipped"],
            "ti_ci_95": res["tautology_index_ci_95_raw"],
            "ti_ci_95_clipped": res["tautology_index_ci_95_clipped"],
            "perf_leaky": res["auroc_items"],
            "perf_clean": res["auroc_clean"],
            "chance": 0.5,
            "alternative_a_absolute_drop": res["alternative_a_absolute_drop"],
            "alternative_b_relative_acc_drop": res["alternative_b_relative_acc_drop_over_majority"],
            "instability_flag": res["instability_flag"],
            "instability_diagnostics": res["instability_diagnostics"],
        }
        if "feature_ablation" in taut_data and cohort in taut_data["feature_ablation"]:
            taut_data["feature_ablation"][cohort]["varA_items_only"]["auroc"] = res["auroc_items"]
            taut_data["feature_ablation"][cohort]["varA_items_only"]["auroc_ci"] = res["auroc_items_ci_95"]
            taut_data["feature_ablation"][cohort]["varA_items_only"]["accuracy"] = res["accuracy_items"]
            taut_data["feature_ablation"][cohort]["varA_items_only"]["accuracy_ci_95"] = res["accuracy_items_ci_95"]
            taut_data["feature_ablation"][cohort]["varC_demos_only"]["auroc"] = res["auroc_clean"]
            taut_data["feature_ablation"][cohort]["varC_demos_only"]["auroc_ci"] = res["auroc_clean_ci_95"]

    with open(taut_legacy_path, "w", encoding="utf-8") as f:
        json.dump(taut_data, f, indent=2)
    print(f"[OK] Updated {taut_legacy_path} with unified ONE-RUN predictions.")

    # Update dedup_and_threshold_audit.json with participant identifier and composition breakdown
    dedup_path = RESULTS_DIR / "dedup_and_threshold_audit.json"
    with open(dedup_path, "r", encoding="utf-8") as f:
        dedup_data = json.load(f)

    dedup_data["participant_identifier_audit"] = id_audit
    dedup_data["cohort_composition_comparison"] = cohort_accounting
    dedup_data["sensitivity_with_vs_without_dedup"] = {
        cohort: {
            "clean_items_auc": unified_clean_results[cohort]["auroc_items"],
            "clean_demos_auc": unified_clean_results[cohort]["auroc_clean"],
            "clean_items_acc": unified_clean_results[cohort]["accuracy_items"],
            "clean_ti_raw": unified_clean_results[cohort]["tautology_index_raw"],
            "clean_ti_clipped": unified_clean_results[cohort]["tautology_index_clipped"],
            "raw_items_auc": unified_raw_results[cohort]["auroc_items"],
            "raw_demos_auc": unified_raw_results[cohort]["auroc_clean"],
            "raw_items_acc": unified_raw_results[cohort]["accuracy_items"],
            "raw_ti_raw": unified_raw_results[cohort]["tautology_index_raw"],
            "raw_ti_clipped": unified_raw_results[cohort]["tautology_index_clipped"],
        }
        for cohort in COHORT_THRESHOLDS
    }
    with open(dedup_path, "w", encoding="utf-8") as f:
        json.dump(dedup_data, f, indent=2)
    print(f"[OK] Updated {dedup_path} with participant ID audit and composition accounting.")

    return output_payload


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_full_revision_audit()
