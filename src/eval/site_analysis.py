"""
Site Analysis Module — Module 3 Extension.

Comprehensive Multi-Center Generalization and Scanner Confound Analysis on Full ABIDE I:
1. Full harmonization comparison across all 20 clinical sites:
   - None (Raw connectome)
   - Site regression deconfounding (in-fold only)
   - Parametric ComBat (correctly fitted strictly inside training folds)
   - Deliberately LEAKY ComBat (fitted on all pooled data before CV) — labeled clearly
2. Random 5-Fold Stratified CV vs Leave-One-Site-Out (LOSO) CV across 5 random seeds
3. Statistical honesty & rigorous inference:
   - Subject-level bootstrap 95% CIs
   - Site-level cluster bootstrap 95% CIs
   - Paired bootstrap differences with Holm-Bonferroni correction between arms
   - Reconciled AUROC CI, within-site label permutation test, and exact binomial test
4. Multi-class site decodability probe (connectome vs majority-site chance level: NYU 19.91%)
5. Confound-only models (motion, age, sex, FIQ) under LOSO
6. Full per-site generalization gap table across all 20 sites
7. Robustness arm on non-QC cohort (N=989, 20 sites)

MEDICAL DISCLAIMER: Research prototype only. Not a diagnostic tool. No clinical claims.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Import ComBat from existing train_abide module
sys.path.insert(0, str(BASE_DIR / "src"))
try:
    from train_abide import parametric_combat_fit, parametric_combat_transform
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False
    print("WARNING: Could not import ComBat from train_abide.py")


# ---------------------------------------------------------------------------
# Statistical & Bootstrap Helpers
# ---------------------------------------------------------------------------

def fast_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute AUROC via Mann-Whitney U for speed."""
    n1 = int(np.sum(y_true == 1))
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(y_prob)
    rank_sum = np.sum(ranks[y_true == 1])
    u = rank_sum - (n1 * (n1 + 1)) / 2.0
    return float(u / (n1 * n0))


def bootstrap_auroc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> Tuple[float, float]:
    """Subject-level bootstrap CI for AUROC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = rng.integers(0, n, size=(n_boot, n))
    aurocs = []
    for idx in indices:
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(fast_auc(yt, yp))
    if not aurocs:
        return 0.5, 0.5
    aurocs = np.array(aurocs)
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def cluster_bootstrap_auroc(
    y_true: np.ndarray, y_prob: np.ndarray, clusters: np.ndarray,
    n_boot: int = 2000, seed: int = 42
) -> Tuple[float, float]:
    """Site-level cluster bootstrap CI for AUROC (resamples sites with replacement)."""
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    aurocs = []
    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = []
        for c in sampled_clusters:
            idx.extend(np.where(clusters == c)[0].tolist())
        idx = np.array(idx)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        aurocs.append(fast_auc(yt, yp))
    if not aurocs:
        return 0.5, 0.5
    aurocs = np.array(aurocs)
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


def within_site_permutation_test(
    y_true: np.ndarray, y_prob: np.ndarray, sites: np.ndarray,
    n_perm: int = 1000, seed: int = 42
) -> Tuple[float, float]:
    """
    Permutation test permuting diagnosis labels ONLY within each site.
    Preserves site sample sizes and within-site class marginals under H0.
    """
    rng = np.random.default_rng(seed)
    obs_auc = fast_auc(y_true, y_prob)
    unique_sites = np.unique(sites)
    perm_aucs = []

    for _ in range(n_perm):
        y_perm = np.zeros_like(y_true)
        for s in unique_sites:
            mask = (sites == s)
            y_perm[mask] = rng.permutation(y_true[mask])
        perm_aucs.append(fast_auc(y_perm, y_prob))

    perm_aucs = np.array(perm_aucs)
    pval = float(np.mean(perm_aucs >= obs_auc))
    return obs_auc, pval


def paired_bootstrap_comparison(
    y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray,
    n_boot: int = 2000, seed: int = 42
) -> Dict[str, Any]:
    """
    Paired bootstrap test comparing model A vs model B.
    Delta = AUROC_A - AUROC_B.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    obs_a = fast_auc(y_true, prob_a)
    obs_b = fast_auc(y_true, prob_b)
    obs_delta = obs_a - obs_b

    indices = rng.integers(0, n, size=(n_boot, n))
    deltas = []
    for idx in indices:
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_a = fast_auc(yt, prob_a[idx])
        auc_b = fast_auc(yt, prob_b[idx])
        deltas.append(auc_a - auc_b)

    deltas = np.array(deltas)
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    # Two-sided empirical p-value for H0: delta == 0
    p_pos = np.mean(deltas >= 0)
    p_neg = np.mean(deltas <= 0)
    pval = float(2 * min(p_pos, p_neg))
    pval = min(1.0, max(1.0 / n_boot, pval))

    excludes_zero = (ci_lo > 0.0) or (ci_hi < 0.0)
    return {
        "observed_delta": float(obs_delta),
        "ci_95": [ci_lo, ci_hi],
        "p_value": pval,
        "ci_excludes_zero": bool(excludes_zero),
    }


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]
    adj_p = np.zeros(m)
    cum_max = 0.0
    for i, p in enumerate(sorted_p):
        step_val = (m - i) * p
        cum_max = max(cum_max, step_val)
        adj_p[i] = min(1.0, cum_max)
    # Restore original order
    rev_indices = np.empty_like(sorted_indices)
    rev_indices[sorted_indices] = np.arange(m)
    return adj_p[rev_indices].tolist()


# ---------------------------------------------------------------------------
# Harmonization Methods
# ---------------------------------------------------------------------------

def site_regression_deconfound(
    X_train: np.ndarray, sites_train: np.ndarray,
    X_test: np.ndarray, sites_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove site mean effects by regression (without diagnosis label).
    Fit strictly on training data only; apply to test.
    """
    unique_sites = np.unique(sites_train)
    grand_mean = np.mean(X_train, axis=0)

    site_means = {}
    for s in unique_sites:
        mask = sites_train == s
        site_means[s] = np.mean(X_train[mask], axis=0)

    X_train_deconf = X_train.copy()
    for s in unique_sites:
        mask = sites_train == s
        X_train_deconf[mask] = X_train[mask] - site_means[s] + grand_mean

    X_test_deconf = X_test.copy()
    for s in np.unique(sites_test):
        mask = sites_test == s
        if s in site_means:
            X_test_deconf[mask] = X_test[mask] - site_means[s] + grand_mean

    return X_train_deconf, X_test_deconf


def run_loso_fold(
    X_train: np.ndarray, y_train: np.ndarray, sites_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, sites_test: np.ndarray,
    harmonization: str = "combat",
    k_features: int = 250,
    seed: int = 42,
    X_all_leaky: Optional[np.ndarray] = None,
    train_idx: Optional[np.ndarray] = None,
    test_idx: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Run one LOSO fold with the specified harmonization strategy."""
    if harmonization == "none":
        X_tr = X_train.copy()
        X_te = X_test.copy()

    elif harmonization == "site_regression":
        X_tr, X_te = site_regression_deconfound(X_train, sites_train, X_test, sites_test)

    elif harmonization == "combat" and COMBAT_AVAILABLE:
        X_tr, combat_model = parametric_combat_fit(X_train, sites_train, None)
        X_te = parametric_combat_transform(X_test, sites_test, None, combat_model)

    elif harmonization == "leaky_combat":
        # Pre-fitted on all data (train+test combined) before CV split
        if X_all_leaky is not None and train_idx is not None and test_idx is not None:
            X_tr = X_all_leaky[train_idx].copy()
            X_te = X_all_leaky[test_idx].copy()
        else:
            X_all = np.vstack([X_train, X_test])
            sites_all = np.concatenate([sites_train, sites_test])
            X_all_harm, _ = parametric_combat_fit(X_all, sites_all, None)
            X_tr = X_all_harm[:len(X_train)]
            X_te = X_all_harm[len(X_train):]
    else:
        X_tr = X_train.copy()
        X_te = X_test.copy()

    # Feature selection inside fold
    if 0 < k_features < X_tr.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_tr = selector.fit_transform(X_tr, y_train)
        X_te = selector.transform(X_te)

    # Scale inside fold
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # L2 regularized Logistic Regression
    clf = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)
    clf.fit(X_tr, y_train)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)

    n_test = len(y_test)
    if len(np.unique(y_test)) < 2:
        auc = float("nan")
        auprc = float("nan")
    else:
        auc = roc_auc_score(y_test, prob)
        auprc = average_precision_score(y_test, prob)

    return {
        "y_true": y_test.tolist(),
        "y_prob": prob.tolist(),
        "y_pred": pred.tolist(),
        "n_test": n_test,
        "accuracy": float(accuracy_score(y_test, pred)),
        "bacc": float(balanced_accuracy_score(y_test, pred)),
        "auroc": float(auc) if not np.isnan(auc) else None,
        "auprc": float(auprc) if not np.isnan(auprc) else None,
        "sensitivity": float(recall_score(y_test, pred, zero_division=0)),
        "specificity": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)) if len(np.unique(pred)) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Harmonization Comparison Runner
# ---------------------------------------------------------------------------

def run_harmonization_comparison(
    X: np.ndarray, meta_df: pd.DataFrame, k_features: int = 250, seed: int = 42
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Compare harmonization strategies under LOSO across all clinical sites."""
    print("\n--- Harmonization Comparison (Leave-One-Site-Out across all sites) ---")
    harmonizations = ["none", "site_regression", "combat"]
    if COMBAT_AVAILABLE:
        harmonizations.append("leaky_combat")

    sites = meta_df["site"].values
    y = meta_df["target"].values
    unique_sites = np.unique(sites)

    # Pre-compute leaky ComBat on all data if evaluated
    X_leaky = None
    if COMBAT_AVAILABLE and "leaky_combat" in harmonizations:
        print("  Pre-computing Leaky ComBat on entire pooled dataset (deliberately wrong arm)...")
        X_leaky, _ = parametric_combat_fit(X, sites, None)

    results = {}
    predictions_map: Dict[str, np.ndarray] = {}

    for harm in harmonizations:
        label = harm.upper()
        if harm == "leaky_combat":
            label = "LEAKY_COMBAT (deliberately wrong — fit before CV to measure inflation)"
        print(f"  Evaluating Arm: {label}")

        all_true: List[int] = []
        all_prob: List[float] = []
        all_pred: List[int] = []
        all_site_order: List[str] = []
        site_metrics: Dict[str, Any] = {}

        for held_site in unique_sites:
            test_mask = (sites == held_site)
            train_mask = ~test_mask
            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            X_tr, X_te = X[train_mask], X[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            s_tr, s_te = sites[train_mask], sites[test_mask]

            fold_res = run_loso_fold(
                X_tr, y_tr, s_tr, X_te, y_te, s_te,
                harmonization=harm, k_features=k_features, seed=seed,
                X_all_leaky=X_leaky, train_idx=train_idx, test_idx=test_idx
            )
            site_metrics[held_site] = {
                "n_test": fold_res["n_test"],
                "accuracy": fold_res["accuracy"],
                "bacc": fold_res["bacc"],
                "auroc": fold_res["auroc"],
                "sensitivity": fold_res["sensitivity"],
                "specificity": fold_res["specificity"],
            }
            all_true.extend(fold_res["y_true"])
            all_prob.extend(fold_res["y_prob"])
            all_pred.extend(fold_res["y_pred"])
            all_site_order.extend([held_site] * len(fold_res["y_true"]))

        all_true_arr = np.array(all_true)
        all_prob_arr = np.array(all_prob)
        all_pred_arr = np.array(all_pred)
        all_sites_arr = np.array(all_site_order)

        predictions_map[harm] = all_prob_arr

        pooled_auroc = roc_auc_score(all_true_arr, all_prob_arr)
        subj_lo, subj_hi = bootstrap_auroc_ci(all_true_arr, all_prob_arr, n_boot=2000, seed=seed)
        clust_lo, clust_hi = cluster_bootstrap_auroc(all_true_arr, all_prob_arr, all_sites_arr, n_boot=2000, seed=seed)
        obs_auc, perm_p = within_site_permutation_test(all_true_arr, all_prob_arr, all_sites_arr, n_perm=1000, seed=seed)

        # Exact binomial test for pooled accuracy vs 0.5
        n_correct = int(np.sum(all_pred_arr == all_true_arr))
        n_total = len(all_true_arr)
        binom_p = float(stats.binomtest(n_correct, n_total, 0.5, alternative="two-sided").pvalue)

        results[harm] = {
            "harmonization": harm,
            "is_leaky": (harm == "leaky_combat"),
            "n_total": n_total,
            "pooled_accuracy": float(accuracy_score(all_true_arr, all_pred_arr)),
            "pooled_bacc": float(balanced_accuracy_score(all_true_arr, all_pred_arr)),
            "pooled_auroc": float(pooled_auroc),
            "subject_bootstrap_ci_95": [subj_lo, subj_hi],
            "site_cluster_bootstrap_ci_95": [clust_lo, clust_hi],
            "within_site_permutation_p": float(perm_p),
            "binomial_p_vs_chance": float(binom_p),
            "pooled_sensitivity": float(recall_score(all_true_arr, all_pred_arr, zero_division=0)),
            "pooled_specificity": float(recall_score(all_true_arr, all_pred_arr, pos_label=0, zero_division=0)),
            "pooled_f1": float(f1_score(all_true_arr, all_pred_arr, zero_division=0)),
            "pooled_mcc": float(matthews_corrcoef(all_true_arr, all_pred_arr)),
            "site_metrics": site_metrics,
        }
        print(f"    Pooled AUROC: {pooled_auroc:.4f} [Subj 95% CI: {subj_lo:.4f}, {subj_hi:.4f}; Cluster 95% CI: {clust_lo:.4f}, {clust_hi:.4f}]")
        print(f"    Within-site Permutation p: {perm_p:.4f} | Binomial p: {binom_p:.4f}")

    return results, predictions_map


# ---------------------------------------------------------------------------
# Paired Bootstrap Tests Between Harmonization Arms
# ---------------------------------------------------------------------------

def run_paired_harmonization_tests(
    y_true: np.ndarray, predictions_map: Dict[str, np.ndarray], seed: int = 42
) -> Dict[str, Any]:
    """
    Run paired bootstrap comparisons between harmonization arms.
    Applies Holm-Bonferroni correction.
    """
    print("\n--- Paired Bootstrap Tests Across Harmonization Arms ---")
    comparisons = [
        ("leaky_combat", "combat", "Leaky ComBat vs In-Fold ComBat"),
        ("combat", "none", "In-Fold ComBat vs No Harmonization"),
        ("combat", "site_regression", "In-Fold ComBat vs Site Regression"),
        ("site_regression", "none", "Site Regression vs No Harmonization"),
    ]

    p_raw = []
    contrast_results = []

    for name_a, name_b, label in comparisons:
        if name_a not in predictions_map or name_b not in predictions_map:
            continue
        prob_a = predictions_map[name_a]
        prob_b = predictions_map[name_b]

        res = paired_bootstrap_comparison(y_true, prob_a, prob_b, n_boot=2000, seed=seed)
        res["contrast"] = label
        res["arm_a"] = name_a
        res["arm_b"] = name_b
        p_raw.append(res["p_value"])
        contrast_results.append(res)

    p_adj = holm_bonferroni_correction(p_raw)
    for res, padj in zip(contrast_results, p_adj):
        res["p_value_holm"] = float(padj)
        ci_lo, ci_hi = res["ci_95"]
        if res["arm_a"] == "leaky_combat" and res["arm_b"] == "combat":
            if res["ci_excludes_zero"]:
                res["conclusion"] = f"Statistically significant leakage inflation detected: Delta={res['observed_delta']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], p_holm={padj:.4f}"
            else:
                res["conclusion"] = f"No detectable difference at this sample size: Delta={res['observed_delta']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], p_holm={padj:.4f} (CI includes 0)"
        else:
            sig_str = "significant" if padj < 0.05 else "not significant"
            res["conclusion"] = f"Delta={res['observed_delta']:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], p_holm={padj:.4f} ({sig_str})"
        print(f"  {res['contrast']}: {res['conclusion']}")

    return {"contrasts": contrast_results}


# ---------------------------------------------------------------------------
# Random 5-Fold Stratified CV Across 5 Seeds
# ---------------------------------------------------------------------------

def run_random_cv_benchmark(
    X: np.ndarray, y: np.ndarray, sites: np.ndarray,
    k_features: int = 250, seeds: List[int] = [42, 43, 44, 45, 46]
) -> Dict[str, Any]:
    """Run 5-Fold Stratified Random CV across multiple seeds to quantify site leakage."""
    print("\n--- Random 5-Fold Stratified CV (Inflated / Site-Leaking Arm) ---")
    aurocs, accs, baccs = [], [], []

    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_probs = np.zeros(len(y))
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            # In-fold feature selection & scaling
            sel = SelectKBest(score_func=f_classif, k=min(k_features, X_tr.shape[1]))
            X_tr_sel = sel.fit_transform(X_tr, y_tr)
            X_te_sel = sel.transform(X_te)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_sel)
            X_te_s = scaler.transform(X_te_sel)

            clf = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)
            clf.fit(X_tr_s, y_tr)
            fold_probs[test_idx] = clf.predict_proba(X_te_s)[:, 1]

        auc = roc_auc_score(y, fold_probs)
        acc = accuracy_score(y, (fold_probs >= 0.5).astype(int))
        bacc = balanced_accuracy_score(y, (fold_probs >= 0.5).astype(int))
        aurocs.append(auc)
        accs.append(acc)
        baccs.append(bacc)

    return {
        "seeds": seeds,
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_bacc": float(np.mean(baccs)),
        "std_bacc": float(np.std(baccs)),
        "seed_aurocs": [float(x) for x in aurocs],
    }


# ---------------------------------------------------------------------------
# Site Decodability Probe
# ---------------------------------------------------------------------------

def run_site_probe(X: np.ndarray, meta_df: pd.DataFrame, k_features: int = 100, seed: int = 42) -> Dict[str, Any]:
    """
    Train classifier to predict SITE_ID from connectome and confounds.
    Chance level = true majority-site base rate (NYU = 170 / 854 = 19.91%).
    """
    print("\n--- Site Decodability Probe: Predicting SITE_ID ---")
    sites = meta_df["site"].values
    le = LabelEncoder()
    sites_encoded = le.fit_transform(sites)
    n_classes = len(le.classes_)

    # Majority site base rate
    counts = pd.Series(sites).value_counts()
    majority_site = counts.index[0]
    majority_count = int(counts.iloc[0])
    majority_base_rate = float(majority_count / len(sites))
    print(f"  Total sites: {n_classes} | Largest site: {majority_site} ({majority_count}/{len(sites)} = {majority_base_rate:.2%})")

    # Connectome site probe
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred_sites = np.zeros(len(sites_encoded))

    for train_idx, test_idx in skf.split(X, sites_encoded):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = sites_encoded[train_idx]

        sel = SelectKBest(score_func=f_classif, k=min(k_features, X_tr.shape[1]))
        X_tr_s = sel.fit_transform(X_tr, y_tr)
        X_te_s = sel.transform(X_te)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_s)
        X_te_sc = scaler.transform(X_te_s)

        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=seed, multi_class="auto")
        clf.fit(X_tr_sc, y_tr)
        pred_sites[test_idx] = clf.predict(X_te_sc)

    site_pred_acc = float(accuracy_score(sites_encoded, pred_sites))
    site_pred_bacc = float(balanced_accuracy_score(sites_encoded, pred_sites))

    print(f"  Connectome Site Decodability Acc: {site_pred_acc:.4f} (Balanced: {site_pred_bacc:.4f}) vs Majority Chance: {majority_base_rate:.4f}")

    # Confound-only probe (age, sex, motion, FIQ)
    confound_cols = [c for c in ["age", "sex", "func_mean_fd", "fiq"] if c in meta_df.columns]
    conf_site_acc, conf_site_bacc = None, None
    if confound_cols:
        X_conf = meta_df[confound_cols].values.astype(float)
        col_means = np.nanmean(X_conf, axis=0)
        X_conf = np.where(np.isnan(X_conf), col_means, X_conf)

        pred_conf_sites = np.zeros(len(sites_encoded))
        for train_idx, test_idx in skf.split(X_conf, sites_encoded):
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_conf[train_idx])
            X_te_s = scaler.transform(X_conf[test_idx])
            clf = LogisticRegression(C=0.1, max_iter=1000, random_state=seed, multi_class="auto")
            clf.fit(X_tr_s, sites_encoded[train_idx])
            pred_conf_sites[test_idx] = clf.predict(X_te_s)

        conf_site_acc = float(accuracy_score(sites_encoded, pred_conf_sites))
        conf_site_bacc = float(balanced_accuracy_score(sites_encoded, pred_conf_sites))
        print(f"  Confound Site Decodability Acc: {conf_site_acc:.4f} (Balanced: {conf_site_bacc:.4f})")

    # Site-specific diagnosis base rates
    site_as_dx = {}
    for s in meta_df["site"].unique():
        mask = (meta_df["site"] == s)
        n_s = int(mask.sum())
        asd_rate = float(meta_df.loc[mask, "target"].mean())
        majority_pred = 1 if asd_rate >= 0.5 else 0
        site_acc = float(accuracy_score(meta_df.loc[mask, "target"].values, np.full(n_s, majority_pred)))
        site_as_dx[s] = {
            "n": n_s,
            "asd_rate": asd_rate,
            "majority_vote_acc": site_acc,
        }

    return {
        "n_sites": n_classes,
        "sites": list(le.classes_),
        "majority_site": majority_site,
        "majority_site_count": majority_count,
        "majority_site_base_rate": majority_base_rate,
        "connectome_site_decodability_acc": site_pred_acc,
        "connectome_site_decodability_bacc": site_pred_bacc,
        "confound_site_decodability_acc": conf_site_acc,
        "confound_site_decodability_bacc": conf_site_bacc,
        "confound_cols": confound_cols,
        "site_as_diagnosis_predictor": site_as_dx,
    }


# ---------------------------------------------------------------------------
# Confound-Only Diagnostic Models
# ---------------------------------------------------------------------------

def run_confound_models(meta_df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    """Train diagnostic models using ONLY motion, age, sex, and FIQ under LOSO."""
    print("\n--- Confound-Only Diagnostic Models (Motion/Age/Sex/FIQ under LOSO) ---")
    confound_cols = [c for c in ["age", "sex", "func_mean_fd", "fiq"] if c in meta_df.columns]
    if not confound_cols:
        return {"note": "No confound columns found"}

    X_conf = meta_df[confound_cols].values.astype(float)
    col_means = np.nanmean(X_conf, axis=0)
    X_conf = np.where(np.isnan(X_conf), col_means, X_conf)
    y = meta_df["target"].values
    sites = meta_df["site"].values
    unique_sites = np.unique(sites)

    all_true, all_prob = [], []
    for held_site in unique_sites:
        test_mask = (sites == held_site)
        train_mask = ~test_mask

        X_tr, X_te = X_conf[train_mask], X_conf[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ])
        pipe.fit(X_tr, y_tr)
        prob = pipe.predict_proba(X_te)[:, 1]
        all_true.extend(y_te.tolist())
        all_prob.extend(prob.tolist())

    all_true_arr = np.array(all_true)
    all_prob_arr = np.array(all_prob)

    auroc = float(roc_auc_score(all_true_arr, all_prob_arr))
    bacc = float(balanced_accuracy_score(all_true_arr, (all_prob_arr >= 0.5).astype(int)))

    corrs = {}
    for col in confound_cols:
        vals = meta_df[col].fillna(meta_df[col].median()).values
        try:
            r, p = stats.pearsonr(vals, y)
            corrs[col] = {"r": float(r), "p": float(p)}
        except Exception:
            corrs[col] = {"r": None, "p": None}

    print(f"  Confound LOSO AUROC: {auroc:.4f} | Balanced Acc: {bacc:.4f}")
    return {
        "confound_cols": confound_cols,
        "loso_auroc": auroc,
        "loso_bacc": bacc,
        "correlations_with_asd": corrs,
    }


# ---------------------------------------------------------------------------
# Per-Site Generalization Gap Table
# ---------------------------------------------------------------------------

def build_per_site_gap_table(
    loso_site_metrics: Dict[str, Any], meta_df: pd.DataFrame,
    X: np.ndarray, y: np.ndarray, k_features: int = 250, seed: int = 42
) -> List[Dict[str, Any]]:
    """Build detailed per-site performance and generalization gap table."""
    print("\n--- Compiling Per-Site Generalization Gap Table ---")
    sites = meta_df["site"].values
    table = []

    # Estimate random CV performance per site using 5-fold OOF predictions
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y))
    for train_idx, test_idx in skf.split(X, y):
        sel = SelectKBest(score_func=f_classif, k=min(k_features, X.shape[1]))
        X_tr_s = sel.fit_transform(X[train_idx], y[train_idx])
        X_te_s = sel.transform(X[test_idx])
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_s)
        X_te_sc = scaler.transform(X_te_s)
        clf = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)
        clf.fit(X_tr_sc, y[train_idx])
        oof_probs[test_idx] = clf.predict_proba(X_te_sc)[:, 1]

    oof_preds = (oof_probs >= 0.5).astype(int)

    for s, metrics in loso_site_metrics.items():
        mask = (sites == s)
        n_s = int(mask.sum())
        y_s = y[mask]
        asd_rate = float(y_s.mean())
        majority_class = 1 if asd_rate >= 0.5 else 0
        majority_acc = float(np.mean(y_s == majority_class))

        rand_acc = float(accuracy_score(y_s, oof_preds[mask]))
        loso_acc = float(metrics["accuracy"])
        loso_bacc = float(metrics.get("bacc", metrics["accuracy"]))
        loso_auc = float(metrics["auroc"]) if metrics["auroc"] is not None else None
        gap = float(rand_acc - loso_acc)

        table.append({
            "site": s,
            "n": n_s,
            "n_asd": int(np.sum(y_s == 1)),
            "n_control": int(np.sum(y_s == 0)),
            "asd_rate": asd_rate,
            "majority_base_rate_acc": majority_acc,
            "random_cv_accuracy": rand_acc,
            "loso_accuracy": loso_acc,
            "loso_bacc": loso_bacc,
            "loso_auroc": loso_auc,
            "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"],
            "generalization_gap": gap,
        })

    # Sort descending by site size N
    table.sort(key=lambda x: x["n"], reverse=True)
    return table


# ---------------------------------------------------------------------------
# Robustness Arm: Non-QC ABIDE I Cohort
# ---------------------------------------------------------------------------

def run_non_qc_robustness_arm(seed: int = 42, k_features: int = 250) -> Optional[Dict[str, Any]]:
    """Evaluate Leave-One-Site-Out ComBat on the non-QC cohort (N=989, 20 sites)."""
    raw_cache_path = DATA_DIR / "abide_raw_cache.npz"
    if not raw_cache_path.exists():
        print("  Non-QC cache abide_raw_cache.npz not found; skipping robustness arm.")
        return None

    print("\n--- Robustness Arm: Evaluating Non-QC ABIDE I Cohort (abide_raw_cache.npz) ---")
    cache = np.load(raw_cache_path, allow_pickle=True)
    mats = cache["matrices"]
    triu_idx = np.triu_indices(200, k=1)
    X_raw = np.array([m[triu_idx] for m in mats])
    y_raw = cache["target"]
    sites_raw = cache["site"]

    meta_raw = pd.DataFrame({"site": sites_raw, "target": y_raw})
    unique_sites = np.unique(sites_raw)

    all_true, all_prob = [], []
    for held_site in unique_sites:
        test_mask = (sites_raw == held_site)
        train_mask = ~test_mask
        X_tr, X_te = X_raw[train_mask], X_raw[test_mask]
        y_tr, y_te = y_raw[train_mask], y_raw[test_mask]
        s_tr, s_te = sites_raw[train_mask], sites_raw[test_mask]

        fold_res = run_loso_fold(
            X_tr, y_tr, s_tr, X_te, y_te, s_te,
            harmonization="combat", k_features=k_features, seed=seed
        )
        all_true.extend(fold_res["y_true"])
        all_prob.extend(fold_res["y_prob"])

    all_true_arr = np.array(all_true)
    all_prob_arr = np.array(all_prob)
    auc = float(roc_auc_score(all_true_arr, all_prob_arr))
    acc = float(accuracy_score(all_true_arr, (all_prob_arr >= 0.5).astype(int)))
    ci_lo, ci_hi = bootstrap_auroc_ci(all_true_arr, all_prob_arr, n_boot=2000, seed=seed)

    print(f"  Non-QC Cohort (N={len(all_true_arr)}, {len(unique_sites)} sites): LOSO AUROC = {auc:.4f} [{ci_lo:.4f}, {ci_hi:.4f}], Acc = {acc:.4f}")
    return {
        "n_subjects": len(all_true_arr),
        "n_sites": len(unique_sites),
        "loso_combat_auroc": auc,
        "loso_combat_accuracy": acc,
        "auroc_ci_95": [ci_lo, ci_hi],
    }


# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------

def run_site_analysis(seed: int = 42, k_features: int = 250) -> Dict[str, Any]:
    print("=" * 70)
    print("  MODULE 3: COMPREHENSIVE SCANNER / SITE GENERALIZATION ANALYSIS")
    print("=" * 70)

    # 1. Load full quality-controlled ABIDE dataset
    cache_path = DATA_DIR / "abide_full_cache.npz"
    if not cache_path.exists():
        cache_path = DATA_DIR / "abide_cache.npz"

    print(f"Loading ABIDE cache from {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    mats = cache["matrices"]
    triu_idx = np.triu_indices(200, k=1)
    X = np.array([m[triu_idx] for m in mats])

    meta_df = pd.DataFrame({
        "site": cache["site"],
        "target": cache["target"],
        "age": cache["age"] if "age" in cache.files else np.zeros(len(cache["target"])),
        "sex": cache["sex"] if "sex" in cache.files else np.zeros(len(cache["target"])),
        "func_mean_fd": cache["func_mean_fd"] if "func_mean_fd" in cache.files else np.zeros(len(cache["target"])),
        "fiq": cache["fiq"] if "fiq" in cache.files else np.zeros(len(cache["target"])),
    })

    n_subjects = len(X)
    n_sites = int(meta_df["site"].nunique())
    print(f"Loaded {n_subjects} subjects across {n_sites} clinical sites.")
    print(f"Diagnosis balance: {dict(meta_df['target'].value_counts())}")

    output: Dict[str, Any] = {
        "module": "3_site_analysis",
        "dataset": "ABIDE_I_CC200_FULL",
        "seed": seed,
        "n_subjects": n_subjects,
        "n_sites": n_sites,
        "sites": list(meta_df["site"].unique()),
        "site_sizes": meta_df["site"].value_counts().to_dict(),
        "diagnosis_balance": {
            "control": int(np.sum(meta_df["target"] == 0)),
            "asd": int(np.sum(meta_df["target"] == 1)),
        },
    }

    # 2. Harmonization comparison (LOSO across all sites)
    harm_results, predictions_map = run_harmonization_comparison(X, meta_df, k_features=k_features, seed=seed)
    output["harmonization_comparison"] = harm_results

    # 3. Paired bootstrap tests between harmonization arms (with Holm correction)
    paired_tests = run_paired_harmonization_tests(meta_df["target"].values, predictions_map, seed=seed)
    output["paired_harmonization_tests"] = paired_tests

    # 4. Random 5-Fold Stratified CV (site-leaking baseline)
    random_cv = run_random_cv_benchmark(X, meta_df["target"].values, meta_df["site"].values, k_features=k_features)
    output["random_cv_benchmark"] = random_cv

    # Calculate overall scanner generalization gap
    combat_loso_auc = harm_results["combat"]["pooled_auroc"]
    random_auc = random_cv["mean_auroc"]
    generalization_gap = float(random_auc - combat_loso_auc)
    output["scanner_generalization_gap"] = {
        "random_cv_auroc": random_auc,
        "loso_combat_auroc": combat_loso_auc,
        "gap": generalization_gap,
        "interpretation": "Positive gap indicates optimistic inflation from scanner hardware leakage in random CV.",
    }
    print(f"\n[SCANNER GENERALIZATION GAP] Random CV ({random_auc:.4f}) vs LOSO ComBat ({combat_loso_auc:.4f}) -> Gap: {generalization_gap:+.4f}")

    # 5. Site decodability probe
    probe_results = run_site_probe(X, meta_df, k_features=min(100, k_features), seed=seed)
    output["site_probe"] = probe_results

    # 6. Confound-only models
    confound_results = run_confound_models(meta_df, seed=seed)
    output["confound_models"] = confound_results

    # 7. Detailed per-site gap table
    site_table = build_per_site_gap_table(
        harm_results["combat"]["site_metrics"], meta_df, X, meta_df["target"].values,
        k_features=k_features, seed=seed
    )
    output["per_site_gap_table"] = site_table

    # 8. Robustness arm on non-QC cohort
    robustness_arm = run_non_qc_robustness_arm(seed=seed, k_features=k_features)
    if robustness_arm:
        output["non_qc_robustness_arm"] = robustness_arm

    # Save results
    out_path = RESULTS_DIR / "site_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Full site analysis saved to {out_path}")

    return output


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_site_analysis(seed=seed)
