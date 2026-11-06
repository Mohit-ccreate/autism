"""
Module 2 Comprehensive Benchmark Leaderboard.
src/eval/benchmark.py

Rigorously benchmarks multiple model families on:
1. Leak-Free Tabular Cohorts (Variant C: Demographics & History only)
   - Adult (N=667), Adolescent (N=97), Child (N=262), Toddler (N=850)
   - Models: L2 Logistic Regression, Linear SVM, Random Forest, Extra Trees, LightGBM, Small MLP
2. Full ABIDE I Neuroimaging Cohort (N=854, 20 sites)
   - Models: L2 Logistic Regression, Linear SVM, Random Forest, Extra Trees, LightGBM, Small MLP,
             Tangent-Space Ridge, Heinsfeld-Style Deep Autoencoder
   - Evaluated under both:
     (a) Leave-One-Site-Out (LOSO) across all 20 sites with in-fold ComBat
     (b) 5-Fold Stratified Random CV across 5 seeds (revealing site leakage gap)

All metrics computed with 95% bootstrap confidence intervals.
Zero cross-fold or test-site leakage.

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
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
    brier_score_loss,
)
import lightgbm as lgb
from nilearn.connectome import ConnectivityMeasure

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "src"))
from train_abide import parametric_combat_fit, parametric_combat_transform
from models.autoencoder import HeinsfeldAutoencoderClassifier


# ---------------------------------------------------------------------------
# Metric & Bootstrap Helpers
# ---------------------------------------------------------------------------

def fast_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n1 = int(np.sum(y_true == 1))
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(y_prob)
    rank_sum = np.sum(ranks[y_true == 1])
    u = rank_sum - (n1 * (n1 + 1)) / 2.0
    return float(u / (n1 * n0))


def bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray, metric_fn, n_boot: int = 1000, seed: int = 42
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = rng.integers(0, n, size=(n_boot, n))
    scores = []
    for idx in indices:
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric_fn(yt, y_prob[idx]))
    if not scores:
        return 0.5, 0.5
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def compute_metrics_dict(y_true: np.ndarray, y_prob: np.ndarray, seed: int = 42) -> Dict[str, Any]:
    y_pred = (y_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
    brier = brier_score_loss(y_true, y_prob)

    auc_lo, auc_hi = bootstrap_ci(y_true, y_prob, fast_auc, n_boot=1000, seed=seed)
    acc_lo, acc_hi = bootstrap_ci(y_true, y_prob, lambda yt, yp: accuracy_score(yt, (yp >= 0.5).astype(int)), n_boot=1000, seed=seed)

    return {
        "auroc": float(auc),
        "auroc_ci_95": [auc_lo, auc_hi],
        "auprc": float(auprc),
        "accuracy": float(acc),
        "accuracy_ci_95": [acc_lo, acc_hi],
        "balanced_accuracy": float(bacc),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "f1": float(f1),
        "mcc": float(mcc),
        "brier_score": float(brier),
    }


# ---------------------------------------------------------------------------
# Part 1: Tabular Leak-Free Benchmark (Variant C: Demographics & History only)
# ---------------------------------------------------------------------------

def load_tabular_cohort(cohort_name: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    from audit.dedup_and_threshold_audit import load_raw_cohort, DEMO_COLS, ITEM_COLS
    df = load_raw_cohort(cohort_name.lower())
    match_cols = ITEM_COLS + [c for c in ["age", "sex", "target"] if c in df.columns]
    exact_dup_mask = df.duplicated(subset=match_cols, keep="first")
    df_clean = df[~exact_dup_mask].copy().reset_index(drop=True)
    X = df_clean[DEMO_COLS].values.astype(float)
    y = df_clean["target"].values.astype(int)
    return X, y, DEMO_COLS


def benchmark_tabular_models(seeds: List[int] = [42, 43, 44, 45, 46]) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: LEAK-FREE TABULAR COHORTS (Demographics / History Only)")
    print("=" * 70)
    cohorts = ["adult", "adolescent", "child", "toddler"]
    tabular_results = {}

    for cohort in cohorts:
        X, y, feats = load_tabular_cohort(cohort)
        n_samples = len(X)
        asd_prev = float(np.mean(y))
        print(f"\nCohort: {cohort.upper()} (N={n_samples}, ASD prevalence={asd_prev:.2%}, Features={feats})")

        cohort_models = {
            "Logistic Regression (L2)": lambda s: LogisticRegression(C=0.1, max_iter=1000, random_state=s),
            "Linear SVM": lambda s: CalibratedClassifierCV(LinearSVC(C=0.1, random_state=s), cv=3),
            "Random Forest": lambda s: RandomForestClassifier(n_estimators=100, max_depth=4, random_state=s),
            "Extra Trees": lambda s: ExtraTreesClassifier(n_estimators=100, max_depth=4, random_state=s),
            "LightGBM": lambda s: lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=s),
            "Small MLP": lambda s: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, alpha=0.01, random_state=s),
        }

        cohort_res = {}
        for model_name, model_fn in cohort_models.items():
            all_true = []
            all_prob = []
            for seed in seeds:
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                oof_prob = np.zeros(len(y))
                for tr_idx, te_idx in skf.split(X, y):
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X[tr_idx])
                    X_te = scaler.transform(X[te_idx])
                    clf = model_fn(seed)
                    clf.fit(X_tr, y[tr_idx])
                    if hasattr(clf, "predict_proba"):
                        prob = clf.predict_proba(X_te)[:, 1]
                    else:
                        prob = clf.decision_function(X_te)
                        prob = 1.0 / (1.0 + np.exp(-prob))
                    oof_prob[te_idx] = prob

                all_true.extend(y.tolist())
                all_prob.extend(oof_prob.tolist())

            metrics = compute_metrics_dict(np.array(all_true), np.array(all_prob), seed=seeds[0])
            cohort_res[model_name] = metrics
            print(f"  {model_name:26s} | AUROC: {metrics['auroc']:.4f} [{metrics['auroc_ci_95'][0]:.4f}, {metrics['auroc_ci_95'][1]:.4f}] | Acc: {metrics['accuracy']:.4f} | Brier: {metrics['brier_score']:.4f}")

        tabular_results[cohort] = {
            "n_samples": n_samples,
            "asd_prevalence": asd_prev,
            "features": feats,
            "models": cohort_res,
        }

    return tabular_results


# ---------------------------------------------------------------------------
# Part 2: ABIDE Full Neuroimaging Benchmark (N=854, 20 sites)
# ---------------------------------------------------------------------------

def benchmark_abide_loso(
    X_vec: np.ndarray, mats: np.ndarray, y: np.ndarray, sites: np.ndarray,
    k_features: int = 250, seed: int = 42
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  BENCHMARK 2A: FULL ABIDE I LEAVE-ONE-SITE-OUT (LOSO) BENCHMARK (N=854, 20 SITES)")
    print("=" * 70)
    unique_sites = np.unique(sites)

    model_factories = {
        "Logistic Regression (L2)": lambda: LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed),
        "Linear SVM": lambda: SVC(kernel="linear", C=0.01, probability=True, random_state=seed),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=seed),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=seed),
        "Small MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, alpha=0.05, random_state=seed),
        "Heinsfeld Autoencoder": lambda: HeinsfeldAutoencoderClassifier(latent_dim=32, pretrain_epochs=20, finetune_epochs=25, seed=seed),
    }

    # Pre-harmonize X_vec in-fold per held site
    site_fold_data = {}
    for held_site in unique_sites:
        test_mask = (sites == held_site)
        train_mask = ~test_mask
        X_tr, X_te = X_vec[train_mask], X_vec[test_mask]
        s_tr, s_te = sites[train_mask], sites[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        X_tr_harm, combat_model = parametric_combat_fit(X_tr, s_tr, None)
        X_te_harm = parametric_combat_transform(X_te, s_te, None, combat_model)

        sel = SelectKBest(score_func=f_classif, k=min(k_features, X_tr_harm.shape[1]))
        X_tr_s = sel.fit_transform(X_tr_harm, y_tr)
        X_te_s = sel.transform(X_te_harm)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_s)
        X_te_sc = scaler.transform(X_te_s)

        site_fold_data[held_site] = (X_tr_sc, y_tr, X_te_sc, y_te)

    loso_results = {}
    for model_name, factory in model_factories.items():
        print(f"  Evaluating LOSO: {model_name}...")
        all_true = []
        all_prob = []
        for held_site in unique_sites:
            X_tr_sc, y_tr, X_te_sc, y_te = site_fold_data[held_site]
            clf = factory()
            clf.fit(X_tr_sc, y_tr)
            prob = clf.predict_proba(X_te_sc)[:, 1]
            all_true.extend(y_te.tolist())
            all_prob.extend(prob.tolist())

        metrics = compute_metrics_dict(np.array(all_true), np.array(all_prob), seed=seed)
        loso_results[model_name] = metrics
        print(f"    {model_name:25s} | LOSO AUROC: {metrics['auroc']:.4f} [{metrics['auroc_ci_95'][0]:.4f}, {metrics['auroc_ci_95'][1]:.4f}] | Acc: {metrics['accuracy']:.4f}")

    # Tangent Space Ridge / SVM
    print("  Evaluating LOSO: Tangent Space Ridge Classifier...")
    all_true_tan = []
    all_prob_tan = []
    # Tangent Space Logistic / Ridge
    print("  Evaluating LOSO: Tangent Space (Log-Euclidean Riemannian) Logistic...")
    reg_mats = mats.copy()
    for i in range(len(reg_mats)):
        np.fill_diagonal(reg_mats[i], 1.0)
        reg_mats[i] += 1e-3 * np.eye(200)

    # Vectorized Log-Euclidean projection
    eigs, vecs = np.linalg.eigh(reg_mats)
    log_mats = np.array([v @ np.diag(np.log(np.clip(e, 1e-4, None))) @ v.T for e, v in zip(eigs, vecs)])
    triu_idx = np.triu_indices(200, k=0)
    X_tan = np.array([m[triu_idx] for m in log_mats])

    all_true_tan = []
    all_prob_tan = []
    for held_site in unique_sites:
        test_mask = (sites == held_site)
        train_mask = ~test_mask
        X_tr_tan, X_te_tan = X_tan[train_mask], X_tan[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        sel = SelectKBest(score_func=f_classif, k=min(k_features, X_tr_tan.shape[1]))
        X_tr_s = sel.fit_transform(X_tr_tan, y_tr)
        X_te_s = sel.transform(X_te_tan)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_s)
        X_te_sc = scaler.transform(X_te_s)

        clf = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)
        clf.fit(X_tr_sc, y_tr)
        prob = clf.predict_proba(X_te_sc)[:, 1]
        all_true_tan.extend(y_te.tolist())
        all_prob_tan.extend(prob.tolist())

    tan_metrics = compute_metrics_dict(np.array(all_true_tan), np.array(all_prob_tan), seed=seed)
    loso_results["Tangent Space Logistic"] = tan_metrics
    print(f"    {'Tangent Space Logistic':25s} | LOSO AUROC: {tan_metrics['auroc']:.4f} [{tan_metrics['auroc_ci_95'][0]:.4f}, {tan_metrics['auroc_ci_95'][1]:.4f}] | Acc: {tan_metrics['accuracy']:.4f}")

    return loso_results


def benchmark_abide_random_cv(
    X_vec: np.ndarray, mats: np.ndarray, y: np.ndarray,
    k_features: int = 250, seeds: List[int] = [42, 43, 44, 45, 46]
) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  BENCHMARK 2B: FULL ABIDE I RANDOM 5-FOLD CV (SITE-LEAKING CONTRAST)")
    print("=" * 70)
    models = {
        "Logistic Regression (L2)": lambda s: LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=s),
        "Linear SVM": lambda s: SVC(kernel="linear", C=0.01, probability=True, random_state=s),
        "Random Forest": lambda s: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=s),
        "Extra Trees": lambda s: ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=s),
        "LightGBM": lambda s: lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=s),
        "Small MLP": lambda s: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, alpha=0.05, random_state=s),
        "Heinsfeld Autoencoder": lambda s: HeinsfeldAutoencoderClassifier(latent_dim=32, pretrain_epochs=15, finetune_epochs=20, seed=s),
    }

    # Pre-compute cross-validation folds across all seeds
    print("  Precomputing feature selection and scaling for 25 CV folds...")
    cv_folds = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for tr_idx, te_idx in skf.split(X_vec, y):
            X_tr, X_te = X_vec[tr_idx], X_vec[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            sel = SelectKBest(score_func=f_classif, k=min(k_features, X_tr.shape[1]))
            X_tr_s = sel.fit_transform(X_tr, y_tr)
            X_te_s = sel.transform(X_te)

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr_s)
            X_te_sc = scaler.transform(X_te_s)
            cv_folds.append((seed, X_tr_sc, y_tr, X_te_sc, y_te))

    random_results = {}
    for model_name, factory in models.items():
        print(f"  Evaluating Random CV: {model_name}...")
        all_true = []
        all_prob = []
        for seed, X_tr_sc, y_tr, X_te_sc, y_te in cv_folds:
            clf = factory(seed)
            clf.fit(X_tr_sc, y_tr)
            prob = clf.predict_proba(X_te_sc)[:, 1]
            all_true.extend(y_te.tolist())
            all_prob.extend(prob.tolist())

        metrics = compute_metrics_dict(np.array(all_true), np.array(all_prob), seed=seeds[0])
        random_results[model_name] = metrics
        print(f"    {model_name:25s} | Random CV AUROC: {metrics['auroc']:.4f} [{metrics['auroc_ci_95'][0]:.4f}, {metrics['auroc_ci_95'][1]:.4f}] | Acc: {metrics['accuracy']:.4f}")

    return random_results


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_benchmark_leaderboard(seed: int = 42) -> Dict[str, Any]:
    print("=" * 70)
    print("  MODULE 2: COMPREHENSIVE BENCHMARK LEADERBOARD")
    print("=" * 70)

    # 1. Tabular Leak-Free Benchmark
    tabular_results = benchmark_tabular_models(seeds=[42, 43, 44, 45, 46])

    # 2. ABIDE Full Benchmark
    cache_path = DATA_DIR / "abide_full_cache.npz"
    if not cache_path.exists():
        cache_path = DATA_DIR / "abide_cache.npz"

    print(f"\nLoading ABIDE dataset from {cache_path}")
    cache = np.load(cache_path, allow_pickle=True)
    mats = cache["matrices"]
    triu_idx = np.triu_indices(200, k=1)
    X_vec = np.array([m[triu_idx] for m in mats])
    y = cache["target"]
    sites = cache["site"]

    loso_results = benchmark_abide_loso(X_vec, mats, y, sites, k_features=250, seed=seed)
    random_cv_results = benchmark_abide_random_cv(X_vec, mats, y, k_features=250, seeds=[42, 43, 44, 45, 46])

    # Compute site leakage gap for each model
    gaps = {}
    for m in loso_results:
        if m in random_cv_results:
            gap = float(random_cv_results[m]["auroc"] - loso_results[m]["auroc"])
            gaps[m] = {
                "random_cv_auroc": random_cv_results[m]["auroc"],
                "loso_auroc": loso_results[m]["auroc"],
                "leakage_gap": gap,
            }

    output = {
        "module": "2_benchmark_leaderboard",
        "description": "Comprehensive benchmark across model families on leak-free tabular and full ABIDE I",
        "seed": seed,
        "tabular_leak_free": tabular_results,
        "abide_full": {
            "n_subjects": len(X_vec),
            "n_sites": int(len(np.unique(sites))),
            "loso_benchmark": loso_results,
            "random_cv_benchmark": random_cv_results,
            "site_leakage_gaps": gaps,
        },
    }

    out_path = RESULTS_DIR / "benchmark_leaderboard.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Benchmark leaderboard results saved to {out_path}")
    return output


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_benchmark_leaderboard(seed=42)
