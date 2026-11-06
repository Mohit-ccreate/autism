"""
Site Analysis Module — Module 3 Extension.

Extends the existing ABIDE LOSO analysis with:
1. Full harmonization comparison:
   - No harmonization
   - Site-as-covariate regression
   - ComBat (correctly inside folds) 
   - Deliberately LEAKY ComBat (fit before CV) — labeled clearly
2. Site probe: predict SITE_ID from connectivity features
3. Site as diagnosis predictor (base-rate probe)
4. Motion/age/sex/FIQ confound-only models
5. Generalization gap visualization data

MEDICAL DISCLAIMER: Research tool only. Not a diagnostic device.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
    from train_abide import parametric_combat_fit, parametric_combat_transform, load_abide_dataset
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False
    print("WARNING: Could not import ComBat from train_abide.py")


# ---------------------------------------------------------------------------
# Bootstrap CI helpers
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
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 42
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
    """Site-level cluster bootstrap CI for AUROC."""
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
        aurocs.append(roc_auc_score(yt, yp))
    aurocs = np.array(aurocs)
    return float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))


# ---------------------------------------------------------------------------
# ComBat leaky variant
# ---------------------------------------------------------------------------

def leaky_combat_all_data(X: np.ndarray, sites: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """
    DELIBERATELY LEAKY: Fit ComBat on ALL data (train + test combined).
    This replicates the methodological error found in some published pipelines.
    Clearly labeled as leaky — used ONLY to quantify inflation.
    """
    if not COMBAT_AVAILABLE:
        return X
    X_harm, _ = parametric_combat_fit(X, sites, covariates)
    return X_harm


# ---------------------------------------------------------------------------
# Site regression deconfounding
# ---------------------------------------------------------------------------

def site_regression_deconfound(X_train: np.ndarray, sites_train: np.ndarray,
                                X_test: np.ndarray, sites_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove site mean effects by regression (without diagnosis label).
    Fit on training data only; apply to test.
    """
    unique_sites = np.unique(sites_train)
    grand_mean = np.mean(X_train, axis=0)

    # Compute site means on training data
    site_means = {}
    for s in unique_sites:
        mask = sites_train == s
        site_means[s] = np.mean(X_train[mask], axis=0)

    # Remove site mean, add grand mean
    X_train_deconf = X_train.copy()
    for s in unique_sites:
        mask = sites_train == s
        X_train_deconf[mask] = X_train[mask] - site_means[s] + grand_mean

    X_test_deconf = X_test.copy()
    for s in np.unique(sites_test):
        mask = sites_test == s
        if s in site_means:
            # Known site — use training mean
            X_test_deconf[mask] = X_test[mask] - site_means[s] + grand_mean
        # Unknown site — no adjustment (site-naive)

    return X_train_deconf, X_test_deconf


# ---------------------------------------------------------------------------
# Single LOSO fold evaluation
# ---------------------------------------------------------------------------

def run_loso_fold(
    X_train: np.ndarray, y_train: np.ndarray, sites_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray, sites_test: np.ndarray,
    harmonization: str = "combat",
    k_features: int = 250,
    seed: int = 42,
) -> Dict:
    """
    Run one LOSO fold with the specified harmonization strategy.

    Parameters
    ----------
    harmonization : "none" | "site_regression" | "combat" | "leaky_combat"
    """
    covariates_train = None  # will be set if metadata available

    if harmonization == "none":
        X_tr = X_train.copy()
        X_te = X_test.copy()

    elif harmonization == "site_regression":
        X_tr, X_te = site_regression_deconfound(X_train, sites_train, X_test, sites_test)

    elif harmonization == "combat" and COMBAT_AVAILABLE:
        X_tr, combat_model = parametric_combat_fit(X_train, sites_train, covariates_train)
        X_te = parametric_combat_transform(X_test, sites_test, covariates_train, combat_model)

    elif harmonization == "leaky_combat" and COMBAT_AVAILABLE:
        # This is the deliberately wrong arm — fit on combined train+test
        X_all = np.vstack([X_train, X_test])
        sites_all = np.concatenate([sites_train, sites_test])
        X_all_harm, _ = parametric_combat_fit(X_all, sites_all, None)
        X_tr = X_all_harm[:len(X_train)]
        X_te = X_all_harm[len(X_train):]

    else:
        X_tr = X_train.copy()
        X_te = X_test.copy()

    # Feature selection inside fold
    if k_features > 0 and k_features < X_tr.shape[1]:
        selector = SelectKBest(score_func=f_classif, k=k_features)
        X_tr = selector.fit_transform(X_tr, y_train)
        X_te = selector.transform(X_te)

    # Scale inside fold
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # Logistic Regression (L2)
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
# Harmonization comparison
# ---------------------------------------------------------------------------

def harmonization_comparison(
    X: np.ndarray, meta_df: pd.DataFrame, k_features: int = 250, seed: int = 42
) -> Dict:
    """
    Compare harmonization strategies under LOSO CV.

    Returns per-strategy aggregate metrics.
    """
    print("\n--- Harmonization Comparison (LOSO CV) ---")
    harmonizations = ["none", "site_regression", "combat"]
    if COMBAT_AVAILABLE:
        harmonizations.append("leaky_combat")

    sites = meta_df["site"].values
    y = meta_df["target"].values
    unique_sites = np.unique(sites)

    results = {}
    for harm in harmonizations:
        label = harm.upper()
        if harm == "leaky_combat":
            label = "LEAKY_COMBAT (deliberately wrong — do not use clinically)"
        print(f"  Harmonization: {label}")

        all_true, all_prob = [], []
        site_metrics = {}

        for held_site in unique_sites:
            test_mask = sites == held_site
            train_mask = ~test_mask

            X_tr = X[train_mask]
            X_te = X[test_mask]
            y_tr = y[train_mask]
            y_te = y[test_mask]
            s_tr = sites[train_mask]
            s_te = sites[test_mask]

            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                print(f"    Skipping site {held_site} (single class)")
                continue

            fold_res = run_loso_fold(
                X_tr, y_tr, s_tr, X_te, y_te, s_te,
                harmonization=harm, k_features=k_features, seed=seed
            )
            site_metrics[held_site] = {
                "n_test": fold_res["n_test"],
                "accuracy": fold_res["accuracy"],
                "auroc": fold_res["auroc"],
                "sensitivity": fold_res["sensitivity"],
                "specificity": fold_res["specificity"],
            }
            all_true.extend(fold_res["y_true"])
            all_prob.extend(fold_res["y_prob"])

        all_true_arr = np.array(all_true)
        all_prob_arr = np.array(all_prob)

        if len(np.unique(all_true_arr)) >= 2:
            pooled_auroc = roc_auc_score(all_true_arr, all_prob_arr)
            auroc_lo, auroc_hi = bootstrap_auroc_ci(all_true_arr, all_prob_arr, n_boot=2000, seed=seed)
        else:
            pooled_auroc = None
            auroc_lo, auroc_hi = None, None

        results[harm] = {
            "harmonization": harm,
            "is_leaky": harm == "leaky_combat",
            "pooled_accuracy": float(accuracy_score(all_true_arr, (all_prob_arr >= 0.5).astype(int))),
            "pooled_auroc": float(pooled_auroc) if pooled_auroc else None,
            "auroc_ci_95": [auroc_lo, auroc_hi],
            "site_metrics": site_metrics,
        }
        print(f"    Pooled AUROC: {pooled_auroc:.4f}" if pooled_auroc else "    AUROC: N/A")

    return results


# ---------------------------------------------------------------------------
# Site probe: predict SITE_ID from features
# ---------------------------------------------------------------------------

def site_probe(X: np.ndarray, meta_df: pd.DataFrame, k_features: int = 100, seed: int = 42) -> Dict:
    """
    Train a classifier to predict SITE_ID from connectivity features.
    If SITE_ID is predictable, features carry site/scanner signatures.
    """
    print("\n--- Site Probe: Predicting SITE_ID ---")
    sites = meta_df["site"].values
    le = LabelEncoder()
    sites_encoded = le.fit_transform(sites)
    n_classes = len(le.classes_)
    print(f"  Sites: {list(le.classes_)} ({n_classes} classes)")

    # Connectivity features
    pipe = Pipeline([
        ("sel", SelectKBest(f_classif, k=min(k_features, X.shape[1]))),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=seed, multi_class="auto")),
    ])
    skf = StratifiedKFold(n_splits=min(5, n_classes), shuffle=True, random_state=seed)
    # Use cross_val_predict for OOF predictions
    try:
        oof_preds = cross_val_predict(pipe, X, sites_encoded, cv=skf)
        site_pred_acc = float(accuracy_score(sites_encoded, oof_preds))
        site_pred_bacc = float(balanced_accuracy_score(sites_encoded, oof_preds))
    except Exception as e:
        print(f"  Site probe failed: {e}")
        site_pred_acc = float("nan")
        site_pred_bacc = float("nan")

    # Confound-only (motion/age/sex) probe
    confound_cols = [c for c in ["age", "sex", "func_mean_fd", "FIQ"] if c in meta_df.columns]
    if confound_cols:
        X_confounds = meta_df[confound_cols].values.astype(float)
        # Replace NaN with column mean
        col_means = np.nanmean(X_confounds, axis=0)
        X_confounds = np.where(np.isnan(X_confounds), col_means, X_confounds)
        pipe_conf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=seed, multi_class="auto")),
        ])
        try:
            oof_preds_conf = cross_val_predict(pipe_conf, X_confounds, sites_encoded, cv=skf)
            conf_site_acc = float(accuracy_score(sites_encoded, oof_preds_conf))
        except Exception:
            conf_site_acc = float("nan")
    else:
        conf_site_acc = float("nan")
        confound_cols = []

    # Base-rate (most common site always predicted)
    from collections import Counter
    most_common = Counter(sites_encoded).most_common(1)[0][0]
    base_rate_acc = float(np.mean(sites_encoded == most_common))

    # Site as diagnosis predictor (base rate per site)
    site_as_dx = {}
    for s in meta_df["site"].unique():
        mask = meta_df["site"] == s
        asd_rate = float(meta_df.loc[mask, "target"].mean())
        # Predict the majority class within site
        majority_pred = 1 if asd_rate >= 0.5 else 0
        site_acc = float(accuracy_score(meta_df.loc[mask, "target"].values,
                                        np.full(mask.sum(), majority_pred)))
        site_as_dx[s] = {
            "n": int(mask.sum()),
            "asd_rate": asd_rate,
            "majority_vote_acc": site_acc,
        }

    return {
        "connectivity_site_pred_accuracy": site_pred_acc,
        "connectivity_site_pred_bacc": site_pred_bacc,
        "confound_site_pred_accuracy": conf_site_acc,
        "confound_cols_used": confound_cols,
        "base_rate_accuracy": base_rate_acc,
        "n_sites": n_classes,
        "site_as_diagnosis_predictor": site_as_dx,
    }


# ---------------------------------------------------------------------------
# Confound-only models
# ---------------------------------------------------------------------------

def confound_only_models(X: np.ndarray, meta_df: pd.DataFrame, seed: int = 42) -> Dict:
    """
    Train models using only motion/age/sex (no connectivity features).
    If these predict ASD, we have a confound problem.
    """
    print("\n--- Confound-Only Models (Motion/Age/Sex) ---")
    confound_cols = [c for c in ["age", "sex", "func_mean_fd", "FIQ"] if c in meta_df.columns]
    if not confound_cols:
        return {"note": "No confound columns found in metadata"}

    X_conf = meta_df[confound_cols].values.astype(float)
    col_means = np.nanmean(X_conf, axis=0)
    X_conf = np.where(np.isnan(X_conf), col_means, X_conf)
    y = meta_df["target"].values

    sites = meta_df["site"].values
    unique_sites = np.unique(sites)

    all_true, all_prob = [], []
    for held_site in unique_sites:
        test_mask = sites == held_site
        train_mask = ~test_mask

        X_tr = X_conf[train_mask]
        X_te = X_conf[test_mask]
        y_tr = y[train_mask]
        y_te = y[test_mask]

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

    if len(np.unique(all_true_arr)) >= 2:
        auroc = float(roc_auc_score(all_true_arr, all_prob_arr))
        bacc = float(balanced_accuracy_score(all_true_arr, (all_prob_arr >= 0.5).astype(int)))
    else:
        auroc = None
        bacc = None

    # Also compute partial correlations with ASD label
    partial_corrs = {}
    for col in confound_cols:
        col_vals = meta_df[col].fillna(meta_df[col].median()).values
        try:
            r, pval = stats.pearsonr(col_vals, y)
            partial_corrs[col] = {"r": float(r), "p": float(pval)}
        except Exception:
            partial_corrs[col] = {"r": None, "p": None}

    return {
        "confound_cols": confound_cols,
        "loso_auroc": auroc,
        "loso_bacc": bacc,
        "partial_correlations_with_asd": partial_corrs,
        "interpretation": (
            "If AUROC > 0.6, confounds may be driving diagnosis prediction. "
            "Values near 0.5 indicate low confound influence."
        )
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_site_analysis(seed: int = 42, k_features: int = 250) -> Dict:
    """
    Run full site/scanner generalization analysis.
    Returns results dict for JSON export.
    """
    print("=" * 70)
    print("  MODULE 3: SCANNER / SITE GENERALIZATION ANALYSIS")
    print("=" * 70)

    # Load cached ABIDE data or from .1D files
    cache_path = DATA_DIR / "abide_cache.npz"
    if cache_path.exists():
        print(f"Loading ABIDE cache from {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        mats = cache["matrices"]  # (120, 200, 200)
        triu_idx = np.triu_indices(200, k=1)
        X = np.array([m[triu_idx] for m in mats])
        meta_df = pd.DataFrame({
            "site": cache["site"],
            "target": cache["target"],
            "age": cache["age"] if "age" in cache.files else np.zeros(len(cache["target"])),
            "sex": cache["sex"] if "sex" in cache.files else np.zeros(len(cache["target"])),
            "func_mean_fd": cache["func_mean_fd"] if "func_mean_fd" in cache.files else np.zeros(len(cache["target"])),
        })
    elif COMBAT_AVAILABLE:
        try:
            X, meta_df, _ = load_abide_dataset()
        except Exception as e:
            print(f"Could not load ABIDE dataset: {e}")
            return {"error": str(e), "note": "ABIDE data not available"}
    else:
        return {"error": "No ABIDE data available", "note": "SKIPPED"}

    print(f"Loaded {len(X)} subjects, {len(meta_df['site'].unique())} sites")
    print(f"Sites: {dict(meta_df['site'].value_counts())}")

    output = {
        "module": "3_site_analysis",
        "seed": seed,
        "n_subjects": len(X),
        "sites": list(meta_df["site"].unique()),
        "n_sites": int(meta_df["site"].nunique()),
        "site_sizes": meta_df["site"].value_counts().to_dict(),
    }

    # 1. Harmonization comparison
    harm_results = harmonization_comparison(X, meta_df, k_features=k_features, seed=seed)
    output["harmonization_comparison"] = harm_results

    # 2. Site probe
    site_probe_results = site_probe(X, meta_df, k_features=min(100, k_features), seed=seed)
    output["site_probe"] = site_probe_results

    # 3. Confound-only models
    confound_results = confound_only_models(X, meta_df, seed=seed)
    output["confound_models"] = confound_results

    # Save
    out_path = RESULTS_DIR / "site_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Site analysis results saved to {out_path}")
    return output


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_site_analysis(seed=seed)
