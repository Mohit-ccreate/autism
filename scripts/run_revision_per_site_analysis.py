"""
Revision Pass: Pooled vs Within-Site Performance & Site-Prevalence Baseline.
Addresses Reviewer Item B.

Computes for all 8 ABIDE models under LOSO:
1. Per-site AUROC for all sites containing both classes (excluded sites listed).
2. Unweighted and n-weighted macro-average AUROCs.
3. Median and range (min, max).
4. Bootstrap 95% CIs resampling sites with replacement (B=2,000).
5. Site-prevalence baseline: pooled AUROC obtained by scoring each subject with their site's true ASD prevalence (oracle).
6. Within-site z-scored baseline: pooled AUROC after z-scoring predicted scores within each held-out site.
7. Quantifies how much of pooled AUROC stems from between-site score and prevalence shifts.
8. Saves results/per_site_auroc.csv.
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
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR / "src"))
from train_abide import parametric_combat_fit, parametric_combat_transform
from models.autoencoder import HeinsfeldAutoencoderClassifier


def fast_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    n1 = int(np.sum(y_true == 1))
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(y_prob)
    rank_sum = np.sum(ranks[y_true == 1])
    u = rank_sum - (n1 * (n1 + 1)) / 2.0
    return float(u / (n1 * n0))


def site_resampling_bootstrap_ci(
    site_names: List[str], site_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    metric: str = "macro_unweighted", n_boot: int = 2000, seed: int = 42
) -> Tuple[float, float]:
    """Bootstrap CI resampling entire sites with replacement."""
    rng = np.random.default_rng(seed)
    n_sites = len(site_names)
    boot_scores = []

    for _ in range(n_boot):
        sampled_sites = rng.choice(site_names, size=n_sites, replace=True)
        if metric == "macro_unweighted":
            vals = []
            for s in sampled_sites:
                yt, yp = site_data[s]
                if len(np.unique(yt)) > 1:
                    vals.append(fast_auc(yt, yp))
            if vals:
                boot_scores.append(float(np.mean(vals)))
        elif metric == "macro_weighted":
            vals = []
            weights = []
            for s in sampled_sites:
                yt, yp = site_data[s]
                if len(np.unique(yt)) > 1:
                    vals.append(fast_auc(yt, yp))
                    weights.append(len(yt))
            if vals and sum(weights) > 0:
                boot_scores.append(float(np.average(vals, weights=weights)))
        elif metric == "pooled":
            y_pool = []
            p_pool = []
            for s in sampled_sites:
                yt, yp = site_data[s]
                y_pool.extend(yt.tolist())
                p_pool.extend(yp.tolist())
            y_pool = np.array(y_pool)
            p_pool = np.array(p_pool)
            if len(np.unique(y_pool)) > 1:
                boot_scores.append(fast_auc(y_pool, p_pool))

    if not boot_scores:
        return 0.5, 0.5
    return float(np.percentile(boot_scores, 2.5)), float(np.percentile(boot_scores, 97.5))


def run_per_site_benchmark() -> None:
    print("=" * 75)
    print("  REVISION AUDIT: POOLED VS WITHIN-SITE PERFORMANCE BENCHMARK")
    print("=" * 75)

    cache_path = DATA_DIR / "abide_full_cache.npz"
    cache = np.load(cache_path, allow_pickle=True)
    mats = cache["matrices"]
    triu_idx = np.triu_indices(200, k=1)
    X_vec = np.array([m[triu_idx] for m in mats])
    y = cache["target"].astype(int)
    sites = cache["site"].astype(str)
    unique_sites = np.unique(sites)

    print(f"Loaded full dataset: N={len(X_vec)} subjects across {len(unique_sites)} clinical sites.")

    # 1. Check sites with both classes vs single-class sites
    excluded_sites = []
    included_sites = []
    site_counts = {}
    for s in unique_sites:
        mask = (sites == s)
        n_asd = int(np.sum(y[mask] == 1))
        n_ctrl = int(np.sum(y[mask] == 0))
        site_counts[s] = {"n_total": int(np.sum(mask)), "n_asd": n_asd, "n_ctrl": n_ctrl}
        if n_asd == 0 or n_ctrl == 0:
            excluded_sites.append(s)
        else:
            included_sites.append(s)

    print(f"Sites with both classes (Included): {len(included_sites)} of {len(unique_sites)}")
    print(f"Single-class sites (Excluded): {excluded_sites} (Count: {len(excluded_sites)})")

    # 2. Site-Prevalence Oracle Baseline
    df_meta = pd.DataFrame({"site": sites, "target": y})
    site_prevalences = df_meta.groupby("site")["target"].transform("mean").values
    site_prev_oracle_auroc = float(roc_auc_score(y, site_prevalences))
    print(f"\n[ORACLE BASELINE] Site Prevalence Oracle Pooled AUROC (uses NO imaging features): {site_prev_oracle_auroc:.4f}")

    # 3. Precompute Harmonization (In-Fold ComBat) and Scaling for standard features
    print("Pre-computing in-fold ComBat and feature selection per site...")
    fold_data = {}
    k_features = 250
    seed = 42

    for held_site in unique_sites:
        test_mask = (sites == held_site)
        train_mask = ~test_mask
        X_tr_raw, y_tr = X_vec[train_mask], y[train_mask]
        X_te_raw, y_te = X_vec[test_mask], y[test_mask]
        s_tr, s_te = sites[train_mask], sites[test_mask]

        X_tr_harm, combat_model = parametric_combat_fit(X_tr_raw, s_tr, None)
        X_te_harm = parametric_combat_transform(X_te_raw, s_te, None, combat_model)

        sel = SelectKBest(score_func=f_classif, k=k_features)
        X_tr_s = sel.fit_transform(X_tr_harm, y_tr)
        X_te_s = sel.transform(X_te_harm)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_s)
        X_te_sc = scaler.transform(X_te_s)

        fold_data[held_site] = (X_tr_sc, y_tr, X_te_sc, y_te)

    # 4. Tangent space features
    print("Pre-computing Log-Euclidean Tangent Space Riemannian features...")
    reg_mats = mats.copy()
    for i in range(len(reg_mats)):
        np.fill_diagonal(reg_mats[i], 1.0)
        reg_mats[i] += 1e-3 * np.eye(200)
    eigs, vecs = np.linalg.eigh(reg_mats)
    log_mats = np.array([v @ np.diag(np.log(np.clip(e, 1e-4, None))) @ v.T for e, v in zip(eigs, vecs)])
    triu_idx0 = np.triu_indices(200, k=0)
    X_tan = np.array([m[triu_idx0] for m in log_mats])

    tan_fold_data = {}
    for held_site in unique_sites:
        test_mask = (sites == held_site)
        train_mask = ~test_mask
        X_tr_t, y_tr = X_tan[train_mask], y[train_mask]
        X_te_t, y_te = X_tan[test_mask], y[test_mask]

        sel = SelectKBest(score_func=f_classif, k=k_features)
        X_tr_s = sel.fit_transform(X_tr_t, y_tr)
        X_te_s = sel.transform(X_te_t)

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_s)
        X_te_sc = scaler.transform(X_te_s)
        tan_fold_data[held_site] = (X_tr_sc, y_tr, X_te_sc, y_te)

    # Define all 8 models
    models_to_evaluate = {
        "Logistic Regression (L2)": lambda: LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed),
        "Linear SVM": lambda: SVC(kernel="linear", C=0.01, probability=True, random_state=seed),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=6, random_state=seed),
        "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=100, max_depth=6, random_state=seed),
        "LightGBM": lambda: lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05, verbose=-1, random_state=seed),
        "Small MLP": lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, alpha=0.05, random_state=seed),
        "Heinsfeld Autoencoder": lambda: HeinsfeldAutoencoderClassifier(latent_dim=32, pretrain_epochs=15, finetune_epochs=20, seed=seed),
        "Tangent Space Logistic": None,  # evaluated using tan_fold_data
    }

    per_site_rows = []
    summary_model_records = {}

    for model_name, factory in models_to_evaluate.items():
        print(f"\n--- Evaluating Model: {model_name} across 20 sites ---")
        site_preds = {}
        all_true = []
        all_prob = []
        all_prob_zscored = []

        for held_site in unique_sites:
            if model_name == "Tangent Space Logistic":
                X_tr, y_tr, X_te, y_te = tan_fold_data[held_site]
                clf = LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)
            else:
                X_tr, y_tr, X_te, y_te = fold_data[held_site]
                clf = factory()

            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_te)[:, 1]

            # Within-site z-scored predictions
            prob_std = np.std(prob)
            if prob_std < 1e-6:
                prob_z = np.zeros_like(prob)
            else:
                prob_z = (prob - np.mean(prob)) / prob_std

            site_preds[held_site] = (y_te, prob)
            all_true.extend(y_te.tolist())
            all_prob.extend(prob.tolist())
            all_prob_zscored.extend(prob_z.tolist())

            n_te = len(y_te)
            n_asd = int(np.sum(y_te == 1))
            n_ctrl = int(np.sum(y_te == 0))
            site_auc = fast_auc(y_te, prob) if (n_asd > 0 and n_ctrl > 0) else float("nan")
            site_acc = float(accuracy_score(y_te, (prob >= 0.5).astype(int)))
            site_brier = float(brier_score_loss(y_te, prob))

            per_site_rows.append({
                "model": model_name,
                "site": held_site,
                "n_subjects": n_te,
                "n_asd": n_asd,
                "n_control": n_ctrl,
                "site_prevalence_asd": float(n_asd / n_te),
                "auroc": round(site_auc, 4) if not np.isnan(site_auc) else None,
                "accuracy": round(site_acc, 4),
                "brier_score": round(site_brier, 4),
            })

        all_true_arr = np.array(all_true)
        all_prob_arr = np.array(all_prob)
        all_prob_z_arr = np.array(all_prob_zscored)

        # 1. Pooled AUROC raw
        pooled_auc_raw = fast_auc(all_true_arr, all_prob_arr)
        # 2. Pooled AUROC after z-scoring within site
        pooled_auc_zscored = fast_auc(all_true_arr, all_prob_z_arr)

        # 3. Per-site statistics across included sites
        site_aucs = [row["auroc"] for row in per_site_rows if row["model"] == model_name and row["auroc"] is not None]
        site_weights = [row["n_subjects"] for row in per_site_rows if row["model"] == model_name and row["auroc"] is not None]

        unweighted_macro = float(np.mean(site_aucs))
        weighted_macro = float(np.average(site_aucs, weights=site_weights))
        median_auc = float(np.median(site_aucs))
        min_auc = float(np.min(site_aucs))
        max_auc = float(np.max(site_aucs))
        auc_range = [min_auc, max_auc]

        # 4. Bootstrap CIs resampling sites
        boot_ci_unw = site_resampling_bootstrap_ci(included_sites, site_preds, metric="macro_unweighted", n_boot=2000, seed=seed)
        boot_ci_w = site_resampling_bootstrap_ci(included_sites, site_preds, metric="macro_weighted", n_boot=2000, seed=seed)
        boot_ci_pool = site_resampling_bootstrap_ci(included_sites, site_preds, metric="pooled", n_boot=2000, seed=seed)

        # 5. Difference between pooled and within-site z-scored
        between_site_shift_contribution = float(pooled_auc_raw - pooled_auc_zscored)

        print(f"  Pooled AUROC (Raw): {pooled_auc_raw:.4f}")
        print(f"  Pooled AUROC (Within-site Z-scored): {pooled_auc_zscored:.4f} (Difference: {between_site_shift_contribution:+.4f})")
        print(f"  Macro AUROC Unweighted: {unweighted_macro:.4f} [95% Site-Resample CI: {boot_ci_unw[0]:.4f}, {boot_ci_unw[1]:.4f}]")
        print(f"  Macro AUROC N-Weighted: {weighted_macro:.4f} [95% Site-Resample CI: {boot_ci_w[0]:.4f}, {boot_ci_w[1]:.4f}]")
        print(f"  Median AUROC: {median_auc:.4f} | Range: [{min_auc:.4f}, {max_auc:.4f}]")

        summary_model_records[model_name] = {
            "model": model_name,
            "pooled_auroc_raw": pooled_auc_raw,
            "pooled_auroc_zscored": pooled_auc_zscored,
            "site_prevalence_oracle_auroc": site_prev_oracle_auroc,
            "between_site_shift_contribution": between_site_shift_contribution,
            "macro_auroc_unweighted": unweighted_macro,
            "macro_auroc_unweighted_ci_95": boot_ci_unw,
            "macro_auroc_n_weighted": weighted_macro,
            "macro_auroc_n_weighted_ci_95": boot_ci_w,
            "median_auroc": median_auc,
            "min_auroc": min_auc,
            "max_auroc": max_auc,
            "site_resample_pooled_ci_95": boot_ci_pool,
            "excluded_sites": excluded_sites,
            "n_included_sites": len(included_sites),
        }

    # Save to results/per_site_auroc.csv
    df_per_site = pd.DataFrame(per_site_rows)
    csv_path = RESULTS_DIR / "per_site_auroc.csv"
    df_per_site.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved per-site detailed table to {csv_path} ({len(df_per_site)} rows).")

    # Save structured summary to results/per_site_summary.json
    summary_path = RESULTS_DIR / "per_site_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "site_prevalence_oracle_baseline_auroc": site_prev_oracle_auroc,
            "excluded_sites": excluded_sites,
            "n_total_sites": len(unique_sites),
            "n_included_sites": len(included_sites),
            "models_summary": summary_model_records,
        }, f, indent=2)
    print(f"[OK] Saved per-site model summary to {summary_path}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run_per_site_benchmark()
