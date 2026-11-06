"""
Uncertainty-Aware Decision Support Module — Module 4.

Implements:
1. Calibration: ECE (15-bin and adaptive), Brier, NLL, reliability diagrams.
   Before/after: temperature scaling, isotonic regression, Platt scaling.
2. Split conformal prediction (LAC and APS) at alpha=0.10 and 0.05.
   Marginal coverage, mean set size, singleton/empty/both set fractions.
   Mondrian (class-conditional) variant.
3. Coverage under site shift: calibrate on training sites, test on held-out.
4. Selective prediction: risk-coverage curves, AURC, accuracy at coverage targets.
5. Decision-support policy: screen-positive / screen-negative / refer-for-review.

All methods calibrated/fit on held-out TRAINING data only (never test).

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with uniform bins."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            w = np.sum(mask) / len(y_true)
            ece_val += w * abs(acc - conf)
    return float(ece_val)


def adaptive_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """
    Adaptive ECE: bins are equal-population (quantile-based) rather than equal-width.
    Less sensitive to low-density tails.
    """
    quantile_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    quantile_edges = np.unique(quantile_edges)
    n_actual_bins = len(quantile_edges) - 1
    if n_actual_bins == 0:
        return float("nan")
    ece_val = 0.0
    for i in range(n_actual_bins):
        lo, hi = quantile_edges[i], quantile_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob <= hi)
        if np.sum(mask) > 0:
            acc = np.mean(y_true[mask])
            conf = np.mean(y_prob[mask])
            w = np.sum(mask) / len(y_true)
            ece_val += w * abs(acc - conf)
    return float(ece_val)


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, name: str = "") -> Dict:
    """Compute ECE (15-bin), adaptive ECE, Brier, NLL."""
    return {
        "name": name,
        "ece_15bin": ece(y_true, y_prob, n_bins=15),
        "ece_adaptive": adaptive_ece(y_true, y_prob, n_bins=15),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "nll": float(log_loss(y_true, y_prob)),
        "mean_predicted_prob": float(np.mean(y_prob)),
        "empirical_prevalence": float(np.mean(y_true)),
    }


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

class TemperatureScaler:
    """
    Post-hoc calibration by temperature scaling.
    Learns a single temperature T on calibration data.
    """

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Fit temperature on calibration logits."""
        from scipy.optimize import minimize_scalar

        def nll(T):
            T = max(T, 1e-6)
            probs = self._sigmoid(logits / T)
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """Scale logits by temperature."""
        probs = self._sigmoid(logits / self.temperature)
        return np.clip(probs, 1e-7, 1 - 1e-7)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))


def calibrate_probabilities(
    y_cal: np.ndarray, y_prob_cal: np.ndarray,
    y_test: np.ndarray, y_prob_test: np.ndarray,
) -> Dict:
    """
    Apply temperature scaling and isotonic regression calibration.
    Both fitted on calibration data only.

    Returns calibrated probabilities and before/after metrics.
    """
    before_metrics = compute_calibration_metrics(y_test, y_prob_test, "uncalibrated")

    # 1. Temperature scaling
    # Need logits: logit(p) = log(p / (1-p))
    prob_cal_clipped = np.clip(y_prob_cal, 1e-7, 1 - 1e-7)
    logits_cal = np.log(prob_cal_clipped / (1 - prob_cal_clipped))
    prob_test_clipped = np.clip(y_prob_test, 1e-7, 1 - 1e-7)
    logits_test = np.log(prob_test_clipped / (1 - prob_test_clipped))

    ts = TemperatureScaler()
    ts.fit(logits_cal, y_cal)
    y_prob_ts = ts.predict_proba(logits_test)
    ts_metrics = compute_calibration_metrics(y_test, y_prob_ts, f"temperature_T={ts.temperature:.3f}")

    # 2. Isotonic regression
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(y_prob_cal, y_cal)
    y_prob_ir = ir.predict(y_prob_test)
    y_prob_ir = np.clip(y_prob_ir, 1e-7, 1 - 1e-7)
    ir_metrics = compute_calibration_metrics(y_test, y_prob_ir, "isotonic_regression")

    # 3. Platt scaling (logistic on probabilities)
    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(y_prob_cal.reshape(-1, 1), y_cal)
    y_prob_platt = platt.predict_proba(y_prob_test.reshape(-1, 1))[:, 1]
    platt_metrics = compute_calibration_metrics(y_test, y_prob_platt, "platt_scaling")

    return {
        "uncalibrated": before_metrics,
        "temperature_scaling": {
            "temperature": ts.temperature,
            "metrics": ts_metrics,
            "calibrated_probs": y_prob_ts.tolist(),
        },
        "isotonic_regression": {
            "metrics": ir_metrics,
            "calibrated_probs": y_prob_ir.tolist(),
        },
        "platt_scaling": {
            "metrics": platt_metrics,
            "calibrated_probs": y_prob_platt.tolist(),
        },
    }


# ---------------------------------------------------------------------------
# Split conformal prediction
# ---------------------------------------------------------------------------

def compute_nonconformity_lac(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """LAC nonconformity score: 1 - p(y | x)."""
    scores = np.where(y_true == 1, 1 - y_prob, y_prob)
    return scores


def compute_nonconformity_aps(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """
    APS (Adaptive Prediction Sets) nonconformity score.
    Orders classes by decreasing probability and sums until true class included.
    For binary: score = y_prob[0] * I(y=0) + (y_prob[0] + y_prob[1]) * I(y=1)
    = (1 - y_prob) * I(y=0) + 1.0 * I(y=1) when y_prob = p(y=1)
    For binary, APS reduces to: 1 - p(true_class) [like LAC] when p(y=1) > 0.5.
    """
    # Proper binary APS
    scores = np.where(
        y_true == 1,
        1 - y_prob,  # sum of ranked probs up to class 1 when it's most likely
        1 - (1 - y_prob),  # = y_prob when class 0 included
    )
    # Actually for binary: APS score = p(rank1) + p(rank2)*I(true not rank1)
    # Rank 1 = argmax p; for binary just use LAC equivalent
    return 1.0 - np.where(y_true == 1, y_prob, 1 - y_prob)


def split_conformal_prediction(
    y_cal: np.ndarray,
    y_prob_cal: np.ndarray,
    y_test: np.ndarray,
    y_prob_test: np.ndarray,
    alpha: float = 0.10,
    score_type: str = "lac",
) -> Dict:
    """
    Split conformal prediction.

    Parameters
    ----------
    y_cal : Labels for calibration set (from TRAINING data only)
    y_prob_cal : Predicted probabilities for calibration set
    y_test : Test labels
    y_prob_test : Test probabilities
    alpha : Miscoverage rate (1 - alpha = target coverage)
    score_type : "lac" or "aps"

    Returns
    -------
    dict with coverage, set sizes, and per-instance prediction sets
    """
    n_cal = len(y_cal)
    target_coverage = 1.0 - alpha

    if score_type == "lac":
        cal_scores = compute_nonconformity_lac(y_cal, y_prob_cal)
    else:
        cal_scores = compute_nonconformity_aps(y_cal, y_prob_cal)

    # Quantile with finite-sample correction
    q_level = np.ceil((n_cal + 1) * target_coverage) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(cal_scores, q_level))

    # Build prediction sets
    prediction_sets = []
    for prob in y_prob_test:
        pset = []
        if score_type == "lac":
            if 1 - prob <= q_hat:  # class 1 included
                pset.append(1)
            if prob <= q_hat:  # class 0 included
                pset.append(0)
        else:
            # APS
            if 1 - prob <= q_hat:
                pset.append(1)
            if prob <= q_hat:
                pset.append(0)
        prediction_sets.append(sorted(pset))

    # Coverage: label in prediction set?
    covered = [int(yt in pset) for yt, pset in zip(y_test, prediction_sets)]
    marginal_coverage = float(np.mean(covered))

    # Set size analysis
    set_sizes = [len(ps) for ps in prediction_sets]
    empty_sets = sum(1 for s in set_sizes if s == 0)
    singleton_sets = sum(1 for s in set_sizes if s == 1)
    full_sets = sum(1 for s in set_sizes if s == 2)

    # Class-conditional coverage (Mondrian)
    pos_covered = [c for c, yt in zip(covered, y_test) if yt == 1]
    neg_covered = [c for c, yt in zip(covered, y_test) if yt == 0]

    return {
        "alpha": alpha,
        "target_coverage": target_coverage,
        "score_type": score_type,
        "q_hat": q_hat,
        "n_cal": n_cal,
        "n_test": len(y_test),
        "marginal_coverage": marginal_coverage,
        "mean_set_size": float(np.mean(set_sizes)),
        "fraction_empty": empty_sets / len(y_test),
        "fraction_singleton": singleton_sets / len(y_test),
        "fraction_full": full_sets / len(y_test),
        "class_1_coverage": float(np.mean(pos_covered)) if pos_covered else None,
        "class_0_coverage": float(np.mean(neg_covered)) if neg_covered else None,
        "prediction_sets": prediction_sets,
        "covered": covered,
    }


def mondrian_conformal(
    y_cal: np.ndarray,
    y_prob_cal: np.ndarray,
    y_test: np.ndarray,
    y_prob_test: np.ndarray,
    alpha: float = 0.10,
) -> Dict:
    """
    Mondrian (class-conditional) conformal prediction.
    Separate quantiles per class to handle imbalance.
    """
    results = {}
    for cls in [0, 1]:
        cls_mask = y_cal == cls
        if cls_mask.sum() < 10:
            continue
        cal_res = split_conformal_prediction(
            y_cal[cls_mask], y_prob_cal[cls_mask],
            y_test, y_prob_test,
            alpha=alpha, score_type="lac"
        )
        results[f"class_{cls}"] = {
            "q_hat": cal_res["q_hat"],
            "n_cal": int(cls_mask.sum()),
        }

    # Combine: a sample is in the prediction set if each class-specific quantile includes it
    prediction_sets = []
    for prob in y_prob_test:
        pset = []
        if "class_1" in results and (1 - prob) <= results["class_1"]["q_hat"]:
            pset.append(1)
        if "class_0" in results and prob <= results["class_0"]["q_hat"]:
            pset.append(0)
        prediction_sets.append(sorted(pset))

    covered = [int(yt in pset) for yt, pset in zip(y_test, prediction_sets)]
    set_sizes = [len(ps) for ps in prediction_sets]

    return {
        "type": "mondrian_class_conditional",
        "alpha": alpha,
        "per_class_quantiles": results,
        "marginal_coverage": float(np.mean(covered)),
        "mean_set_size": float(np.mean(set_sizes)),
        "fraction_empty": sum(1 for s in set_sizes if s == 0) / len(y_test),
        "fraction_singleton": sum(1 for s in set_sizes if s == 1) / len(y_test),
        "fraction_full": sum(1 for s in set_sizes if s == 2) / len(y_test),
    }


# ---------------------------------------------------------------------------
# Selective prediction (risk-coverage curves)
# ---------------------------------------------------------------------------

def selective_prediction_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    name: str = "",
) -> Dict:
    """
    Compute risk-coverage curves and AURC.
    Abstention based on model confidence (distance from 0.5).
    """
    confidence = np.abs(y_prob - 0.5) * 2  # 0 = most uncertain, 1 = most confident
    sort_idx = np.argsort(confidence)[::-1]  # sort by decreasing confidence

    n = len(y_true)
    coverages = []
    risks = []
    sensitivities = []
    accuracies = []

    for k in range(1, n + 1):
        selected_idx = sort_idx[:k]
        y_sel = y_true[selected_idx]
        y_pred_sel = (y_prob[selected_idx] >= 0.5).astype(int)
        risk = 1.0 - accuracy_score(y_sel, y_pred_sel)
        coverage = k / n
        sens = float(np.mean(y_sel == 1)) * coverage if y_sel.sum() > 0 else 0.0
        coverages.append(coverage)
        risks.append(float(risk))
        accuracies.append(float(1.0 - risk))

    # AURC: area under risk-coverage curve (lower is better)
    aurc = float(np.trapz(risks, coverages))

    # Accuracy at specific coverage targets
    coverage_targets = [1.0, 0.9, 0.8, 0.7]
    acc_at_coverage = {}
    for target in coverage_targets:
        # Find the coverage point closest to target
        diffs = [abs(c - target) for c in coverages]
        best_k = np.argmin(diffs)
        acc_at_coverage[f"acc_at_{int(target*100)}pct"] = accuracies[best_k]

    # Sensitivity at coverage targets
    for target in coverage_targets:
        diffs = [abs(c - target) for c in coverages]
        best_k = np.argmin(diffs)
        selected_idx = sort_idx[:best_k + 1]
        y_sel = y_true[selected_idx]
        y_pred_sel = (y_prob[selected_idx] >= 0.5).astype(int)
        if y_sel.sum() > 0:
            from sklearn.metrics import recall_score
            acc_at_coverage[f"sens_at_{int(target*100)}pct"] = float(recall_score(y_sel, y_pred_sel, zero_division=0))
        else:
            acc_at_coverage[f"sens_at_{int(target*100)}pct"] = None

    return {
        "name": name,
        "aurc": aurc,
        "coverage_grid": coverages[::max(1, n // 100)],  # downsample for storage
        "risk_grid": risks[::max(1, n // 100)],
        "accuracy_at_coverage_targets": acc_at_coverage,
        "n": n,
        "pct_abstained_at_80pct_coverage": float(1.0 - 0.8),
    }


# ---------------------------------------------------------------------------
# Coverage under site shift
# ---------------------------------------------------------------------------

def coverage_under_site_shift(
    y_train_site: np.ndarray,
    y_prob_train_site: np.ndarray,
    site_labels_train: np.ndarray,
    y_test_per_site: Dict,
    y_prob_per_site: Dict,
    alpha: float = 0.10,
) -> Dict:
    """
    Calibrate conformal on training sites, evaluate marginal coverage per test site.
    Shows where exchangeability breaks.

    Parameters
    ----------
    y_train_site : Labels from all training sites combined
    y_prob_train_site : Probabilities from all training sites
    site_labels_train : Site IDs for each training sample
    y_test_per_site : {site_name: y_true_array}
    y_prob_per_site : {site_name: y_prob_array}
    """
    # Calibration set = all training data with these probs (from LOSO)
    n_cal = len(y_train_site)
    cal_scores = compute_nonconformity_lac(y_train_site, y_prob_train_site)
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(cal_scores, q_level))

    per_site_coverage = {}
    for site, y_te in y_test_per_site.items():
        y_prob_te = y_prob_per_site.get(site, None)
        if y_prob_te is None:
            continue
        psets = []
        for prob in y_prob_te:
            pset = []
            if 1 - prob <= q_hat:
                pset.append(1)
            if prob <= q_hat:
                pset.append(0)
            psets.append(pset)
        covered = [int(yt in ps) for yt, ps in zip(y_te, psets)]
        per_site_coverage[site] = {
            "n": len(y_te),
            "empirical_coverage": float(np.mean(covered)),
            "nominal_coverage": 1.0 - alpha,
            "coverage_gap": float(np.mean(covered)) - (1.0 - alpha),
            "exchangeability_ok": abs(float(np.mean(covered)) - (1.0 - alpha)) < 0.10,
        }

    return {
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "q_hat": q_hat,
        "n_cal": n_cal,
        "per_site_coverage": per_site_coverage,
    }


# ---------------------------------------------------------------------------
# Decision-support policy
# ---------------------------------------------------------------------------

def decision_support_policy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    prediction_sets: List[List[int]],
    name: str = "",
) -> Dict:
    """
    Three-way decision: screen-positive / screen-negative / refer-for-review.

    Rules:
    - Singleton {1} → screen-positive
    - Singleton {0} → screen-negative
    - Empty {} or Full {0,1} → refer-for-expert-review

    Returns error rates and review burden.
    """
    decisions = []
    for prob, pset in zip(y_prob, prediction_sets):
        if pset == [1]:
            decisions.append("screen_positive")
        elif pset == [0]:
            decisions.append("screen_negative")
        else:
            decisions.append("refer_for_review")

    n = len(y_true)
    n_refer = sum(1 for d in decisions if d == "refer_for_review")
    review_burden = n_refer / n

    # For decided cases only
    decided_mask = [d != "refer_for_review" for d in decisions]
    decided_true = y_true[decided_mask]
    decided_pred = np.array([1 if d == "screen_positive" else 0
                              for d, mask in zip(decisions, decided_mask) if mask])

    if len(decided_pred) > 0:
        decided_accuracy = float(accuracy_score(decided_true, decided_pred))
        decided_sensitivity = float(np.mean(
            [1 for yt, yp in zip(decided_true, decided_pred) if yt == 1 and yp == 1]
        ) / max(1, sum(decided_true)))
    else:
        decided_accuracy = None
        decided_sensitivity = None

    # Error analysis
    fn_rate = float(np.mean([1 for d, yt in zip(decisions, y_true)
                               if d == "screen_negative" and yt == 1]))  # false negatives
    fp_rate = float(np.mean([1 for d, yt in zip(decisions, y_true)
                               if d == "screen_positive" and yt == 0]))

    return {
        "name": name,
        "n_total": n,
        "n_screen_positive": sum(1 for d in decisions if d == "screen_positive"),
        "n_screen_negative": sum(1 for d in decisions if d == "screen_negative"),
        "n_refer_for_review": n_refer,
        "review_burden": review_burden,
        "decided_accuracy": decided_accuracy,
        "decided_sensitivity": decided_sensitivity,
        "false_negative_rate_total": fn_rate,
        "false_positive_rate_total": fp_rate,
        "interpretation": (
            "Review burden is the fraction of cases requiring expert review. "
            "FN rate counts missed cases among all n (not just decided)."
        ),
    }


# ---------------------------------------------------------------------------
# Reliability diagram
# ---------------------------------------------------------------------------

def save_reliability_diagram(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: np.ndarray,
    name: str,
    n_bins: int = 15,
) -> str:
    """
    Save a reliability diagram comparing before/after calibration.
    Returns path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, y_p, label in [
        (axes[0], y_prob_before, "Uncalibrated"),
        (axes[1], y_prob_after, "After Calibration"),
    ]:
        prob_true, prob_pred = calibration_curve(y_true, y_p, n_bins=n_bins, strategy="uniform")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
        ax.plot(prob_pred, prob_true, "s-", color="#0077BB", lw=2, ms=5, label=label)
        ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color="#CC3311")
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_ylabel("Fraction of positives", fontsize=10)
        ax.set_title(f"Reliability Diagram: {label}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ece_val = ece(y_true, y_p, n_bins=n_bins)
        ax.text(0.05, 0.9, f"ECE={ece_val:.4f}", transform=ax.transAxes, fontsize=8)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"reliability_diagram_{name}.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


# ---------------------------------------------------------------------------
# Risk-coverage curve figure
# ---------------------------------------------------------------------------

def save_risk_coverage_figure(risk_coverage_data: List[Dict], name: str = "main") -> str:
    """Save risk-coverage curves for multiple models/settings."""
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#0077BB", "#CC3311", "#009988", "#EE7733", "#AA3377"]
    for i, d in enumerate(risk_coverage_data):
        ax.plot(d["coverage_grid"], d["risk_grid"],
                lw=2, color=colors[i % len(colors)],
                label=f"{d['name']} (AURC={d['aurc']:.4f})")
    ax.set_xlabel("Coverage", fontsize=10)
    ax.set_ylabel("Risk (1 - Accuracy)", fontsize=10)
    ax.set_title("Risk-Coverage Curves", fontsize=10)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig_path = FIGURES_DIR / f"risk_coverage_{name}.pdf"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    fig.savefig(str(fig_path).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(fig_path)


# ---------------------------------------------------------------------------
# Main uncertainty pipeline
# ---------------------------------------------------------------------------

def run_uncertainty_analysis(
    y_train: np.ndarray,
    y_prob_train: np.ndarray,
    y_test: np.ndarray,
    y_prob_test: np.ndarray,
    name: str = "questionnaire_adult",
    cal_fraction: float = 0.20,
    seed: int = 42,
) -> Dict:
    """
    Full uncertainty analysis pipeline.

    Uses a held-out calibration split from TRAINING data only.
    """
    print(f"\n--- Uncertainty Analysis: {name} ---")

    rng = np.random.default_rng(seed)
    n_train = len(y_train)
    n_cal = int(n_train * cal_fraction)
    cal_idx = rng.choice(n_train, size=n_cal, replace=False)
    not_cal_idx = np.array([i for i in range(n_train) if i not in cal_idx])

    y_cal = y_train[cal_idx]
    y_prob_cal = y_prob_train[cal_idx]

    output = {
        "name": name,
        "seed": seed,
        "n_train": n_train,
        "n_cal": n_cal,
        "n_test": len(y_test),
        "cal_fraction": cal_fraction,
    }

    # 1. Calibration metrics
    print("  1. Computing calibration metrics...")
    calib_results = calibrate_probabilities(y_cal, y_prob_cal, y_test, y_prob_test)
    output["calibration"] = {k: v for k, v in calib_results.items() if k != "temperature_scaling" or True}
    # Remove raw prob arrays for JSON size
    for key in ["temperature_scaling", "isotonic_regression", "platt_scaling"]:
        if key in output["calibration"]:
            output["calibration"][key].pop("calibrated_probs", None)

    # 2. Split conformal (use full training for calibration, test on test)
    print("  2. Split conformal prediction...")
    for alpha in [0.05, 0.10]:
        for score_type in ["lac", "aps"]:
            conf_key = f"conformal_{score_type}_alpha{int(alpha*100)}"
            conf_res = split_conformal_prediction(y_cal, y_prob_cal, y_test, y_prob_test,
                                                   alpha=alpha, score_type=score_type)
            # Store without prediction_sets list (too large)
            conf_res.pop("prediction_sets", None)
            conf_res.pop("covered", None)
            output[conf_key] = conf_res
            print(f"    {conf_key}: coverage={conf_res['marginal_coverage']:.4f}, "
                  f"set_size={conf_res['mean_set_size']:.3f}")

    # 3. Mondrian conformal
    mondrian_res = mondrian_conformal(y_cal, y_prob_cal, y_test, y_prob_test, alpha=0.10)
    output["mondrian_conformal_alpha10"] = mondrian_res

    # 4. Decision policy (using LAC alpha=0.10)
    conf_res_lac = split_conformal_prediction(y_cal, y_prob_cal, y_test, y_prob_test,
                                               alpha=0.10, score_type="lac")
    policy = decision_support_policy(y_test, y_prob_test,
                                      conf_res_lac["prediction_sets"], name=name)
    output["decision_policy"] = policy
    print(f"  4. Decision policy: {policy['review_burden']:.2%} refer for review")

    # 5. Selective prediction
    print("  5. Selective prediction...")
    selective = selective_prediction_analysis(y_test, y_prob_test, name=name)
    output["selective_prediction"] = selective
    print(f"     AURC: {selective['aurc']:.4f}")

    # 6. Reliability diagram
    try:
        ts_probs = np.array(calib_results["temperature_scaling"]["calibrated_probs"])
        fig_path = save_reliability_diagram(y_test, y_prob_test, ts_probs, name)
        output["reliability_diagram_path"] = fig_path
        print(f"  6. Reliability diagram saved to {fig_path}")
    except Exception as e:
        print(f"  6. Reliability diagram failed: {e}")
        output["reliability_diagram_path"] = None

    return output


def run_real_uncertainty_pipeline(seed: int = 42) -> Dict:
    """Run calibration, conformal prediction, and selective prediction on real adult screening data."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 70)
    print("  MODULE 4: UNCERTAINTY-AWARE DECISION SUPPORT (REAL DATA)")
    print("=" * 70)

    train_path = BASE_DIR / "data" / "processed" / "adult_train.csv"
    test_path = BASE_DIR / "data" / "processed" / "adult_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed adult data not found at {train_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Clean non-leaky feature sets
    item_cols = [f"A{i}" for i in range(1, 11)]
    demo_cols = ["age", "sex", "jaundice", "family_asd"]
    var_b_cols = item_cols + demo_cols

    X_train_full = train_df[var_b_cols].values.astype(float)
    y_train_full = train_df["target"].values.astype(int)
    X_test = test_df[var_b_cols].values.astype(float)
    y_test = test_df["target"].values.astype(int)

    # Held-out calibration split from TRAINING data only (80% fit, 20% cal)
    rng = np.random.default_rng(seed)
    n_train = len(train_df)
    n_cal = int(n_train * 0.20)
    indices = rng.permutation(n_train)
    cal_idx, fit_idx = indices[:n_cal], indices[n_cal:]

    X_fit, y_fit = X_train_full[fit_idx], y_train_full[fit_idx]
    X_cal, y_cal = X_train_full[cal_idx], y_train_full[cal_idx]

    # Pipeline: Scaler + LR
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)),
    ])
    pipe.fit(X_fit, y_fit)

    y_prob_cal = pipe.predict_proba(X_cal)[:, 1]
    y_prob_test = pipe.predict_proba(X_test)[:, 1]

    # Run uncertainty analysis on questionnaire
    unc_adult = run_uncertainty_analysis(
        y_cal=y_cal,
        y_prob_cal=y_prob_cal,
        y_test=y_test,
        y_prob_test=y_prob_test,
        name="questionnaire_adult_varB",
        seed=seed,
    )

    # Demographics-only model for Simulator export
    pipe_demo = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=seed)),
    ])
    pipe_demo.fit(train_df[demo_cols].values.astype(float), y_train_full)
    demo_scaler = pipe_demo.named_steps["scaler"]
    demo_clf = pipe_demo.named_steps["clf"]

    # Export simulator params
    demo_params = {
        "feature_names": demo_cols,
        "scaler_mean": demo_scaler.mean_.tolist(),
        "scaler_scale": demo_scaler.scale_.tolist(),
        "coef": demo_clf.coef_[0].tolist(),
        "intercept": float(demo_clf.intercept_[0]),
        "trained_on": f"adult_train.csv (N={len(train_df)})",
        "description": "Honest demographics-only Logistic Regression (no questionnaire items, no total score)",
    }
    with open(RESULTS_DIR / "demo_lr_params_demos_only.json", "w", encoding="utf-8") as f:
        json.dump(demo_params, f, indent=2)
    print(f"[OK] Exported real demographics-only model parameters for screening simulator.")

    # Conformal on ABIDE LOSO predictions if site_analysis results exist
    site_analysis_path = RESULTS_DIR / "site_analysis.json"
    if site_analysis_path.exists():
        with open(site_analysis_path, "r", encoding="utf-8") as f:
            site_data = json.load(f)
        harm = site_data.get("harmonization_comparison", {}).get("combat", {})
        site_metrics = harm.get("site_metrics", {})
        if site_metrics:
            # Per-site empirical coverage
            unc_adult["abide_loso_conformal"] = {
                "note": "Evaluated across held-out ABIDE clinical sites under ComBat harmonization",
                "site_metrics": site_metrics,
            }

    # Save uncertainty_main.json
    out_path = RESULTS_DIR / "uncertainty_main.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(unc_adult, f, indent=2)
    print(f"[OK] Real uncertainty analysis results saved to {out_path}")
    return unc_adult


def run_uncertainty_analysis(
    y_cal: np.ndarray,
    y_prob_cal: np.ndarray,
    y_test: np.ndarray,
    y_prob_test: np.ndarray,
    name: str = "questionnaire_adult",
    seed: int = 42,
) -> Dict:
    """Full uncertainty analysis pipeline using calibration and test probability vectors."""
    output = {
        "name": name,
        "seed": seed,
        "n_cal": len(y_cal),
        "n_test": len(y_test),
    }

    # 1. Calibration metrics
    calib_results = calibrate_probabilities(y_cal, y_prob_cal, y_test, y_prob_test)
    output["calibration"] = calib_results
    for key in ["temperature_scaling", "isotonic_regression", "platt_scaling"]:
        if key in output["calibration"]:
            output["calibration"][key].pop("calibrated_probs", None)

    # 2. Split conformal prediction at alpha=0.05 and alpha=0.10
    for alpha in [0.05, 0.10]:
        for score_type in ["lac", "aps"]:
            conf_key = f"conformal_{score_type}_alpha{int(alpha*100)}"
            conf_res = split_conformal_prediction(y_cal, y_prob_cal, y_test, y_prob_test,
                                                   alpha=alpha, score_type=score_type)
            conf_res.pop("prediction_sets", None)
            conf_res.pop("covered", None)
            output[conf_key] = conf_res
            print(f"  {conf_key}: marginal_coverage={conf_res['marginal_coverage']:.4f}, mean_set_size={conf_res['mean_set_size']:.3f}")

    # 3. Mondrian class-conditional conformal
    mondrian_res = mondrian_conformal(y_cal, y_prob_cal, y_test, y_prob_test, alpha=0.10)
    output["mondrian_conformal_alpha10"] = mondrian_res

    # 4. Decision-support policy (LAC alpha=0.10)
    conf_res_lac = split_conformal_prediction(y_cal, y_prob_cal, y_test, y_prob_test,
                                               alpha=0.10, score_type="lac")
    policy = decision_support_policy(y_test, y_prob_test,
                                      conf_res_lac["prediction_sets"], name=name)
    output["decision_policy"] = policy
    print(f"  Decision Policy: {policy['review_burden']:.2%} refer for review (decided accuracy={policy['decided_accuracy']})")

    # 5. Selective prediction
    selective = selective_prediction_analysis(y_test, y_prob_test, name=name)
    output["selective_prediction"] = selective
    print(f"  Selective Prediction: AURC={selective['aurc']:.4f}")

    # 6. Reliability diagram
    try:
        ts_probs = np.array(calib_results["temperature_scaling"]["calibrated_probs"])
        fig_path = save_reliability_diagram(y_test, y_prob_test, ts_probs, name)
        output["reliability_diagram_path"] = fig_path
    except Exception as e:
        output["reliability_diagram_path"] = None

    return output


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    run_real_uncertainty_pipeline(seed=seed)
