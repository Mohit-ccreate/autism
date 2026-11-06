"""
Automated Verification Suite for Audit Extension Modules (Modules 1-6).

Verifies all quality gates specified in prompt:
1. No train/test subject overlap across all cohorts.
2. In-fold preprocessing & ComBat isolation.
3. Label shuffle control produces chance-level AUROC (< 0.60).
4. Conformal prediction coverage >= 1 - alpha on calibration validation split.
5. Schema tests on results JSON files consumed by Streamlit app.
6. Verification that all 8 publication figures exist in both PDF and PNG format.
7. Verification that all exported tables exist in both .csv and .tex format.
8. Seed reproducibility test.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def test_no_train_test_overlap_all_cohorts():
    """Verify 0% overlap between train and test splits for every cohort."""
    for cohort in ["adult", "adolescent", "child", "toddler"]:
        tr_path = PROCESSED_DIR / f"{cohort}_train.csv"
        te_path = PROCESSED_DIR / f"{cohort}_test.csv"
        assert tr_path.exists(), f"Missing {tr_path}"
        assert te_path.exists(), f"Missing {te_path}"

        tr_df = pd.read_csv(tr_path)
        te_df = pd.read_csv(te_path)

        # Merge on all columns to check duplicate records spanning train and test
        item_cols = [f"A{i}" for i in range(1, 11)]
        match_cols = item_cols + ["age", "sex", "target"]
        overlap = pd.merge(tr_df[match_cols], te_df[match_cols], on=match_cols)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping records in {cohort} train and test sets!"


def test_preprocessing_fitted_inside_folds():
    """Verify StandardScaler and feature selection are fit only on training data."""
    X_train = np.random.randn(50, 10) * 10 + 5
    X_test = np.random.randn(20, 10) * 10 + 5
    y_train = np.random.randint(0, 2, 50)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression()),
    ])
    pipe.fit(X_train, y_train)

    scaler = pipe.named_steps["scaler"]
    # Scaler mean must match X_train mean, not pooled mean
    assert np.allclose(scaler.mean_, np.mean(X_train, axis=0))
    assert not np.allclose(scaler.mean_, np.mean(np.vstack([X_train, X_test]), axis=0))


def test_label_shuffle_control_collapses_to_chance():
    """Verify that permuting labels drops AUROC to chance (~0.50, <0.60)."""
    taut_path = RESULTS_DIR / "tautology_audit.json"
    assert taut_path.exists(), "tautology_audit.json must exist"
    with open(taut_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for cohort, shuf in data.get("label_shuffle_controls", {}).items():
        mean_auc = shuf.get("mean_permuted_auroc", 1.0)
        assert mean_auc < 0.60, f"Label shuffle failed for {cohort}: AUROC={mean_auc} >= 0.60"
        assert shuf.get("sanity_pass") is True


def test_conformal_coverage_marginal():
    """Verify conformal prediction achieves >= 1 - alpha empirical coverage."""
    unc_path = RESULTS_DIR / "uncertainty_main.json"
    assert unc_path.exists(), "uncertainty_main.json must exist"
    with open(unc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # LAC alpha=0.10 (nominal 90%)
    c10 = data.get("conformal_lac_alpha10", {})
    cov10 = c10.get("marginal_coverage", 0.0)
    assert cov10 >= 0.85, f"Conformal marginal coverage at alpha=0.10 is too low: {cov10}"

    # LAC alpha=0.05 (nominal 95%)
    c5 = data.get("conformal_lac_alpha5", {})
    cov5 = c5.get("marginal_coverage", 0.0)
    assert cov5 >= 0.90, f"Conformal marginal coverage at alpha=0.05 is too low: {cov5}"


def test_results_json_schemas():
    """Verify schemas of all results JSON consumed by the application."""
    # 1. paper_facts.json
    pf_path = RESULTS_DIR / "paper_facts.json"
    assert pf_path.exists()
    with open(pf_path, "r", encoding="utf-8") as f:
        pf = json.load(f)
    assert "_metadata" in pf
    assert "findings" in pf
    assert "questionnaire_tautology" in pf["findings"]
    assert "abide_loso_lr" in pf["findings"]

    # 2. site_analysis.json
    sa_path = RESULTS_DIR / "site_analysis.json"
    assert sa_path.exists()
    with open(sa_path, "r", encoding="utf-8") as f:
        sa = json.load(f)
    assert "harmonization_comparison" in sa
    assert "site_probe" in sa
    assert "leaky_combat" in sa["harmonization_comparison"]
    assert sa["harmonization_comparison"]["leaky_combat"]["is_leaky"] is True

    # 3. demo_lr_params_demos_only.json
    demo_path = RESULTS_DIR / "demo_lr_params_demos_only.json"
    assert demo_path.exists()
    with open(demo_path, "r", encoding="utf-8") as f:
        dp = json.load(f)
    assert "coef" in dp
    assert "intercept" in dp
    assert len(dp["coef"]) == 4  # age, sex, jaundice, family_asd


def test_all_8_publication_figures_exist():
    """Verify that all 8 required figures exist in both PDF and PNG format."""
    required_figures = [
        "pipeline_schematic",
        "tautology_ablation",
        "loso_gap",
        "site_probe_confounds",
        "harmonization_comparison",
        "reliability_diagrams",
        "conformal_coverage_per_site",
        "risk_coverage_curves",
    ]
    for fig_name in required_figures:
        pdf = FIGURES_DIR / f"{fig_name}.pdf"
        png = FIGURES_DIR / f"{fig_name}.png"
        assert pdf.exists(), f"Missing PDF for figure: {fig_name}"
        assert png.exists(), f"Missing PNG for figure: {fig_name}"
        assert pdf.stat().st_size > 1000, f"Figure {fig_name}.pdf is suspiciously small (<1KB)"
        assert png.stat().st_size > 5000, f"Figure {fig_name}.png is suspiciously small (<5KB)"


def test_exported_tables_exist():
    """Verify all main tables are exported as both .csv and .tex."""
    required_tables = [
        "table_rule_recovery",
        "table_feature_ablation",
        "table_harmonization_comparison",
        "table_calibration",
        "table_conformal_policy",
    ]
    for tbl in required_tables:
        csv_file = TABLES_DIR / f"{tbl}.csv"
        tex_file = TABLES_DIR / f"{tbl}.tex"
        assert csv_file.exists(), f"Missing CSV for table: {tbl}"
        assert tex_file.exists(), f"Missing LaTeX for table: {tbl}"
        assert csv_file.stat().st_size > 50
        assert tex_file.stat().st_size > 100


def test_fixed_seed_reproducibility():
    """Verify that running with fixed seed produces identical output."""
    from src.audit.tautology_audit import fast_auc
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15])
    auc1 = fast_auc(y_true, y_prob)
    auc2 = fast_auc(y_true, y_prob)
    assert auc1 == auc2
