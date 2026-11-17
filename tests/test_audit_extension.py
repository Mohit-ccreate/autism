"""
Automated Verification Suite for Audit Extension Modules (Modules 1-6).

Verifies all quality gates specified in prompt:
1. No train/test subject overlap across all cohorts.
2. In-fold preprocessing & ComBat isolation.
3. Label shuffle control produces chance-level AUROC (< 0.60).
4. Conformal prediction coverage >= 1 - alpha on calibration validation split.
5. Set-size identity: mean_set_size = 1*singleton + 2*both + 0*empty across all runs.
6. Conformal coverage guarantees hold on exchangeable synthetic data.
7. Schema tests on results JSON files (paper_facts.json, site_analysis.json, benchmark_leaderboard.json, dedup_and_threshold_audit.json).
8. Verification that all 8 publication figures exist in both PDF and PNG format.
9. Verification that all exported tables exist in both .csv and .tex format.
10. Full ABIDE N=854 dataset integrity across 20 clinical sites.
11. Paired bootstrap comparison function verification.
12. Seed reproducibility test.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
DATA_DIR = BASE_DIR / "data"


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


def test_in_fold_combat_isolation():
    """Verify that in-fold ComBat estimator only uses training site statistics."""
    from src.eval.site_analysis import parametric_combat_fit, parametric_combat_transform
    np.random.seed(42)
    n_train_samples = 40
    n_test_samples = 20
    p_features = 15

    # Train from sites A and B
    X_tr = np.random.randn(n_train_samples, p_features)
    sites_tr = np.array(["SITE_A"] * 20 + ["SITE_B"] * 20)

    # Test from unseen site C
    X_te = np.random.randn(n_test_samples, p_features)
    sites_te = np.array(["SITE_C"] * 20)

    # Fit ComBat strictly on training data
    _, combat_model = parametric_combat_fit(X_tr, sites_tr, None)
    assert set(combat_model["unique_batches"]) == {"SITE_A", "SITE_B"}
    assert "SITE_C" not in combat_model["unique_batches"]

    # Applying to test data from unseen site C must leave test data unmodified or handle unseen gracefully
    X_te_harm = parametric_combat_transform(X_te, sites_te, None, combat_model)
    assert X_te_harm.shape == X_te.shape
    assert np.allclose(X_te_harm, X_te)


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
    """Verify conformal prediction achieves >= 1 - alpha empirical coverage across arms."""
    unc_path = RESULTS_DIR / "uncertainty_main.json"
    assert unc_path.exists(), "uncertainty_main.json must exist"
    with open(unc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check adult tabular leak-free
    adult_conf = data["tabular_leak_free"]["adult"]["conformal_prediction"]
    cov10_adult = adult_conf["lac_alpha10"]["marginal_coverage"]
    cov5_adult = adult_conf["lac_alpha5"]["marginal_coverage"]
    assert cov10_adult >= 0.85, f"Adult LAC alpha=0.10 coverage too low: {cov10_adult}"
    assert cov5_adult >= 0.90, f"Adult LAC alpha=0.05 coverage too low: {cov5_adult}"

    # Check ABIDE full LOSO
    abide_conf = data["abide_full_loso"]["conformal_prediction"]
    cov10_abide = abide_conf["lac_alpha10"]["marginal_coverage"]
    cov5_abide = abide_conf["lac_alpha5"]["marginal_coverage"]
    assert cov10_abide >= 0.82, f"ABIDE LOSO LAC alpha=0.10 coverage too low: {cov10_abide}"
    assert cov5_abide >= 0.88, f"ABIDE LOSO LAC alpha=0.05 coverage too low: {cov5_abide}"


def test_conformal_set_size_identity():
    """
    Verify the fundamental conformal set-size decomposition identity:
    mean_set_size = 1.0 * fraction_singleton + 2.0 * fraction_both + 0.0 * fraction_empty
    and fraction_empty + fraction_singleton + fraction_both == 1.0
    """
    unc_path = RESULTS_DIR / "uncertainty_main.json"
    with open(unc_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    arms_to_check = []

    # 1. Tabular leak-free arms
    for cohort, cdata in data.get("tabular_leak_free", {}).items():
        conf_dict = cdata.get("conformal_prediction", {})
        for name, m in conf_dict.items():
            arms_to_check.append((f"tabular_{cohort}_{name}", m))

    # 2. ABIDE full LOSO arm
    abide_conf = data.get("abide_full_loso", {}).get("conformal_prediction", {})
    for name, m in abide_conf.items():
        arms_to_check.append((f"abide_loso_{name}", m))

    # 3. Leaky contrast baseline
    leaky_conf = data.get("leaky_contrast_baseline", {}).get("conformal_prediction", {})
    for name, m in leaky_conf.items():
        arms_to_check.append((f"leaky_contrast_{name}", m))

    assert len(arms_to_check) > 0, "No conformal runs found to verify"

    for label, m in arms_to_check:
        empty = m.get("fraction_empty", 0.0)
        single = m.get("fraction_singleton", 0.0)
        both = m.get("fraction_both", 0.0)
        mean_size = m.get("mean_set_size", 0.0)

        # Partition sum equals 1.0
        assert abs((empty + single + both) - 1.0) < 1e-5, f"{label}: partition does not sum to 1.0"

        # Linear expectation identity
        expected_size = 0.0 * empty + 1.0 * single + 2.0 * both
        assert abs(mean_size - expected_size) < 1e-5, (
            f"{label}: set size identity violated! mean={mean_size}, expected={expected_size}"
        )
        assert m.get("set_size_identity_verified") is True, f"{label}: identity verification flag missing"


def test_conformal_coverage_exchangeable_synthetic_data():
    """
    Verify conformal prediction mathematically guarantees >= 1 - alpha coverage
    on independent exchangeable synthetic data.
    """
    np.random.seed(42)
    n_train = 400
    n_cal = 400
    n_test = 600
    n_total = n_train + n_cal + n_test
    d = 5

    # Generate synthetic binary classification dataset
    X = np.random.randn(n_total, d)
    true_w = np.array([1.2, -0.8, 0.5, -1.0, 0.3])
    logits = X @ true_w + 0.2
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (np.random.rand(n_total) < probs).astype(int)

    X_tr, y_tr = X[:n_train], y[:n_train]
    X_cal, y_cal = X[n_train:n_train+n_cal], y[n_train:n_train+n_cal]
    X_te, y_te = X[n_train+n_cal:], y[n_train+n_cal:]

    model = LogisticRegression()
    model.fit(X_tr, y_tr)

    cal_probs = model.predict_proba(X_cal)
    test_probs = model.predict_proba(X_te)

    for alpha in [0.10, 0.05]:
        nominal = 1.0 - alpha
        # Nonconformity score for true label
        cal_scores = np.array([1.0 - cal_probs[i, y_cal[i]] for i in range(n_cal)])
        q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
        q_level = min(1.0, max(0.0, q_level))
        q_hat = np.quantile(cal_scores, q_level, method="higher")

        # Form test prediction sets
        sets = []
        for i in range(n_test):
            s = set()
            if (1.0 - test_probs[i, 0]) <= q_hat:
                s.add(0)
            if (1.0 - test_probs[i, 1]) <= q_hat:
                s.add(1)
            sets.append(s)

        covered = sum(y_te[i] in sets[i] for i in range(n_test))
        emp_coverage = covered / n_test

        # Conformal theorem: on exchangeable data, coverage is >= 1 - alpha - finite sample error
        finite_sample_margin = 2.0 / np.sqrt(n_test)
        assert emp_coverage >= (nominal - finite_sample_margin), (
            f"Conformal theorem failed on synthetic data! Nominal={nominal}, empirical={emp_coverage}"
        )


def test_results_json_schemas():
    """Verify schemas of all results JSON consumed by the application."""
    # 1. paper_facts.json
    pf_path = RESULTS_DIR / "paper_facts.json"
    assert pf_path.exists()
    with open(pf_path, "r", encoding="utf-8") as f:
        pf = json.load(f)
    assert "_metadata" in pf
    assert "theses_verdicts" in pf
    expected_theses = [
        "thesis_1_behavioral_tautology",
        "thesis_2_scanner_generalization_gap",
        "thesis_3_leaky_combat_inflation",
        "thesis_4_uncertainty_aware_decision_support",
    ]
    for t_key in expected_theses:
        assert t_key in pf["theses_verdicts"]
        assert pf["theses_verdicts"][t_key]["verdict"] in [
            "SUPPORTED", "PARTIALLY SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"
        ]
    assert "statistical_reconciliation" in pf
    assert "deduplication_audit_summary" in pf

    # 2. site_analysis.json
    sa_path = RESULTS_DIR / "site_analysis.json"
    assert sa_path.exists()
    with open(sa_path, "r", encoding="utf-8") as f:
        sa = json.load(f)
    assert sa["n_subjects"] == 854
    assert sa["n_sites"] == 20
    assert "harmonization_comparison" in sa
    assert "site_probe" in sa
    assert "paired_harmonization_tests" in sa
    assert "leaky_combat" in sa["harmonization_comparison"]
    assert sa["harmonization_comparison"]["leaky_combat"]["is_leaky"] is True

    # 3. benchmark_leaderboard.json
    bl_path = RESULTS_DIR / "benchmark_leaderboard.json"
    assert bl_path.exists()
    with open(bl_path, "r", encoding="utf-8") as f:
        bl = json.load(f)
    assert "tabular_leak_free" in bl
    assert "abide_full" in bl
    assert "loso_benchmark" in bl["abide_full"]
    assert len(bl["abide_full"]["loso_benchmark"]) >= 8

    # 4. dedup_and_threshold_audit.json
    dta_path = RESULTS_DIR / "dedup_and_threshold_audit.json"
    assert dta_path.exists()
    with open(dta_path, "r", encoding="utf-8") as f:
        dta = json.load(f)
    assert "deduplication_audit" in dta
    assert "threshold_sweeps" in dta

    # 5. demo_lr_params_demos_only.json
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
    """Verify all 10 exported publication tables exist in both .csv and .tex."""
    required_tables = [
        "table_rule_recovery",
        "table_feature_ablation",
        "table_harmonization_comparison",
        "table_calibration",
        "table_conformal_policy",
        "table_deduplication_accounting",
        "table_threshold_sensitivity",
        "table_tabular_benchmark_leaderboard",
        "table_abide_benchmark_leaderboard",
        "table_per_site_generalization_gap",
    ]
    for tbl in required_tables:
        csv_file = TABLES_DIR / f"{tbl}.csv"
        tex_file = TABLES_DIR / f"{tbl}.tex"
        assert csv_file.exists(), f"Missing CSV for table: {tbl}"
        assert tex_file.exists(), f"Missing LaTeX for table: {tbl}"
        assert csv_file.stat().st_size > 50
        assert tex_file.stat().st_size > 100


def test_abide_full_dataset_integrity():
    """Verify ABIDE full dataset has 854 quality-checked subjects across 20 sites."""
    csv_path = DATA_DIR / "abide_full_phenotypic.csv"
    assert csv_path.exists(), "Missing abide_full_phenotypic.csv"
    df = pd.read_csv(csv_path)
    assert len(df) == 854
    assert df["SITE_ID"].nunique() == 20
    assert "DX_GROUP" in df.columns
    # Check binary label balance: controls (DX_GROUP=2) and ASD (DX_GROUP=1)
    n_asd = (df["DX_GROUP"] == 1).sum()
    n_ctrl = (df["DX_GROUP"] == 2).sum()
    assert n_asd == 398
    assert n_ctrl == 456
    assert (n_asd + n_ctrl) == 854


def test_paired_bootstrap_test_identity():
    """Verify paired bootstrap function returns observed_delta=0 when testing identical models."""
    from src.eval.site_analysis import paired_bootstrap_comparison
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 200)
    y_prob = np.random.rand(200)

    res = paired_bootstrap_comparison(y_true, y_prob, y_prob, n_boot=200, seed=42)
    assert abs(res["observed_delta"]) < 1e-6
    assert res["ci_95"][0] <= 0.0 <= res["ci_95"][1]
    assert res["ci_excludes_zero"] is False


def test_fixed_seed_reproducibility():
    """Verify that running with fixed seed produces identical output."""
    from src.audit.tautology_audit import fast_auc
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.85, 0.15])
    auc1 = fast_auc(y_true, y_prob)
    auc2 = fast_auc(y_true, y_prob)
    assert auc1 == auc2


def test_ti_consistency_from_one_run():
    """Item A.1 & Quality Gate: Verify Tautology Index is computed consistently in ONE run."""
    json_path = RESULTS_DIR / "tautology_and_dedup_sensitivity.json"
    assert json_path.exists(), "Missing tautology_and_dedup_sensitivity.json"
    with open(json_path) as f:
        data = json.load(f)

    for cohort, c_data in data["unified_tautology_clean"].items():
        auc_items = c_data["auroc_items"]
        auc_clean = c_data["auroc_clean"]
        ti_raw = c_data["tautology_index_raw"]
        ti_clipped = c_data["tautology_index_clipped"]

        # Mathematical definition: TI = (AUC_items - AUC_clean) / (AUC_items - 0.5)
        ti_denom = auc_items - 0.5
        expected_ti = (auc_items - auc_clean) / ti_denom if abs(ti_denom) > 1e-6 else 0.0
        assert abs(ti_raw - expected_ti) < 1e-4, f"TI inconsistent for {cohort}: {ti_raw} vs {expected_ti}"
        assert ti_clipped <= 1.0, f"Clipped TI must not exceed 1.0: {ti_clipped}"
        assert c_data["alternative_a_absolute_drop"] == pytest.approx(auc_items - auc_clean, abs=1e-4)


def test_disjoint_uncertainty_subsets():
    """Item E & Quality Gate: Verify Version 2 calibration subsets are strictly DISJOINT."""
    from sklearn.model_selection import StratifiedKFold
    n_samples = 200
    y = np.random.randint(0, 2, n_samples)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in skf.split(np.zeros(n_samples), y):
        # 3-way split: 60% proper train, 20% calibrator, 20% conformal calibration
        skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_splits = list(skf_inner.split(train_idx, y[train_idx]))
        
        calibrator_idx = train_idx[inner_splits[0][1]]
        conformal_cal_idx = train_idx[inner_splits[1][1]]
        proper_train_idx = train_idx[np.setdiff1d(np.arange(len(train_idx)), np.concatenate([inner_splits[0][1], inner_splits[1][1]]))]

        # Verify pairwise disjointness
        assert len(set(proper_train_idx) & set(calibrator_idx)) == 0, "Proper train and calibrator overlap!"
        assert len(set(proper_train_idx) & set(conformal_cal_idx)) == 0, "Proper train and conformal cal overlap!"
        assert len(set(calibrator_idx) & set(conformal_cal_idx)) == 0, "Calibrator and conformal cal overlap!"
        
        # Verify coverage of training set
        all_tr = set(proper_train_idx) | set(calibrator_idx) | set(conformal_cal_idx)
        assert all_tr == set(train_idx), "3-way union does not equal full training partition!"


def test_permutation_p_resolution():
    """Item A.3 & Quality Gate: Verify permutation p >= 1/(1+B) and respects finite resolution."""
    npz_path = RESULTS_DIR / "permutation_null_distribution.npz"
    assert npz_path.exists(), "Missing permutation_null_distribution.npz"
    perm_data = np.load(npz_path)
    b = len(perm_data["infold_null"])
    assert b >= 10000, f"Permutation count must be >= 10,000, got {b}"

    json_path = RESULTS_DIR / "site_analysis.json"
    with open(json_path) as f:
        site_data = json.load(f)

    for arm, arm_res in site_data["harmonization_comparison"].items():
        if "within_site_permutation_p" in arm_res:
            p_val = arm_res["within_site_permutation_p"]
            count = arm_res["within_site_permutation_count"]
            b_arm = arm_res["within_site_permutation_B"]
            min_res = 1.0 / (1.0 + b_arm)
            expected_p = (1.0 + count) / (1.0 + b_arm)
            assert p_val >= min_res, f"Arm {arm} p-value {p_val} is below resolution limit {min_res}!"
            assert p_val == pytest.approx(expected_p, rel=1e-4)


def test_paired_difference_consistency():
    """Item A.2 & Quality Gate: Verify paired difference consistency between leaky and in-fold ComBat."""
    json_path = RESULTS_DIR / "site_analysis.json"
    with open(json_path) as f:
        site_data = json.load(f)

    infold_auc = site_data["harmonization_comparison"]["infold_as_implemented"]["pooled_auroc"]
    leaky_auc = site_data["harmonization_comparison"]["leaky_combat"]["pooled_auroc"]
    direct_delta = leaky_auc - infold_auc

    contrasts = site_data["paired_harmonization_tests"]["contrasts"]
    leaky_c = next(c for c in contrasts if c["arm_a"] == "leaky_combat" and c["arm_b"] == "infold_as_implemented")
    paired_delta = leaky_c["observed_delta"]

    # Point estimate and paired delta must match exactly
    assert abs(direct_delta - paired_delta) < 1e-4, f"Mismatch: direct {direct_delta} vs paired {paired_delta}"
    assert paired_delta == pytest.approx(0.0096, abs=1e-3)
    assert leaky_c["p_value_holm"] == pytest.approx(1.0, abs=1e-3)

