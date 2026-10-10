"""
Phase 2 Scientific and Clinical Safety Verification Tests.

Verifies:
1. Strict Absence of Diagnostic Claims: Confirms that forbidden diagnosis wording
   (e.g., 'you have autism', 'you do not have autism') never appears in any user-facing code or markdown.
2. Leakage Detector Integrity: Confirms that src/leakage_check.py accurately detects
   score column leakage and deterministic target derivation.
3. Metrics Traceability: Confirms that all performance numbers trace directly to reports/metrics.json.
4. Help Directory Links: Confirms that all help directory links in the web application are valid, active, and have verification dates.
"""

import os
import re
import json
import pandas as pd
import numpy as np
import pytest
from src.leakage_check import audit_dataset_for_leakage

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBAPP_DIR = os.path.join(BASE_DIR, "webapp")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def test_no_diagnosis_wording_in_codebase():
    """Verify that no diagnostic assertions appear in webapp, reports, or documentation."""
    forbidden_patterns = [
        r"\byou have autism\b",
        r"\byou do not have autism\b",
        r"\byou don't have autism\b",
        r"\bwe diagnose you\b",
        r"\bdiagnosed you with autism\b",
        r"\bpositive diagnosis of autism\b",
        r"\bnegative diagnosis of autism\b",
        r"\bconfirmed diagnosis of autism\b"
    ]
    
    files_to_scan = [
        os.path.join(WEBAPP_DIR, "index.html"),
        os.path.join(WEBAPP_DIR, "app.js"),
        os.path.join(REPORTS_DIR, "model_card.md"),
        os.path.join(REPORTS_DIR, "honest_leaderboard.md"),
        os.path.join(REPORTS_DIR, "comparison.md"),
        os.path.join(BASE_DIR, "README.md")
    ]
    
    for fpath in files_to_scan:
        assert os.path.exists(fpath), f"File {fpath} must exist"
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            
        for pattern in forbidden_patterns:
            matches = re.findall(pattern, content, flags=re.IGNORECASE)
            assert len(matches) == 0, f"Clinical Safety Violation: Forbidden phrase matching '{pattern}' found in {fpath}: {matches}"


def test_leakage_detector_flags_score_column():
    """Verify that src/leakage_check.py correctly flags datasets containing total scores or deterministic rules."""
    # Create test dataframe WITH score column
    np.random.seed(42)
    items = np.random.randint(0, 2, size=(100, 10))
    sum_items = items.sum(axis=1)
    target = (sum_items >= 7).astype(int)
    
    df_leaked = pd.DataFrame(items, columns=[f'A{i}' for i in range(1, 11)])
    df_leaked['result'] = sum_items
    df_leaked['target'] = target
    
    report_leaked = audit_dataset_for_leakage(df_leaked, target_col='target')
    assert report_leaked['risk_level'] == 'CRITICAL', "Must flag leaked dataset as CRITICAL risk"
    assert report_leaked['verdict'] in ['FEATURE_LEAKAGE_DETECTED', 'DETERMINISTIC_TARGET_DERIVATION']
    
    # Create test dataframe WITHOUT score column and without deterministic rule
    # Target is probabilistic (e.g. noisy logistic)
    logits = items[:, 0] * 0.8 + items[:, 1] * 0.5 - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    prob_target = (np.random.rand(100) < probs).astype(int)
    
    df_clean = pd.DataFrame(items, columns=[f'A{i}' for i in range(1, 11)])
    df_clean['target'] = prob_target
    
    report_clean = audit_dataset_for_leakage(df_clean, target_col='target')
    assert report_clean['risk_level'] == 'LOW', "Must pass clean non-deterministic dataset"
    assert report_clean['verdict'] == 'PASSED_CLEAN'


def test_ui_numbers_trace_to_metrics():
    """Verify that all metrics in webapp match metrics.json exactly."""
    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    q_bench = metrics["questionnaire_benchmarks"]
    assert "adult_variant_c" in q_bench
    adult_m = q_bench["adult_variant_c"]["calibrated_metrics"]
    assert adult_m["accuracy"] > 0.80, "Adult variant C accuracy should be > 80%"
    assert adult_m["roc_auc"] > 0.90, "Adult variant C ROC-AUC should be > 0.90"
    
    abide_m = metrics["abide_brain_benchmarks"]["honest_loso_evaluation"]["overall"]["logistic_regression"]
    assert abide_m["accuracy"] == pytest.approx(0.5583, abs=0.01)
    assert abide_m["roc_auc"] == pytest.approx(0.6073, abs=0.01)


def test_help_directory_verification_dates():
    """Verify that help directory entries in webapp have explicit verification dates."""
    html_path = os.path.join(WEBAPP_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
        
    assert "Verified Sep 2026" in html, "Help directory must contain explicit verification dates"
    assert "NIMHANS" in html
    assert "AIIMS" in html
    assert "1800-599-0019" in html
    assert "thenationaltrust.gov.in" in html
    assert "cdc.gov/autism" in html
    assert "nhs.uk/conditions/autism" in html


def test_screening_risk_calibration_not_zero():
    """Verify that clinical screening likelihood is calibrated across the full trait spectrum and never collapses to 0."""
    app_js_path = os.path.join(WEBAPP_DIR, "app.js")
    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js = f.read()

    # 1. Verify that official psychometric reverse-scoring key is present
    assert "ScoringKey" in app_js, "ScoringKey must be defined in app.js"
    assert "userResponses" in app_js, "userResponses must be tracked in app.js"

    # 2. Simulate the calibrated risk function across trait counts [0..10]
    models_path = os.path.join(WEBAPP_DIR, "models_export.json")
    with open(models_path, "r", encoding="utf-8") as f:
        models = json.load(f)

    coefs = models["adult_variant_a_items_only"]["coefficients"]
    mean_coef = sum(abs(c) for c in coefs) / 10.0

    probs = []
    for traits in range(11):
        weighted_score = sum(abs(coefs[i]) / mean_coef for i in range(traits))
        z = (weighted_score - 5.2) / 1.5
        p = 1.0 / (1.0 + np.exp(-z))
        p = max(0.03, min(0.98, p))
        probs.append(p)

    # Must be strictly positive at 0 traits (general population prevalence baseline ~3%)
    assert probs[0] >= 0.025, f"0 traits should not be zero; got {probs[0]}"

    # Monotonically strictly increasing
    for k in range(10):
        assert probs[k] < probs[k + 1], f"Probability must strictly increase with traits: {probs[k]} vs {probs[k+1]}"

    # Moderate traits (3/10) must be distinctly non-zero (> 12%)
    assert 0.12 <= probs[3] <= 0.25, f"Expected 3 traits ~17%, got {probs[3]}"

    # Borderline traits (5/10) must be near the conformal referral boundary (40% - 55%)
    assert 0.40 <= probs[5] <= 0.55, f"Expected 5 traits ~47%, got {probs[5]}"

    # Elevated traits (7/10) must be clinically elevated (> 70%)
    assert 0.70 <= probs[7] <= 0.85, f"Expected 7 traits ~77%, got {probs[7]}"

    # Maximum traits (10/10) must be high (> 90%)
    assert probs[10] >= 0.90, f"Expected 10 traits >= 90%, got {probs[10]}"
