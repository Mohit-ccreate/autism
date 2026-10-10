"""
Unit and Integration Tests for Phase 3:
- Grounded Help Assistant (Safety Intercepts, Crisis Lifelines, Refusals, Citations)
- Population Drift Detection (Population Stability Index on baseline vs drifted cohorts)
- PWA Assets (manifest.json, sw.js, icons)
- Webapp HTML Structure & JavaScript Syntax Verification
"""

import os
import sys
import json
import subprocess
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.drift_detector import compute_psi, audit_batch_drift



def test_pwa_manifest_and_assets():
    """Verify PWA manifest and service worker configuration."""
    manifest_path = os.path.join(PROJECT_ROOT, "webapp", "manifest.json")
    sw_path = os.path.join(PROJECT_ROOT, "webapp", "sw.js")
    icon192 = os.path.join(PROJECT_ROOT, "webapp", "icon-192.png")
    icon512 = os.path.join(PROJECT_ROOT, "webapp", "icon-512.png")

    assert os.path.exists(manifest_path), "manifest.json missing"
    assert os.path.exists(sw_path), "sw.js missing"
    assert os.path.exists(icon192), "icon-192.png missing"
    assert os.path.exists(icon512), "icon-512.png missing"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "name" in manifest
    assert "icons" in manifest
    assert len(manifest["icons"]) >= 2
    assert manifest["display"] == "standalone"

    with open(sw_path, "r", encoding="utf-8") as f:
        sw_content = f.read()

    assert "CACHE_NAME" in sw_content
    assert "caches.open" in sw_content
    assert "fetch" in sw_content


def test_drift_detector_psi_calculation():
    """Verify PSI calculation for stable vs drifted cohorts."""
    # 1. Identical distributions should yield PSI near 0
    rng = np.random.RandomState(42)
    base = rng.binomial(1, 0.6, 500)
    same_dist = rng.binomial(1, 0.6, 500)
    psi_stable = compute_psi(base, same_dist)
    assert psi_stable < 0.05, f"Expected stable PSI < 0.05, got {psi_stable}"

    # 2. Shifted distribution should yield significant drift PSI >= 0.25
    shifted_dist = rng.binomial(1, 0.1, 500)
    psi_shifted = compute_psi(base, shifted_dist)
    assert psi_shifted >= 0.25, f"Expected high PSI >= 0.25, got {psi_shifted}"


def test_batch_drift_audit():
    """Verify audit_batch_drift on adult test data vs inverted distribution."""
    test_csv = os.path.join(PROJECT_ROOT, "data", "processed", "adult_test.csv")
    if not os.path.exists(test_csv):
        pytest.skip("adult_test.csv not found")

    df_test = pd.read_csv(test_csv)
    stable_report = audit_batch_drift(df_test, cohort_name='adult')
    assert stable_report["drift_status"] == "STABLE"
    assert stable_report["overall_psi"] < 0.10

    # Invert binary answers to simulate strong domain shift
    df_inverted = df_test.copy()
    for i in range(1, 11):
        df_inverted[f"A{i}"] = 1 - df_inverted[f"A{i}"]

    shifted_report = audit_batch_drift(df_inverted, cohort_name='adult')
    assert "ALERT" in shifted_report["drift_status"] or "WARNING" in shifted_report["drift_status"]
    assert shifted_report["overall_psi"] > 0.15


def test_webapp_javascript_syntax():
    """Verify webapp/app.js syntax using node --check."""
    app_js = os.path.join(PROJECT_ROOT, "webapp", "app.js")
    assert os.path.exists(app_js)
    res = subprocess.run(["node", "--check", app_js], capture_output=True, text=True)
    assert res.returncode == 0, f"app.js syntax error: {res.stderr}"


def test_grounded_assistant_safety_rules_in_app_js():
    """Verify that app.js contains strict crisis intercepts, diagnostic refusals, and citations."""
    app_js = os.path.join(PROJECT_ROOT, "webapp", "app.js")
    with open(app_js, "r", encoding="utf-8") as f:
        code = f.read()

    # Check crisis lifeline numbers
    assert "1800-599-0019" in code, "KIRAN helpline number missing in assistant"
    assert "988" in code, "988 helpline missing in assistant"
    assert "14416" in code, "Tele-MANAS missing in assistant"

    # Check diagnostic refusal logic
    assert "Non-Diagnostic Policy" in code or "non-diagnostic" in code.lower()
    assert "cannot provide a medical diagnosis" in code

    # Check GroundedKnowledge base presence
    assert "GroundedKnowledge" in code
    assert "RPwD Act 2016" in code
    assert "ADOS-2" in code
    assert "NICE Guidelines" in code


def test_webapp_html_phase3_elements():
    """Verify that all Phase 3 DOM elements and tabs exist in webapp/index.html."""
    html_path = os.path.join(PROJECT_ROOT, "webapp", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    required_ids = [
        "consent-modal",
        "btn-accept-consent",
        "btn-dyslexia-font",
        "btn-high-contrast",
        "btn-text-size",
        "btn-read-aloud",
        "tab-history",
        "tbody-history",
        "btn-clear-history",
        "btn-export-history-json",
        "btn-export-history-csv",
        "tab-learning",
        "tab-clinician",
        "chart-clinician-dist",
        "btn-export-clinician-report",
        "tab-monitoring",
        "drift-drop-zone",
        "drift-file-input",
        "drift-status-banner",
        "tbody-drift-items",
        "btn-toggle-assistant",
        "assistant-modal",
        "assistant-chat-body",
        "assistant-user-input",
        "btn-send-assistant"
    ]

    for elem_id in required_ids:
        assert f'id="{elem_id}"' in html, f"Element id='{elem_id}' missing in index.html"
