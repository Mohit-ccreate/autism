"""
Unit and Integration Tests for Phase 4 & Phase 5:
- Deep Learning Research Lab Benchmarks (ABIDE I CC200 Atlas)
- Scientific Safeguards Verification (Label-Shuffle, Site Predictability, Motion Decorrelation)
- Multimodal Graceful Degradation & Browser Distillation
- Auto-Generated Paper Draft & v2 Summary
- Google Colab GPU Notebook Validity
- Research Lab Web UI & Ethical Warnings Integrity
- Non-Diagnostic Phrasing Compliance
"""

import os
import sys
import json
import re
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_lab_metrics_existence_and_parity():
    """Verify reports/lab_metrics.json and webapp/lab_metrics.json exist and match."""
    rep_path = os.path.join(PROJECT_ROOT, "reports", "lab_metrics.json")
    web_path = os.path.join(PROJECT_ROOT, "webapp", "lab_metrics.json")

    assert os.path.exists(rep_path), "reports/lab_metrics.json missing"
    assert os.path.exists(web_path), "webapp/lab_metrics.json missing"

    with open(rep_path, "r", encoding="utf-8") as f:
        rep_data = json.load(f)
    with open(web_path, "r", encoding="utf-8") as f:
        web_data = json.load(f)

    assert rep_data == web_data, "Parity mismatch between reports/ and webapp/ lab_metrics.json"

    # Core required sections
    required_sections = [
        "metadata",
        "loso_model_benchmarks",
        "scientific_safeguards",
        "self_supervised_data_efficiency",
        "multimodal_robustness",
        "browser_distillation",
        "test_time_adaptation",
        "ablations",
        "statistical_tests"
    ]
    for sec in required_sections:
        assert sec in rep_data, f"Missing section '{sec}' in lab_metrics.json"


def test_scientific_safeguards():
    """Verify that all 4 scientific safeguards passed and values reflect reality."""
    rep_path = os.path.join(PROJECT_ROOT, "reports", "lab_metrics.json")
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    safeguards = data["scientific_safeguards"]

    # 1. Label shuffle must collapse to chance (50% ± 8%)
    shuf_acc = safeguards["label_shuffle_test"]["shuffled_accuracy"]
    assert 0.40 <= shuf_acc <= 0.55, f"Expected label shuffle accuracy near chance, got {shuf_acc}"
    assert "PASS" in safeguards["label_shuffle_test"]["status"]

    # 2. Scanner hardware predictability must drop significantly with ComBat
    site_audit = safeguards["site_classifier_audit"]
    assert site_audit["site_predictability_raw"] > 0.85, "Expected high raw site predictability"
    assert site_audit["site_predictability_after_combat"] < 0.50, "Expected ComBat to drop site predictability"
    assert site_audit["predictability_drop"] > 0.40, "Expected predictability drop > 40%"

    # 3. Head motion artifact correlation (|r| < 0.15)
    fd_r = abs(safeguards["confound_correlations"]["head_motion_fd_correlation"])
    assert fd_r < 0.15, f"Predictions correlate too strongly with head motion FD: {fd_r}"

    # 4. Age confound correlation (|r| < 0.25)
    age_r = abs(safeguards["confound_correlations"]["age_correlation"])
    assert age_r < 0.25, f"Predictions correlate too strongly with age: {age_r}"


def test_honest_loso_benchmarks():
    """Verify deep learning model results are honest and uninflated under LOSO CV."""
    rep_path = os.path.join(PROJECT_ROOT, "reports", "lab_metrics.json")
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    benchmarks = data["loso_model_benchmarks"]

    # Classical linear baseline
    lr = benchmarks["logistic_regression_baseline"]
    assert 0.50 <= lr["accuracy"] <= 0.65, f"Unexpected LR accuracy: {lr['accuracy']}"
    assert lr["roc_auc"] > 0.55, f"Unexpected LR ROC-AUC: {lr['roc_auc']}"

    # Deep architectures under LOSO CV on N=120 must be honest (< 0.60)
    for model_key in ["brain_network_transformer", "connectivity_vit", "connectivity_cnn", "brain_gat"]:
        assert model_key in benchmarks
        m = benchmarks[model_key]
        assert m["accuracy"] < 0.60, f"{model_key} shows suspiciously inflated accuracy: {m['accuracy']}"


def test_multimodal_and_distillation():
    """Verify multimodal attention fusion and browser distillation payloads."""
    rep_path = os.path.join(PROJECT_ROOT, "reports", "lab_metrics.json")
    with open(rep_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Multimodal degradation
    mm = data["multimodal_robustness"]
    assert mm["full_multimodal_accuracy"] > 0.90
    assert mm["brain_only_accuracy"] > 0.80
    assert mm["questionnaire_only_accuracy"] > 0.65
    assert "PASS" in mm["status"]

    # Browser distillation
    dist = data["browser_distillation"]
    assert dist["model_footprint_kb"] < 10.0, f"Footprint too large: {dist['model_footprint_kb']} KB"
    assert dist["inference_latency_ms"] < 5.0, f"Latency too high: {dist['inference_latency_ms']} ms"


def test_paper_draft_and_v2_summary_generated():
    """Verify generated scientific paper draft and v2 summary exist and are populated."""
    paper_path = os.path.join(PROJECT_ROOT, "reports", "paper_draft.md")
    v2_path = os.path.join(PROJECT_ROOT, "reports", "v2_summary.md")

    assert os.path.exists(paper_path), "reports/paper_draft.md missing"
    assert os.path.exists(v2_path), "reports/v2_summary.md missing"

    with open(paper_path, "r", encoding="utf-8") as f:
        paper_text = f.read()
    with open(v2_path, "r", encoding="utf-8") as f:
        v2_text = f.read()

    assert len(paper_text) > 3000, "paper_draft.md is suspiciously short"
    assert len(v2_text) > 1500, "v2_summary.md is suspiciously short"

    # Verify real numbers are present
    assert "55.00%" in paper_text or "55.0%" in paper_text
    assert "45.83%" in paper_text or "45.8%" in paper_text
    assert "CC200" in paper_text
    assert "ComBat" in paper_text
    assert "Leave-One-Site-Out" in paper_text

    assert "55.0%" in v2_text or "55.00%" in v2_text
    assert "What We Tried That Didn't Work" in v2_text


def test_colab_gpu_notebook():
    """Verify Google Colab GPU notebook exists and is valid JSON."""
    nb_path = os.path.join(PROJECT_ROOT, "notebooks", "Phase4_Deep_Research_Lab.ipynb")
    assert os.path.exists(nb_path), "Phase4_Deep_Research_Lab.ipynb missing"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    assert "cells" in nb_data
    assert len(nb_data["cells"]) >= 10
    assert nb_data["metadata"]["accelerator"] == "GPU"


def test_webapp_phase5_panels_and_ethical_notice():
    """Verify webapp/index.html includes all Phase 5 tabs and ethical notice."""
    html_path = os.path.join(PROJECT_ROOT, "webapp", "index.html")
    assert os.path.exists(html_path)

    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert 'id="tab-brain-explorer"' in html
    assert 'id="tab-arena"' in html
    assert 'id="tab-disagreement"' in html
    assert 'id="tab-efficiency"' in html
    assert 'id="tab-robustness"' in html
    assert 'id="tab-negative"' in html

    # Ethical AI Notice
    assert "WHAT THE MODEL USED, NOT WHERE AUTISM IS" in html


def test_non_diagnostic_phrasing_in_reports():
    """Verify that reports and code never assert definitive diagnosis."""
    files_to_check = [
        os.path.join(PROJECT_ROOT, "reports", "paper_draft.md"),
        os.path.join(PROJECT_ROOT, "reports", "v2_summary.md"),
        os.path.join(PROJECT_ROOT, "src", "export_paper_draft.py"),
        os.path.join(PROJECT_ROOT, "src", "research_lab.py")
    ]

    # Forbidden phrases that claim certainty or direct medical diagnosis
    forbidden_patterns = [
        re.compile(r"diagnoses\s+autism", re.IGNORECASE),
        re.compile(r"you\s+have\s+autism", re.IGNORECASE),
        re.compile(r"100%\s+accurate\s+diagnosis", re.IGNORECASE),
        re.compile(r"replaces\s+clinical\s+evaluation", re.IGNORECASE)
    ]

    for path in files_to_check:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            for pat in forbidden_patterns:
                match = pat.search(content)
                assert match is None, f"Forbidden diagnostic phrasing '{match.group(0)}' found in {path}"
