import json
from pathlib import Path
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

def test_lit_audit_csv_schema_and_counts():
    csv_path = REPO_ROOT / "docs" / "lit_audit_template.csv"
    assert csv_path.exists(), f"Missing {csv_path}"
    
    df = pd.read_csv(csv_path)
    # Target: 30-40 real, published papers
    assert 30 <= len(df) <= 40, f"Expected 30-40 papers, found {len(df)}"
    
    expected_cols = [
        "paper", "first_author", "year", "venue", "source_url",
        "dataset", "n", "split_type", "preprocessing_fit_in_fold",
        "leaking_features", "harmonization", "calibration_reported",
        "uncertainty_reported", "external_validation", "reported_accuracy",
        "evidence_level", "needs_human_check", "notes"
    ]
    assert list(df.columns) == expected_cols, f"Column mismatch: {list(df.columns)}"
    
    # Check URLs and non-empty metadata
    for idx, row in df.iterrows():
        assert str(row["source_url"]).startswith("http"), f"Row {idx} invalid URL: {row['source_url']}"
        assert row["evidence_level"] in ["full_text", "abstract_only"], f"Row {idx} invalid evidence_level: {row['evidence_level']}"
        assert row["needs_human_check"] in ["Y", "N"], f"Row {idx} invalid needs_human_check: {row['needs_human_check']}"
        if row["evidence_level"] == "abstract_only":
            assert row["needs_human_check"] == "Y", f"Row {idx} abstract_only must have needs_human_check=Y"

def test_lit_audit_summary_json():
    json_path = REPO_ROOT / "results" / "lit_audit_summary.json"
    assert json_path.exists(), f"Missing {json_path}"
    
    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
        
    meta = summary["meta"]
    assert meta["total_papers"] == 33
    assert meta["tabular_papers"] == 16
    assert meta["neuroimaging_papers"] == 17
    assert meta["evidence_levels"]["full_text"]["count"] == 31
    assert meta["evidence_levels"]["abstract_only"]["count"] == 2
    assert meta["needs_human_check"]["count"] == 3

def test_lit_audit_artifacts_exist():
    assert (REPO_ROOT / "figures" / "lit_meta_audit_summary.png").exists()
    assert (REPO_ROOT / "figures" / "lit_meta_audit_summary.pdf").exists()
    assert (REPO_ROOT / "docs" / "lit_audit_for_human_review.md").exists()
    assert (REPO_ROOT / "results" / "lit_search_log.md").exists()
