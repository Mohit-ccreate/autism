"""
Master Orchestration Script for ASD Screening Research Platform.

Runs the complete pipeline end-to-end:
1. Ingests all datasets (UCI Adult, Adolescent, Child; Toddler Q-CHAT-10; ABIDE I CC200).
2. Cleans data, performs deduplication, and creates stratified train/test partitions.
3. Trains, tunes, calibrates questionnaire models with conformal prediction & explainability.
4. Evaluates ABIDE functional connectivity under Leave-One-Site-Out CV and random splits.
5. Generates reports/metrics.json (Single Source of Truth), model_card.md, honest_leaderboard.md, comparison.md.
6. Evaluates Phase 4 Research Lab (BNT, ViT, CNN, GAT, Self-Supervised, Distillation, Safeguards).
7. Compiles Phase 5 scientific paper draft (paper_draft.md) and executive summary (v2_summary.md).
8. Runs complete 23-test pytest suite confirming zero leakage, train/test isolation, safeguards, and UI assets.
"""

import os
import sys
import subprocess
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
if not os.path.exists(PYTHON_EXE):
    PYTHON_EXE = sys.executable


def run_step(step_name, command_args):
    print(f"\n======================================================================")
    print(f"  RUNNING STEP: {step_name}")
    print(f"======================================================================")
    start_time = time.time()
    res = subprocess.run([PYTHON_EXE] + command_args, cwd=BASE_DIR)
    elapsed = time.time() - start_time
    if res.returncode != 0:
        print(f"\n[ERROR] Step '{step_name}' failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"  [OK] {step_name} completed successfully in {elapsed:.1f}s")


def main():
    print("======================================================================")
    print("  ASD CLINICAL SCREENING & NEURAL CONNECTIVITY PLATFORM: RUN_ALL")
    print("======================================================================")
    total_start = time.time()

    # Step 1: Data Ingestion
    run_step("1. Automated Data Ingestion", ["src/data_ingestion.py"])

    # Step 2: Data Preprocessing & Splitting
    run_step("2. Data Preprocessing & Leakage Removal", ["src/data_preprocessing.py"])

    # Step 3: Questionnaire Machine Learning Pipeline
    run_step("3. Questionnaire ML Training & Calibration", ["src/train_questionnaire.py"])

    # Step 4: ABIDE Functional Connectivity Baseline
    run_step("4. ABIDE I Brain Connectivity LOSO Baseline", ["src/train_abide.py"])

    # Step 5: Metrics & Documentation Generation
    run_step("5. Single Source of Truth Compilation", ["src/export_metrics.py"])

    # Step 6: Phase 4 Advanced Research Lab (Deep Models, SSL, Safeguards)
    run_step("6. Phase 4 Deep Learning Research Lab & Safeguards", ["src/research_lab.py"])

    # Step 7: Phase 5 Paper Draft & v2 Summary Compilation
    run_step("7. Scientific Paper Draft & v2 Summary Compilation", ["src/export_paper_draft.py"])

    # Step 8: Full Test Suite Verification (All 5 Phases)
    run_step("8. Full Pytest Verification Suite (24 Tests)", ["-m", "pytest", "tests/", "-v"])

    total_elapsed = time.time() - total_start
    print(f"\n======================================================================")
    print(f"  ALL PIPELINE STEPS COMPLETED IN {total_elapsed:.1f}s")
    print(f"  Web application ready in webapp/ (open webapp/index.html or start http.server)")
    print(f"======================================================================")


if __name__ == "__main__":
    main()
