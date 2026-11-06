"""
Download and Cache Full ABIDE I Dataset — scripts/fetch_full_abide.py

Fetches CC200 ROI time series for all available subjects (QC=True, ~871 subjects across 17+ sites)
using nilearn.datasets.fetch_abide_pcp.
Computes Pearson correlation and Fisher-z transformed connectivity matrices.
Saves to data/abide_full_cache.npz.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ABIDE_DIR = DATA_DIR / "abide"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def fetch_and_build_cache(quality_checked: bool = True, max_subjects: int = None):
    import nilearn.datasets as nd

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    qc_label = "QC=True" if quality_checked else "QC=False"
    print("=" * 70)
    print(f"  FETCHING FULL ABIDE I DATASET ({qc_label})")
    print("=" * 70)

    start_time = time.time()
    try:
        print(f"Calling nilearn.datasets.fetch_abide_pcp (pipeline='cpac', filt_global, rois_cc200, {qc_label})...")
        abide_data = nd.fetch_abide_pcp(
            data_dir=str(ABIDE_DIR),
            n_subjects=max_subjects,
            pipeline="cpac",
            band_pass_filtering=True,
            global_signal_regression=True,
            derivatives=["rois_cc200"],
            quality_checked=quality_checked,
            verbose=1,
        )
    except Exception as e:
        print(f"[ERROR] Failed to fetch ABIDE data: {e}")
        return False, str(e), 0, 0

    elapsed = time.time() - start_time
    pheno = pd.DataFrame(abide_data.phenotypic)
    rois_files = abide_data.rois_cc200
    n_files = len(rois_files)
    print(f"[OK] Downloaded/Verified {n_files} subjects in {elapsed:.1f}s.")
    print(f"Sites represented ({pheno['SITE_ID'].nunique()} sites):")
    site_counts = pheno["SITE_ID"].value_counts().to_dict()
    for s, c in site_counts.items():
        print(f"  {s}: {c}")

    # Build correlation matrices
    print("\nComputing Pearson correlation and Fisher-z transformed matrices...")
    matrices = []
    valid_idx = []
    
    for i, fpath in enumerate(rois_files):
        try:
            ts = np.loadtxt(fpath)
            if ts.ndim != 2 or ts.shape[1] != 200 or ts.shape[0] < 10:
                print(f"  Warning: Subject {i} invalid timeseries shape {ts.shape}, skipping")
                continue
            # Pearson correlation
            corr = np.corrcoef(ts.T)
            np.fill_diagonal(corr, 0.0)
            # Fisher-z transform
            corr = np.clip(corr, -0.999999, 0.999999)
            z_mat = np.arctanh(corr)
            np.fill_diagonal(z_mat, 0.0)
            
            # Check for NaNs
            if np.isnan(z_mat).any():
                print(f"  Warning: Subject {i} contains NaNs, skipping")
                continue
                
            matrices.append(z_mat)
            valid_idx.append(i)
        except Exception as err:
            print(f"  Error reading {fpath}: {err}")
            continue

    matrices = np.array(matrices, dtype=np.float32)
    valid_pheno = pheno.iloc[valid_idx].reset_index(drop=True)

    # Standardize column mappings
    # DX_GROUP: 1 = Autism, 2 = Control -> map to target: 1 = ASD, 0 = Control
    dx_group = valid_pheno["DX_GROUP"].values.astype(int)
    target = np.where(dx_group == 1, 1, 0)
    sites = valid_pheno["SITE_ID"].values
    sub_ids = valid_pheno["SUB_ID"].values

    ages = valid_pheno["AGE_AT_SCAN"].values.astype(float) if "AGE_AT_SCAN" in valid_pheno.columns else np.zeros(len(target))
    sexes = valid_pheno["SEX"].values.astype(int) if "SEX" in valid_pheno.columns else np.zeros(len(target))
    func_mean_fd = valid_pheno["func_mean_fd"].values.astype(float) if "func_mean_fd" in valid_pheno.columns else np.zeros(len(target))
    fiq = valid_pheno["FIQ"].values.astype(float) if "FIQ" in valid_pheno.columns else np.zeros(len(target))

    cache_filename = "abide_full_cache.npz" if quality_checked else "abide_raw_cache.npz"
    cache_path = DATA_DIR / cache_filename
    np.savez_compressed(
        cache_path,
        matrices=matrices,
        sub_id=sub_ids,
        site=sites,
        dx_group=dx_group,
        target=target,
        age=ages,
        sex=sexes,
        func_mean_fd=func_mean_fd,
        fiq=fiq,
    )
    print(f"\n[OK] Saved {len(matrices)} correlation matrices ({matrices.shape}) to {cache_path}")
    print(f"Total sites: {len(np.unique(sites))}, Diagnosis balance: {dict(pd.Series(target).value_counts())}")
    
    # Save phenotypic CSV
    pheno_csv_name = "abide_full_phenotypic.csv" if quality_checked else "abide_raw_phenotypic.csv"
    valid_pheno.to_csv(DATA_DIR / pheno_csv_name, index=False)
    
    return True, "", len(matrices), len(np.unique(sites))


if __name__ == "__main__":
    qc = True
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["false", "0", "noqc"]:
        qc = False
    max_sub = int(sys.argv[2]) if len(sys.argv) > 2 else None
    success, err, n_subj, n_sites = fetch_and_build_cache(quality_checked=qc, max_subjects=max_sub)
    if not success:
        sys.exit(1)
