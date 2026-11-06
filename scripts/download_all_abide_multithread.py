"""
Multithreaded Resilient Downloader for Full ABIDE I CC200 Dataset.
scripts/download_all_abide_multithread.py

Downloads all available CC200 ROI time series files from Amazon S3
into data/abide/ABIDE_pcp/cpac/filt_global/{FILE_ID}_rois_cc200.1D.
Uses 16 worker threads with exponential backoff and timeout handling.
Then computes Pearson correlation + Fisher-z matrices and saves to data/abide_full_cache.npz.
"""

from __future__ import annotations

import concurrent.futures
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ABIDE_DIR = DATA_DIR / "abide"
DEST_DIR = ABIDE_DIR / "ABIDE_pcp" / "cpac" / "filt_global"
DEST_DIR.mkdir(parents=True, exist_ok=True)
PHENO_PATH = ABIDE_DIR / "ABIDE_pcp" / "Phenotypic_V1_0b_preprocessed1.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ASD-Screening-Research/2.0"}


def download_single_file(file_id: str, max_retries: int = 3) -> Tuple[str, bool, str]:
    fname = f"{file_id}_rois_cc200.1D"
    target = DEST_DIR / fname
    if target.exists() and target.stat().st_size > 1000:
        return file_id, True, "cached"

    url = f"https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/rois_cc200/{fname}"
    last_err = ""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) < 500:
                raise ValueError("File payload too small")
            with open(target, "wb") as f:
                f.write(data)
            return file_id, True, "downloaded"
        except Exception as e:
            last_err = str(e)
            time.sleep(1.5 ** attempt)
    return file_id, False, last_err


def download_and_build(quality_checked: bool = True, max_workers: int = 16):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print(f"  DOWNLOADING FULL ABIDE I CC200 DATASET (quality_checked={quality_checked})")
    print("=" * 70)

    if not PHENO_PATH.exists():
        print(f"[ERROR] Phenotypic CSV not found at {PHENO_PATH}")
        return False, "Phenotypic CSV not found", 0, 0

    df = pd.read_csv(PHENO_PATH)
    if quality_checked:
        mask = (
            (df["qc_rater_1"] == "OK")
            & (df["qc_anat_rater_2"].isin(["OK", "maybe"]))
            & (df["qc_func_rater_2"].isin(["OK", "maybe"]))
            & (df["qc_anat_rater_3"] == "OK")
            & (df["qc_func_rater_3"] == "OK")
            & (df["FILE_ID"] != "no_filename")
        )
    else:
        mask = df["FILE_ID"] != "no_filename"

    target_df = df[mask].copy().reset_index(drop=True)
    file_ids = target_df["FILE_ID"].tolist()
    total_targets = len(file_ids)
    print(f"Target subjects to acquire: {total_targets}")
    print(f"Sites in cohort ({target_df['SITE_ID'].nunique()} sites):")
    for s, count in target_df["SITE_ID"].value_counts().items():
        print(f"  {s}: {count}")

    start_time = time.time()
    succeeded = 0
    failed = 0
    fail_details = []

    print(f"\nStarting download pool with {max_workers} worker threads...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(download_single_file, fid): fid for fid in file_ids}
        done_count = 0
        for future in concurrent.futures.as_completed(future_map):
            fid = future_map[future]
            done_count += 1
            try:
                fid_res, ok, msg = future.result()
                if ok:
                    succeeded += 1
                else:
                    failed += 1
                    fail_details.append((fid, msg))
            except Exception as e:
                failed += 1
                fail_details.append((fid, str(e)))

            if done_count % 50 == 0 or done_count == total_targets:
                elapsed = time.time() - start_time
                print(f"  Progress: {done_count}/{total_targets} processed ({succeeded} ok, {failed} fail) in {elapsed:.1f}s")

    elapsed_total = time.time() - start_time
    print(f"\n[DOWNLOAD COMPLETE in {elapsed_total:.1f}s]")
    print(f"Successfully available: {succeeded}/{total_targets} subjects")
    if failed > 0:
        print(f"Failed downloads: {failed}")
        for fid, err in fail_details[:5]:
            print(f"  - {fid}: {err}")

    # Build correlation matrices
    print("\nProcessing timeseries into Pearson correlation & Fisher-z matrices...")
    matrices = []
    valid_indices = []

    for idx, row in target_df.iterrows():
        fid = row["FILE_ID"]
        fpath = DEST_DIR / f"{fid}_rois_cc200.1D"
        if not fpath.exists() or fpath.stat().st_size < 1000:
            continue
        try:
            ts = np.loadtxt(fpath)
            if ts.ndim != 2 or ts.shape[1] != 200 or ts.shape[0] < 10:
                continue
            corr = np.corrcoef(ts.T)
            np.fill_diagonal(corr, 0.0)
            corr = np.clip(corr, -0.999999, 0.999999)
            z_mat = np.arctanh(corr)
            np.fill_diagonal(z_mat, 0.0)
            if np.isnan(z_mat).any():
                continue
            matrices.append(z_mat)
            valid_indices.append(idx)
        except Exception:
            continue

    matrices = np.array(matrices, dtype=np.float32)
    clean_pheno = target_df.iloc[valid_indices].reset_index(drop=True)

    dx_group = clean_pheno["DX_GROUP"].values.astype(int)
    target = np.where(dx_group == 1, 1, 0)
    sites = clean_pheno["SITE_ID"].values
    sub_ids = clean_pheno["SUB_ID"].values
    ages = clean_pheno["AGE_AT_SCAN"].values.astype(float) if "AGE_AT_SCAN" in clean_pheno.columns else np.zeros(len(target))
    sexes = clean_pheno["SEX"].values.astype(int) if "SEX" in clean_pheno.columns else np.zeros(len(target))
    func_mean_fd = clean_pheno["func_mean_fd"].values.astype(float) if "func_mean_fd" in clean_pheno.columns else np.zeros(len(target))
    fiq = clean_pheno["FIQ"].values.astype(float) if "FIQ" in clean_pheno.columns else np.zeros(len(target))

    cache_name = "abide_full_cache.npz" if quality_checked else "abide_raw_cache.npz"
    cache_path = DATA_DIR / cache_name
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
    print(f"[OK] Saved {len(matrices)} matrices ({matrices.shape}) to {cache_path}")
    print(f"Total clinical sites: {len(np.unique(sites))}")
    print(f"Site distribution: {dict(pd.Series(sites).value_counts())}")
    print(f"Diagnostic class balance: {dict(pd.Series(target).value_counts())}")

    pheno_name = "abide_full_phenotypic.csv" if quality_checked else "abide_raw_phenotypic.csv"
    clean_pheno.to_csv(DATA_DIR / pheno_name, index=False)
    print(f"[OK] Saved clean phenotypic data to {DATA_DIR / pheno_name}")

    return True, "", len(matrices), len(np.unique(sites))


if __name__ == "__main__":
    qc = True
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["false", "0", "noqc"]:
        qc = False
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    download_and_build(quality_checked=qc, max_workers=workers)
