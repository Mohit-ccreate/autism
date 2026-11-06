"""
ABIDE II External Validation Acquisition Attempt.
scripts/attempt_abide_ii.py

Rigorously tests network accessibility and availability of preprocessed ABIDE II derivatives.
Logs HTTP status, endpoint accessibility, and authentication requirements.
"""

import urllib.request
import urllib.error
import ssl
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ENDPOINTS = [
    {
        "name": "ABIDE II S3 Preprocessed Root",
        "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/Outputs/",
        "description": "Amazon S3 bucket for preprocessed derivatives"
    },
    {
        "name": "ABIDE II NITRC Project Page",
        "url": "https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html",
        "description": "Official ABIDE II portal"
    },
    {
        "name": "ABIDE II S3 Phenotypic Table",
        "url": "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE2/Phenotypic_V1_0b.csv",
        "description": "ABIDE II Phenotypic manifest on S3"
    },
    {
        "name": "ABIDE II NITRC Phenotypic Download",
        "url": "https://www.nitrc.org/frs/download.php/8404/ABIDEII_Composite_Phenotypic.csv",
        "description": "Direct composite phenotypic download on NITRC"
    }
]

def check_endpoints():
    ctx = ssl.create_default_context()
    results = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ASD-Screening-Research/2.0"
    }
    
    for ep in ENDPOINTS:
        url = ep["url"]
        print(f"Testing endpoint: {ep['name']} ({url})...")
        req = urllib.request.Request(url, headers=headers)
        status_info = {
            "name": ep["name"],
            "url": url,
            "description": ep["description"],
            "accessible": False,
            "status_code": None,
            "message": "",
        }
        try:
            with urllib.request.urlopen(req, timeout=10, context=ctx) as response:
                status_code = response.getcode()
                status_info["status_code"] = status_code
                status_info["accessible"] = (status_code == 200)
                status_info["message"] = f"HTTP {status_code} OK"
                print(f"  -> SUCCESS: HTTP {status_code}")
        except urllib.error.HTTPError as e:
            status_info["status_code"] = e.code
            status_info["message"] = f"HTTP Error {e.code}: {e.reason}"
            print(f"  -> FAILED: HTTP {e.code} ({e.reason})")
        except urllib.error.URLError as e:
            status_info["message"] = f"URL Error: {e.reason}"
            print(f"  -> FAILED: URL Error {e.reason}")
        except Exception as e:
            status_info["message"] = f"Exception: {str(e)}"
            print(f"  -> FAILED: {str(e)}")
            
        results.append(status_info)
        
    out_file = RESULTS_DIR / "abide_ii_attempt.json"
    summary = {
        "attempt_timestamp": "2026-11-06T16:50:00+05:30",
        "status": "SKIPPED",
        "reason": (
            "ABIDE II does not provide standardized CPAC/CC200 preprocessed time-series on public S3 "
            "comparable to the ABIDE I Preprocessed Connectome Project (PCP). The preprocessed derivatives "
            "require manual raw data preprocessing through CPAC/fMRIPrep or authenticated NITRC downloads "
            "which cannot be completed non-interactively without credentials."
        ),
        "endpoints_tested": results
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nLogged ABIDE II verification attempt to {out_file}")
    return summary

if __name__ == "__main__":
    check_endpoints()
