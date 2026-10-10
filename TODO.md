# TODO & Unresolved Items (Final Pass)

This document lists strictly the items that could not be independently verified from existing workspace logs/code and require user supply or future consortium resources.

---

## 1. Unverified Items Requiring User Input

1. **`\url{TODO}`: Persistent Code and Preprocessing Repository URL**
   - *Status:* Unverified placeholder in Section I and Section VII.
   - *Action for User:* Provide the official public GitHub or Zenodo URL hosting the evaluation scripts, or confirm anonymous submission URL.

2. **`\todo{SELECTION: exact Nilearn CPAC cache filtering rule}` (Section III-C)**
   - *Status:* Subject counts across the four sites (OHSU: 25/28, OLIN: 28/36, PITT: 50/57, SDSU: 17/36; Total 120, 58 ASD, 62 TD) reflect Nilearn's local cache from `fetch_abide_pcp(n_subjects=120, ...)`. The exact sequential indexing or quality-filtering criterion executed internally by Nilearn that produced these site-specific balanced subsets cannot be reconstructed from the ingestion script.
   - *Action for User:* Verify whether a specific subject ID inclusion list was applied or document Nilearn's default index order.

3. **`\todo{QC: prospective criteria}` (Section III-C)**
   - *Status:* The retrospective benchmark reports mean framewise displacement ($0.14 \pm 0.08$~mm across all 120 subjects), but explicit prospective quality control thresholds (e.g., maximum allowable FD cutoff, minimum continuous BOLD volumes) were not logged.
   - *Action for User:* Supply prospective clinical exclusion thresholds for future cohorts.

4. **`\todo{RERUN: prospective multi-site ABIDE scaling}` (Section VI)**
   - *Status:* The current benchmark evaluates $N = 120$ subjects across 4 scanning centers. Power analysis confirms the cohort is underpowered below 62.7% accuracy. Scaling to the full ABIDE-I cohort ($N = 1{,}112$ across 17 sites) under 17-fold Leave-Center-Out validation was not run in this workspace.
   - *Action for User:* Run full 17-site Leave-Center-Out validation if consortium-scale verification is required.

5. **`\todo{PAGE LIMIT}` (Section VII)**
   - *Status:* Conference page budget constraints (IEEE 6-page vs 8-page format).
   - *Action for User:* Verify target conference page limits during final typesetting.
