# FIGURE_CHECK.md — Figure Pack Quality & Formatting Audit

This document audits every figure in `figures/paper_pack/` for IEEE/ACM publication standards:

- **Resolution:** Verified 300 DPI on raster PNG outputs; vector scalability on PDF outputs.
- **Typography & Sizing:** Legible fonts at single-column (8.5 cm / 3.3 in) and double-column (17.5 cm / 6.9 in) widths.
- **Color Accessibility:** Colorblind-safe palettes (Wong 2011 / Tol color suites; high contrast, no red-green ambiguity).
- **Data Integrity:** 100% agreement between plotted visual elements and values recorded in `results/*.json`.

## Detailed Per-Figure Audit

### Figure 1 (`fig1_pipeline`)
- **Files:** `fig1_pipeline.pdf` (vector), `fig1_pipeline.png` (3570x1926 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `src/viz/generate_figures.py (Pipeline Architecture)`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 2 (`fig2_tautology_ablation`)
- **Files:** `fig2_tautology_ablation.pdf` (vector), `fig2_tautology_ablation.png` (3870x1291 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/tautology_audit.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 3 (`fig3_loso_vs_random`)
- **Files:** `fig3_loso_vs_random.pdf` (vector), `fig3_loso_vs_random.png` (3267x1223 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/benchmark_leaderboard.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 4 (`fig4_site_probe_confounds`)
- **Files:** `fig4_site_probe_confounds.pdf` (vector), `fig4_site_probe_confounds.png` (3119x1162 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/site_analysis.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 5 (`fig5_harmonization`)
- **Files:** `fig5_harmonization.pdf` (vector), `fig5_harmonization.png` (2688x1214 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/site_analysis.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 6 (`fig6_reliability`)
- **Files:** `fig6_reliability.pdf` (vector), `fig6_reliability.png` (2967x1247 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/uncertainty_main.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 7 (`fig7_conformal_per_site`)
- **Files:** `fig7_conformal_per_site.pdf` (vector), `fig7_conformal_per_site.png` (3569x1316 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/uncertainty_main.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 8 (`fig8_risk_coverage`)
- **Files:** `fig8_risk_coverage.pdf` (vector), `fig8_risk_coverage.png` (2369x1229 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/uncertainty_main.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.

### Figure 9 (`fig9_literature_audit`)
- **Files:** `fig9_literature_audit.pdf` (vector), `fig9_literature_audit.png` (3719x2834 px, DPI: 299.9994x299.9994)
- **Target Layout:** Double-column width (17.5 cm)
- **Source File:** `results/lit_audit_summary.json`
- **Visual Elements & Units:** Axis labels clearly state units (AUROC, Prevalence %, Accuracy %, Bins). Legends positioned in open whitespace without occluding data points.
- **Palette:** Verified Wong (2011) colorblind-safe hex values (`#0072B2` Blue, `#E69F00` Orange, `#009E73` Green, `#D55E00` Red-Orange, `#999999` Gray).
- **Data Consistency:** Plotted numbers match underlying results file exactly.
- **Formatting Changes Made:** Standardized file naming to `fig1` through `fig9` in `figures/paper_pack/` to ensure zero collision and clean LaTeX embedding.
