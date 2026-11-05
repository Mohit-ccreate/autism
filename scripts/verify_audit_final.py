"""
Comprehensive verification of all 14 submission criteria.
"""
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "main.tex"), "r", encoding="utf-8") as f:
    tex = f.read()

with open(os.path.join(BASE_DIR, "references.bib"), "r", encoding="utf-8") as f:
    bib = f.read()

checks = []

# 1. Math and glyph artifacts in main.tex
checks.append(("Abstract summation glyph", r"($\sum_{i=1}^{10} A_i \ge \tau$, 100.0\% concordance)" in tex))
checks.append(("Abstract p-value standard text", r"($p = 0.315$, two-sided)" in tex))
checks.append(("Equation (2) Pearson correlation closed radicals", r"R_{ij} = \frac{\sum_{t=1}^T (s_{i,t} - \bar{s}_i)(s_{j,t} - \bar{s}_j)}{\sqrt{\sum_{t=1}^T (s_{i,t} - \bar{s}_i)^2} \sqrt{\sum_{t=1}^T (s_{j,t} - \bar{s}_j)^2}}" in tex))
checks.append(("Conformal Quantile Equation after Eq 8", r"where $\hat{q} = \text{Quantile}\left(\{s_i\}_{i=1}^{n_{\text{cal}}}, \frac{\lceil(n_{\text{cal}}+1)(1-\alpha)\rceil}{n_{\text{cal}}}\right)$." in tex))

# 2. Markdown wrapper junk in URLs
checks.append(("No markdown link wrapper in main.tex", r"\url{[https" not in tex and r"[\url" not in tex))
checks.append(("Clean GitHub repository URL", r"\url{https://github.com/Mohit-ccreate/autism}" in tex))
checks.append(("Clean Kaggle Toddler URL in bib", "https://www.kaggle.com/datasets/fabdelja/autism-screening-for-toddlers" in bib and "url" in bib))
checks.append(("Clean PCP URL in bib", r"howpublished = {\url{http://preprocessed-connectomes-project.org/abide/}}" in bib))

# 3. Author attribution in references.bib
checks.append(("Yin-Shan Fan in craddock2013pcp", "Yin-Shan Fan" in bib))
checks.append(("No Yassine Fan", "Yassine" not in bib))

# 4. Deduplication and concordance tables
checks.append(("Narrow-key matching in Table II", "Narrow-key matching" in tex))
checks.append(("Threshold concordance table label", "tab:threshold_concordance" in tex))
checks.append(("Shortened column headers (Adult, Adol., Child, Toddler)", r"\textbf{Adult} & \textbf{Adol.} & \textbf{Child} & \textbf{Toddler}" in tex))
checks.append(("Sample sizes in caption (Adult N=667, Adol. N=97, Child N=262, Toddler N=850)", "(Adult $N=667$, Adol. $N=97$, Child $N=262$, Toddler $N=850$)" in tex))

# 5. Abstract word count <= 200
match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.DOTALL)
clean_abs = re.sub(r"\\[a-zA-Z]+", " ", match.group(1))
clean_abs = re.sub(r"[\$\{\}]", " ", clean_abs)
word_count = len(clean_abs.split())
checks.append((f"Abstract word count <= 200 (actual: {word_count})", word_count <= 200))

# 6. Zero banned phrases
banned = ["zero incremental value", "Platt temperature scaling", "bounds diagnostic uncertainty", "0.45 ms", "Chrome v122"]
checks.append(("Zero banned phrases", not any(b.lower() in tex.lower() for b in banned)))

all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"[{status}] {name}")

if all_pass:
    print("\n>>> ALL CRITERIA VERIFIED AND PASSED SUCCESSFULLY! <<<")
else:
    print("\n>>> SOME CRITERIA FAILED! <<<")
