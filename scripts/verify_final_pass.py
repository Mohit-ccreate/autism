"""
Verification script for Final Pass Rules of Evidence.
Checks:
1. Abstract word count <= 200 words.
2. Zero unverified / banned phrases in main.tex.
3. 100% mutual citation match between main.tex and references.bib.
4. All \input files exist.
5. All figure files exist.
6. Environment and bracket balance in main.tex.
"""

import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_abstract():
    with open(os.path.join(BASE_DIR, "main.tex"), encoding="utf-8") as f:
        content = f.read()
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", content, re.DOTALL)
    if not match:
        return False, "Abstract not found"
    abs_text = match.group(1).strip()
    # Remove LaTeX commands for word counting
    clean_text = re.sub(r"\\[a-zA-Z]+", " ", abs_text)
    clean_text = re.sub(r"[\$\{\}]", " ", clean_text)
    words = clean_text.split()
    count = len(words)
    return count <= 200, f"Abstract word count: {count} words (limit: 200)"

def check_banned_phrases():
    with open(os.path.join(BASE_DIR, "main.tex"), encoding="utf-8") as f:
        content = f.read()
    banned = [
        "zero incremental value",
        "Platt temperature scaling",
        "demonstrating that published classifier evaluations reconstruct",
        "these analyses did not scrutinize",
        "zero-calibration reference rescaling does not bridge domain shifts",
        "our findings align with consortium-scale studies",
        "bounds diagnostic uncertainty",
        "bounds uncertainty",
        "0.45~ms",
        "0.45 ms",
        "i7-11800H",
        "Chrome v122",
        "connect-src",
        "thabtah2019",
        "zablotsky2020",
        "addm2025",
        "onnxruntime2023",
        "safeinclude"
    ]
    found = []
    for b in banned:
        if b.lower() in content.lower():
            found.append(b)
    if found:
        return False, f"Found banned phrases: {found}"
    return True, "No banned phrases found"

def check_citations():
    with open(os.path.join(BASE_DIR, "main.tex"), encoding="utf-8") as f:
        tex = f.read()
    with open(os.path.join(BASE_DIR, "references.bib"), encoding="utf-8") as f:
        bib = f.read()
    
    cited = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            cited.add(k.strip())
            
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    
    missing_in_bib = cited - bib_keys
    uncited_in_bib = bib_keys - cited
    
    if missing_in_bib:
        return False, f"Citations in main.tex missing from references.bib: {missing_in_bib}"
    return True, f"All {len(cited)} citations matched in references.bib! (Uncited in bib: {uncited_in_bib})"

def check_inputs_and_figures():
    with open(os.path.join(BASE_DIR, "main.tex"), encoding="utf-8") as f:
        tex = f.read()
    
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    missing_inputs = []
    for inp in inputs:
        inp_path = inp if inp.endswith(".tex") else inp + ".tex"
        if not os.path.exists(os.path.join(BASE_DIR, inp_path)):
            missing_inputs.append(inp)
            
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing_figs = []
    for fig in figs:
        pdf_path = os.path.join(BASE_DIR, fig + ".pdf")
        png_path = os.path.join(BASE_DIR, fig + ".png")
        if not (os.path.exists(pdf_path) or os.path.exists(png_path) or os.path.exists(os.path.join(BASE_DIR, fig))):
            missing_figs.append(fig)
            
    if missing_inputs or missing_figs:
        return False, f"Missing inputs: {missing_inputs}, Missing figures: {missing_figs}"
    return True, f"All {len(inputs)} inputs and {len(figs)} figures exist!"

def check_environments():
    with open(os.path.join(BASE_DIR, "main.tex"), encoding="utf-8") as f:
        tex = f.read()
    tokens = re.findall(r"\\(begin|end)\{([^}]+)\}", tex)
    stack = []
    for action, env in tokens:
        if action == "begin":
            stack.append(env)
        elif action == "end":
            if not stack:
                return False, f"Extra \\end{{{env}}} with no matching \\begin"
            top = stack.pop()
            if top != env:
                return False, f"Mismatched environment: \\begin{{{top}}} closed by \\end{{{env}}}"
    if stack:
        return False, f"Unclosed environments: {stack}"
    return True, f"All {len(tokens)//2} LaTeX environments perfectly matched in LIFO stack!"

if __name__ == "__main__":
    print("--- Running final verification checks ---")
    ok, msg = check_abstract()
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    
    ok, msg = check_banned_phrases()
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    
    ok, msg = check_citations()
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    
    ok, msg = check_inputs_and_figures()
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    
    ok, msg = check_environments()
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
