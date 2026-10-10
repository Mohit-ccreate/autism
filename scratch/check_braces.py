with open("main.tex", "r", encoding="utf-8") as f:
    text = f.read()

lines = text.splitlines()
cleaned_lines = []
for l in lines:
    l_safe = l.replace(r"\%", "__PERCENT__").replace(r"\{", "__LBRACE__").replace(r"\}", "__RBRACE__")
    if "%" in l_safe:
        l_safe = l_safe.split("%")[0]
    cleaned_lines.append(l_safe)

full_clean = "\n".join(cleaned_lines)
o = full_clean.count("{")
c = full_clean.count("}")
print(f"Open: {o}, Close: {c}, Diff: {o - c}")
if o != c:
    balance = 0
    for i, line in enumerate(cleaned_lines, 1):
        cur_o = line.count("{")
        cur_c = line.count("}")
        balance += (cur_o - cur_c)
        if cur_o != cur_c:
            print(f"Line {i:3d} (delta {cur_o - cur_c:+2d}, cum {balance:2d}): {line.strip()[:60]}")
