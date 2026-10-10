import re

with open("main.tex", "r", encoding="utf-8") as f:
    text = f.read()

begins = re.findall(r"\\begin\{([^}]+)\}", text)
ends = re.findall(r"\\end\{([^}]+)\}", text)

print(f"Total \\begin: {len(begins)}, Total \\end: {len(ends)}")

from collections import Counter
c_beg = Counter(begins)
c_end = Counter(ends)

all_envs = set(begins) | set(ends)
mismatch = False
for env in sorted(all_envs):
    b_count = c_beg[env]
    e_count = c_end[env]
    if b_count != e_count:
        print(f"MISMATCH for '{env}': begin={b_count}, end={e_count}")
        mismatch = True
    else:
        print(f"OK: {env} ({b_count})")

if not mismatch:
    print("ALL LaTeX environments perfectly matched!")
