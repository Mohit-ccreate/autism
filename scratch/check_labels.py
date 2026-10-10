import re

with open("main.tex", "r", encoding="utf-8") as f:
    text = f.read()

labels = set(re.findall(r"\\label\{([^}]+)\}", text))
refs = set(re.findall(r"\\ref\{([^}]+)\}", text))

print(f"Total labels defined: {len(labels)}")
print(f"Total refs called: {len(refs)}")

missing = refs - labels
if missing:
    print(f"MISSING LABELS: {missing}")
else:
    print("ALL \\ref calls have matching \\label! 0 broken references.")
