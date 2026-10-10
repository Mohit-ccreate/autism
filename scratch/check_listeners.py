import re

with open('webapp/index.html', encoding='utf-8') as f:
    html = f.read()

with open('webapp/app.js', encoding='utf-8') as f:
    js = f.read()

# 1. Check ID-based buttons
ids = re.findall(r'<button[^>]*id=["\']([^"\']+)["\']', html)
print("=== Checking ID-based Buttons ===")
for btn_id in ids:
    found = btn_id in js
    print(f"[{'OK' if found else 'MISSING'}] #{btn_id}")

# 2. Check Class-based buttons
classes = set(re.findall(r'<button[^>]*class=["\']([^"\']+)["\']', html))
all_classes = set()
for c in classes:
    for sub in c.split():
        all_classes.add(sub)

print("\n=== Checking Class-based Button Selectors in app.js ===")
for cls in sorted(all_classes):
    found = cls in js
    print(f"[{'OK' if found else 'NOT_REFERENCED'}] .{cls}")
