import re

with open('webapp/index.html', encoding='utf-8') as f:
    c = f.read()

tab_btns = re.findall(r'data-tab=["\']([^"\']+)["\']', c)
tab_panels = re.findall(r'id=["\'](tab-[^"\']+)["\']', c)

print(f"Found {len(tab_btns)} tab buttons and {len(tab_panels)} tab panels:")
for tb in tab_btns:
    if tb in tab_panels:
        print(f"  [OK] {tb}")
    else:
        print(f"  [ERROR] {tb} has NO corresponding panel in HTML!")

for p in tab_panels:
    if p not in tab_btns:
        print(f"  [ORPHAN PANEL] {p}")
