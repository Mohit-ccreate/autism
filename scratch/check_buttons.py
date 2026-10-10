import re

with open('webapp/index.html', encoding='utf-8') as f:
    html = f.read()

buttons = re.findall(r'<button\s+([^>]+)>', html)
print(f"Total buttons found: {len(buttons)}")
for b in buttons:
    id_m = re.search(r'id=["\']([^"\']+)["\']', b)
    class_m = re.search(r'class=["\']([^"\']+)["\']', b)
    btn_id = id_m.group(1) if id_m else 'NO_ID'
    btn_cls = class_m.group(1) if class_m else 'NO_CLASS'
    print(f"ID: {btn_id:32s} | Class: {btn_cls}")
