import re

with open(r'C:\Users\mohit\.gemini\antigravity\brain\34e10c40-45e6-406b-95cf-42dc27709a64\.system_generated\steps\66\content.md', encoding='utf-8') as f:
    text = f.read()

for m in re.finditer(r'href=[\'"]([^\'"]+)[\'"]', text):
    link = m.group(1)
    if any(k in link.lower() for k in ['static', 'download', 'zip', 'arff', 'data', 'api']):
        print(link)
