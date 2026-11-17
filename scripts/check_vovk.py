import urllib.request
import json

req = urllib.request.Request('https://api.crossref.org/works/10.1007/b106715', headers={'User-Agent': 'ResearchBot/1.0 (mailto:test@example.com)'})
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        d = json.loads(r.read())['message']
        print("Keys:", d.keys())
        print("Title:", d.get('title'))
        print("Author:", d.get('author'))
        print("Editor:", d.get('editor'))
        print("Publisher:", d.get('publisher'))
        print("Issued:", d.get('issued'))
except Exception as e:
    print("Error:", e)
