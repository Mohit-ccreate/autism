import json
import time
import urllib.request

dois = {
    'allison2012': '10.1016/j.jaac.2011.11.003',
    'raj2020': '10.1016/j.procs.2020.03.399',
    'heinsfeld2018': '10.1016/j.nicl.2017.08.017',
    'angelopoulos2023': '10.1561/2200000101',
    'vovk2005': '10.1007/b106715'
}

with open('scripts/crossref_results.json', 'r', encoding='utf-8') as f:
    results = json.load(f)

for key, doi in dois.items():
    print(f"Fetching {key} ({doi})...", flush=True)
    url = f'https://api.crossref.org/works/{doi}'
    req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot/1.0 (mailto:test@example.com)'})
    try:
        time.sleep(1.5)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))['message']
            authors = data.get('author', [])
            formatted_authors = []
            for a in authors:
                family = a.get('family', '')
                given = a.get('given', '')
                if family and given:
                    formatted_authors.append(f"{family}, {given}")
                elif family:
                    formatted_authors.append(family)
            
            # handle date
            year = None
            for date_field in ['published-print', 'published-online', 'issued', 'created']:
                if date_field in data and 'date-parts' in data[date_field] and data[date_field]['date-parts']:
                    parts = data[date_field]['date-parts'][0]
                    if parts and parts[0]:
                        year = parts[0]
                        break

            results[key] = {
                'title': data.get('title', [''])[0] if isinstance(data.get('title'), list) else data.get('title'),
                'container_title': data.get('container-title', [''])[0] if isinstance(data.get('container-title'), list) else data.get('container-title'),
                'volume': data.get('volume'),
                'issue': data.get('issue'),
                'page': data.get('page'),
                'doi': doi,
                'authors': formatted_authors,
                'author_count': len(formatted_authors),
                'year': year,
                'publisher': data.get('publisher')
            }
            print(f"  [OK] {key}: {len(formatted_authors)} authors, vol {data.get('volume')}, year {year}", flush=True)
    except Exception as e:
        print(f"  [ERR] {key} ({doi}): {e}", flush=True)

with open('scripts/crossref_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print("\nDone.", flush=True)
