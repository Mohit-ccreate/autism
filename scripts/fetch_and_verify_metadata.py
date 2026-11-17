import json
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor

dois = {
    'maenner2023': '10.15585/mmwr.ss7202a1',
    'allison2012': '10.1016/j.jaac.2011.11.003',
    'di2014': '10.1038/mp.2013.78',
    'thabtah2018': '10.1016/j.ijmedinf.2018.06.009',
    'raj2020': '10.1016/j.procs.2020.03.399',
    'heinsfeld2018': '10.1016/j.nicl.2017.08.017',
    'eslami2019': '10.3389/fninf.2019.00070',
    'abraham2017': '10.1016/j.neuroimage.2016.10.045',
    'johnson2007': '10.1093/biostatistics/kxj037',
    'craddock2012': '10.1002/hbm.21333',
    'angelopoulos2023': '10.1561/2200000101',
    'vovk2005': '10.1007/b106715'
}

def fetch_crossref(item):
    key, doi = item
    url = f'https://api.crossref.org/works/{doi}'
    req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot/1.0 (mailto:test@example.com)'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
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
            
            entry = {
                'title': data.get('title', [''])[0] if isinstance(data.get('title'), list) else data.get('title'),
                'container_title': data.get('container-title', [''])[0] if isinstance(data.get('container-title'), list) else data.get('container-title'),
                'volume': data.get('volume'),
                'issue': data.get('issue'),
                'page': data.get('page'),
                'doi': doi,
                'authors': formatted_authors,
                'author_count': len(formatted_authors),
                'year': data.get('issued', {}).get('date-parts', [[None]])[0][0],
                'publisher': data.get('publisher')
            }
            print(f"[OK] {key}: {len(formatted_authors)} authors, {entry['container_title']}", flush=True)
            return key, entry
    except Exception as e:
        print(f"[ERR] {key} ({doi}): {e}", flush=True)
        return key, None

with ThreadPoolExecutor(max_workers=6) as executor:
    results = dict(executor.map(fetch_crossref, dois.items()))

with open('scripts/crossref_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f"Successfully processed {len([k for k, v in results.items() if v])} records.", flush=True)
