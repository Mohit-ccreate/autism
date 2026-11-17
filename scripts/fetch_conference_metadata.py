import urllib.request
import re

urls = {
    'kan2022': 'https://proceedings.neurips.cc/paper_files/paper/2022/hash/a487920150993952f41b44ecddb7dfa8-Abstract-Conference.html',
    'guo2017': 'https://proceedings.mlr.press/v70/guo17a.html',
    'romano2020': 'https://proceedings.neurips.cc/paper_files/paper/2020/hash/26365b2114d2417730e12d1b033ad885-Abstract.html'
}

for key, url in urls.items():
    print(f"Fetching {key} from {url}...")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            print(f"  Got {len(html)} bytes")
            # find title and authors
            # PMLR has <meta name="citation_author" content="..."> or similar
            authors = re.findall(r'meta name="citation_author" content="([^"]+)"', html)
            if not authors:
                authors = re.findall(r'citation_author" content="([^"]+)"', html)
            title_m = re.search(r'citation_title" content="([^"]+)"', html)
            title = title_m.group(1) if title_m else None
            pdf_url = re.search(r'citation_pdf_url" content="([^"]+)"', html)
            print(f"  Title: {title}")
            print(f"  Authors ({len(authors)}): {authors}")
    except Exception as e:
        print(f"  Error: {e}")
