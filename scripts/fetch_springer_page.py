import urllib.request
import re

req = urllib.request.Request('https://link.springer.com/book/10.1007/b106715', headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
try:
    with urllib.request.urlopen(req, timeout=15) as r:
        html = r.read().decode('utf-8')
        print("Length:", len(html))
        # Authors
        authors = re.findall(r'itemprop="name">([^<]+)<', html)
        print("Itemprop name:", authors)
        citation_authors = re.findall(r'name="citation_author"\s+content="([^"]+)"', html)
        print("Citation authors:", citation_authors)
        title = re.findall(r'name="citation_title"\s+content="([^"]+)"', html)
        print("Citation title:", title)
        publisher = re.findall(r'name="citation_publisher"\s+content="([^"]+)"', html)
        print("Citation publisher:", publisher)
        date = re.findall(r'name="citation_publication_date"\s+content="([^"]+)"', html)
        print("Citation date:", date)
        doi = re.findall(r'name="citation_doi"\s+content="([^"]+)"', html)
        print("Citation doi:", doi)
except Exception as e:
    print("Error:", e)
