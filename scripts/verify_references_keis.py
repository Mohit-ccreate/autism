"""
Verification script for references_keis.bib
Tests parsing with bibtexparser, verifies 15 entries, checks formatting rules.
"""

from pathlib import Path
import bibtexparser

EXPECTED_KEYS = [
    "maenner2023",
    "allison2012",
    "di2014",
    "thabtah2018",
    "raj2020",
    "heinsfeld2018",
    "eslami2019",
    "abraham2017",
    "kan2022",
    "johnson2007",
    "craddock2012",
    "guo2017",
    "romano2020",
    "angelopoulos2023",
    "vovk2005",
]

def verify():
    bib_file = Path("references_keis.bib")
    assert bib_file.exists(), "references_keis.bib does not exist!"

    library = bibtexparser.parse_file(str(bib_file))
    print(f"Total entries parsed: {len(library.entries)}")
    assert len(library.entries) == 15, f"Expected exactly 15 entries, got {len(library.entries)}"

    parsed_keys = [e.key for e in library.entries]
    assert parsed_keys == EXPECTED_KEYS, f"Keys mismatch!\nExpected: {EXPECTED_KEYS}\nFound: {parsed_keys}"

    print("\n--- Per-Entry Details and Author Counts ---")
    for e in library.entries:
        # Extract author
        author_val = e.get("author").value if hasattr(e.get("author"), "value") else str(e.get("author"))
        # Strip potential wrapping braces
        author_val = author_val.strip("{} ")
        authors = [a.strip() for a in author_val.split(" and ")]
        
        # Check rule 1: complete author list in Last, First format
        for a in authors:
            assert "," in a, f"Author '{a}' in key '{e.key}' missing comma (must be 'Last, First')!"
            assert "et al." not in a.lower(), f"'et al.' detected in author '{a}' of '{e.key}'!"
            assert "others" not in a.lower(), f"'others' detected in author '{a}' of '{e.key}'!"
        
        # Check rule 2: entry types
        assert e.entry_type in ["article", "inproceedings", "book"], f"Invalid entry type '{e.entry_type}' in '{e.key}'"
        
        # Check required fields
        title = e.get("title").value if hasattr(e.get("title"), "value") else str(e.get("title"))
        year = e.get("year").value if hasattr(e.get("year"), "value") else str(e.get("year"))
        assert title, f"Missing title in {e.key}"
        assert year, f"Missing year in {e.key}"

        doi_val = e.get("doi")
        doi_str = (doi_val.value if hasattr(doi_val, "value") else str(doi_val)) if doi_val else "None"

        print(f"Key: {e.key:<18} | Type: {e.entry_type:<14} | Authors: {len(authors):<2} | DOI: {doi_str}")

    print("\nSUCCESS: All 15 entries parsed and passed all criteria!")

if __name__ == "__main__":
    verify()
