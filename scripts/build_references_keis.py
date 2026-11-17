"""
Build references_keis.bib with exactly 15 verified entries.
Strictly conforms to rules:
1. Complete author list in "Last, First and Last, First" format (no 'et al.', no 'others').
2. Entry types: article, inproceedings, book.
3. Abbreviated journal titles, LaTeX escaped special characters (e.g. Cand{\`e}s).
4. Volume, number, pages/article number, year, doi where one exists.
5. Original references.bib is untouched.
6. Parses with bibtexparser, verifying 15 entries and author counts.
"""

from pathlib import Path
import bibtexparser

ENTRIES = [
    {
        "type": "article",
        "key": "maenner2023",
        "fields": {
            "author": (
                "Maenner, Matthew J. and Warren, Zachary and Williams, Ashley Robinson and "
                "Amoakohene, Esther and Bakian, Amanda V. and Bilder, Deborah A. and "
                "Durkin, Maureen S. and Fitzgerald, Robert T. and Furnier, Sarah M. and "
                "Hughes, Michelle M. and Ladd-Acosta, Christine M. and McArthur, Dedria and "
                "Pas, Elise T. and Salinas, Angelica and Vehorn, Alison and Williams, Susan and "
                "Esler, Amy and Grzybowski, Andrea and Hall-Lande, Jennifer and "
                "Nguyen, Ruby H. N. and Pierce, Karen and Zahorodny, Walter and "
                "Hudson, Allison and Hallas, Libby and Mancilla, Kristen Clancy and "
                "Patrick, Mary and Shenouda, Josephine and Sidwell, Kate and "
                "DiRienzo, Monica and Gutierrez, Johanna and Spivey, Margaret H. and "
                "Lopez, Maya and Pettygrove, Sydney and Schwenk, Yvette D. and "
                "Washington, Anita and Shaw, Kelly A."
            ),
            "title": "{Prevalence and Characteristics of Autism Spectrum Disorder Among Children Aged 8 Years --- {Autism and Developmental Disabilities Monitoring Network}, 11 Sites, {United States}, 2020}",
            "journal": "{MMWR Surveill. Summ.}",
            "volume": "{72}",
            "number": "{2}",
            "pages": "{1--14}",
            "year": "{2023}",
            "doi": "{10.15585/mmwr.ss7202a1}",
        },
    },
    {
        "type": "article",
        "key": "allison2012",
        "fields": {
            "author": "Allison, Carrie and Auyeung, Bonnie and Baron-Cohen, Simon",
            "title": "{Toward Brief ``Red Flags'' for Autism Screening: The Short {Autism Spectrum Quotient} and the Short {Quantitative Checklist} in 1,000 Cases and 3,000 Controls}",
            "journal": "{J. Am. Acad. Child Adolesc. Psychiatry}",
            "volume": "{51}",
            "number": "{2}",
            "pages": "{202--212.e7}",
            "year": "{2012}",
            "doi": "{10.1016/j.jaac.2011.11.003}",
        },
    },
    {
        "type": "article",
        "key": "di2014",
        "fields": {
            "author": (
                "Di Martino, Adriana and Yan, Chao-Gan and Li, Qingyang and Denio, Erin and "
                "Castellanos, Francisco Xavier and Alaerts, Kaat and Anderson, Jeffrey S. and "
                "Assaf, Michal and Bookheimer, Susan Y. and Dapretto, Mirella and "
                "Deen, Benjamin and Delmonte, Sarah and Dinstein, Ilan and "
                "Ertl-Wagner, Birgit and Fair, Damien A. and Gallagher, Louise and "
                "Kennedy, David P. and Keown, Cheryl L. and Keysers, Christian and "
                "Lainhart, Janet E. and Lord, Catherine and Luna, Beatriz and "
                "Menon, Vinod and Minshew, Nancy J. and Monk, Christopher S. and "
                "Mueller, Sven and M{\\\"u}ller, Ralph-Axel and Nebel, Mary Beth and "
                "Nigg, Joel T. and O'Hearn, Kirstin and Pelphrey, Kevin A. and "
                "Peltier, Scott J. and Rudie, Jeffrey D. and Sunaert, Stefan and "
                "Thioux, Marc and Tyszka, J. Michael and Uddin, Lucina Q. and "
                "Verhoeven, Judith S. and Wenderoth, Nicole and Wiggins, Jillian Lee and "
                "Mostofsky, Stewart H. and Milham, Michael P."
            ),
            "title": "{The {Autism Brain Imaging Data Exchange}: Towards a Large-Scale Evaluation of the Intrinsic Brain Architecture in Autism}",
            "journal": "{Mol. Psychiatry}",
            "volume": "{19}",
            "number": "{6}",
            "pages": "{659--667}",
            "year": "{2014}",
            "doi": "{10.1038/mp.2013.78}",
        },
    },
    {
        "type": "article",
        "key": "thabtah2018",
        "fields": {
            "author": "Thabtah, Fadi and Kamalov, Firuz and Rajab, Khairan",
            "title": "{A New Computational Intelligence Approach to Detect Autistic Features for Autism Screening}",
            "journal": "{Int. J. Med. Inform.}",
            "volume": "{117}",
            "pages": "{112--124}",
            "year": "{2018}",
            "doi": "{10.1016/j.ijmedinf.2018.06.009}",
        },
    },
    {
        "type": "article",
        "key": "raj2020",
        "fields": {
            "author": "Raj, Suman and Masood, Sarfaraz",
            "title": "{Analysis and Detection of Autism Spectrum Disorder Using Machine Learning Techniques}",
            "journal": "{Procedia Comput. Sci.}",
            "volume": "{167}",
            "pages": "{994--1004}",
            "year": "{2020}",
            "doi": "{10.1016/j.procs.2020.03.399}",
        },
    },
    {
        "type": "article",
        "key": "heinsfeld2018",
        "fields": {
            "author": "Heinsfeld, Anibal S{\\'o}lon and Franco, Alexandre Rosa and Craddock, R. Cameron and Buchweitz, Augusto and Meneguzzi, Felipe",
            "title": "{Identification of Autism Spectrum Disorder Using Deep Learning and the {ABIDE} Dataset}",
            "journal": "{NeuroImage Clin.}",
            "volume": "{17}",
            "pages": "{16--23}",
            "year": "{2018}",
            "doi": "{10.1016/j.nicl.2017.08.017}",
        },
    },
    {
        "type": "article",
        "key": "eslami2019",
        "fields": {
            "author": "Eslami, Taban and Mirjalili, Vahid and Fong, Alvis and Laird, Angela R. and Saeed, Fahad",
            "title": "{{ASD-DiagNet}: A Hybrid Learning Approach for Detection of Autism Spectrum Disorder Using {fMRI} Data}",
            "journal": "{Front. Neuroinform.}",
            "volume": "{13}",
            "pages": "{70}",
            "year": "{2019}",
            "doi": "{10.3389/fninf.2019.00070}",
        },
    },
    {
        "type": "article",
        "key": "abraham2017",
        "fields": {
            "author": "Abraham, Alexandre and Milham, Michael P. and Di Martino, Adriana and Craddock, R. Cameron and Samaras, Dimitris and Thirion, Bertrand and Varoquaux, Ga{\\\"e}l",
            "title": "{Deriving Reproducible Biomarkers from Multi-Site Resting-State {fMRI}: {Dictionary Learning} and Benchmarks on the {ABIDE} Dataset}",
            "journal": "{NeuroImage}",
            "volume": "{147}",
            "pages": "{736--745}",
            "year": "{2017}",
            "doi": "{10.1016/j.neuroimage.2016.10.045}",
        },
    },
    {
        "type": "inproceedings",
        "key": "kan2022",
        "fields": {
            "author": "Kan, Xuan and Dai, Wei and Cui, Hejie and Zhang, Zilong and Guo, Ying and Yang, Carl",
            "title": "{Brain Network Transformer}",
            "booktitle": "{Advances in Neural Information Processing Systems}",
            "volume": "{35}",
            "pages": "{25586--25599}",
            "year": "{2022}",
        },
    },
    {
        "type": "article",
        "key": "johnson2007",
        "fields": {
            "author": "Johnson, W. Evan and Li, Cheng and Rabinovic, Ariel",
            "title": "{Adjusting Batch Effects in Microarray Expression Data Using Empirical {Bayes} Methods}",
            "journal": "{Biostatistics}",
            "volume": "{8}",
            "number": "{1}",
            "pages": "{118--127}",
            "year": "{2007}",
            "doi": "{10.1093/biostatistics/kxj037}",
        },
    },
    {
        "type": "article",
        "key": "craddock2012",
        "fields": {
            "author": "Craddock, R. Cameron and James, G. Andrew and Holtzheimer, Paul E. and Hu, Xiaoping P. and Mayberg, Helen S.",
            "title": "{A Whole Brain {fMRI} Atlas Generated via Spatially Constrained Spectral Clustering}",
            "journal": "{Hum. Brain Mapp.}",
            "volume": "{33}",
            "number": "{8}",
            "pages": "{1914--1928}",
            "year": "{2012}",
            "doi": "{10.1002/hbm.21333}",
        },
    },
    {
        "type": "inproceedings",
        "key": "guo2017",
        "fields": {
            "author": "Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.",
            "title": "{On Calibration of Modern Neural Networks}",
            "booktitle": "{Proceedings of the 34th International Conference on Machine Learning}",
            "series": "{Proceedings of Machine Learning Research}",
            "volume": "{70}",
            "pages": "{1321--1330}",
            "year": "{2017}",
        },
    },
    {
        "type": "inproceedings",
        "key": "romano2020",
        "fields": {
            "author": "Romano, Yaniv and Sesia, Matteo and Cand{\\`e}s, Emmanuel J.",
            "title": "{Classification with Valid and Adaptive Coverage}",
            "booktitle": "{Advances in Neural Information Processing Systems}",
            "volume": "{33}",
            "pages": "{3585--3595}",
            "year": "{2020}",
        },
    },
    {
        "type": "article",
        "key": "angelopoulos2023",
        "fields": {
            "author": "Angelopoulos, Anastasios N. and Bates, Stephen",
            "title": "{Conformal Prediction: A Gentle Introduction}",
            "journal": "{Found. Trends Mach. Learn.}",
            "volume": "{16}",
            "number": "{4}",
            "pages": "{494--591}",
            "year": "{2023}",
            "doi": "{10.1561/2200000101}",
        },
    },
    {
        "type": "book",
        "key": "vovk2005",
        "fields": {
            "author": "Vovk, Vladimir and Gammerman, Alex and Shafer, Glenn",
            "title": "{Algorithmic Learning in a Random World}",
            "publisher": "{Springer}",
            "address": "{New York, NY}",
            "year": "{2005}",
            "doi": "{10.1007/b106715}",
        },
    },
]

def format_bibtex(entries):
    lines = []
    for entry in entries:
        lines.append(f"@{entry['type']}{{{entry['key']},")
        for k, v in entry["fields"].items():
            # If not already wrapped in braces, wrap it
            val_str = str(v).strip()
            if not (val_str.startswith("{") and val_str.endswith("}")):
                val_str = f"{{{val_str}}}"
            lines.append(f"  {k:<12}= {val_str},")
        # Remove trailing comma on last field for neatness or keep standard BibTeX syntax
        lines[-1] = lines[-1].rstrip(",")
        lines.append("}\n")
    return "\n".join(lines)

def main():
    repo_root = Path(__file__).resolve().parent.parent
    target_path = repo_root / "references_keis.bib"
    
    bib_content = format_bibtex(ENTRIES)
    target_path.write_text(bib_content, encoding="utf-8")
    print(f"Wrote {target_path}")

    # Validate parsing
    library = bibtexparser.parse_file(str(target_path))
    print(f"\nParsed {len(library.entries)} entries.")
    assert len(library.entries) == 15, f"Expected 15 entries, got {len(library.entries)}"

    print("\nEntry details and author counts:")
    for entry in library.entries:
        # Bibtexparser entry
        author_val = entry.get("author").value if hasattr(entry.get("author"), "value") else str(entry.get("author"))
        authors = [a.strip() for a in author_val.split(" and ")]
        print(f"  - Key: {entry.key:<18} Type: {entry.entry_type:<14} Authors: {len(authors):<3}")
        for a in authors:
            assert "," in a, f"Author '{a}' in {entry.key} is not in 'Last, First' format!"
            assert "et al." not in a.lower(), f"'et al.' found in {a}"
            assert "others" not in a.lower(), f"'others' found in {a}"

    print("\nAll 15 entries validated successfully!")

if __name__ == "__main__":
    main()
