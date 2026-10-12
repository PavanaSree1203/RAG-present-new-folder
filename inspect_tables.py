"""
Stage 4 (inspection only): see how financial statements are laid out in the PDFs and what three
extraction methods return for them. Nothing is saved; this guides the extractor design.

Usage:  python inspect_tables.py                 # default sample years
        python inspect_tables.py 2005 2012 2022  # chosen years
"""
import re
import sys
from pathlib import Path

import pdfplumber

DATA_DIR = Path("data")
DEFAULT_YEARS = [2005, 2012, 2022]
# Standard SEC statement titles (generic to any 10-K, not company specific)
TITLES = {
    "OPERATIONS": re.compile(r"CONSOLIDATED\s+STATEMENTS?\s+OF\s+(OPERATIONS|INCOME|EARNINGS)", re.I),
    "BALANCE": re.compile(r"CONSOLIDATED\s+BALANCE\s+SHEETS?", re.I),
}
MIN_DIGITS = 150        # a real statement page is number-heavy; the index/TOC page is not
ROWS_TO_SHOW = 16


def find_statement_page(pdf, title_re):
    for i, page in enumerate(pdf.pages):
        text = page.extract_text() or ""
        if title_re.search(text[:600]) and sum(ch.isdigit() for ch in text) >= MIN_DIGITS:
            return i
    return None


def rows_from_words(page, gap=6.0, ytol=3.0):
    """Method 3: rebuild rows from word coordinates; a new cell starts at a horizontal gap > `gap` points."""
    words = page.extract_words(x_tolerance=1.5, y_tolerance=2)
    buckets = {}
    for w in words:
        buckets.setdefault(round(w["top"] / ytol), []).append(w)
    rows = []
    for key in sorted(buckets):
        ws = sorted(buckets[key], key=lambda w: w["x0"])
        cells, cur, last = [], [ws[0]["text"]], ws[0]["x1"]
        for w in ws[1:]:
            if w["x0"] - last > gap:
                cells.append(" ".join(cur)); cur = [w["text"]]
            else:
                cur.append(w["text"])
            last = w["x1"]
        cells.append(" ".join(cur))
        rows.append(cells)
    return rows


def show_tables(label, tables):
    print(f"  [{label}] tables found: {len(tables)}")
    if tables:
        t = max(tables, key=lambda t: len(t.extract()))
        data = t.extract()
        print(f"      largest: {len(data)} rows x {max(len(r) for r in data)} cols")
        for r in data[:8]:
            print("      ", [(c or "")[:22] for c in r])


def inspect(year):
    files = sorted(DATA_DIR.glob(f"*_{year}.pdf"))
    if not files:
        print(f"\n##### {year}: no PDF found in {DATA_DIR}")
        return
    with pdfplumber.open(files[0]) as pdf:
        for name, title_re in TITLES.items():
            idx = find_statement_page(pdf, title_re)
            print("\n" + "=" * 100)
            if idx is None:
                print(f"{year} {name}: no page matched (garbled/OCR year or different title)")
                continue
            page = pdf.pages[idx]
            print(f"{year} {name}: PDF page {idx + 1} | ruling lines={len(page.lines)} rects={len(page.rects)}")
            show_tables("1 pdfplumber lines strategy", page.find_tables())
            show_tables("2 pdfplumber text strategy",
                        page.find_tables({"vertical_strategy": "text", "horizontal_strategy": "text"}))
            rows = rows_from_words(page)
            start = next((i for i, r in enumerate(rows) if title_re.search(" ".join(r))), 0)
            print(f"  [3 word-gap rows] total rows on page: {len(rows)}; from the title row:")
            for r in rows[start:start + ROWS_TO_SHOW]:
                print("      ", " | ".join(r)[:150])


if __name__ == "__main__":
    years = [int(a) for a in sys.argv[1:]] or DEFAULT_YEARS
    for y in years:
        inspect(y)
