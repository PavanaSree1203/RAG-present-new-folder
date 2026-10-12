"""Diagnostic: where do statement titles appear in one filing? Usage: python diag_find_title.py 1997"""
import re
import sys
from pathlib import Path

import pymupdf

year = sys.argv[1] if len(sys.argv) > 1 else "1997"
path = next(Path("data").glob(f"*_{year}.pdf"))
keys = ["balancesheet", "statementsofoperations", "statementofoperations", "statementsofcashflow",
        "statementofcashflow", "statementsofincome", "financialstatements"]
doc = pymupdf.open(path)
print(f"{path.name}: {len(doc)} pages")
for i, page in enumerate(doc):
    text = page.get_text()
    letters = re.sub(r"[^a-z]", "", text.lower())
    hits = [k for k in keys if k in letters]
    if hits:
        digits = sum(ch.isdigit() for ch in text)
        print(f"  p{i + 1:<3} digits={digits:<5} {hits} | {text[:90].replace(chr(10), ' / ')!r}")
