"""
Stage 2 content validation (run after stage1_extract_chunk.py).

1. Anchor check: each filing should contain its own year's net sales figure (USD millions).
   Tests that numbers survived extraction/OCR intact. Anchors are Amazon-specific TEST DATA
   only; the pipeline itself is company-agnostic.
2. Garbled-text check: prose chunks with almost no common English words.
3. Prints sample chunks from OCR and non-OCR years for manual reading.

Usage:  python validate_corpus.py
"""
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

CHUNKS = Path("data/processed/all_chunks.json")

# Reported net sales, USD millions (from the original filings). Used only to test extraction quality.
NET_SALES = {2005: "8,490", 2006: "10,711", 2007: "14,835", 2008: "19,166", 2009: "24,509",
             2010: "34,204", 2011: "48,077", 2012: "61,093", 2013: "74,452", 2014: "88,988",
             2015: "107,006", 2016: "135,987", 2017: "177,866", 2018: "232,887", 2019: "280,522",
             2020: "386,064", 2021: "469,822", 2022: "513,983", 2023: "574,785", 2024: "637,959"}

STOP = {"the", "and", "of", "to", "in", "a", "is", "for", "on", "that", "by", "with", "as", "are",
        "our", "we", "or", "be", "from", "this", "at", "an", "not", "have", "has", "its", "may"}


def main():
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    by_year = defaultdict(list)
    for c in chunks:
        by_year[c["doc_year"]].append(c)

    print("=== 1. NET SALES ANCHOR CHECK ===")
    misses = []
    for yr, anchor in sorted(NET_SALES.items()):
        cs = by_year.get(yr, [])
        hits = [c for c in cs if anchor in c["text"]]
        in_fin = [c for c in hits if c["item"] in ("7", "8")]
        ocr = "OCR" if any(c.get("ocr_used") for c in cs) else "   "
        status = "ok " if hits else "MISS"
        print(f"  {yr} {ocr} {status} '{anchor}' found in {len(hits)} chunks ({len(in_fin)} in Items 7/8)")
        if not hits:
            misses.append(yr)
    print("  Misses:", misses or "none")

    print("\n=== 2. GARBLED PROSE CHUNKS (few common English words) ===")
    bad = defaultdict(int)
    for c in chunks:
        if c["numeric_ratio"] > 0.3:
            continue
        toks = re.findall(r"[A-Za-z]+", c["text"].lower())
        if len(toks) >= 60 and sum(t in STOP for t in toks) / len(toks) < 0.05:
            bad[c["doc_year"]] += 1
    print("  flagged per year:", dict(bad) or "none")

    print("\n=== 3. WORDS PER YEAR (sanity: OCR years should be comparable) ===")
    for yr in (2017, 2018, 2019, 2020, 2021, 2022):
        w = sum(c["word_count"] for c in by_year.get(yr, []))
        print(f"  {yr}: {w:,} words in {len(by_year.get(yr, []))} chunks")

    print("\n=== 4. SAMPLE CHUNKS (read these) ===")
    random.seed(7)
    for yr in (2019, 2020, 2022):
        pool = [c for c in by_year.get(yr, []) if c["item"] in ("7", "8")]
        if pool:
            c = random.choice(pool)
            print(f"\n--- {c['chunk_id']} | {c['section_path']} | pages {c['page_start']}-{c['page_end']} "
                  f"| ocr={c.get('ocr_used')} ---")
            print(c["text"][:700])


if __name__ == "__main__":
    if not CHUNKS.exists():
        sys.exit("Run stage1_extract_chunk.py first.")
    main()
