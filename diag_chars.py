"""Diagnostic: which unusual non-ASCII characters appear in the corpus, how often, and in which years."""
import json
import unicodedata
from collections import Counter, defaultdict

chunks = json.load(open("data/processed/all_chunks.json", encoding="utf-8"))
count, years, example = Counter(), defaultdict(set), {}
for c in chunks:
    t = c["text"]
    for i, ch in enumerate(t):
        if ord(ch) > 127:
            count[ch] += 1
            years[ch].add(c["doc_year"])
            example.setdefault(ch, t[max(0, i - 25): i + 25].replace("\n", " "))

print(f"{'char':5} {'hex':7} {'count':>7}  {'years':11} name / example")
for ch, n in count.most_common(30):
    ys = sorted(years[ch])
    name = unicodedata.name(ch, "?")
    print(f"{ch!r:5} U+{ord(ch):04X} {n:>7}  {ys[0]}-{ys[-1]:<6} {name} | ...{example[ch]}...")
