"""
Small spelling-tolerance helper shared by Stage 6 and Stage 7 (no external dictionary, no company-specific rules).

The vocabulary is built from words the system already knows: line-item labels in the validated tables, the query
grammar, company names.  A query word that is NOT in that vocabulary but is a close match to one word in it
("assests" -> "assets") is corrected, and every correction is reported so it is never silent.
"""
import difflib
import re

WORD = re.compile(r"[A-Za-z][A-Za-z']*")


def build_vocab(strings):
    v = set()
    for s in strings:
        v.update(w.lower().strip("'") for w in WORD.findall(s or "") if len(w) >= 3)
    return v


def correct(text, vocab, min_len=5, cutoff=0.8):
    """Return (corrected_text, [(wrong, right), ...]).  Only words >= min_len letters, not in vocab, are touched."""
    fixes, by_len = [], {}
    for w in vocab:
        by_len.setdefault(w[0], []).append(w)

    def fix(m):
        w = m.group(0)
        lw = w.lower()
        if len(lw) < min_len or lw in vocab:
            return w
        if lw.endswith("s") and lw[:-1] in vocab:           # plural of a known word
            return w
        cands = [c for c in by_len.get(lw[0], []) if abs(len(c) - len(lw)) <= 2]
        best = difflib.get_close_matches(lw, cands, n=1, cutoff=cutoff)
        if not best:
            return w
        fixes.append((w, best[0]))
        return best[0]

    return WORD.sub(fix, text), fixes