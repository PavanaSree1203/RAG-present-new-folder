"""
Shared canonical row labelling, used by Stage 5 (table index) and Stage 6 (line-item catalog) so that both stages
see exactly the same label for every table row.

Three defects in older filings are repaired here, without any company-specific rule:
  1. leader dots / legacy-font residue / letter-spaced headings in labels      ("Total assets....._þß....")
  2. a section heading merged into the first row of the section                ("Current Assets: Cash and cash equivalents",
                                                                                  "OPERATING ACTIVITIES Net loss")
  3. a total row with an EMPTY label whose meaning is carried by its section   (the net EPS row under "Diluted earnings per share")
"""
import re

LEADER = re.compile(r"(?:\s*[.Ï_]){3,}")
COLON_HEAD = re.compile(r"^(?P<h>[A-Za-z][A-Za-z ,&/'’()\-]{2,70}?):\s+(?P<l>[A-Z(].*)$")
CAPS_HEAD = re.compile(r"^(?P<h>[A-Z]{3,}(?:[ &/]+[A-Z]{3,}){0,3})\s+(?P<l>[A-Z][a-z].*)$")


def clean_label(label: str) -> str:
    label = LEADER.sub(" ", label or "")
    label = re.sub(r"[þß]", "", label)                        # legacy-font residue
    toks = label.split()
    if len(toks) >= 4 and sum(len(t) == 1 for t in toks) / len(toks) > 0.6:
        label = "".join(toks)                                           # letter-spaced heading "A S S E T S"
    return re.sub(r"\s+", " ", label).strip(" :")


def split_heading(label: str):
    """('Current Assets', 'Cash and cash equivalents') for a merged heading+row label, else (None, label)."""
    m = COLON_HEAD.match(label)
    if m and len(m.group("h").split()) <= 8 and not re.search(r"\d", m.group("h")):
        return clean_label(m.group("h")), m.group("l").strip()
    m = CAPS_HEAD.match(label)
    if m:
        return m.group("h").strip(), m.group("l").strip()
    return None, label


def canonical(row: dict, section=None):
    """Return (label, section, how) for a table row dict from Stage 4.
    how is "" (unchanged), "heading_split" or "label_from_section"."""
    sec = clean_label(section if section is not None else (row.get("section") or ""))
    label = clean_label(row.get("label", ""))
    how = ""
    head, rest = split_heading(label)
    if head:
        label, sec, how = clean_label(rest), head, "heading_split"
    if not label and sec and row.get("values") and not re.match(r"^\(\d+\)", sec):      # not a footnote block
        label, how = sec, "label_from_section"                          # e.g. the net EPS row printed without a label
    return label, sec, how
