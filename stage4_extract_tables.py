"""
Stage 4: direct table extraction from the original PDFs + validation.

Company-agnostic. Finds the primary financial statements (operations/income, balance sheet, cash flows)
in every <Company>_<YYYY>.pdf, rebuilds rows from WORD COORDINATES, maps every number to a FINANCIAL-YEAR column
by x-position, and validates each table with accounting identities before anything is embedded (Stage 5).

v2 changes (from the first full run):
  * EVERY candidate page is tried; the best-validating table wins (v1 picked the page with most digits, which was
    often an exhibit/schedule page).
  * Pages with a garbled text layer (e.g. 2019/2020) are OCR'd with PyMuPDF/Tesseract; word boxes feed the same
    row-rebuilding logic.
  * Footnote markers like "(1)" left of the year columns belong to the label, not to a year.
  * Units found from the title block even when split across rows; '§' and similar dash glyphs handled.
  * Real identities: assets == liabilities+equity, operating income = base - opex, cash begin + net change = end.

Usage:
    python stage4_extract_tables.py                       # all PDFs -> data/processed/all_tables.json
    python stage4_extract_tables.py --years 2017 2019     # only these filing years (merged into the output)
    python stage4_extract_tables.py --show 2022           # also print the extracted tables
    python stage4_extract_tables.py --validate-only       # re-validate the saved JSON
"""
import argparse
import json
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

import pdfplumber

try:
    import pymupdf as fitz
except ImportError:
    fitz = None

try:   # reuse the garbled-text detector and Tesseract-data locator from Stage 1
    from stage1_extract_chunk import page_is_garbled, resolve_tessdata, LEGACY_MAP
except Exception:
    LEGACY_MAP = {}

    def page_is_garbled(text):
        return False

    def resolve_tessdata(user_path=None):
        return user_path

FILE_RE = re.compile(r"^(?P<company>.+?)_(?P<year>\d{4})\.pdf$", re.I)

# Standard SEC statement titles, matched on letters only so letter-spaced / split titles still match.
STATEMENTS = {
    # "consolidated" is optional (a company's first 10-K may have no subsidiaries); "...Data" rows such as
    # "Balance Sheet Data:" on the Selected Financial Data page are not statements.
    "operations": re.compile(r"(consolidated)?(statements?of(operations|income|earnings)|incomestatements?)(?!data)"),
    "balance_sheet": re.compile(r"(consolidated)?(balancesheets?|statements?offinancial(position|condition)|"
                                r"statements?ofcondition)(?!data)"),
    "cash_flows": re.compile(r"(consolidated)?statements?ofcashflows?(?!data)"),
}
MIN_DIGITS = 150            # a real statement page is number-heavy
OCR_WINDOW = 16             # garbled years: OCR only this many pages after the start of Item 8
YEAR_CELL = re.compile(r"^(19[89]\d|20[0-4]\d)$")
NUM_CELL = re.compile(r"^\(?-?\$?\(?\d[\d,]*\.?\d*\)?%?$")
DASH_CELL = re.compile(r"^[\u2014\u2013\-\u00a7]+$")
END_RE = re.compile(r"^(see|the)\s+(accompanying|notes)", re.I)
WRAP_END = re.compile(r"(,|\b(of|and|in|for|to|from|the|on|by|with|related|less|including)|\()\s*$", re.I)
STOP_ROW = re.compile(r"^(operating (income|loss|\(loss\)|\(income\))|income \(loss\)|(loss|income) from operations|net income|net loss|"
                      r"gross (profit|margin)|(total )?net sales)")

GP_RX = re.compile(r"^gross (profit|margin)")
COS_RX = re.compile(r"^cost of (sales|goods sold|revenues?|net sales)")
SALES_RX = re.compile(r"^(total )?(net sales|net revenues?|revenues?)")
OPEX_RX = re.compile(r"^total (operating expenses|costs and expenses)")
OI_RX = re.compile(r"^(operating (income|loss|\(loss\)|\(income\))|income \(loss\) from operations|\(loss\) income from operations|"
                   r"loss from operations|income from operations)")
CF_BEGIN = re.compile(r"cash.*beginning of (the )?(period|year)")
CF_END = re.compile(r"cash.*end of (the )?(period|year)")
CF_NET = re.compile(r"^(net )?(\(?increase|\(?decrease|change)")


def letters_only(s: str) -> str:
    return re.sub(r"[^a-z]", "", s.lower())


# ------------------------------------------------------------ page access
class PageSource:
    """Gives word boxes for any page: native text via pdfplumber, garbled pages via OCR (cached)."""

    def __init__(self, pdf_path, tessdata=None, item8_start=None):
        self.pdf = pdfplumber.open(pdf_path)
        self.doc = fitz.open(pdf_path) if fitz else None
        self.tessdata = tessdata
        self.item8_start = item8_start
        self.cache = {}
        self.legacy = False          # old PDFs whose fonts map ligatures/dashes/dot-leaders to accented capitals
        if self.doc is not None:
            n = 0
            for i in range(len(self.doc)):
                t = self.doc[i].get_text()
                n += t.count("\u00cf") + t.count("\u00d1")
            self.legacy = n >= 30

    def _clean_words(self, ws, native):
        out = []
        for w in ws:
            t = w["text"]
            if native:
                t = unicodedata.normalize("NFKC", re.sub(r"\(cid:\d+\)", "", t))
                if self.legacy:
                    t = re.sub("\u00cf+", "", t)                      # dot leaders
                    t = "".join(LEGACY_MAP.get(ch, ch) for ch in t)    # fi/ff/ffi/fl, dash
            t = t.strip()
            if not t or re.fullmatch(r"[._]+", t):                      # leader dots / underscores
                continue
            out.append({**w, "text": t})
        return out

    def close(self):
        self.pdf.close()

    def n_pages(self):
        return len(self.pdf.pages)

    def words(self, pno):
        if pno in self.cache:
            return self.cache[pno]
        if self.doc is not None and page_is_garbled(self.doc[pno].get_text()):
            td = self.tessdata or resolve_tessdata()
            if not td:
                raise SystemExit("This PDF has garbled pages that need OCR, but eng.traineddata was not found. "
                                 "Re-run with --tessdata \"<folder containing eng.traineddata>\"")
            page = self.doc[pno]
            print(f"      OCR page {pno + 1} ...", end="\r", flush=True)
            tp = page.get_textpage_ocr(language="eng", dpi=300, full=True, tessdata=td)
            ws = [{"text": w[4], "x0": w[0], "x1": w[2], "top": w[1], "bottom": w[3]}
                  for w in page.get_text("words", textpage=tp) if w[4].strip() not in ("", "|", "_")]
            res = (self._clean_words(ws, native=False), True)
        else:
            res = (self._clean_words(self.pdf.pages[pno].extract_words(x_tolerance=1.5, y_tolerance=2), native=True), False)
        self.cache[pno] = res
        return res


def candidate_pages(src):
    """All pages where a statement title appears AND the page is number-heavy. Pages are verified later by
    actually building and validating the table."""
    cands = {n: [] for n in STATEMENTS}
    for pno in range(src.n_pages()):
        if src.doc is not None:
            text = src.doc[pno].get_text()
            garbled = page_is_garbled(text)
        else:
            text, garbled = src.pdf.pages[pno].extract_text() or "", False
        if garbled:
            start = (src.item8_start - 1) if src.item8_start else 30
            if not (start - 2 <= pno <= start + OCR_WINDOW):
                continue
            text = " ".join(w["text"] for w in src.words(pno)[0])
        if sum(ch.isdigit() for ch in text) < MIN_DIGITS:
            continue
        L = letters_only(text)
        for name, rx in STATEMENTS.items():
            if rx.search(L):
                cands[name].append(pno)
    return cands


# ------------------------------------------------------ rows from words
def parse_number(cell: str):
    """'(75)' -> -75.0, '$ 1,263' -> 1263.0, '—' -> 0.0, '12.5%' -> 12.5. None if not numeric."""
    t = cell.replace("$", "").replace(" ", "")
    if DASH_CELL.match(t):
        return 0.0
    if not NUM_CELL.match(t):
        return None
    neg = t.startswith("(") or t.startswith("-")
    t = t.strip("()-%").replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None
    return -v if neg else v


def cluster_rows(words):
    if not words:
        return []
    med_h = statistics.median(w["bottom"] - w["top"] for w in words)
    tol = max(2.5, 0.45 * med_h)
    items = sorted(words, key=lambda w: (w["top"] + w["bottom"]) / 2)
    rows, cur, centre = [], [items[0]], (items[0]["top"] + items[0]["bottom"]) / 2
    for w in items[1:]:
        c = (w["top"] + w["bottom"]) / 2
        if abs(c - centre) <= tol:
            cur.append(w)
            centre = statistics.mean((x["top"] + x["bottom"]) / 2 for x in cur)
        else:
            rows.append(cur)
            cur, centre = [w], c
    rows.append(cur)
    return rows


def split_cells(row_words, gap=6.0):
    """Split one row into cells at horizontal gaps; a lone '$' is dropped, a '$' prefix stripped."""
    ws = sorted(row_words, key=lambda w: w["x0"])
    cells, cur, last = [], [], None
    for w in ws:
        txt = w["text"]
        if "$" in txt and re.fullmatch(r"[$\u00a7]+", txt):    # currency marker, incl. OCR noise like '§$'
            continue
        txt = re.sub(r"^[$\u00a7]+(?=[\d(\-\u2014\u2013])", "", txt)    # '$32,173' / '§$32,173' -> '32,173'
        if cur and w["x0"] - last > gap:
            cells.append(cur)
            cur = []
        cur.append((txt, w["x0"], w["x1"], w["top"]))
        last = w["x1"]
    if cur:
        cells.append(cur)
    return [{"text": " ".join(t for t, *_ in c), "x0": c[0][1], "x1": c[-1][2],
             "xc": (c[0][1] + c[-1][2]) / 2, "top": min(t for *_, t in c)} for c in cells]


def extract_rows(words):
    return [split_cells(r) for r in cluster_rows(words)]


def is_num_cell(c):
    return parse_number(c["text"]) is not None


# --------------------------------------------------------- build a table
def find_year_header(rows, start):
    for i in range(start, min(start + 14, len(rows))):
        years = [c for c in rows[i] if YEAR_CELL.match(c["text"])]
        others = [c for c in rows[i] if not YEAR_CELL.match(c["text"])]
        if len(years) >= 2 and len(others) <= 1:
            return i, [(int(c["text"]), c["xc"]) for c in years]
    return None, []


def assign_monotonic(xs, cs):
    """Order-preserving, one-to-one assignment of numbers (positions xs) to columns (positions cs, len(xs) <= len(cs))
    minimising total distance. Returns the column index for each number."""
    k, m, INF = len(xs), len(cs), float("inf")
    best = [[INF] * (m + 1) for _ in range(k + 1)]
    for j in range(m + 1):
        best[0][j] = 0.0
    for i in range(1, k + 1):
        for j in range(1, m + 1):
            best[i][j] = min(best[i][j - 1], best[i - 1][j - 1] + abs(xs[i - 1] - cs[j - 1]))
    out, i, j = [], k, m
    while i > 0:
        if best[i][j] == best[i][j - 1]:
            j -= 1
        else:
            out.append(j - 1)
            i, j = i - 1, j - 1
    return out[::-1]


def select_numbers(xs, cs):
    """More numbers than columns: pick len(cs) numbers (order-preserving) that best fit the columns.
    Returns the chosen indices into xs, one per column."""
    n, m, INF = len(xs), len(cs), float("inf")
    best = [[INF] * (n + 1) for _ in range(m + 1)]
    for i in range(n + 1):
        best[0][i] = 0.0
    for j in range(1, m + 1):
        for i in range(j, n + 1):
            best[j][i] = min(best[j][i - 1], best[j - 1][i - 1] + abs(xs[i - 1] - cs[j - 1]))
    out, i, j = [], n, m
    while j > 0:
        if best[j][i] == best[j][i - 1]:
            i -= 1
        else:
            out.append(i - 1)
            i, j = i - 1, j - 1
    return out[::-1]


def build_table(rows, title_re):
    # title row = a row that is (almost) only the title, so prose mentions of the statement are ignored
    t_idx = None
    for i, r in enumerate(rows):
        L = letters_only(" ".join(c["text"] for c in r))
        m = title_re.search(L)
        if m and len(L) - (m.end() - m.start()) <= 30:
            t_idx = i
            break
    if t_idx is None:
        return None
    h_idx, cols = find_year_header(rows, t_idx + 1)
    if not cols:
        return None
    title = " ".join(c["text"] for c in rows[t_idx])
    ctx = " ".join(c["text"] for r in rows[max(0, t_idx - 2):h_idx + 2] for c in r)
    um = re.search(r"in(millions|thousands|billions)", letters_only(ctx))
    units = f"USD {um.group(1)}" if um else None
    left_limit = min(xc for _, xc in cols) - 50

    cols = sorted(cols, key=lambda yc: yc[1])
    cx = [xc for _, xc in cols]
    raw_rows = []
    for r in rows[h_idx + 1:]:
        if END_RE.match(" ".join(c["text"] for c in r)):
            break
        num_cells = [c for c in r if is_num_cell(c) and c["xc"] >= left_limit]
        label_cells = [c for c in r if c not in num_cells]     # incl. footnote markers like "(1)"
        if num_cells:
            first_x = min(c["x0"] for c in num_cells)
            label_cells = [c for c in label_cells if c["x0"] < first_x]
        raw_rows.append((" ".join(c["text"] for c in label_cells).strip(), sorted(num_cells, key=lambda c: c["xc"])))

    # Financial numbers are normally right-aligned under centred year headers, so a short and a long number in the
    # same column have different centres but the same right edge. Use whichever edge (right x1 or centre xc) is more
    # consistent within each column across complete rows, then learn each column's offset from those rows.
    def spread(k):
        total = 0.0
        for j in range(len(cx)):
            v = [nc[j][k] for _, nc in raw_rows if len(nc) == len(cx)]
            if len(v) >= 3:
                med = statistics.median(v)
                total += statistics.mean(abs(x - med) for x in v)
        return total
    key = "x1" if spread("x1") <= spread("xc") else "xc"
    offs = [[] for _ in cx]
    for _, nc in raw_rows:
        if len(nc) == len(cx):
            for j, c in enumerate(nc):
                offs[j].append(c[key] - cx[j])
    adj = [cx[j] + (statistics.median(offs[j]) if offs[j] else 0.0) for j in range(len(cx))]
    spacing = min((b - a) for a, b in zip(adj, adj[1:])) if len(adj) > 1 else 90.0
    tol = max(20.0, 0.6 * spacing)
    geometry = {"header_x": [round(x, 1) for x in cx], "adjusted_x": [round(x, 1) for x in adj], "tol": round(tol, 1),
                "alignment": "right" if key == "x1" else "center"}

    parsed, unassigned, unassigned_detail = [], 0, []
    for label, nc in raw_rows:
        vals, raw, xs = {}, {}, {}
        if len(nc) <= len(adj):
            pairs = list(zip(nc, assign_monotonic([c[key] for c in nc], adj)))
        else:     # more numbers than year columns: keep the best-fitting subset, report the extras
            pick = select_numbers([c[key] for c in nc], adj)
            pairs = [(nc[i], j) for j, i in enumerate(pick)]
            for i, c in enumerate(nc):
                if i not in pick:
                    unassigned += 1
                    if len(unassigned_detail) < 6:
                        unassigned_detail.append(f"{label[:28]!r}: extra number {c['text']!r}; row has "
                                                 f"{len(nc)} numbers for {len(adj)} year columns: {[x['text'] for x in nc]}")
        for c, j in pairs:
            yr, dx = cols[j][0], abs(adj[j] - c[key])
            if dx > tol or str(yr) in vals:
                unassigned += 1
                if len(unassigned_detail) < 6:
                    unassigned_detail.append(f"{label[:28]!r} {c['text']} (dx={dx:.0f}"
                                             f"{', duplicate' if str(yr) in vals else ''})")
                continue
            vals[str(yr)], raw[str(yr)], xs[str(yr)] = parse_number(c["text"]), c["text"], round(c[key], 1)
        parsed.append({"label": label, "values": vals, "raw": raw, "xs": xs})

    merged = []
    for row in parsed:
        prev = merged[-1] if merged else None
        if prev and not prev["values"] and prev["label"] and not row["label"] and row["values"]:
            prev["values"], prev["raw"], prev["xs"] = row["values"], row["raw"], row["xs"]
            continue
        if prev and not prev["values"] and prev["label"] and row["label"] and not prev["label"].endswith(":") \
                and (row["label"][0].islower() or WRAP_END.search(prev["label"])):
            prev["label"] += " " + row["label"]
            if row["values"]:
                prev["values"], prev["raw"], prev["xs"] = row["values"], row["raw"], row["xs"]
            continue
        if row["label"] or row["values"]:
            merged.append(row)

    section = None
    for row in merged:
        if row["label"] and not row["values"]:
            section = row["label"].rstrip(":")
            row["kind"] = "heading"
        else:
            row["kind"] = "total" if row["label"].lower().startswith("total") else "line"
            row["section"] = section
    return {"title": title, "units": units, "per_share_note": "pershare" in letters_only(ctx),
            "column_years": sorted(y for y, _ in cols), "rows": merged, "unassigned_numbers": unassigned,
            "unassigned_detail": unassigned_detail, "geometry": geometry}


def camelot_fallback(pdf_path, page_no):
    try:
        import camelot
    except ImportError:
        return "camelot not installed"
    try:
        t = camelot.read_pdf(str(pdf_path), pages=str(page_no + 1), flavor="stream")
        return "camelot grid available - inspect manually" if len(t) else "camelot found no table"
    except Exception as e:
        return f"camelot error: {e}"


# ------------------------------------------------------------ validation
def _find(rows, rx):
    return next((r for r in rows if r["kind"] != "heading" and rx.search(r["label"].lower())), None)


def check_table(tbl):
    """Company-agnostic accounting checks. Returns list of (level, message); levels FAIL / WARN / INFO."""
    issues = []
    years = [str(y) for y in tbl["column_years"]]
    rows = tbl["rows"]
    data_rows = [r for r in rows if r["kind"] != "heading"]
    n_lines = len(data_rows)
    if n_lines < 8:
        issues.append(("FAIL", f"only {n_lines} numeric rows"))
    if tbl["unassigned_numbers"]:
        issues.append(("WARN", f"{tbl['unassigned_numbers']} numbers not near / duplicated in a year column: "
                               + "; ".join(tbl.get("unassigned_detail", [])[:3])))
    ragged = sum(1 for r in data_rows if r["values"] and len(r["values"]) != len(years))
    cf_verified = False
    if tbl["units"] is None:
        issues.append(("WARN", "units (millions/thousands) not found in the title block"))

    st = tbl["statement"]
    if st == "balance_sheet":
        ta = _find(rows, re.compile(r"^total assets"))
        tl = _find(rows, re.compile(r"^total liabilities and (stock|share)"))
        if not ta or not tl:
            issues.append(("FAIL", "total assets / total liabilities-and-equity row not found"))
        else:
            for y in years:
                a, b = ta["values"].get(y), tl["values"].get(y)
                if a is None or b is None or abs(a - b) > 1:
                    issues.append(("FAIL", f"assets != liabilities+equity for {y}: {a} vs {b}"))
    elif st == "operations":
        base = _find(rows, GP_RX) or _find(rows, SALES_RX)
        opex, oi, cos = _find(rows, OPEX_RX), _find(rows, OI_RX), _find(rows, COS_RX)
        if base and opex and oi:
            for y in years:
                a, b, c = (base["values"].get(y), opex["values"].get(y), oi["values"].get(y))
                if None in (a, b, c):
                    continue
                tol = max(1.0, 0.002 * abs(c))
                k = cos["values"].get(y) if cos else None      # cost of sales listed outside operating expenses
                if abs(a - b - c) > tol and not (k is not None and abs(a - k - b - c) <= tol):
                    issues.append(("WARN", f"{base['label'][:20]} - opex != operating income for {y}: {a}-{b} vs {c}"))
        else:
            issues.append(("INFO", "operating-income identity not checked (row labels not recognised)"))
    elif st == "cash_flows":
        b, e, n = (_find(rows, CF_BEGIN), _find(rows, CF_END), _find(rows, CF_NET))
        if b and e and n:
            checked = bad_cf = 0
            for y in years:
                bv, ev, nv = b["values"].get(y), e["values"].get(y), n["values"].get(y)
                if None not in (bv, ev, nv):
                    checked += 1
                    if abs(bv + nv - ev) > 1:
                        bad_cf += 1
                        issues.append(("FAIL", f"cash begin + net change != end for {y}: {bv}+{nv} vs {ev}"))
            cf_verified = checked >= 2 and bad_cf == 0
        else:
            issues.append(("INFO", f"cash-flow identity not checked (beginning row found={bool(b)}, "
                                   f"end row found={bool(e)}, net-change row found={bool(n)})"))

    if ragged > max(3, 0.25 * n_lines) and not cf_verified:
        issues.append(("WARN", f"{ragged} rows have fewer values than year columns"))

    # 'Total X' == sum of the lines above it (back to the previous heading) -> informational only
    ok = bad = 0
    shown = 0
    for i, r in enumerate(rows):
        if r["kind"] != "total":
            continue
        for y in years:
            if y not in r["values"]:
                continue
            s, got, j = 0.0, 0, i - 1
            while j >= 0 and rows[j]["kind"] != "heading":
                if STOP_ROW.match(rows[j]["label"].lower()):
                    break
                if y in rows[j]["values"]:
                    s += rows[j]["values"][y]
                    got += 1
                if rows[j]["kind"] == "total":
                    break
                j -= 1
            if got < 2:
                continue
            if abs(s - r["values"][y]) <= max(1.0, 0.002 * abs(r["values"][y])):
                ok += 1
            else:
                bad += 1
                if shown < 3:
                    issues.append(("INFO", f"subtotal '{r['label'][:40]}' {y}: lines sum {s:,.0f} vs reported {r['values'][y]:,.0f}"))
                    shown += 1
    tbl["subtotal_checks"] = {"ok": ok, "mismatch": bad}
    return issues


def anchor_check(tables):
    """Compare reported net sales (validate_corpus.NET_SALES; Amazon TEST data only) with the operations tables."""
    try:
        from validate_corpus import NET_SALES
    except Exception:
        return {}
    res = {}
    for t in tables:
        want = NET_SALES.get(t["doc_year"]) if t["statement"] == "operations" else None
        if not want:
            continue
        v = float(want.replace(",", ""))
        res[t["doc_year"]] = any(abs(r["values"].get(str(t["doc_year"]), -1) - v) < 0.5
                                 for r in t["rows"] if re.search(r"net sales|revenue", r["label"], re.I))
    return res


# ------------------------------------------------------------------- main
def item8_starts(chunks_path):
    try:
        chunks = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    d = {}
    for c in chunks:
        if c.get("item") == "8":
            d[c["doc_year"]] = min(d.get(c["doc_year"], 10 ** 6), c["page_start"])
    return d


def level_of(issues):
    if any(lv == "FAIL" for lv, _ in issues):
        return "FAIL"
    return "warn" if any(lv == "WARN" for lv, _ in issues) else "ok"


def process_pdf(pdf_path: Path, tessdata, item8):
    m = FILE_RE.match(pdf_path.name)
    company, doc_year = m.group("company").replace("-", " ").strip(), int(m.group("year"))
    src = PageSource(pdf_path, tessdata, item8.get(doc_year))
    out = []
    try:
        cands = candidate_pages(src)
        for name, rx in STATEMENTS.items():
            best = None
            for pno in cands[name]:
                words, ocr = src.words(pno)
                tbl = build_table(extract_rows(words), rx)
                if tbl is None:
                    continue
                tbl.update({"table_id": f"{company.lower().replace(' ', '-')}_{doc_year}_{name}",
                            "company": company, "doc_year": doc_year, "statement": name, "page": pno + 1,
                            "source_file": pdf_path.name, "ocr_used": ocr,
                            "extractor": "ocr-words" if ocr else "pdfplumber-words"})
                issues = check_table(tbl)
                cons = "consolidated" in letters_only(tbl["title"])   # prefer the consolidated statement if both exist
                score = (cons, -sum(lv == "FAIL" for lv, _ in issues), -sum(lv == "WARN" for lv, _ in issues), len(tbl["rows"]))
                if best is None or score > best[0]:
                    best = (score, tbl, issues)
            if best is None:
                hint = f"candidates {[p + 1 for p in cands[name]]}" if cands[name] else "no candidate page"
                fb = camelot_fallback(pdf_path, cands[name][0]) if cands[name] else ""
                print(f"  {doc_year} {name:13} NOT FOUND ({hint}) {fb}")
                continue
            _, tbl, issues = best
            tbl["issues"] = [f"{lv}: {msg}" for lv, msg in issues]
            sc = tbl["subtotal_checks"]
            print(f"  {doc_year} {name:13} p{tbl['page']:<3} {level_of(issues):4} years={tbl['column_years']} "
                  f"rows={len(tbl['rows'])} units={tbl['units']} subtotals ok/bad={sc['ok']}/{sc['mismatch']} "
                  f"{'[OCR] ' if tbl['ocr_used'] else ''}(chosen from {len(cands[name])} candidate pages)")
            for lv, msg in issues:
                print(f"        {lv}: {msg}")
            out.append(tbl)
    finally:
        src.close()
    return out


def show(tables, year, only=None):
    for t in tables:
        if t["doc_year"] != year or (only and t["statement"] not in only):
            continue
        print("\n" + "=" * 100)
        print(f"{t['doc_year']} {t['statement']} | page {t['page']} | {t['title']} | {t['units']} | years {t['column_years']}")
        g = t.get("geometry")
        if g:
            print(f"  geometry: header_x={g['header_x']} adjusted_x={g['adjusted_x']} tol={g['tol']}")
        for r in t["rows"]:
            vals = " | ".join(f"{y}:{r['raw'].get(str(y), '')}" for y in t["column_years"]) if r["values"] else ""
            xs = r.get("xs") or {}
            xtxt = ("   x=" + ",".join(f"{xs[str(y)]:.0f}" for y in t["column_years"] if str(y) in xs)) if xs else ""
            print(f"  {'[H] ' if r['kind'] == 'heading' else '    '}{r['label'][:50]:50} {vals}{xtxt}")


def summarize(tables, all_years):
    print("\n=== STAGE 4 SUMMARY ===")
    by = defaultdict(dict)
    for t in tables:
        by[t["doc_year"]][t["statement"]] = t
    print(f"{'year':6}" + "".join(f"{s:15}" for s in STATEMENTS))
    for y in sorted(all_years):
        line = f"{y:<6}"
        for s in STATEMENTS:
            t = by[y].get(s)
            lv = "MISSING" if not t else level_of([(i.split(':')[0], i) for i in t["issues"]])
            line += f"{lv:15}"
        print(line)
    anchors = anchor_check(tables)
    if anchors:
        print("\nNet sales anchor found in operations table:",
              {y: ("ok" if v else "MISS") for y, v in sorted(anchors.items())})
    print("\nDo not start Stage 5 until every year shows ok (or a warn you have inspected) for all three statements.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data")
    ap.add_argument("--output", default="data/processed/all_tables.json")
    ap.add_argument("--chunks", default="data/processed/all_chunks.json")
    ap.add_argument("--tessdata", default=None, help="folder containing eng.traineddata (for garbled-text years)")
    ap.add_argument("--years", type=int, nargs="*")
    ap.add_argument("--show", type=int, nargs="*")
    ap.add_argument("--only", nargs="*", help="with --show: only these statements (operations balance_sheet cash_flows)")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.output)
    pdfs = sorted(p for p in Path(a.input_dir).glob("*.pdf") if FILE_RE.match(p.name))
    all_years = [int(FILE_RE.match(p.name).group("year")) for p in pdfs]
    if a.validate_only:
        tables = json.loads(out.read_text(encoding="utf-8"))
        for t in tables:
            t["issues"] = [f"{lv}: {m}" for lv, m in check_table(t)]
        summarize(tables, all_years)
        for y in a.show or []:
            show(tables, y, a.only)
        return
    if a.years:
        pdfs = [p for p in pdfs if int(FILE_RE.match(p.name).group("year")) in a.years]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {a.input_dir}")
    item8 = item8_starts(a.chunks)
    tables = []
    for p in pdfs:
        print(p.name)
        tables.extend(process_pdf(p, a.tessdata, item8))
    if a.years and out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        tables = [t for t in old if t["doc_year"] not in a.years] + tables
    tables.sort(key=lambda t: (t["company"], t["doc_year"], t["statement"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(tables, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(tables)} tables -> {out}")
    for y in a.show or []:
        show(tables, y, a.only)
    summarize(tables, all_years)


if __name__ == "__main__":
    main()