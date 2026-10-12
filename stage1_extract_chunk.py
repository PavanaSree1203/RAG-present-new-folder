"""
Stage 1 + 2: PDF extraction, structure detection, hierarchical chunking, validation.

Company-agnostic: files must be named  <Company>_<YYYY>.pdf  (e.g. Amazon_2024.pdf).
YYYY from the filename = "document year" (doc_year). The real financial year the
filing covers is detected from the cover page ("fiscal year ended ...") and stored
separately as filing_fiscal_year. Financial years mentioned in each chunk are
stored as financial_years_mentioned.

Usage:
    python stage1_extract_chunk.py --input-dir data --output data/processed/all_chunks.json
    python stage1_extract_chunk.py --validate-only --output data/processed/all_chunks.json
"""
import argparse
import json
import re
import unicodedata
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

# ----------------------------------------------------------------- settings
# Sizes are measured in BGE tokens (not words): BGE-base silently truncates beyond 512 tokens,
# and numeric/table text uses far more tokens per word than prose.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
EMBED_LIMIT = 500        # max tokens of embedding_text (context header + chunk); model limit is 512
TARGET_TOKENS = 330      # flush a chunk once it reaches this size
MAX_TOKENS = 430         # ceiling for the chunk body (reduced further by the context-header length)
MIN_CHUNK_TOKENS = 45    # chunks smaller than this are merged into a neighbour in the same Item
MIN_TAIL_TOKENS = 80     # merge smaller trailing chunks into the previous one
OVERLAP_MAX_TOKENS = 60  # last sentence carried into the next chunk (prose only)
HEADER_FOOTER_ZONE = 0.07
TOC_ITEM_THRESHOLD = 5  # page with >= this many ITEM lines is a table of contents

FILE_RE = re.compile(r"^(?P<company>.+?)_(?P<year>\d{4})\.pdf$", re.I)
PART_RE = re.compile(r"^PART\s+(IV|I{1,3})\b[\.\s]*(.*)$", re.I)
ITEM_RE = re.compile(r"^ITEM\s+(\d{1,2}[A-C]?)\s*[\.\:\-\u2013\u2014]?\s*(.*)$", re.I)
PAGE_NUM_RE = re.compile(r"^(page\s*)?\d{1,3}(\s*of\s*\d+)?$|^-\s*\d+\s*-$", re.I)
FY_RE = re.compile(r"fiscal\s+year\s+ended\s+([A-Za-z]+\s+\d{1,2},?\s+(\d{4}))", re.I)
YEAR_RE = re.compile(r"\b(19[89]\d|20[0-3]\d)\b")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"\u201c])")
STOPWORDS = {"the", "and", "of", "to", "in", "a", "is", "for", "on", "that", "by", "with", "as", "are",
             "our", "we", "or", "be", "from", "this", "at", "an", "not", "have", "has", "its", "may"}


def page_is_garbled(text: str) -> bool:
    """True if the PDF text layer is unreadable (broken font encoding) -> page needs OCR."""
    if len(text) < 200:
        return False
    ctrl = sum(1 for ch in text if (ord(ch) < 32 and ch not in "\n\t\r\x0c") or 0x7f <= ord(ch) <= 0x9f)
    if ctrl / len(text) > 0.01:
        return True
    toks = re.findall(r"[A-Za-z]+", text.lower())
    return len(toks) >= 60 and sum(t in STOPWORDS for t in toks) / len(toks) < 0.03


def fix_ocr_line(t: str) -> str:
    """Repair common Tesseract mistakes in heading keywords."""
    t = re.sub(r"^[lI1!|]tem\b", "Item", t)
    t = re.sub(r"(?<!\d)\u00a2\s", "\u2022 ", t)   # OCR reads bullets as the cent sign
    m = re.match(r"^(PART)\s+([IVl|1!]{1,3})(?![A-Za-z0-9])(.*)$", t, re.I)
    if m:
        roman = re.sub(r"[l|1!]", "I", m.group(2)).upper()
        t = f"{m.group(1)} {roman}{m.group(3)}"
    return t


def resolve_tessdata(user_path=None):
    if user_path:
        return user_path
    try:   # look for eng.traineddata anywhere inside the project folder first
        hit = next(Path(__file__).resolve().parent.rglob("eng.traineddata"), None)
        if hit:
            return str(hit.parent)
    except Exception:
        pass
    try:
        return fitz.get_tessdata()
    except Exception:
        return None


# ------------------------------------------------------------ small helpers
def clean(text: str) -> str:
    """NFKC-normalise (ligatures fi/fl, fraction slash, odd spaces) and collapse whitespace and dot leaders."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2044", "/")   # fraction slash: 43\u20444% -> 43/4%
    text = re.sub(r"(?<=\d)\u2030", "\u2020", text).replace("\u2030", "\u2022")  # legacy bullet/dagger glyph
    text = re.sub(r"[\ue000-\uf8ff]", " ", text)   # private-use symbol-font glyphs (checkboxes, bullets)
    text = re.sub(r"\.{4,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def numeric_ratio(text: str) -> float:
    toks = text.split()
    if not toks:
        return 0.0
    num = sum(1 for t in toks if re.fullmatch(r"[\$\(\)\-\u2014%,.\d]+", t))
    return num / len(toks)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


# ----------------------------------------------------------- PDF extraction
def _is_ocr_junk(t: str) -> bool:
    """Symbol-soup lines OCR produces from charts/graphics (never occur in real financial text)."""
    weird = sum(ch in "~=<>*^|`" for ch in t)
    return len(t) > 20 and weird >= 4 and weird / len(t) > 0.1


def extract_lines(pdf_path: Path, tessdata=None):
    """Return (pages, body_font_size, ocr_pages). Pages with garbled text layers are OCR'd."""
    doc = fitz.open(pdf_path)
    size_counter = Counter()
    pages, ocr_pages = [], set()
    for pno, page in enumerate(doc, 1):
        h = page.rect.height
        d, ocr = None, False
        if page_is_garbled(page.get_text()):
            tessdata = tessdata or resolve_tessdata()
            if not tessdata:
                raise SystemExit(
                    f"Page {pno} of {pdf_path.name} has a garbled text layer and needs OCR, but Tesseract "
                    "language data was not found. Re-run with --tessdata \"<folder containing eng.traineddata>\"")
            try:
                print(f"    OCR page {pno}", end="\r", flush=True)
                tp = page.get_textpage_ocr(language="eng", dpi=300, full=True, tessdata=tessdata)
                d, ocr = page.get_text("dict", textpage=tp), True
                ocr_pages.add(pno)
            except Exception as e:
                print(f"\n    [warn] OCR failed on page {pno}: {e}")
        if d is None:
            d = page.get_text("dict")
        blocks_out = []
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            entries = []
            for l in b["lines"]:
                spans = [s for s in l["spans"] if s["text"].strip()]
                if not spans:
                    continue
                text = clean("".join(s["text"] for s in l["spans"]))
                if not text:
                    continue
                if ocr:   # OCR gives no reliable font info -> no font-based sub-heading detection
                    size, bold = -1.0, False
                else:
                    total = sum(len(s["text"]) for s in spans) or 1
                    bold_chars = sum(len(s["text"]) for s in spans
                                     if (s["flags"] & 16) or "bold" in s["font"].lower())
                    size, bold = max(s["size"] for s in spans), bold_chars / total > 0.6
                    size_counter[round(size, 1)] += len(text)
                entries.append({"text": text, "size": size, "bold": bold, "x0": l["bbox"][0],
                                "y0": l["bbox"][1], "y1": l["bbox"][3], "page": pno, "h": h})
            if ocr and entries:   # OCR splits one visual line into fragments -> re-join by baseline
                entries.sort(key=lambda e: e["y0"])
                groups = []
                for e in entries:
                    if groups and abs(e["y0"] - groups[-1][0]["y0"]) < 0.5 * max(e["y1"] - e["y0"], 1):
                        groups[-1].append(e)
                    else:
                        groups.append([e])
                entries = []
                for g in groups:
                    g.sort(key=lambda e: e["x0"])
                    entries.append({**g[0], "text": fix_ocr_line(" ".join(e["text"] for e in g)),
                                    "y1": max(e["y1"] for e in g)})
            if ocr:   # drop OCR junk lines (charts/graphics read as punctuation soup)
                entries = [e for e in entries if not _is_ocr_junk(e["text"])]
            if entries:
                blocks_out.append(entries)
        pages.append(blocks_out)
    body = size_counter.most_common(1)[0][0] if size_counter else 10.0
    return pages, body, ocr_pages


# Some older PDFs (e.g. 1999-2002 filings) use a legacy font encoding in which ligatures, dashes and
# dot leaders come out as accented capitals ("eÅorts" = "efforts", "oÇce" = "office").
LEGACY_MAP = {"\u00d1": "fi", "\u00c5": "ff", "\u00c7": "ffi", "\u00d6": "fl",
              "\u00cc": "\u2014", "\u201a": "\u2022", "\u2030": "\u2020"}


def repair_legacy_encoding(pages) -> bool:
    """Detect the legacy encoding by its dot-leader signature and repair it. Returns True if applied."""
    joined = "".join(ln["text"] for blocks in pages for blk in blocks for ln in blk)
    if joined.count("\u00cf") + joined.count("\u00d1") < 30:
        return False
    for blocks in pages:
        for blk in blocks:
            for ln in blk:
                t = re.sub("\u00cf{2,}", " ", ln["text"]).replace("\u00cf", ".")
                ln["text"] = clean("".join(LEGACY_MAP.get(ch, ch) for ch in t))
    return True


def remove_headers_footers(pages):
    """Drop page numbers and lines repeated in top/bottom zones across many pages."""
    n = len(pages)
    seen = defaultdict(set)
    for pi, blocks in enumerate(pages):
        for blk in blocks:
            for ln in blk:
                if ln["y0"] < ln["h"] * HEADER_FOOTER_ZONE or ln["y1"] > ln["h"] * (1 - HEADER_FOOTER_ZONE):
                    seen[re.sub(r"\d+", "#", ln["text"].lower())].add(pi)
    repeated = {k for k, v in seen.items() if len(v) >= max(3, 0.3 * n)}
    out = []
    for blocks in pages:
        nb = []
        for blk in blocks:
            kept = []
            for ln in blk:
                in_zone = ln["y0"] < ln["h"] * HEADER_FOOTER_ZONE or ln["y1"] > ln["h"] * (1 - HEADER_FOOTER_ZONE)
                if PAGE_NUM_RE.match(ln["text"]):
                    continue
                if in_zone and re.sub(r"\d+", "#", ln["text"].lower()) in repeated:
                    continue
                kept.append(ln)
            if kept:
                nb.append(kept)
        out.append(nb)
    return out


# --------------------------------------------------------------- structure
def page_is_toc(blocks) -> bool:
    return sum(1 for blk in blocks for ln in blk if ITEM_RE.match(ln["text"])) >= TOC_ITEM_THRESHOLD


def is_item_heading(ln) -> bool:
    m = ITEM_RE.match(ln["text"])
    if not m or len(ln["text"]) > 140:
        return False
    title = m.group(2).strip().lower()
    # reject cross-references such as "Item 7 of Part II ..." or "Item 1A, Risk Factors"
    if re.match(r"^(of|in|and|to|for|under|,|\(|herein|above|below)\b", title):
        return False
    return True


def subheading_level(ln, body):
    """Return 1 (larger font), 2 (bold, body size) or 0 (not a heading)."""
    t = ln["text"]
    words = t.split()
    if not (2 <= len(t) <= 120) or len(words) > 15 or len(t) < 4:
        return 0
    if t[-1] in ".,;:" or t[0].islower() or numeric_ratio(t) > 0.3:
        return 0
    if ln["size"] >= body * 1.15:
        return 1
    if ln["bold"]:
        return 2
    return 0


def build_paragraphs(pages, body):
    """Walk lines in reading order -> list of paragraphs with hierarchy state."""
    st = {"part": None, "item": None, "item_title": None, "sub": None, "subsub": None, "await": False}
    paras = []

    def emit(text, page):
        text = clean(text)
        if text:
            paras.append({"page": page, "text": text, "part": st["part"], "item": st["item"],
                          "item_title": st["item_title"], "sub": st["sub"], "subsub": st["subsub"]})

    for pi, blocks in enumerate(pages, 1):
        toc = page_is_toc(blocks)
        for blk in blocks:
            buf, buf_page = [], pi
            for ln in blk:
                t = ln["text"]
                if toc:                       # skip table-of-contents pages entirely
                    continue
                pm, im = PART_RE.match(t), ITEM_RE.match(t)
                if pm and len(t) < 60 and (t[:4].isupper() or ln["bold"]):
                    emit(" ".join(buf), buf_page); buf = []
                    st.update(part="PART " + pm.group(1).upper(), item=None, item_title=None, sub=None, subsub=None)
                    continue
                if im and is_item_heading(ln):
                    emit(" ".join(buf), buf_page); buf = []
                    title = im.group(2).strip()
                    st.update(item=im.group(1).upper(), item_title=title or None, sub=None, subsub=None,
                              **{"await": not title})
                    continue
                if st["await"] and len(t.split()) <= 20 and numeric_ratio(t) < 0.3:
                    st["item_title"], st["await"] = t, False
                    continue
                lvl = subheading_level(ln, body)
                if lvl:
                    emit(" ".join(buf), buf_page); buf = []
                    if lvl == 1:
                        st["sub"], st["subsub"] = t, None
                    else:
                        st["subsub"] = t
                    continue
                if not buf:
                    buf_page = pi
                buf.append(t)
            emit(" ".join(buf), buf_page)
    return paras


# ---------------------------------------------------------------- chunking
def split_sentences(text):
    return [s for s in SENT_SPLIT_RE.split(text) if s.strip()]


_TOK = None


def ntok(text: str) -> int:
    """Number of BGE tokens in text (falls back to a conservative estimate if the tokenizer is unavailable)."""
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(EMBED_MODEL)
            _TOK.model_max_length = 10 ** 7   # silence the >512 warning; we measure only
        except Exception:
            _TOK = False
            print("[warn] BGE tokenizer unavailable; using approximate token counts (chars/2.8)")
    if _TOK:
        return len(_TOK.encode(text, add_special_tokens=False))
    return int(len(text) / 2.8) + 1


def hard_split(text, max_tok):
    """Recursively halve text until every piece fits (for table-like text with no sentence boundaries)."""
    if ntok(text) <= max_tok:
        return [text]
    words = text.split()
    if len(words) < 2:
        mid = len(text) // 2
        return hard_split(text[:mid], max_tok) + hard_split(text[mid:], max_tok)
    mid = len(words) // 2
    return hard_split(" ".join(words[:mid]), max_tok) + hard_split(" ".join(words[mid:]), max_tok)


def to_units(paras, max_tok):
    """Split over-long paragraphs (by sentence, then by words) so no unit exceeds max_tok tokens."""
    units = []
    for p in paras:
        if ntok(p["text"]) <= max_tok:
            units.append((p["text"], p["page"]))
            continue
        cur, w = [], 0
        for sent in split_sentences(p["text"]):
            for piece in hard_split(sent, max_tok):
                pw = ntok(piece)
                if cur and w + pw > max_tok:
                    units.append((" ".join(cur), p["page"])); cur, w = [], 0
                cur.append(piece); w += pw
        if cur:
            units.append((" ".join(cur), p["page"]))
    return units


def pack(paras, max_tok, target_tok):
    """Greedy-pack paragraphs of ONE section into chunks. Returns list of (text, p_start, p_end)."""
    chunks, cur, curw, pages = [], [], 0, []

    def flush():
        nonlocal cur, curw, pages
        if cur:
            chunks.append(("\n".join(cur), min(pages), max(pages)))
        cur, curw, pages = [], 0, []

    for text, page in to_units(paras, max_tok):
        w = ntok(text)
        if cur and (curw + w > max_tok or curw >= target_tok):
            prev = cur[-1]
            flush()
            if numeric_ratio(prev) < 0.3:     # carry one sentence of overlap (prose only)
                sents = split_sentences(prev)
                last = sents[-1] if sents else ""
                lw = ntok(last) if last else 0
                if 0 < lw <= OVERLAP_MAX_TOKENS and lw + w <= max_tok:
                    cur, curw, pages = [last], lw, [page]
        cur.append(text); curw += w; pages.append(page)
    flush()
    # merge tiny tail
    if len(chunks) >= 2 and ntok(chunks[-1][0]) < MIN_TAIL_TOKENS:
        a, b = chunks[-2], chunks[-1]
        if ntok(a[0]) + ntok(b[0]) <= max_tok:
            chunks[-2:] = [(a[0] + "\n" + b[0], a[1], b[2])]
    return chunks


def merge_tiny(raw):
    """Merge chunks under MIN_CHUNK_TOKENS into a neighbour in the same Part/Item.
    Fixes stray table cells ('AWS', 'Number', 'Beginning') that were detected as headings."""
    merged, carry = [], None
    for r in raw:
        if carry is not None:
            if carry["key"][:2] == r["key"][:2] and ntok(carry["text"]) + ntok(r["text"]) <= r["max_tok"]:
                r = {**r, "text": carry["text"] + "\n" + r["text"], "ps": min(carry["ps"], r["ps"])}
            else:
                merged.append(carry)
            carry = None
        if ntok(r["text"]) < MIN_CHUNK_TOKENS:
            prev = merged[-1] if merged else None
            if prev and prev["key"][:2] == r["key"][:2] and ntok(prev["text"]) + ntok(r["text"]) <= prev["max_tok"]:
                prev["text"] += "\n" + r["text"]
                prev["pe"] = max(prev["pe"], r["pe"])
            else:
                carry = r
        else:
            merged.append(r)
    if carry is not None:
        merged.append(carry)
    return merged


def detect_fiscal_year(pages, fallback):
    for blocks in pages[:4]:
        txt = " ".join(ln["text"] for blk in blocks for ln in blk)
        m = FY_RE.search(txt)
        if m:
            return int(m.group(2))
    return fallback


def process_pdf(pdf_path: Path, tessdata=None):
    m = FILE_RE.match(pdf_path.name)
    if not m:
        print(f"  [skip] {pdf_path.name}: name must look like Company_YYYY.pdf")
        return []
    company, doc_year = m.group("company").replace("-", " ").strip(), int(m.group("year"))
    pages, body, ocr_pages = extract_lines(pdf_path, tessdata)
    legacy = repair_legacy_encoding(pages)
    n_pages = len(pages)
    fiscal_year = detect_fiscal_year(pages, doc_year)
    pages = remove_headers_footers(pages)
    paras = build_paragraphs(pages, body)

    # group consecutive paragraphs sharing the same hierarchy
    groups, key = [], None
    for p in paras:
        k = (p["part"], p["item"], p["item_title"], p["sub"], p["subsub"])
        if k != key:
            groups.append((k, [])); key = k
        groups[-1][1].append(p)

    raw = []
    for (part, item, ititle, sub, subsub), gp in groups:
        path = " > ".join(x for x in [part, f"Item {item}" if item else None,
                                      ititle, sub, subsub] if x) or "Cover/Preamble"
        ctx = f"{company} 10-K (filing year {doc_year}) | {path}"
        max_tok = max(120, min(MAX_TOKENS, EMBED_LIMIT - ntok(ctx) - 4))
        target_tok = min(TARGET_TOKENS, int(max_tok * 0.78))
        for text, ps, pe in pack(gp, max_tok, target_tok):
            raw.append({"key": (part, item, ititle, sub, subsub), "path": path, "ctx": ctx,
                        "max_tok": max_tok, "text": text, "ps": ps, "pe": pe})
    raw = merge_tiny(raw)

    out = []
    for idx, r in enumerate(raw, 1):
        part, item, ititle, sub, subsub = r["key"]
        text, ps, pe, path, ctx = r["text"], r["ps"], r["pe"], r["path"], r["ctx"]
        out.append({
            "chunk_id": f"{slug(company)}_{doc_year}_c{idx:05d}",
            "chunk_type": "text",
            "company": company,
            "doc_year": doc_year,
            "filing_fiscal_year": fiscal_year,
            "financial_years_mentioned": sorted({int(y) for y in YEAR_RE.findall(text)}),
            "part": part, "item": item, "item_title": ititle,
            "section": sub, "subsection": subsub, "section_path": path,
            "page_start": ps, "page_end": pe,
            "source_file": pdf_path.name,
            "ocr_used": any(ps <= p <= pe for p in ocr_pages),
            "word_count": len(text.split()),
            "token_count": ntok(f"{ctx}\n{text}"),
            "numeric_ratio": round(numeric_ratio(text), 3),
            "text": text,
            "embedding_text": f"{ctx}\n{text}",
        })
    print(f"  {pdf_path.name}: {n_pages} pages -> {len(out)} chunks (FY detected: {fiscal_year}, OCR pages: {len(ocr_pages)}{", legacy-encoding repaired" if legacy else ""})")
    return out


# -------------------------------------------------------------- validation
def validate(chunks):
    print("\n=== STAGE 2: CORPUS VALIDATION ===")
    by_year = defaultdict(list)
    for c in chunks:
        by_year[(c["company"], c["doc_year"])].append(c)
    print(f"{'company':10} {'doc_yr':6} {'FY':5} {'chunks':6} {'no-item%':8} {'med_w':6} {'max_tok':7} "
          f"{'pages':5} {'ocr%':5} key items present")
    problems = []
    for (comp, yr), cs in sorted(by_year.items()):
        wc = [c["word_count"] for c in cs]
        items = {c["item"] for c in cs if c["item"]}
        no_item = 100 * sum(1 for c in cs if not c["item"]) / len(cs)
        ocr_pct = 100 * sum(1 for c in cs if c.get("ocr_used")) / len(cs)
        pages = max(c["page_end"] for c in cs)
        key = [i for i in ("1", "1A", "7", "7A", "8") if i in items]
        print(f"{comp[:10]:10} {yr:<6} {cs[0]['filing_fiscal_year']:<5} {len(cs):<6} {no_item:<8.1f} "
              f"{int(statistics.median(wc)):<6} {max(c['token_count'] for c in cs):<7} {pages:<5} {ocr_pct:<5.0f} {','.join(key)}")
        if no_item > 30: problems.append(f"{comp} {yr}: {no_item:.0f}% chunks have no ITEM")
        if "7" not in items: problems.append(f"{comp} {yr}: Item 7 (MD&A) not detected")
        if "8" not in items: problems.append(f"{comp} {yr}: Item 8 (financials) not detected")
        n_over = sum(1 for c in cs if c["token_count"] > 512)
        if n_over: problems.append(f"{comp} {yr}: {n_over} chunks exceed 512 BGE tokens")
        if cs[0]["filing_fiscal_year"] != yr:
            problems.append(f"{comp} {yr}: detected fiscal year {cs[0]['filing_fiscal_year']} != filename year")
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        problems.append("duplicate chunk_ids")
    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Tiny chunks (<15 words): {sum(1 for c in chunks if c['word_count'] < 15)}")
    print("\nPotential problems:" if problems else "\nNo structural problems flagged.")
    for p in problems:
        print("  -", p)
    print("\nInspect samples manually: python -c \"import json,random;"
          "c=json.load(open('data/processed/all_chunks.json'));print(json.dumps(random.choice(c),indent=1))\"")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="data")
    ap.add_argument("--output", default="data/processed/all_chunks.json")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", help="only (re)process these doc years and merge into existing output")
    ap.add_argument("--tessdata", default=None, help="folder containing eng.traineddata (for OCR)")
    a = ap.parse_args()
    out = Path(a.output)
    if a.validate_only:
        validate(json.loads(out.read_text(encoding="utf-8")))
        return
    pdfs = sorted(Path(a.input_dir).glob("*.pdf"))
    if a.years:
        pdfs = [p for p in pdfs if FILE_RE.match(p.name) and int(FILE_RE.match(p.name).group("year")) in a.years]
    if not pdfs:
        raise SystemExit(f"No PDFs found in {a.input_dir}")
    chunks = []
    for p in pdfs:
        chunks.extend(process_pdf(p, a.tessdata))
    if a.years and out.exists():
        old = json.loads(out.read_text(encoding="utf-8"))
        chunks = [c for c in old if c["doc_year"] not in a.years] + chunks
        chunks.sort(key=lambda c: (c["company"], c["doc_year"], c["chunk_id"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(chunks)} chunks -> {out}")
    validate(chunks)


if __name__ == "__main__":
    main()