"""
Stage 12: ask a question from an IMAGE (a screenshot, a photo of a printed question, a scanned exam line).

  python stage12_ocr_question.py question.png                 # read the image, print the recognised question
  python stage12_ocr_question.py question.png --ask           # ... and answer it with the full pipeline (Chroma + BGE + Ollama)
  python stage12_ocr_question.py question.png --ask --offline --no-llm
  python stage12_ocr_question.py --selftest                   # draws test images and checks the OCR + cleaning (needs Tesseract)

The web UI (stage14_ui.py) uses the same functions: the paperclip button / pasting an image reads the question into the
text box so you can correct it before sending.

Needs: Pillow, pytesseract and the Tesseract program (the same you used to OCR the scanned filings).
On Windows, if Tesseract is not on PATH:  set TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe

What it does (no model, no network):
  1. preprocess: grey-scale, invert dark backgrounds, enlarge small text, boost contrast, sharpen
  2. run Tesseract with two page-segmentation modes and keep the more confident reading
  3. clean up: join wrapped lines, remove bullets/numbering, repair digits that OCR confuses with letters inside numbers
     (20l8 -> 2018, 1O0 -> 100) - every repair is reported
  4. report the confidence; a low-confidence reading is flagged so the person checks it before the answer is trusted
"""
import argparse
import io
import os
import re
import sys

LOW_CONFIDENCE = 70.0


def _tess():
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError("pytesseract is not installed: pip install pytesseract pillow") from e
    
    cmd = os.environ.get("TESSERACT_CMD")
    if not cmd:
        # Automatically look in your D drive folder first
        common_paths = [
            r"D:\New folder\tesseract\tesseract.exe",
            r"D:\New folder\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for p in common_paths:
            if os.path.exists(p):
                cmd = p
                break
                
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        
    return pytesseract


def load_image(src):
    """src: a path, raw bytes, or a PIL image."""
    from PIL import Image, ImageOps
    if isinstance(src, Image.Image):
        im = src
    elif isinstance(src, (bytes, bytearray)):
        im = Image.open(io.BytesIO(src))
    else:
        im = Image.open(src)
    im = ImageOps.exif_transpose(im)                      # phone photos carry a rotation flag
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        im = im.convert("RGBA")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    return im.convert("RGB")


def preprocess(im, variant="plain"):
    from PIL import ImageFilter, ImageOps, ImageStat
    g = ImageOps.grayscale(im)
    if ImageStat.Stat(g).mean[0] < 110:                   # light text on a dark background (dark-mode screenshots)
        g = ImageOps.invert(g)
    w, h = g.size
    if w < 1400:                                          # Tesseract reads best at ~300 dpi; screenshots are smaller
        f = 1400 / w
        g = g.resize((int(w * f), int(h * f)), resample=3)
    g = ImageOps.autocontrast(g, cutoff=1)
    if variant == "sharp":
        g = g.filter(ImageFilter.SHARPEN)
        g = g.point(lambda p: 255 if p > 150 else 0)      # hard threshold: helps faint photos
    return g


def _read(g, psm):
    pt = _tess()
    data = pt.image_to_data(g, config=f"--psm {psm}", output_type=pt.Output.DICT)
    words, confs, lines = [], [], {}
    for t, c, b, p, l in zip(data["text"], data["conf"], data["block_num"], data["par_num"], data["line_num"]):
        t = (t or "").strip()
        try:
            c = float(c)
        except (TypeError, ValueError):
            c = -1
        if t and c >= 0:
            lines.setdefault((b, p, l), []).append(t)
            confs.append(c)
            words.append(t)
    text = "\n".join(" ".join(v) for v in lines.values())
    return text, (sum(confs) / len(confs) if confs else 0.0), len(words)


def ocr_text(src):
    """Return (raw_text, mean_confidence_0_100) from the better of two readings."""
    im = load_image(src)
    best = ("", 0.0, 0)
    for variant, psm in (("plain", 6), ("plain", 3), ("sharp", 6)):
        text, conf, n = _read(preprocess(im, variant), psm)
        if n and (conf * min(n, 30) > best[1] * min(best[2], 30)):
            best = (text, conf, n)
        if best[1] >= 90 and best[2] >= 3:                # already clean: no need for more passes
            break
    return best[0], best[1]


_NUMLIKE = re.compile(r"(?<![A-Za-z])[0-9][0-9OolI|,.]{1,}[0-9OolI|](?![A-Za-z])|(?<![A-Za-z])[0-9][OolI|](?![A-Za-z])|(?<![A-Za-z])[OolI|][0-9]{2,}(?![A-Za-z])")
_FIX = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"})


def clean_question(raw):
    """Join OCR line breaks, drop bullets and numbering, repair digit/letter confusions inside numbers. Returns (text, [fixes])."""
    fixes = []
    t = raw.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    lines = [re.sub(r"^\s*(?:[-*\u2022\u25cf]|\(?\d{1,2}[.)])\s+", "", ln).strip() for ln in t.split("\n")]
    t = " ".join(ln for ln in lines if ln)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"(\w)- (\w)", r"\1\2", t)                 # hyphenated line break

    def repair(m):
        s = m.group(0)
        r = s.translate(_FIX)
        if r != s and re.fullmatch(r"[0-9][0-9,.]*", r):
            fixes.append((s, r))
            return r
        return s
    t = _NUMLIKE.sub(repair, t)
    t = re.sub(r"\s+([?.!,;:])", r"\1", t)
    return t.strip(), fixes


def questions_in(text):
    """If the image holds several questions, split them (a question ends with '?')."""
    qs = [q.strip() for q in re.findall(r"[^?]+\?", text) if len(q.split()) >= 3]
    return qs or ([text] if text else [])


def read_question(src):
    """Main entry: image -> {"text", "questions", "confidence", "fixes", "warnings", "needs_review"}."""
    out = {"text": "", "questions": [], "confidence": 0.0, "fixes": [], "warnings": [], "needs_review": True}
    try:
        raw, conf = ocr_text(src)
    except Exception as e:
        name = type(e).__name__
        hint = " Install Tesseract and set TESSERACT_CMD if it is not on PATH." if "TesseractNotFound" in name or "pytesseract" in str(e) else ""
        out["warnings"].append(f"could not read the image ({name}: {str(e)[:120]}).{hint}")
        return out
    text, fixes = clean_question(raw)
    out.update(text=text, questions=questions_in(text), confidence=round(conf, 1), fixes=fixes)
    if not text:
        out["warnings"].append("no text was found in the image; try a sharper or larger picture")
    elif conf < LOW_CONFIDENCE:
        out["warnings"].append(f"the reading is uncertain (confidence {conf:.0f}%): please check the text before sending")
    if len(out["questions"]) > 1:
        out["warnings"].append(f"the image contains {len(out['questions'])} questions; the first one is used (the rest are listed)")
    if fixes:
        out["warnings"].append("digits repaired: " + ", ".join(f"'{a}' -> '{b}'" for a, b in fixes))
    out["needs_review"] = (not text) or conf < LOW_CONFIDENCE or bool(fixes)
    return out


# ----------------------------------------------------------------------------- self-test
def _draw(text, size=34, dark=False, rotate=0, scale=1.0, noise=False):
    from PIL import Image, ImageDraw, ImageFont
    font = None
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "/Library/Fonts/Arial.ttf"):
        if os.path.exists(p):
            font = ImageFont.truetype(p, size)
            break
    font = font or ImageFont.load_default()
    lines = text.split("\n")
    w = max(int(font.getlength(l)) for l in lines) + 60
    im = Image.new("RGB", (w, len(lines) * (size + 14) + 40), (20, 22, 25) if dark else "white")
    d = ImageDraw.Draw(im)
    for i, l in enumerate(lines):
        d.text((30, 20 + i * (size + 14)), l, font=font, fill=(235, 235, 230) if dark else (20, 20, 20))
    if noise:
        import random
        random.seed(2)
        for _ in range(900):
            x, y = random.randrange(im.width), random.randrange(im.height)
            d.point((x, y), fill=(120, 120, 120))
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    return im.rotate(rotate, expand=True, fillcolor=(255, 255, 255)) if rotate else im


def selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))
        ok = ok and cond
    # cleaning (no Tesseract needed)
    t, fx = clean_question("1. What was total net sales\nin 20l8 ?")
    check("line join, numbering removed, 20l8 -> 2018", t == "What was total net sales in 2018?" and ("20l8", "2018") in fx, t)
    t, fx = clean_question("Free cash flow in 2O18 and 2019")
    check("2O18 -> 2018", "2018" in t and fx, t)
    t, fx = clean_question("Compare Q1 sales with COVID-19 and Item 1A")
    check("words with digits are left alone", t == "Compare Q1 sales with COVID-19 and Item 1A" and not fx, t)
    t, fx = clean_question("Net income was $1,0O0 million")
    check("1,0O0 -> 1,000", "1,000" in t, t)
    check("several questions split", len(questions_in("What was net sales in 2018? And what was net income in 2019?")) == 2)
    check("hyphen line break joined", clean_question("Percent- age change in net sales")[0] == "Percentage change in net sales")
    try:
        _tess().get_tesseract_version()
    except Exception as e:
        print("SKIP OCR tests: Tesseract not available (" + type(e).__name__ + ")")
        return ok
    cases = [("plain", "What was total net sales in 2018?", {}),
             ("small screenshot", "Percentage change in net income from 2017 to 2018", {"size": 18}),
             ("dark mode", "Free cash flow in 2018", {"dark": True}),
             ("two lines", "What are the main risks related\nto competition?", {}),
             ("slightly rotated + noise", "Compare 2017 and 2018", {"rotate": 2, "noise": True}),
             ("low resolution", "What was operating income in 2021?", {"size": 28, "scale": 0.5})]
    for name, q, kw in cases:
        r = read_question(_draw(q, **kw))
        got = re.sub(r"[^a-z0-9 ]", "", r["text"].lower())
        want = re.sub(r"[^a-z0-9 ]", "", q.lower().replace("\n", " "))
        check(f"OCR {name}", got == want, f"{r['text']!r} conf {r['confidence']}")
    r = read_question(Image_blank())
    check("blank image gives a warning, not a crash", r["needs_review"] and r["warnings"])
    return ok


def Image_blank():
    from PIL import Image
    return Image.new("RGB", (600, 200), "white")


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--ask", action="store_true", help="answer the recognised question with the full pipeline")
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--rows", default="data/processed/table_rows.json")
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--chunks", default="data/processed/all_chunks.json")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.image:
        ap.error("give an image file (or --selftest)")
    r = read_question(a.image)
    print(f"Recognised text   : {r['text']!r}")
    print(f"Confidence        : {r['confidence']}%")
    for w in r["warnings"]:
        print(f"  ! {w}")
    if len(r["questions"]) > 1:
        for i, q in enumerate(r["questions"], 1):
            print(f"  question {i}: {q}")
    if a.ask and r["text"]:
        sys.path.insert(0, ".")
        from stage10_answer import Pipeline
        pipe = Pipeline(a.db, a.rows, a.tables, a.model, a.ollama_url, not a.no_llm, not a.offline, a.chunks, a.offline)
        q = r["questions"][0] if r["questions"] else r["text"]
        res = pipe.ask(q)
        print("\nQuestion:", q, "\n")
        print(res["answer"]["answer"])


if __name__ == "__main__":
    main()
