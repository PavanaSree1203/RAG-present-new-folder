"""
Stage 13: test-set generation and accuracy evaluation.

  python stage13_evaluate.py make                       # writes data/eval/testset.json (generated from YOUR tables/chunks)
  python stage13_evaluate.py run                        # retrieval + numeric accuracy for all retrieval modes, no LLM needed
  python stage13_evaluate.py run --llm                  # also checks the final written answer (Ollama must be running)
  python stage13_evaluate.py run --offline --modes bm25 full    # quick run without Chroma/BGE
  python stage13_evaluate.py run --limit 40             # smoke test on a subset

Gold answers are computed straight from table_rows.json (the filing's own row, as reported) with plain arithmetic in this
file - NOT by the pipeline - so the pipeline is graded against an independent reference.

Question families
  fact        "What was total net sales in 2018?"                         gold: one row value
  pct_change  "Percentage change in net income from 2017 to 2018"        gold: (b-a)/|a|*100
  difference  "What was the difference in total assets between ..."      gold: b-a
  cagr        "CAGR of net sales from 2014 to 2018"                       gold: (b/a)^(1/n)-1
  margin      "Operating margin in 2019"                                  gold: operating income / net sales
  paraphrase  the same facts in looser wording and with a typo           (robustness)
  unanswerable years / metrics / companies that are not in the corpus     gold: must NOT produce a number
  text        "What does the 2017 filing say about <section>?"            gold: a chunk from that filing + section
  keyword     "In the 2017 filing, what is said about <terms>?"          gold: the chunk the terms came from

Metrics: Hit@1/3/5/8 and MRR (retrieval), numeric accuracy (value within tolerance), refusal accuracy, end-to-end
answer accuracy (with --llm: the written answer contains the gold number), per family and per retrieval mode (ablation),
plus a failure list (failures.csv) for error analysis.
"""
import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

CONCEPT_ROWS = {                      # concept -> (statement, label regex on the lower-cased cleaned label, spoken name)
    "revenue": ("operations", r"^(total )?(net )?(sales|revenues?)$", "total net sales"),
    "cost_of_sales": ("operations", r"^cost of (sales|goods sold|revenues?)$", "cost of sales"),
    "operating_income": ("operations", r"^(operating (income|profit|earnings|loss)( loss)?|(income|loss)( loss)? from operations)$", "operating income"),
    "net_income": ("operations", r"^net (income|earnings|loss|profit)( loss)?$", "net income"),
    "total_assets": ("balance_sheet", r"^total assets$", "total assets"),
    "current_assets": ("balance_sheet", r"^total current assets$", "total current assets"),
    "current_liabilities": ("balance_sheet", r"^total current liabilities$", "total current liabilities"),
    "equity": ("balance_sheet", r"^total (stockholders|shareholders|shareowners|owners)['’]? ?(equity|deficit)( \(?(equity|deficit)\)?)?$|^total (equity|deficit)$", "total stockholders equity"),
}
SYNONYMS = {"revenue": ["revenue", "sales", "turnover"], "net_income": ["profit", "net earnings", "bottom line"],
            "operating_income": ["operating profit", "income from operations"], "total_assets": ["assets", "total assets"]}
UNANSWERABLE = [
    ("What was net sales in 1985?", "year before the corpus"),
    ("Total net sales in 2040", "future year"),
    ("What was the number of employees on Mars in 2018?", "unknown metric"),
    ("Net income of Microsoft in 2019", "other company"),
    ("What was Tesla's revenue in 2020?", "other company"),
    ("Dividend per share in 2019", "Amazon pays no dividend; no such row"),
    ("Percentage change in net sales from 1980 to 1990", "years not in corpus"),
    ("CAGR of net income from 2030 to 2035", "future years"),
    ("What was the carbon footprint per customer in 2015?", "metric not in statements"),
    ("Stock price on 1 March 2018", "not in the filings' statements"),
]


THEORY = [   # (question, regex alternatives that a relevant passage should contain)
    ("What are the main risks related to competition?", r"competit"),
    ("How does the company describe its business and strategy?", r"customer|strateg|selection|price"),
    ("What does the company say about seasonality?", r"seasonal|holiday|fourth quarter"),
    ("What are the risks from foreign currency exchange rates?", r"foreign currenc|exchange rate"),
    ("How does the company recognize revenue?", r"recogni[sz]e|performance obligation|revenue recognition"),
    ("What does the company say about its employees?", r"employee"),
    ("What legal proceedings is the company involved in?", r"legal proceeding|litigation|lawsuit|patent|claims"),
    ("What are the company's liquidity and capital resources?", r"liquidity|credit facilit|capital resources|cash and cash equivalents"),
    ("What does the company say about income taxes?", r"income tax|tax rate|tax"),
    ("What does the company say about intellectual property?", r"patent|trademark|intellectual property|copyright"),
    ("How does the company describe its operating segments?", r"segment"),
    ("What are the cybersecurity and data security risks?", r"cyber|security|breach|unauthorized"),
    ("How does regulation affect the business?", r"regulat|law|compliance"),
    ("What debt obligations does the company have?", r"debt|notes|borrowing|interest"),
    ("What does the company say about fulfillment and shipping costs?", r"fulfillment|shipping"),
    ("What does the company say about stock-based compensation?", r"stock-based|restricted stock|compensation"),
    ("What properties does the company own or lease?", r"propert|lease|facilit"),
    ("Why did net sales increase?", r"net sales|increase"),
    ("What are the risks of international operations?", r"international|foreign|countries"),
    ("What does management say about future outlook and guidance?", r"expect|outlook|anticipate|future"),
]


# ----------------------------------------------------------------------------- helpers
def clean(label):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9()'’ ]+", " ", (label or "").lower())).strip()


def load_rows(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def own_values(rows):
    """{(concept, filing_year): (value_in_USD_millions, row_id, ocr)} using the filing's OWN row for its OWN year.
    A concept is skipped for a filing when two different rows match (ambiguous), so the gold is never a guess."""
    found = defaultdict(dict)
    for r in rows:
        m = r["meta"]
        if m.get("row_kind") not in ("total", "line") or m.get("value_type") != "amount" or not m.get("units_known", True):
            continue
        lab = clean(m.get("row_label"))
        for c, (stmt, rx, _) in CONCEPT_ROWS.items():
            if m.get("statement") == stmt and re.search(rx, lab):
                vals = json.loads(m.get("values_std_json") or "{}")
                y = str(m.get("doc_year"))
                if y in vals and vals[y] is not None:
                    found[(c, int(y))][r["id"]] = (float(vals[y]), bool(m.get("ocr_used")))
    out = {}
    for k, d in found.items():
        if len({round(v[0], 6) for v in d.values()}) == 1:                 # same number under several ids -> still unambiguous
            rid = sorted(d)[0]
            out[k] = (d[rid][0], sorted(d), d[rid][1])
    return out


def typo(word, rng):
    if len(word) < 6:
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


# ----------------------------------------------------------------------------- make
def make(a):
    rng = random.Random(a.seed)
    rows = load_rows(a.rows)
    vals = own_values(rows)
    years_of = defaultdict(list)
    for (c, y) in vals:
        years_of[c].append(y)
    cases = []

    def add(fam, q, **kw):
        cases.append({"id": f"{fam}-{len(cases) + 1:03d}", "family": fam, "question": q, **kw})

    speak = {c: v[2] for c, v in CONCEPT_ROWS.items()}
    # fact
    pool = [(c, y) for (c, y) in vals]
    rng.shuffle(pool)
    for c, y in pool[:a.n_fact]:
        v, ids, ocr = vals[(c, y)]
        add("fact", f"What was {speak[c]} in {y}?", op="value", concept=c, years=[y], gold=v, tol=0.005, gold_ids=ids, ocr=ocr)
    # paraphrase (looser wording + a typo)
    for c, y in pool[a.n_fact:a.n_fact + a.n_para]:
        v, ids, ocr = vals[(c, y)]
        name = rng.choice(SYNONYMS.get(c, [speak[c]]))
        q = rng.choice([f"how much {name} did the company report for {y}", f"{name} {y}", f"tell me the {name} for fiscal {y}"])
        if rng.random() < .5:
            w = [t for t in q.split() if len(t) >= 6]
            if w:
                q = q.replace(w[0], typo(w[0], rng), 1)
        add("paraphrase", q, op="value", concept=c, years=[y], gold=v, tol=0.005, gold_ids=ids, ocr=ocr, expect_concept=c)
    # pairs
    pairs = [(c, y1, y2) for c in CONCEPT_ROWS for y1 in years_of[c] for y2 in years_of[c] if y2 == y1 + 1]
    rng.shuffle(pairs)
    for c, y1, y2 in pairs[:a.n_calc]:
        (v1, i1, o1), (v2, i2, o2) = vals[(c, y1)], vals[(c, y2)]
        if v1 == 0:
            continue
        add("pct_change", f"Percentage change in {speak[c]} from {y1} to {y2}", op="pct_change", concept=c, years=[y1, y2],
            gold=(v2 - v1) / abs(v1) * 100, tol=0.01, gold_ids=i1 + i2, ocr=o1 or o2)
    rng.shuffle(pairs)
    for c, y1, y2 in pairs[:a.n_calc]:
        (v1, i1, o1), (v2, i2, o2) = vals[(c, y1)], vals[(c, y2)]
        add("difference", f"What was the difference in {speak[c]} between {y1} and {y2}?", op="difference", concept=c, years=[y1, y2],
            gold=v2 - v1, tol=0.005, gold_ids=i1 + i2, ocr=o1 or o2)
    spans = [(c, y1, y2) for c in ("revenue", "total_assets", "equity") for y1 in years_of[c] for y2 in years_of[c]
             if 3 <= y2 - y1 <= 8 and vals[(c, y1)][0] > 0 and vals[(c, y2)][0] > 0]
    rng.shuffle(spans)
    for c, y1, y2 in spans[:a.n_calc]:
        (v1, i1, o1), (v2, i2, o2) = vals[(c, y1)], vals[(c, y2)]
        add("cagr", f"CAGR of {speak[c]} from {y1} to {y2}", op="cagr", concept=c, years=[y1, y2],
            gold=((v2 / v1) ** (1 / (y2 - y1)) - 1) * 100, tol=0.01, gold_ids=i1 + i2, ocr=o1 or o2)
    ys = sorted(y for y in years_of["revenue"] if ("operating_income", y) in vals)
    rng.shuffle(ys)
    for y in ys[:a.n_calc]:
        (oi, i1, o1), (rv, i2, o2) = vals[("operating_income", y)], vals[("revenue", y)]
        if rv:
            add("margin", f"Operating margin in {y}", op="value", concept="operating_margin", years=[y], gold=oi / rv * 100, tol=0.01,
                gold_ids=i1 + i2, ocr=o1 or o2)
    for q, why in UNANSWERABLE:
        add("unanswerable", q, why=why)
    # text families from the chunk file
    if Path(a.chunks).exists():
        chunks = json.loads(Path(a.chunks).read_text(encoding="utf-8"))
        bysec = defaultdict(list)
        for c in chunks:
            if c.get("chunk_type", "text") == "text" and len(c["text"].split()) >= 60:
                bysec[(int(c["doc_year"]), c.get("section_path", ""))].append(c)
        KNOWN = {"properties", "legal proceedings", "employees", "competition", "seasonality", "business", "risk factors", "liquidity and capital resources"}
        BAD = re.compile(r"\b(total|net income|net sales|value at|operating income|cash|balance|assets|liabilities|expenses?)\b|\d", re.I)

        def good_heading(h):
            hl = h.strip().lower()
            return hl in KNOWN or (len(h.split()) >= 3 and not BAD.search(h) and len(h) <= 120)
        keys = [k for k in bysec if len(k[1].split(" > ")) >= 2 and good_heading(k[1].split(" > ")[-1])]
        rng.shuffle(keys)
        for (y, sec) in keys[:a.n_text]:
            head = sec.split(" > ")[-1]
            add("text", f"What does the {y} 10-K say about {head}?", year=y, section=sec, gold_chunks=[c["chunk_id"] for c in bysec[(y, sec)]],
            words=[w for w in re.findall(r"[a-z]{4,}", head.lower()) if w not in ("what", "does", "that", "this", "with", "from", "have", "your")])
        stop = set("the of and to in a for is that on with as by be are or from at this an our we it its may which have has can not their "
                   "these such other will would could any also than more all been were was but if into over under during each".split())
        df = Counter(w for c in chunks for w in set(re.findall(r"[a-z]{5,}", c["text"].lower())) if w not in stop)
        rng.shuffle(keys)
        for (y, sec) in keys[:a.n_text]:
            ch = rng.choice(bysec[(y, sec)])
            toks = Counter(w for w in re.findall(r"[a-z]{5,}", ch["text"].lower()) if w not in stop and df[w] <= 12)
            top = [w for w, _ in toks.most_common(3)]
            if len(top) >= 2:
                add("keyword", f"In the {y} filing, what is said about {', '.join(top)}?", year=y, section=sec, gold_chunks=[ch["chunk_id"]], words=top)
    for q, rx in THEORY:
        add("theory", q, terms=rx)
    if Path(a.chunks).exists():
        yrs = sorted({int(c["doc_year"]) for c in json.loads(Path(a.chunks).read_text(encoding="utf-8"))})
        rng.shuffle(yrs)
        for y in yrs[:a.n_year]:
            add("year_text", rng.choice([f"What are the discussions in the year {y}?", f"What did the company discuss in {y}?"]), year=y)
        for y in yrs[:a.n_year // 2]:
            t, rx = rng.choice(THEORY)
            add("theory_year", t.rstrip("?") + f" in {y}?", year=y, terms=rx)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cases, indent=1), encoding="utf-8")
    print(f"wrote {len(cases)} test questions to {a.out}")
    for f, n in Counter(c["family"] for c in cases).items():
        print(f"  {f:13s} {n}")
    if not any(c["family"] in ("text", "keyword") for c in cases):
        print("  (no text questions: all_chunks.json not found - pass --chunks)")


# ----------------------------------------------------------------------------- scoring
def close(got, gold, tol):
    if got is None:
        return False
    return abs(got - gold) <= max(abs(gold) * tol, 0.011 if tol >= 0.01 else 0.0)


def numeric_outcome(case, res):
    """Look for a computed result for the right operation and years in the verification output (not in the prose)."""
    ver = res.get("verification") or {}
    want = case["op"]
    yrs = sorted(case["years"])
    best = None
    for r in ver.get("results", []):
        if r.get("refused") or r.get("value") is None:
            continue
        if r["op"] == want and sorted(r.get("years") or []) == yrs:
            if want in ("value", "difference", "pct_change", "cagr") and r.get("concept") != case["concept"]:
                continue
            best = r
            if close(r["value"], case["gold"], case["tol"]):
                return True, r["value"]
    return False, (best["value"] if best else None)


def number_in_text(text, gold, tol):
    """True if the written answer contains the gold number (1,234 / 1234.5 / 12.3% / $12.3 billion)."""
    for m in re.finditer(r"-?\$?\d[\d,]*\.?\d*", text or ""):
        try:
            x = float(m.group(0).replace(",", "").replace("$", ""))
        except ValueError:
            continue
        for scale in (1, 1e-3, 1e3):
            if close(x / scale, gold, max(tol, 0.0015)):
                return True
    return False


def rank_of(case, res):
    gold = set(case.get("gold_ids") or case.get("gold_chunks") or [])
    for i, e in enumerate(res["retrieval"]["results"], 1):
        if e["id"] in gold:
            return i
    return None


def text_rank(case, res, strict=False):
    """text/keyword: first retrieved text chunk from the right filing that is (a) the gold chunk / same section, or - unless strict -
    (b) a passage of that filing that contains the question's key words (the same text is often filed under a neighbouring section)."""
    gold = set(case["gold_chunks"])
    words = [w.lower() for w in case.get("words") or []]
    for i, e in enumerate(res["retrieval"]["results"], 1):
        if e["coll"] != "text_chunks":
            continue
        m = e["meta"]
        if e["id"] in gold or (int(m.get("doc_year", 0)) == case["year"] and str(m.get("section_path", "")) == case["section"]):
            return i
        if not strict and words and int(m.get("doc_year", 0)) == case["year"]:
            t = (e.get("text") or "").lower()
            need = len(words) if case["family"] == "keyword" else max(1, (len(words) + 1) // 2)
            if sum(w in t for w in words) >= need:
                return i
    return None


TEXT_FAMS = ("theory", "year_text", "theory_year")


def text_chunks(res):
    return [e for e in res["retrieval"]["results"] if e["coll"] == "text_chunks"]


def first_relevant(case, res):
    """Rank (among text chunks) of the first passage that matches the topic terms (and the year, if the question names one)."""
    rx = re.compile(case["terms"], re.I) if case.get("terms") else None
    for i, e in enumerate(text_chunks(res), 1):
        m = e["meta"]
        if case.get("year") and int(m.get("doc_year", 0)) != case["year"] and str(case["year"]) not in str(m.get("fy_mentioned", "")):
            continue
        if rx is None or rx.search(e["text"] if "text" in e else ""):
            return i
    return None


def year_purity(case, res):
    """Share of retrieved text passages that come from the filing of the requested year (the others are other years' filings)."""
    ch = text_chunks(res)[:8]
    if not ch or not case.get("year"):
        return None
    return sum(1 for e in ch if int(e["meta"].get("doc_year", 0)) == case["year"]) / len(ch)


def repeats(text):
    """True if the answer says the same thing twice (identical or >=80% word-overlapping sentences/bullets)."""
    parts = [re.findall(r"[a-z0-9]+", p.lower()) for p in re.split(r"[\n\u2022*\-]+|(?<=[.!?])\s+", text or "") if len(p.split()) >= 5]
    sets = [set(p) for p in parts]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            if len(sets[i] & sets[j]) / max(1, min(len(sets[i]), len(sets[j]))) >= 0.8:
                return True
    return False


def no_number_produced(res):
    ver = res.get("verification") or {}
    return not any((not r.get("refused")) and r.get("value") is not None for r in ver.get("results", []))


# ----------------------------------------------------------------------------- run
def run(a):
    sys.path.insert(0, ".")
    from stage10_answer import Pipeline
    cases = json.loads(Path(a.testset).read_text(encoding="utf-8"))
    if a.limit:
        random.Random(1).shuffle(cases)
        cases = cases[:a.limit]
    pipe = Pipeline(a.db, a.rows, a.tables, a.model, a.ollama_url, a.llm, not a.offline, a.chunks, a.offline)
    ks = (1, 3, 5, 8)
    agg = defaultdict(lambda: defaultdict(lambda: {"n": 0, "rank": [], "num": 0, "num_n": 0, "ans": 0, "ans_n": 0, "ref": 0, "ref_n": 0, "trank": [], "srank": [], "pur": [], "llm_n": 0, "grounded": 0, "cov": 0, "norep": 0, "cited": 0}))
    fails = []
    import time
    try:
      for mode in a.modes:
        llm_used = Counter()
        for n, case in enumerate(cases, 1):
            fam = case["family"]
            t0 = time.time()
            # the LLM only runs in 'full' mode (ablations are about retrieval), and for at most --llm-per-family questions per family
            want_llm = mode == "full" and pipe.llm is not None and llm_used[fam] < a.llm_per_family
            if want_llm:
                llm_used[fam] += 1
            if not want_llm and pipe.llm is not None:
                saved, pipe.llm = pipe.llm, None
            else:
                saved = None
            try:
                res = pipe.ask(case["question"], k=max(ks), mode=mode, policy="as_reported")
            except Exception as e:
                fails.append([mode, case["id"], fam, case["question"], f"EXCEPTION {type(e).__name__}: {e}"])
                if saved:
                    pipe.llm = saved
                continue
            if saved:
                pipe.llm = saved
            s = agg[mode][fam]
            s["n"] += 1
            if fam == "unanswerable":
                ok = no_number_produced(res)
                s["ref_n"] += 1
                s["ref"] += ok
                if not ok:
                    fails.append([mode, case["id"], fam, case["question"], "produced a number for an unanswerable question: " + case["why"]])
            elif fam in TEXT_FAMS:
                r = first_relevant(case, res)
                s["trank"].append(r)
                pur = year_purity(case, res)
                if pur is not None:
                    s["pur"].append(pur)
                if r is None:
                    fails.append([mode, case["id"], fam, case["question"], "no relevant passage (topic terms" + (" + year" if case.get("year") else "") + ") retrieved"])
                if want_llm:
                    ans = res["answer"]
                    body = ans.get("body") or ans.get("answer") or ""
                    s["llm_n"] += 1
                    g = ans.get("mode") == "llm"
                    s["grounded"] += g
                    c = bool(re.search(r"\[S\d+\]", body))
                    s["cited"] += c
                    rep = repeats(body)
                    s["norep"] += (not rep)
                    cov = bool(case.get("terms") and re.search(case["terms"], body, re.I)) if case.get("terms") else True
                    s["cov"] += cov
                    for ok_, why in ((g, "LLM answer rejected by grounding checks (template shown): " + str(ans.get("llm_status"))), (not rep, "answer repeats a point"),
                                     (c, "no [S#] citation in the answer"), (cov, "answer does not mention the topic terms")):
                        if not ok_:
                            fails.append([mode, case["id"], fam, case["question"], why])
            elif fam in ("text", "keyword"):
                r = text_rank(case, res)
                s["rank"].append(r)
                s["srank"].append(text_rank(case, res, strict=True))
                if r is None:
                    fails.append([mode, case["id"], fam, case["question"], "right filing + section/key words not retrieved"])
            else:
                r = rank_of(case, res)
                s["rank"].append(r)
                ok, got = numeric_outcome(case, res)
                s["num_n"] += 1
                s["num"] += ok
                if not ok:
                    fails.append([mode, case["id"], fam, case["question"], f"gold {case['gold']:.4f} got {got} | years {case['years']} | status {res['verification'].get('status')}"
                                  + (" | OCR filing" if case.get("ocr") else "")])
                if a.llm and mode == "full":
                    s["ans_n"] += 1
                    ans_ok = number_in_text(res["answer"].get("body") or res["answer"].get("answer"), case["gold"], case["tol"])
                    s["ans"] += ans_ok
                    if not ans_ok:
                        fails.append([mode, case["id"], fam, case["question"], f"written answer lacks gold {case['gold']:.4f} (llm_status {res['answer'].get('llm_status')})"])
            if want_llm or n % 10 == 0:
                print(f"  [{mode}] {n}/{len(cases)}  {fam:12s} {time.time() - t0:5.1f}s" + ("  (with LLM)" if want_llm else ""), file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted: writing a report from the questions finished so far.", file=sys.stderr)
    report(a, agg, ks, fails, len(cases))


def pct(x, n):
    return f"{100 * x / n:.1f}%" if n else "-"


def retr_stats(ranks, ks):
    n = len(ranks)
    hits = {k: sum(1 for r in ranks if r and r <= k) for k in ks}
    mrr = sum(1 / r for r in ranks if r) / n if n else 0
    return n, hits, mrr


def report(a, agg, ks, fails, ncases):
    out = [f"# Evaluation report ({ncases} questions, k up to {max(ks)})", ""]
    fams = ["fact", "paraphrase", "pct_change", "difference", "cagr", "margin", "text", "keyword", "unanswerable"]   # theory families have their own table
    summary = {}
    for mode in a.modes:
        out += [f"## Retrieval mode: {mode}", "", "| family | n | Hit@1 | Hit@3 | Hit@5 | Hit@8 | MRR | numeric accuracy | answer accuracy | refusal accuracy |", "|---|---|---|---|---|---|---|---|---|---|"]
        tot = defaultdict(float)
        allr = []
        for f in fams:
            s = agg[mode].get(f)
            if not s:
                continue
            n, hits, mrr = retr_stats(s["rank"], ks)
            allr += s["rank"]
            row = [f, str(s["n"])] + ([pct(hits[k], n) for k in ks] + [f"{mrr:.3f}"] if n else ["-"] * 5)
            row += [pct(s["num"], s["num_n"]), pct(s["ans"], s["ans_n"]) if s["ans_n"] else "n/a", pct(s["ref"], s["ref_n"])]
            out.append("| " + " | ".join(row) + " |")
            for k in ("num", "num_n", "ans", "ans_n", "ref", "ref_n"):
                tot[k] += s[k]
        n, hits, mrr = retr_stats(allr, ks)
        summary[mode] = {"hit": {k: hits[k] / n if n else 0 for k in ks}, "mrr": mrr, "num": tot["num"] / tot["num_n"] if tot["num_n"] else 0,
                         "ans": tot["ans"] / tot["ans_n"] if tot["ans_n"] else None, "ref": tot["ref"] / tot["ref_n"] if tot["ref_n"] else None}
        out += ["", f"Overall: Hit@1 {pct(hits[1], n)}, Hit@3 {pct(hits[3], n)}, Hit@5 {pct(hits[5], n)}, Hit@8 {pct(hits[8], n)}, MRR {mrr:.3f}, "
                f"numeric accuracy {pct(tot['num'], tot['num_n'])}, refusal accuracy {pct(tot['ref'], tot['ref_n'])}"
                + (f", written-answer accuracy {pct(tot['ans'], tot['ans_n'])}" if tot["ans_n"] else ""), ""]
    for mode in a.modes:
        tf = [(f, agg[mode][f]) for f in TEXT_FAMS if f in agg[mode]]
        if not tf:
            continue
        out += [f"## Theoretical (non-metric) questions, retrieval mode: {mode}", "",
                "| family | n | relevant Hit@1 | Hit@3 | Hit@5 | MRR | year purity | LLM n | grounded (no template fallback) | mentions topic | no repetition | cited |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for f, s_ in tf:
            n, hits, mrr = retr_stats(s_["trank"], (1, 3, 5))
            pur = f"{100 * sum(s_['pur']) / len(s_['pur']):.1f}%" if s_["pur"] else "-"
            ln = s_["llm_n"]
            out.append("| " + " | ".join([f, str(s_["n"])] + [pct(hits[k], n) for k in (1, 3, 5)] + [f"{mrr:.3f}", pur, str(ln)]
                                          + [pct(s_[k], ln) if ln else "n/a" for k in ("grounded", "cov", "norep", "cited")]) + " |")
        out.append("")
    if len(a.modes) > 1:
        out += ["## Ablation summary", "", "| mode | Hit@1 | Hit@5 | MRR | numeric accuracy |", "|---|---|---|---|---|"]
        for m, s in summary.items():
            out.append(f"| {m} | {100 * s['hit'][1]:.1f}% | {100 * s['hit'][5]:.1f}% | {s['mrr']:.3f} | {100 * s['num']:.1f}% |")
        out.append("")
    for mode in a.modes:
        sr = [r for f in ("text", "keyword") for r in agg[mode][f]["srank"]] if mode in agg else []
        if sr:
            n, hits, mrr = retr_stats(sr, (1, 3, 5))
            out.append(f"Strict exact-section match, mode {mode} (text+keyword, n={n}): Hit@1 {pct(hits[1], n)}, Hit@3 {pct(hits[3], n)}, Hit@5 {pct(hits[5], n)}, MRR {mrr:.3f}")
    out.append("")
    out += ["Text questions: the main table counts a hit when a retrieved passage is the gold chunk, from the same section, or from that filing and contains the key words "
            "(the same text is often filed under a neighbouring section). The strict line above requires the exact section.", ""]
    out += ["Theory notes: 'relevant' means a retrieved text passage contains the topic terms (and comes from / mentions the requested year); this is a proxy, "
            "not a human judgement of answer quality - read 10-20 answers yourself. Year purity = share of the top-8 text passages from the requested year's own filing "
            "(a low value on 'discussions in year Y' means other years' filings leak in).", ""]
    out += ["Notes: numeric accuracy is read from the verified result (operation, years and value within tolerance), not from the prose. "
            "Gold values come from the filings' own rows (as reported) and are computed in this script, independent of the pipeline. "
            "Questions on OCR-read filings (2019, 2020) are marked 'OCR filing' in failures.csv.", ""]
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "report.md").write_text("\n".join(out), encoding="utf-8")
    (od / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    with open(od / "failures.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "id", "family", "question", "reason"])
        w.writerows(fails)
    print("\n".join(out))
    print(f"\nwrote {od / 'report.md'}, {od / 'summary.json'}, {od / 'failures.csv'} ({len(fails)} failure rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["make", "run"])
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--rows", default="data/processed/table_rows.json")
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--chunks", default="data/processed/all_chunks.json")
    ap.add_argument("--testset", default="data/eval/testset.json")
    ap.add_argument("--out", default="data/eval/testset.json")
    ap.add_argument("--outdir", default="data/eval")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-fact", type=int, default=40)
    ap.add_argument("--n-para", type=int, default=15)
    ap.add_argument("--n-calc", type=int, default=12)
    ap.add_argument("--n-text", type=int, default=12)
    ap.add_argument("--n-year", type=int, default=10, help="'discussions in the year Y' questions")
    ap.add_argument("--llm-per-family", type=int, default=6, help="with --llm: how many questions per family get a Llama answer (it is slow)")
    ap.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid", "full"])
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else run(a)


if __name__ == "__main__":
    main()