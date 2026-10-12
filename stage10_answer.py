"""
Stage 10: grounded answer generation with a local Llama (Ollama, llama3.2:3b) + the end-to-end Pipeline.

The LLM never calculates and never supplies a figure.  It only phrases an answer from
  [R#] verified results (Stage 9: numbers, formulas, inputs, flags)   and   [S#] retrieved passages / table rows (Stage 7).
Guards:
  1. number grounding   every number in the generated answer must occur in the context (or be a rounding / unit conversion
                         of a verified value, a year, or a tiny count); otherwise the answer is regenerated once, then
                         replaced by a deterministic template answer built directly from the verified results.
  2. citation check     cited ids must exist; sources actually provided are always appended.
  3. refusals/flags     refused calculations, restatements, split adjustments and OCR reads are passed to the model and
                         always appended to the answer, whatever the model wrote.
  4. fallback           Ollama not running / model missing / timeout -> the template answer (mode "template").

Usage (needs Stages 1-9 outputs in data/, and `ollama serve` + `ollama pull llama3.2:3b`):
  python stage10_answer.py "difference between 2017 and 2018 sales"
  python stage10_answer.py "why did net sales increase in 2013" --show-context
  python stage10_answer.py "total assets in 2013" --no-llm           # deterministic template only
  python stage10_answer.py --selftest                                 # no data, no Ollama needed
"""
import argparse
import json
import re
import sys
import urllib.request
import urllib.error

MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434"
MAX_CONTEXT_CHARS = 9000            # ~2.3k tokens: leaves room in llama3.2:3b's 4k window for the question and answer
PASSAGE_CHARS = 700

SYSTEM = """You are a careful financial analyst answering questions about SEC 10-K filings.
Rules:
1. Use ONLY the VERIFIED RESULTS and SOURCES given. Never use outside knowledge.
2. Every number in your answer must be copied from VERIFIED RESULTS or quoted from a SOURCE. Do NOT calculate, round differently, or estimate.
3. Cite what you use with its tag, e.g. [R1] for a verified result and [S2] for a source.
4. If a result is marked restated or split-adjusted, say so in one short clause. If a calculation was REFUSED or a year is unavailable, say that plainly and do not substitute a figure.
5. If the sources do not contain the answer, say so. Do not guess.
6. Format: start with a one-sentence direct answer. Then, if useful, add at most 4 short bullet points (each starting with "- "). Put the key figures in **bold**. No tables, no headings, no more than about 120 words.
7. When many results are given (a comparison of several line items), summarise the 3-5 most significant changes with their figures and say the full numbers are in the tables."""


# ----------------------------------------------------------------------------- context
def _fmtv(r):
    return r.get("display") or f"{r.get('value')}"


def _res_line(tag, r, alt=False, compact=False):
    yrs = " to ".join(f"FY{y}" for y in ([r["years"][0], r["years"][-1]] if len(r["years"]) > 1 and r["op"] in ("difference", "pct_change", "cagr") else r["years"][:1] if r["op"] in ("value", "max", "min") else r["years"][:6]))
    s = f"[{tag}] {'(alternate basis) ' if alt else ''}{r['label']} - {r['op']} {yrs} = {_fmtv(r)}. Formula: {r['formula']}."
    if r.get("ranking"):
        s += " Ranking: " + "; ".join(f"#{x['rank']} FY{x['fy']} {x['display']}" for x in r["ranking"]) + "."
    ins = "" if compact and r["op"] != "value" else "; ".join(f"FY{i['fy']} {i['raw'] or i['value']} ({i['filing']} 10-K p{i['page']}" + (f", {i['status']}" if i["status"] not in (None, "ok") else "") + ")"
                    for i in r.get("inputs", [])[:4])
    if ins:
        s += f" Inputs: {ins}."
    for n in r.get("notes", [])[:2]:
        s += f" Note: {n}."
    return s


def build_context(verification, retrieval, max_results=16, max_passages=4, max_rows=4, passage_chars=PASSAGE_CHARS):
    """Return (context_text, tags) where tags maps 'R1'/'S2' -> descriptive dict used for citations."""
    tags, lines = {}, []
    if verification:
        lines.append("VERIFIED RESULTS (computed by Python from the filings; status: " + verification.get("status", "?") + ")")
        n = 0
        many = len(verification.get("results", [])) > 8
        for r in verification.get("results", [])[:max_results]:
            n += 1
            tags[f"R{n}"] = {"kind": "result", "text": _res_line(f"R{n}", r, compact=many), "result": r}
            lines.append(tags[f"R{n}"]["text"])
        alt = verification.get("alternate_basis")
        if alt:
            lines.append(f"ALTERNATE BASIS - {alt['note']}")
            for r in alt["results"][:max(2, max_results // 4)]:
                n += 1
                tags[f"R{n}"] = {"kind": "result", "text": _res_line(f"R{n}", r, True, many), "result": r}
                lines.append(tags[f"R{n}"]["text"])
        for r in verification.get("refusals", []):
            lines.append(f"REFUSED: {r['op']} {r.get('concept') or ''} - {r['reason']}")
        for w in verification.get("warnings", [])[:6]:
            lines.append(f"WARNING: {w}")
        if verification.get("note"):
            lines.append(f"NOTE: {verification['note']}")
        lines.append("")
    lines.append("SOURCES")
    s = 0
    texts = [e for e in (retrieval or {}).get("results", []) if e["coll"] == "text_chunks"][:max_passages]
    rows = [e for e in (retrieval or {}).get("results", []) if e["coll"] == "table_rows" and e["meta"].get("row_label")][:max_rows]
    for e in texts:
        s += 1
        m = e["meta"]
        body = re.sub(r"\s+", " ", e["text"]).strip()[:passage_chars]
        line = f"[S{s}] ({m.get('doc_year')} 10-K, p{m.get('page_start')}, {str(m.get('section_path', ''))[-70:]}) {body}"
        tags[f"S{s}"] = {"kind": "text", "text": line, "id": e["id"], "doc_year": m.get("doc_year"), "page": m.get("page_start"), "section": m.get("section_path")}
        lines.append(line)
    for e in rows:
        s += 1
        m = e["meta"]
        vals = json.loads(m.get("raw_json") or "{}")
        line = f"[S{s}] ({m.get('doc_year')} 10-K, p{m.get('page')}, {m.get('statement')}, {m.get('units')}) {m.get('row_label')}: " + \
               "; ".join(f"FY{y} {v}" for y, v in sorted(vals.items()))
        tags[f"S{s}"] = {"kind": "table", "text": line, "id": e["id"], "doc_year": m.get("doc_year"), "page": m.get("page"), "section": m.get("statement")}
        lines.append(line)
    if s == 0:
        lines.append("(none)")
    text = "\n".join(lines)
    return text[:MAX_CONTEXT_CHARS], tags


# ----------------------------------------------------------------------------- number grounding
TAG = re.compile(r"\[(?:R|S)\d+\]")
NUM = re.compile(r"(?<![\w.])\(?-?\$?\d[\d,]*(?:\.\d+)?\)?")


def parse_nums(text):
    """-> list of (float_abs, decimals, raw)"""
    out = []
    for m in NUM.finditer(TAG.sub(" ", text)):
        raw = m.group(0)
        t = raw.strip("()$-").replace(",", "")
        t = t.rstrip(".")
        if not t or not re.match(r"^\d+(\.\d+)?$", t):
            continue
        dec = len(t.split(".")[1]) if "." in t else 0
        out.append((abs(float(t)), dec, raw))
    return out


def allowed_numbers(context, verification):
    allowed = [a for a, _, _ in parse_nums(context)]
    vals = []
    for r in (verification or {}).get("results", []) + [x for x in ((verification or {}).get("alternate_basis") or {}).get("results", [])]:
        vals.append(abs(float(r["value"])))
        for i in r.get("inputs", []):
            vals.append(abs(float(i["value"])))
        for x in r.get("ranking", []) or []:
            vals.append(abs(float(x["value"])))
    for v in vals:
        allowed += [v, v / 1000.0, v / 1e6]                  # USD millions -> billions -> trillions
    return allowed


def grounded(raw_value, dec, allowed):
    if raw_value in range(0, 13) and dec == 0:               # small counts ("5 years", "2 segments")
        return True
    if dec == 0 and 1990 <= raw_value <= 2035:               # years
        return True
    for a in allowed:
        if round(a, dec) == round(raw_value, dec) or abs(a - raw_value) <= 1e-9 * max(1.0, abs(a)):
            return True
    return False


def check_numbers(answer, context, verification):
    allowed = allowed_numbers(context, verification)
    bad = [raw for v, d, raw in parse_nums(answer) if not grounded(v, d, allowed)]
    return bad


def check_citations(answer, tags):
    cited = re.findall(r"\[((?:R|S)\d+)\]", answer)
    return [c for c in cited if c not in tags]


# ----------------------------------------------------------------------------- deterministic template answer
def _short(d):
    return re.sub(r"\.00(?= |%|$)", "", d)


def overview_answer(verification, tags):
    names = ", ".join(c.replace("_", " ") for c in verification["overview"])
    groups = {}
    for k, v in tags.items():
        if v["kind"] != "result" or "(alternate basis)" in v["text"]:
            continue
        r = v["result"]
        g = groups.setdefault(r["label"], {"vals": [], "diff": None, "pct": None, "tags": []})
        g["tags"].append(k)
        if r["op"] == "value":
            g["vals"].append((r["years"][0], _short(r["display"])))
        elif r["op"] == "difference":
            g["diff"] = _short(r["display"])
        elif r["op"] in ("pct_change", "yoy"):
            g["pct"] = _short(r["display"])
    lines = []
    for label, g in groups.items():
        vals = " -> ".join(f"{v} (FY{y})" for y, v in sorted(g["vals"]))
        chg = ", ".join(x for x in ((g["diff"] and f"change {g['diff']}"), (g["pct"] and f"{g['pct']}")) if x)
        lines.append(f"- **{label}**: {vals}" + (f" ({chg})" if chg else "") + " " + "".join(f"[{t}]" for t in g["tags"]))
    return (f"No specific metric was named, so here is how the key figures compare ({names}):\n" + "\n".join(lines) +
            "\nAsk about one item, e.g. 'total net sales in 2020', for a focused answer.")


def template_answer(question, verification, retrieval, tags):
    if (verification or {}).get("overview"):
        return overview_answer(verification, tags)
    parts = []
    res = [(k, v["result"]) for k, v in tags.items() if v["kind"] == "result" and "(alternate basis)" not in v["text"]]
    for k, r in res[:12]:
        y = r["years"]
        op, lab = r["op"], r["label"]
        if op == "value":
            parts.append(f"{lab} for FY{y[0]} was {r['display']} [{k}].")
        elif op == "difference":
            parts.append(f"{lab} changed by {r['display']} between FY{y[0]} and FY{y[-1]} [{k}].")
        elif op in ("pct_change", "yoy"):
            parts.append(f"{lab} changed by {r['display']} from FY{y[0]} to FY{y[-1]} [{k}].")
        elif op == "cagr":
            parts.append(f"The compound annual growth rate of {lab} from FY{y[0]} to FY{y[-1]} was {r['display']} [{k}].")
        elif op in ("max", "min"):
            parts.append(f"The {'highest' if op == 'max' else 'lowest'} {lab} was in FY{r.get('year', y[0])}: {r['display']} [{k}].")
        elif op == "rank":
            parts.append("Ranking by " + lab + ": " + "; ".join(f"FY{x['fy']} {x['display']}" for x in r["ranking"]) + f" [{k}].")
        elif op in ("sum", "average"):
            parts.append(f"The {op} of {lab} over FY{y[0]}-FY{y[-1]} was {r['display']} [{k}].")
        else:
            parts.append(f"{lab}: {r['display']} [{k}].")
    if verification:
        for r in verification.get("refusals", []):
            parts.append(f"I could not calculate {r['op']}{' for ' + r['concept'] if r.get('concept') else ''}: {r['reason']}")
    explain = re.search(r"\b(why|explain|reason|driver|factor|risk|cause|how did|what drove|due to)\b", question.lower())
    if not parts or explain:
        srcs = [(k, v) for k, v in tags.items() if v["kind"] == "text"][:(3 if not parts else 2)]
        if srcs:
            parts.append("The most relevant passages from the filings are:" if not any(not p.startswith("- ") for p in parts)
                         else "\nRelevant passages from the filings:")
            for k, v in srcs:
                body = re.sub(r"^\[S\d+\]\s*", "", v["text"])
                parts.append("- " + body[:320].rstrip() + (" ..." if len(body) > 320 else "") + f" [{k}]")
        else:
            parts.append("The indexed filings do not contain information that answers this question.")
    return " ".join(parts) if not any(p.startswith("- ") for p in parts) else "\n".join(parts)


def dedupe_points(text):
    """Drop a bullet/sentence that repeats an earlier one (small local models sometimes say the same thing twice)."""
    def words(x):
        return set(re.findall(r"[a-z0-9]+", re.sub(r"\[(?:R|S)\d+\]", "", x.lower())))
    kept, seen = [], []
    for line in text.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", line) if not re.match(r"\s*[-*\u2022]", line) else [line]
        out = []
        for p in parts:
            w = words(p)
            if len(w) >= 5 and any(len(w & o) / max(1, min(len(w), len(o))) >= 0.8 for o in seen):
                continue
            seen.append(w)
            out.append(p)
        if out or not line.strip():
            kept.append(" ".join(out) if out else line)
    return "\n".join(kept).strip()


def trailer(verification, tags, used_tags):
    """Flags and sources that are ALWAYS shown, whatever the model wrote."""
    out = []
    if verification:
        if verification.get("flagged_inputs"):
            out.append("Note: some inputs differ between filings (restated or split-adjusted); see the alternate basis in the details.")
        ocr_years = sorted({m for w in verification.get("warnings", []) if "OCR" in w for m in re.findall(r"FY(\d{4})", w)})
        if ocr_years:
            out.append("Note: values for " + ", ".join(f"FY{y}" for y in ocr_years) + " were read by OCR from scanned pages; check them against the filing.")
        for w in verification.get("warnings", []):
            if "split" in w or "components come from different filings" in w:
                out.append(f"Note: {w}")
        for r in verification.get("refusals", []):
            out.append(f"Not calculated: {r['op']}{' ' + r['concept'] if r.get('concept') else ''} - {r['reason']}")
    seen, src = set(), []
    for k in sorted(used_tags | {t for t in tags if t.startswith("S")}, key=lambda t: (t[0] != "R", int(t[1:]))):
        v = tags.get(k)
        if not v or k in seen:
            continue
        seen.add(k)
        if v["kind"] == "result":
            for i in v["result"].get("inputs", [])[:4]:
                src.append(f"{i['filing']} 10-K p{i['page']}")
        else:
            src.append(f"{v['doc_year']} 10-K p{v['page']} ({str(v['section'])[-40:]})")
    if src:
        uniq = list(dict.fromkeys(re.sub(r" \(.*\)$", "", x) for x in src))        # one entry per filing page
        out.append("Sources: " + "; ".join(uniq[:8]))
    return out


# ----------------------------------------------------------------------------- LLM clients
class LLMUnavailable(Exception):
    pass


class OllamaLLM:
    def __init__(self, model=MODEL, url=OLLAMA_URL, timeout=300, num_ctx=4096):
        self.model, self.url, self.timeout, self.num_ctx = model, url.rstrip("/"), timeout, num_ctx

    def __call__(self, messages):
        body = json.dumps({"model": self.model, "messages": messages, "stream": False,
                           "options": {"temperature": 0, "seed": 7, "num_ctx": self.num_ctx, "num_predict": 400}}).encode()
        req = urllib.request.Request(self.url + "/api/chat", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            raise LLMUnavailable(f"Ollama returned HTTP {e.code} (is the model pulled? `ollama pull {self.model}`)") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise LLMUnavailable(f"Ollama not reachable at {self.url} ({e}); start it with `ollama serve`") from e


# ----------------------------------------------------------------------------- answer
def is_narrative(question, retrieval):
    intent = ((retrieval or {}).get("plan") or {}).get("intent")
    return intent == "narrative" or bool(re.search(r"\b(why|explain|describe|discuss|summar\w+|risks?|strateg\w+|tell (?:me )?(?:about|more)|what (?:is|are|does|do)|how (?:does|do|did))\b", question.lower()))


NARRATIVE_SYSTEM = SYSTEM + """
8. For explanatory or broad questions, write a short synthesis (up to 6 sentences or bullets) in your own words from the SOURCES, citing each claim with its [S#] tag. Quote only short phrases.
9. Each sentence or bullet must cover a DIFFERENT topic or section. Never repeat a point. Start with "Based on the retrieved passages" and do not claim to list everything the filing says."""


def answer(question, verification, retrieval, llm=None):
    narrative = is_narrative(question, retrieval)
    ctx, tags = build_context(verification, retrieval, max_passages=8 if narrative else 4, passage_chars=800 if narrative else PASSAGE_CHARS)
    out = {"question": question, "mode": "template", "answer": "", "citations": [], "warnings": [], "grounding": {}, "context": ctx, "attempts": [],
           "narrative": narrative}
    text = None
    out["llm_status"] = "disabled (--no-llm)" if llm is None else "pending"
    if llm is not None:
        msgs = [{"role": "system", "content": NARRATIVE_SYSTEM if narrative else SYSTEM},
                {"role": "user", "content": f"{ctx}\n\nQUESTION: {question}\n\nAnswer using only the context above."}]
        try:
            for attempt in (1, 2):
                cand = llm(msgs)
                out["attempts"].append(cand)
                bad, badc = check_numbers(cand, ctx, verification), check_citations(cand, tags)
                out["grounding"][f"attempt{attempt}"] = {"ungrounded_numbers": bad, "bad_citations": badc}
                if not bad and not badc and cand.strip():
                    text, out["mode"], out["llm_status"] = dedupe_points(cand.strip()), "llm", "used"
                    break
                msgs = msgs + [{"role": "assistant", "content": cand},
                               {"role": "user", "content": "Your answer contained " + (f"numbers not in the context: {bad}. " if bad else "") +
                                (f"citations that do not exist: {badc}. " if badc else "") +
                                "Rewrite it using only numbers copied from the context and only existing tags."}]
        except LLMUnavailable as e:
            out["warnings"].append(str(e))
            out["llm_status"] = "unavailable: " + str(e)[:90]
    if text is None:
        if llm is not None and not out["warnings"]:
            why = "; ".join(f"attempt {k[-1]}: " + (f"numbers not in the context {v['ungrounded_numbers']} " if v["ungrounded_numbers"] else "") +
                            (f"bad citations {v['bad_citations']}" if v["bad_citations"] else "") + ("" if v["ungrounded_numbers"] or v["bad_citations"] else "empty answer")
                            for k, v in out["grounding"].items())
            out["warnings"].append("the model's answer was rejected by the grounding checks (" + why + "); the template answer is shown instead")
            out["llm_status"] = "rejected by grounding checks"
        text = template_answer(question, verification, retrieval, tags)
    used = set(re.findall(r"\[((?:R|S)\d+)\]", text))
    tr = trailer(verification, tags, used)
    out["body"], out["trailer"] = text, tr                    # the UI shows body + separate flag/source panels; the CLI shows the full text
    out["answer"] = text + ("\n\n" + "\n".join(tr) if tr else "")
    out["citations"] = [{"tag": k, "kind": v["kind"], "text": v["text"][:300]} for k, v in tags.items() if k in used or v["kind"] != "result"]
    return out


# ----------------------------------------------------------------------------- follow-up questions
FOLLOWUP = re.compile(r"\b(it|its|that|this|those|these|they|them|same|also|too|again|then|previous|before|earlier|what about|how about|and)\b", re.I)
CONDENSE_SYSTEM = ("Rewrite the user's LAST question as a complete standalone question, using the chat history only to fill in what it leaves out "
                   "(the company metric, the year, what 'it' or 'that' refers to). If the last question is already complete, return it unchanged. "
                   "Never add years, numbers or facts that are not in the chat. Output only the question, on one line.")


def condense(question, history, llm):
    """Turn a follow-up ('and 2019?', 'why did it increase?') into a standalone question.  Falls back to the question itself
    whenever there is no history, no LLM, the question looks complete, or the rewrite introduces years not found in the chat."""
    if not history or llm is None:
        return question
    words = question.split()
    if len(words) > 12 and not FOLLOWUP.search(question):
        return question
    if len(words) > 12 and re.search(r"\b(19|20)\d{2}\b", question) and not re.search(r"\b(it|that|this|those|they|them|same)\b", question, re.I):
        return question
    chat = "\n".join(f"User: {h['q']}\nAssistant: {str(h.get('a', ''))[:400]}" for h in history[-3:])
    try:
        out = llm([{"role": "system", "content": CONDENSE_SYSTEM},
                   {"role": "user", "content": f"CHAT HISTORY\n{chat}\n\nLAST QUESTION: {question}\n\nStandalone question:"}])
    except LLMUnavailable:
        return question
    out = out.strip().splitlines()[0].strip().strip('"') if out.strip() else ""
    seen = set(re.findall(r"\b(?:19|20)\d{2}\b", question + " " + " ".join(h["q"] for h in history)))
    if not out or len(out) > 300 or not set(re.findall(r"\b(?:19|20)\d{2}\b", out)) <= seen:
        return question
    return out


# ----------------------------------------------------------------------------- end-to-end pipeline
_NOT_COMPANY = set("""january february march april may june july august september october november december fy q1 q2 q3 q4 us usd gaap sec
the a an this that these those our your their its all any each every total net gross operating free what which who how why when where""".split())


def foreign_company(question, known_companies, vocab=()):
    """Name of a company the user asks about that is NOT in the corpus ('Net income of Microsoft in 2019', "Tesla's revenue").
    Only capitalised words right after of/for/at/by or before 's, that are not financial words, are considered."""
    known = [c.lower() for c in known_companies]
    cand = re.findall(r"\b(?:of|for|at|by|from)\s+([A-Z][A-Za-z&.\-]{2,})", question) + re.findall(r"\b([A-Z][A-Za-z&.\-]{2,})['\u2019]s\b", question)
    for c in cand:
        lc = c.lower()
        if lc in _NOT_COMPANY or lc in vocab or any(lc in k or k in lc for k in known):
            continue
        return c
    return None


class Pipeline:
    """Stages 6 -> 7 -> 8/9 -> 10 behind one call, reused by the UI (Stage 14)."""

    def __init__(self, db="data/chroma", rows="data/processed/table_rows.json", tables="data/processed/all_tables.json",
                 model=MODEL, ollama_url=OLLAMA_URL, use_llm=True, use_dense=True, chunks="data/processed/all_chunks.json", offline=False):
        """offline=True reads table_rows.json / all_chunks.json directly (BM25 + label channels only): no Chroma, no BGE model."""
        from stage6_query_parser import Catalog, QueryParser, CONCEPTS
        try:                                                 # the file may be saved as stage7_hybrid_retrieval.py or stage7_retrieve.py
            import stage7_hybrid_retrieval as s7
        except ImportError:
            import stage7_retrieve as s7
        import stage8_align as s8
        self.parser = QueryParser(Catalog.from_json(tables))
        if offline:
            corpus, dense = s7.Corpus(_offline_records(rows, chunks)), _NoDense()
        else:
            corpus, dense = s7.Corpus.from_chroma(db), (s7.ChromaDense(db) if use_dense else _NoDense())
        self.retriever = s7.Retriever(corpus, dense, self.parser)
        self.aligner = s8.Aligner(rows, concepts=CONCEPTS)
        self.llm = OllamaLLM(model, ollama_url) if use_llm else None

    def ask(self, question, k=8, mode="full", policy="as_reported", history=None):
        """history: [{"q": earlier question, "a": earlier answer text}, ...] (most recent last) for follow-up questions."""
        import stage9_verify as s9
        standalone = condense(question, history, self.llm)
        plan = self.parser.parse(standalone)
        vocab = set(getattr(self.retriever.c, "label_vocab", ())) | {w for f in ("revenue", "income", "assets", "sales", "equity", "cash") for w in [f]}
        other = foreign_company(standalone, self.parser.cat.companies, vocab)
        if other:
            have = ", ".join(self.parser.cat.companies)
            msg = f"I only have the 10-K filings of {have} in this knowledge base, so I cannot answer about {other}."
            return {"question": question, "interpreted": None, "plan": plan,
                    "retrieval": {"results": [], "evidence_ids": [], "plan": plan, "missing_years": []},
                    "verification": {"status": "refused", "results": [], "refusals": [{"op": "lookup", "concept": None, "refused": True, "reason": msg}],
                                     "warnings": [], "flagged_inputs": {}, "primary_basis": None},
                    "answer": {"question": question, "mode": "template", "answer": msg, "body": msg, "trailer": [], "citations": [], "warnings": [],
                               "grounding": {}, "context": "", "attempts": [], "narrative": False, "llm_status": "not needed (company not in corpus)"}}
        retrieval = self.retriever.retrieve(standalone, mode, k, plan=plan)
        verification = s9.verify(plan, self.aligner, policy, evidence_ids=set(retrieval["evidence_ids"]), parser=self.parser)
        ans = answer(standalone, verification, retrieval, self.llm)
        return {"question": question, "interpreted": standalone if standalone != question else None, "plan": plan, "retrieval": retrieval,
                "verification": verification, "answer": ans}


def _offline_records(rows_path, chunks_path):
    from pathlib import Path
    recs = [{"id": r["id"], "text": r["text"], "meta": r["meta"], "coll": "table_rows"}
            for r in json.loads(Path(rows_path).read_text(encoding="utf-8"))]
    if Path(chunks_path).exists():
        for c in json.loads(Path(chunks_path).read_text(encoding="utf-8")):
            recs.append({"id": c["chunk_id"], "text": c["text"], "coll": "text_chunks", "meta": {
                "chunk_type": c.get("chunk_type"), "company": c["company"], "doc_year": int(c["doc_year"]),
                "filing_fiscal_year": int(c.get("filing_fiscal_year", c["doc_year"])),
                "fy_mentioned": ",".join(str(y) for y in c.get("financial_years_mentioned", [])), "part": c.get("part") or "",
                "item": c.get("item") or "", "item_title": c.get("item_title") or "", "section_path": c.get("section_path", ""),
                "page_start": int(c.get("page_start", 0)), "page_end": int(c.get("page_end", 0)), "source_file": c.get("source_file", ""),
                "ocr_used": bool(c.get("ocr_used", False)), "numeric_ratio": float(c.get("numeric_ratio", 0.0)),
                "word_count": int(c.get("word_count", 0))}})
    return recs


class _NoDense:
    def search(self, *a, **k):
        return []


# ----------------------------------------------------------------------------- self-test (no data, no Ollama)
def _ver():
    r1 = {"op": "difference", "concept": "revenue", "label": "Total net sales", "years": [2017, 2018], "value": 55021.0, "unit": "USD millions",
          "display": "55,021.00 USD millions", "formula": "FY2018 - FY2017", "steps": [], "cross_check": "passed", "notes": [],
          "inputs": [{"concept": "revenue", "fy": 2017, "value": 177866.0, "unit": "USD millions", "filing": 2017, "page": 48, "raw": "177,866", "status": "ok"},
                     {"concept": "revenue", "fy": 2018, "value": 232887.0, "unit": "USD millions", "filing": 2018, "page": 45, "raw": "232,887", "status": "ok"}]}
    r2 = dict(r1, op="pct_change", value=30.9335, display="30.93%", formula="(FY2018 - FY2017) / |FY2017| x 100")
    return {"status": "verified", "results": [r1, r2], "refusals": [], "warnings": [], "flagged_inputs": {}, "primary_basis": "as_reported"}


def _retr():
    return {"results": [
        {"id": "c1", "coll": "text_chunks", "text": "Net sales increased 31% in 2018 compared with 2017, primarily driven by increased unit sales.",
         "meta": {"doc_year": 2018, "page_start": 30, "section_path": "PART II > Item 7 > MD&A"}},
        {"id": "t1", "coll": "table_rows", "text": "x", "meta": {"doc_year": 2018, "page": 45, "statement": "operations", "units": "USD millions",
                                                              "row_label": "Total net sales", "raw_json": json.dumps({"2016": "135,987", "2017": "177,866", "2018": "232,887"})}}]}


class Stub:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), 0

    def __call__(self, messages):
        self.calls += 1
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class Down:
    def __call__(self, messages):
        raise LLMUnavailable("Ollama not reachable")


def selftest():
    chk = []

    def check(name, cond, extra=""):
        chk.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   {extra}"))

    v, rt = _ver(), _retr()
    ctx, tags = build_context(v, rt)
    check("context has [R1],[R2] and [S1],[S2] and stays within the size limit", all(t in tags for t in ("R1", "R2", "S1", "S2")) and len(ctx) <= MAX_CONTEXT_CHARS)
    good = "Total net sales rose by 55,021.00 USD millions (about $55.0 billion), a 30.93% increase from FY2017 to FY2018 [R1][R2]. Management attributes it to unit sales [S1]."
    check("grounded answer passes the number check (commas, billions, percent)", check_numbers(good, ctx, v) == [], check_numbers(good, ctx, v))
    check("an invented number is caught", check_numbers("Net sales rose by 60,000 million [R1].", ctx, v) == ["60,000"])
    check("rounded variant of a verified value is accepted (30.9%)", check_numbers("Growth was 30.9% [R2].", ctx, v) == [])
    check("a wrong rounding is rejected (31.5%)", check_numbers("Growth was 31.5% [R2].", ctx, v) == ["31.5"])
    check("years and small counts are allowed", check_numbers("Over 2 years, from 2017 to 2018 [R1].", ctx, v) == [])
    check("a non-existent citation is caught", check_citations("see [S9]", tags) == ["S9"])

    a = answer("q", v, rt, Stub(good))
    check("grounded model answer is used (mode llm) and sources are appended", a["mode"] == "llm" and "Sources:" in a["answer"] and "2017 10-K p48" in a["answer"], a["answer"])
    st = Stub("Net sales rose by 60,000 million [R1].", "Net sales rose by 55,000 million [R1].")
    a = answer("q", v, rt, st)
    check("two hallucinated answers -> template answer from verified results, after exactly 2 calls", a["mode"] == "template" and st.calls == 2 and "55,021.00" in a["answer"] and "60,000" not in a["answer"], a["answer"])
    st = Stub("Net sales rose by 60,000 million [R1].", good)
    a = answer("q", v, rt, st)
    check("a hallucination followed by a corrected answer is accepted on the retry", a["mode"] == "llm" and st.calls == 2)
    a = answer("q", v, rt, Down())
    check("Ollama down -> template answer and an explanatory warning", a["mode"] == "template" and a["warnings"] and "55,021.00" in a["answer"])
    a = answer("q", v, rt, None)
    check("no LLM requested -> deterministic template, no warnings", a["mode"] == "template" and not a["warnings"])

    vr = {"status": "refused", "results": [], "warnings": [], "flagged_inputs": {}, "refusals": [{"op": "cagr", "concept": "revenue", "reason": "CAGR is undefined for non-positive endpoints"}]}
    a = answer("q", vr, {"results": []}, Stub("The CAGR is 12.5% [R1]."))
    check("refusal: invented figure rejected and the refusal is stated", a["mode"] == "template" and "could not calculate" in a["answer"] and "12.5" not in a["answer"], a["answer"])
    vf = dict(_ver(), flagged_inputs={"total_assets": [2015]}, warnings=["a stock split changes per-share figures between filings"])
    a = answer("q", vf, rt, Stub(good))
    check("restated/split flags are always appended even if the model ignores them", "differ between filings" in a["answer"] and "stock split" in a["answer"])
    vn = {"status": "not_applicable", "results": [], "refusals": [], "warnings": [], "flagged_inputs": {}, "note": "no metric"}
    a = answer("what are the risks", vn, {"results": [_retr()["results"][0]]}, None)
    check("text-only question: extractive answer quotes the passage with its tag", "[S1]" in a["answer"] and "Net sales increased 31%" in a["answer"], a["answer"])
    a = answer("anything", vn, {"results": []}, None)
    check("no evidence at all: says the filings do not contain it", "do not contain" in a["answer"])
    ctx2, _ = build_context(v, {"results": [dict(_retr()["results"][0], text="x" * 50000)] * 10})
    check("very long passages are truncated to protect the model's context window", len(ctx2) <= MAX_CONTEXT_CHARS)
    vn2 = {"status": "not_applicable", "results": [], "refusals": [], "warnings": [], "flagged_inputs": {}, "note": "no metric"}
    rt2 = {"plan": {"intent": "narrative"}, "results": [_retr()["results"][0]]}
    synth = "Competition and growth are linked: net sales increased 31% in 2018, driven by unit sales [S1]."
    st = Stub(synth)
    a = answer("why did sales rise in 2018", vn2, rt2, st)
    check("general question: the LLM writes the answer (mode llm, status used) and its system prompt asks for synthesis",
          a["mode"] == "llm" and a["llm_status"] == "used" and a["attempts"] == [synth] and a["narrative"])
    ov = {"status": "verified", "results": _ver()["results"], "refusals": [], "warnings": [], "flagged_inputs": {}, "overview": ["revenue"]}
    a = answer("compare 2017 and 2018", ov, {"plan": {"intent": "comparison"}, "results": []}, Stub("Net sales rose by 55,021.00 USD millions, or 30.93% [R1][R2]."))
    check("comparison without a named metric: the LLM now writes the answer", a["mode"] == "llm" and a["llm_status"] == "used")
    a = answer("compare 2017 and 2018", ov, {"plan": {"intent": "comparison"}, "results": []}, Down())
    check("...and if Ollama is down the key-figures list is the fallback", a["mode"] == "template" and "key figures compare" in a["answer"])
    a = answer("tell me about the risks", ov, {"plan": {"intent": "narrative"}, "results": [_retr()["results"][0]]}, Stub("Risks include competition [S1]."))
    check("'tell me about ...' goes to the LLM even when an overview exists", a["mode"] == "llm")
    a = answer("why", vn2, rt2, Stub("Sales rose 99% [S1]."))
    check("rejected answer reports the reason in warnings and llm_status", a["llm_status"] == "rejected by grounding checks" and "99" in a["warnings"][0])
    a = answer("why", vn2, rt2, None)
    check("no LLM -> llm_status says disabled", a["llm_status"].startswith("disabled"))

    hist = [{"q": "net sales in 2017", "a": "Total net sales in FY2017 were 177,866 USD millions."}]
    check("follow-up 'and 2018?' is rewritten into a standalone question", condense("and 2018?", hist, Stub("What were total net sales in 2018?")) == "What were total net sales in 2018?")
    check("a rewrite that invents a year is rejected", condense("and 2018?", hist, Stub("What were total net sales in 2019?")) == "and 2018?")
    check("a complete question is not rewritten (no LLM call)", condense("What was operating income in 2019 compared with 2018 for the whole company?", hist, Down()) == "What was operating income in 2019 compared with 2018 for the whole company?")
    check("no history -> unchanged", condense("and 2018?", [], Stub("x")) == "and 2018?")
    check("LLM unavailable during rewriting -> unchanged", condense("and 2018?", hist, Down()) == "and 2018?")
    ovv = dict(_ver(), overview=["revenue"], results=[dict(_ver()["results"][0], op="value", years=[2017], display="177,866.00 USD millions"),
                                                      dict(_ver()["results"][0], op="value", years=[2018], display="232,887.00 USD millions"),
                                                      _ver()["results"][0], _ver()["results"][1]])
    txt = overview_answer(ovv, build_context(ovv, {"results": []})[1])
    check("key-figures fallback is grouped per line item with the change on the same line", txt.count("**Total net sales**") == 1 and "177,866 USD millions (FY2017) -> 232,887 USD millions (FY2018)" in txt and "30.93%" in txt, txt)

    print(f"\n{sum(chk)}/{len(chk)} passed")
    return 0 if all(chk) else 1


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--rows", default="data/processed/table_rows.json")
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--offline", action="store_true", help="no Chroma / BGE: read the JSON files directly (BM25 + label channels)")
    ap.add_argument("--chunks", default="data/processed/all_chunks.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--policy", default="as_reported", choices=["as_reported", "latest", "same_filing"])
    ap.add_argument("--show-context", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    print("Loading corpus and models ...")
    P = Pipeline(a.db, a.rows, a.tables, a.model, a.ollama_url, not a.no_llm, not a.no_dense, a.chunks, a.offline)
    qs = [" ".join(a.question)] if a.question else None
    while True:
        q = qs[0] if qs else input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        r = P.ask(q, a.k, policy=a.policy)
        ans = r["answer"]
        if a.json:
            print(json.dumps({"answer": ans["answer"], "mode": ans["mode"], "warnings": ans["warnings"], "grounding": ans["grounding"]}, indent=1))
        else:
            print(f"\nQ: {q}   [answer mode: {ans['mode']}]")
            for n in r["plan"].get("notes", []):
                print("  note:", n)
            print("\n" + ans["answer"])
            for w in ans["warnings"]:
                print("  warning:", w)
            if a.show_context:
                print("\n---- context sent to the model ----\n" + ans["context"])
        if qs:
            break


if __name__ == "__main__":
    main()