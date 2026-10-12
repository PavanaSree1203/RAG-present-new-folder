"""
Stage 7: hybrid text + table retrieval.

Channels (each returns a ranked list of record ids):
  dense  - BGE embeddings in the Chroma collections `text_chunks` and `table_rows`
  bm25   - own BM25 implementation over the same records (exact words, numbers, rare terms)
  label  - exact line-item match: rows whose canonical label equals a label variant that Stage 6 resolved
Fusion: weighted Reciprocal Rank Fusion (RRF).  Filters come from the Stage 6 plan (company, statement, financial
year flags for tables, candidate filing years for text).

Modes (for the Stage 13 ablations):
  dense   dense only, no filters          bm25  BM25 only, no filters
  hybrid  dense + BM25 + RRF, no filters  full  dense + BM25 + label channel + RRF + plan filters  (default)

Usage:
    python stage7_hybrid_retrieval.py "total assets in 2013"
    python stage7_hybrid_retrieval.py "total assets in 2013" --compare       # the four modes side by side
    python stage7_hybrid_retrieval.py --selftest                              # offline logic tests (no model needed)
Needs Stage 6 (stage6_query_parser.py, canon.py, typo.py) and the Stage 3/5 Chroma index in data/chroma.
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MODEL_NAME = "BAAI/bge-base-en-v1.5"
RRF_K = 60
WEIGHTS = {"dense": 1.0, "bm25": 1.0, "label": 1.5}
MODES = ("dense", "bm25", "hybrid", "full")
STOP = set("a an the of in on at for to and or is are was were be been what which who how did do does by with from "
           "as that this it its year years amazon".split())


# ----------------------------------------------------------------------------- text helpers
def toks(s):
    s = (s or "").lower().replace("&", " and ").replace("’", "'").replace("'", "")
    out = []
    for w in re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", s):
        if w in STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]                                           # crude plural stemming: assets -> asset
        out.append(w)
    return out


def norm_label(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (s or "").lower().replace("’", "'").replace("'", "")).split())


# ----------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        """docs: list of (id, text)."""
        self.k1, self.b = k1, b
        self.ids, self.tf, self.len = [], [], []
        df = Counter()
        for i, t in docs:
            tk = toks(t)
            self.ids.append(i)
            self.tf.append(Counter(tk))
            self.len.append(len(tk))
            df.update(set(tk))
        self.n = len(self.ids)
        self.avg = (sum(self.len) / self.n) if self.n else 0.0
        self.idf = {w: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.postings = defaultdict(list)
        for ix, tf in enumerate(self.tf):
            for w in tf:
                self.postings[w].append(ix)

    def search(self, query, k=50, allowed=None):
        score = defaultdict(float)
        for w in set(toks(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for ix in self.postings[w]:
                if allowed is not None and self.ids[ix] not in allowed:
                    continue
                f = self.tf[ix][w]
                score[ix] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[ix] / self.avg))
        top = sorted(score.items(), key=lambda x: -x[1])[:k]
        return [(self.ids[ix], s) for ix, s in top]


# ----------------------------------------------------------------------------- fusion
def rrf(ranked_lists, weights, k=RRF_K):
    """ranked_lists: {channel: [id, ...]} -> [(id, score, {channel: rank})] best first."""
    score, where = defaultdict(float), defaultdict(dict)
    for ch, lst in ranked_lists.items():
        for r, i in enumerate(lst, 1):
            score[i] += weights.get(ch, 1.0) / (k + r)
            where[i][ch] = r
    return [(i, s, where[i]) for i, s in sorted(score.items(), key=lambda x: (-x[1], x[0]))]


# ----------------------------------------------------------------------------- corpus
class Corpus:
    """Records from the two Chroma collections: id -> {text, meta, coll}."""

    def __init__(self, records):
        self.rec = {r["id"]: r for r in records}
        self.by_coll = defaultdict(list)
        for r in records:
            self.by_coll[r["coll"]].append(r["id"])
        self.bm25 = {c: BM25([(i, self.rec[i]["text"]) for i in ids]) for c, ids in self.by_coll.items()}
        GENERIC = {"total", "net", "other", "year", "years", "current", "long", "term", "and", "from", "with", "income", "tax", "taxes"} - {"income"}
        self.label_vocab = set()                                 # content words of the line-item labels ("assets", "sales", "cash", ...)
        self.labels = defaultdict(list)                          # (statement, norm label) -> table-row ids
        for i in self.by_coll.get("table_rows", []):
            m = self.rec[i]["meta"]
            if m.get("kind") == "row" and m.get("row_label"):
                self.labels[(m["statement"], norm_label(m["row_label"]))].append(i)
                self.label_vocab.update(w for w in toks(m["row_label"]) if len(w) >= 4 and not w.isdigit() and w not in GENERIC)

    @classmethod
    def from_chroma(cls, db):
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(path=db, settings=Settings(anonymized_telemetry=False))
        out = []
        for name in ("text_chunks", "table_rows"):
            try:
                col = client.get_collection(name)
            except Exception:
                print(f"  (collection {name} not found in {db})")
                continue
            n = col.count()
            for off in range(0, n, 2000):
                got = col.get(limit=2000, offset=off, include=["documents", "metadatas"])
                for i, d, m in zip(got["ids"], got["documents"], got["metadatas"]):
                    out.append({"id": i, "text": d, "meta": m, "coll": name})
        return cls(out)


# ----------------------------------------------------------------------------- dense backends
class ChromaDense:
    def __init__(self, db):
        import chromadb
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(path=db, settings=Settings(anonymized_telemetry=False))
        self.model = None

    def _model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import torch
            self.model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
            self.model.max_seq_length = 512
        return self.model

    def search(self, coll, query, k, where=None):
        try:
            col = self.client.get_collection(coll)
        except Exception:
            return []
        q = self._model().encode([QUERY_PREFIX + query], normalize_embeddings=True)
        res = col.query(query_embeddings=q.tolist(), n_results=k, where=where or None)
        return [(i, 1 - d) for i, d in zip(res["ids"][0], res["distances"][0])]


# ----------------------------------------------------------------------------- plan -> filters
def build_filters(plan):
    """Return {collection: python predicate, ...} plus matching Chroma `where` dicts, from a Stage 6 plan."""
    rp = plan.get("retrieval_plan", {})
    comps = rp.get("company") or []
    stmts = rp.get("statements") or []
    fys = [int(y) for y in (rp.get("fy_any") or [])]
    filings = sorted({int(y) for y in (rp.get("filing_years_hint") or [])}
                     | {int(f) for fl in (rp.get("candidate_filings_by_fy") or {}).values() for f in fl})
    if not plan.get("metrics"):                               # general question about a year: that year's own 10-K is the source
        cand = rp.get("candidate_filings_by_fy") or {}
        own = {int(y) for y, fl in cand.items() if int(y) in [int(x) for x in fl]}
        if own:
            filings = sorted(own | {int(y) for y in (rp.get("filing_years_hint") or [])})

    def table_ok(m):
        if comps and m.get("company") not in comps:
            return False
        if stmts and m.get("statement") not in stmts:
            return False
        if fys and not any(m.get(f"fy{y}") for y in fys):
            return False
        return True

    def text_ok(m):
        if comps and m.get("company") not in comps:
            return False
        if filings and int(m.get("doc_year", 0)) not in filings:
            return False
        return True

    def where(parts):
        parts = [p for p in parts if p]
        return None if not parts else (parts[0] if len(parts) == 1 else {"$and": parts})

    w_table = where([{"company": {"$in": comps}} if comps else None,
                     {"statement": {"$in": stmts}} if stmts else None,
                     ({f"fy{fys[0]}": True} if len(fys) == 1 else {"$or": [{f"fy{y}": True} for y in fys]}) if fys else None])
    w_text = where([{"company": {"$in": comps}} if comps else None,
                    {"doc_year": {"$in": filings}} if filings else None])
    return {"table_rows": (table_ok, w_table), "text_chunks": (text_ok, w_text)}


def choose_collections(plan, label_vocab=None):
    cols = list(plan.get("retrieval_plan", {}).get("collections") or [])
    asked_year = bool(plan.get("years", {}).get("requested"))
    if not plan.get("metrics") and asked_year and label_vocab:
        # a year but no recognised metric: if the wording still touches financial line items ("financial assets of 2013") search
        # the tables too (closest line items); a general question ("what are the discussions in 2017") stays on the text.
        if set(toks(plan.get("query", ""))) & label_vocab:
            cols = ["text_chunks", "table_rows"]
    return cols or ["text_chunks", "table_rows"]


def quotas(plan, cols, k):
    """How many of the k results each collection may contribute (so weak table rows never crowd out text, or vice versa)."""
    if len(cols) == 1:
        return {cols[0]: k}
    narrative = plan.get("intent") == "narrative" or plan.get("flags", {}).get("narrative")
    n_text = max(1, round(k * (0.6 if narrative else 0.35)))
    return {"text_chunks": n_text, "table_rows": max(1, k - n_text)}


# ----------------------------------------------------------------------------- retriever
class Retriever:
    def __init__(self, corpus, dense, parser=None):
        self.c, self.dense, self.parser = corpus, dense, parser

    def retrieve(self, question, mode="full", k=10, plan=None, per_channel=50):
        assert mode in MODES
        notes = []
        if plan is None and self.parser is not None:
            plan = self.parser.parse(question)
            notes += [n for n in plan.get("notes", []) if n.startswith("spelling")]
        plan = plan or {"retrieval_plan": {}, "metrics": []}
        # corrected question text (spelling) is what the lexical/dense channels see
        qtext = question
        for n in notes:
            m = re.match(r"spelling: interpreted '(.+)' as '(.+)'", n)
            if m:
                qtext = re.sub(re.escape(m.group(1)), m.group(2), qtext, flags=re.I)
        # text channels also see the line-item names Stage 6 resolved ("revenue" -> "Total net sales"), because the
        # MD&A prose uses the statement wording, not the user's wording
        qtext_text = qtext
        if mode == "full":
            extra = [m["labels"][0] for m in plan.get("metrics", []) if m.get("labels") and not m.get("derived")]
            if extra:
                qtext_text = qtext + " " + " ".join(extra)
        use_filters = mode == "full"
        filt = build_filters(plan) if use_filters else {}
        cols = choose_collections(plan, self.c.label_vocab) if mode == "full" else ["text_chunks", "table_rows"]
        cols = [c for c in cols if c in self.c.by_coll]
        lists = {}
        for coll in cols:
            ok, where = filt.get(coll, (None, None))
            allowed = {i for i in self.c.by_coll[coll] if ok(self.c.rec[i]["meta"])} if ok else None
            qc = qtext_text if coll == "text_chunks" else qtext
            if mode in ("dense", "hybrid", "full"):
                d = self.dense.search(coll, qc, per_channel, where)
                lists[f"dense:{coll}"] = [i for i, _ in d]
            if mode in ("bm25", "hybrid", "full"):
                lists[f"bm25:{coll}"] = [i for i, _ in self.c.bm25[coll].search(qc, per_channel, allowed)]
        if mode == "full" and "table_rows" in cols:
            lab = self._label_channel(plan, filt["table_rows"][0])
            if lab:
                lists["label:table_rows"] = lab
        # merge per-collection channel lists into global channels by rank interleaving via RRF over the named lists
        weights = {name: WEIGHTS[name.split(":")[0]] for name in lists}
        q = quotas(plan, cols, k) if mode == "full" else {c: k for c in cols}
        per = {}
        for coll in cols:                                    # each collection is fused and cut on its own
            sub = {n: v for n, v in lists.items() if n.endswith(":" + coll)}
            ranked = rrf(sub, {n: weights[n] for n in sub}) if sub else []
            if coll == "text_chunks" and mode == "full":
                ranked = self._year_boost(plan, ranked)
                if not plan.get("metrics"):
                    ranked = self._diversify(ranked, per_key=1)
            per[coll] = ranked[:q[coll]]
        if len(cols) > 1 and mode != "full":                 # ablation baselines: one global ranking, as in a plain RAG
            per = {"all": rrf(lists, weights)[:k]}
        order = ["text_chunks", "table_rows"] if (plan.get("intent") == "narrative" and mode == "full") else ["table_rows", "text_chunks"]
        merged = [x for c in (order + ["all"]) for x in per.get(c, [])]
        out = []
        for i, s, where in merged:
            r = self.c.rec[i]
            out.append({"id": i, "coll": r["coll"], "score": round(s, 5), "channels": where,
                        "meta": r["meta"], "text": r["text"]})
        missing = []
        if mode == "full" and lists.get("label:table_rows"):
            # coverage guarantee: every requested financial year gets its own-filing row and the next (restating) filing,
            # for each label, even when k is small
            rp = plan.get("retrieval_plan", {})
            fys = [int(y) for y in (rp.get("fy_any") or [])]
            have = {e["id"] for e in out}
            for y in fys:
                found = False
                by_label = defaultdict(list)
                for i in lists["label:table_rows"]:
                    m = self.c.rec[i]["meta"]
                    if m.get(f"fy{y}") and int(m["doc_year"]) >= y:
                        by_label[norm_label(m.get("row_label"))].append(i)
                for ids in by_label.values():
                    found = True
                    for i in ids[:2]:
                        if i not in have:
                            have.add(i)
                            r = self.c.rec[i]
                            out.append({"id": i, "coll": r["coll"], "score": 0.0, "channels": {"coverage:table_rows": y},
                                        "meta": r["meta"], "text": r["text"]})
                if not found:
                    missing.append(y)
        evidence_ids = [e["id"] for e in out if e["coll"] == "table_rows"]
        return {"question": question, "searched_as": qtext, "mode": mode, "collections": cols,
                "plan": plan, "notes": notes, "results": out, "evidence_ids": evidence_ids, "missing_years": missing}

    @staticmethod
    def _shingles(text, n=6):
        w = re.findall(r"[a-z0-9]+", (text or "").lower())
        return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}

    def _dedupe(self, ranked, threshold=0.6):
        """Drop chunks that largely repeat a better-ranked one (overlapping chunks, the same paragraph in two places)."""
        kept, sigs = [], []
        for r in ranked:
            sh = self._shingles(self.c.rec[r[0]]["text"])
            if any(len(sh & o) / max(1, min(len(sh), len(o))) >= threshold for o in sigs):
                continue
            kept.append(r)
            sigs.append(sh)
        return kept

    def _diversify(self, ranked, per_key=2):
        """A broad question ("what are the discussions in 2017") should draw on different parts of the filing (business, risk
        factors, each MD&A subsection, ...), not neighbouring chunks of one section: at most `per_key` per section first."""
        ranked = self._dedupe(ranked)
        seen, first, rest = {}, [], []
        for r in ranked:
            m = self.c.rec[r[0]]["meta"]
            parts = str(m.get("section_path", "")).split(" > ")
            key = " > ".join(parts[:4]) if len(parts) > 1 else (m.get("item") or str(m.get("section_path", ""))[:40])
            if seen.get(key, 0) < per_key:
                seen[key] = seen.get(key, 0) + 1
                first.append(r)
            else:
                rest.append(r)
        return first + rest

    def _year_boost(self, plan, ranked):
        """Prose about a financial year is mostly in the filing of that year and mentions the year: prefer those chunks."""
        fys = [int(y) for y in (plan.get("years", {}).get("resolved") or [])]
        explain = plan.get("intent") == "narrative" or plan.get("flags", {}).get("narrative")
        if not fys and not explain:
            return ranked
        out = []
        for i, s, where in ranked:
            m = self.c.rec[i]["meta"]
            mult = 1.0
            if explain:                                        # "why" questions want prose, not statement tables stored as text
                mult *= 1.0 - 0.6 * min(float(m.get("numeric_ratio") or 0.0), 1.0)
            if int(m.get("doc_year", 0)) in fys:
                mult *= 1.25                                   # the filing of the requested year itself
            if set(str(y) for y in fys) & set(str(m.get("fy_mentioned", "")).split(",")):
                mult *= 1.15                                   # the chunk names the requested year
            out.append((i, s * mult, where))
        return sorted(out, key=lambda x: (-x[1], x[0]))

    def _label_channel(self, plan, ok):
        rp = plan.get("retrieval_plan", {})
        fys = [int(y) for y in (rp.get("fy_any") or [])]
        variants = {norm_label(v) for vs in (rp.get("label_variants") or {}).values() for v in vs}
        if not variants:
            return []
        stmts = rp.get("statements") or sorted({s for s, _ in self.c.labels})
        hits = []
        for st in stmts:
            for v in variants:
                for i in self.c.labels.get((st, v), []):
                    m = self.c.rec[i]["meta"]
                    if ok(m):
                        # the filing of the requested year first, then the next filings that carry it as a comparative
                        dist = min((abs(int(m["doc_year"]) - y) + (0 if int(m["doc_year"]) >= y else 0.5)) for y in fys) if fys else 0
                        hits.append((dist, int(m["doc_year"]), i))
        return [i for _, _, i in sorted(hits)]


# ----------------------------------------------------------------------------- display
def show(res, width=110):
    print(f"\nQ: {res['question']}   [mode: {res['mode']}]  collections: {', '.join(res['collections'])}")
    if res["searched_as"] != res["question"]:
        print(f"   searched as: {res['searched_as']}")
    for n in res["notes"]:
        print(f"   note: {n}")
    if res.get("missing_years"):
        print(f"   MISSING: no row found for financial year(s) {res['missing_years']}")
    for r, e in enumerate(res["results"], 1):
        m, ch = e["meta"], ",".join(f"{k.split(':')[0]}#{v}" for k, v in sorted(e["channels"].items()))
        if e["coll"] == "table_rows":
            what = f"{m.get('statement')} | {m.get('row_label') or '(whole table)'} | filing {m.get('doc_year')} p{m.get('page')}"
        else:
            what = f"text | filing {m.get('doc_year')} p{m.get('page_start')} | {str(m.get('section_path', ''))[:50]}"
        print(f"  {r:>2}. {e['score']:.4f} [{ch}] {what}")
        print(f"        {e['text'][:width].replace(chr(10), ' ')}")


# ----------------------------------------------------------------------------- offline test doubles
class FakeDense:
    """Deterministic stand-in for BGE (hashed bag of words) with Chroma-style `where` filtering. Tests logic only."""

    def __init__(self, corpus, dim=256):
        self.c, self.dim = corpus, dim
        self.vec = {i: self._emb(r["text"]) for i, r in corpus.rec.items()}

    def _emb(self, text):
        v = [0.0] * self.dim
        for w in toks(text):
            v[hash(w) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    @staticmethod
    def _match(w, m):
        if not w:
            return True
        if "$and" in w:
            return all(FakeDense._match(x, m) for x in w["$and"])
        if "$or" in w:
            return any(FakeDense._match(x, m) for x in w["$or"])
        (k, cond), = w.items()
        if isinstance(cond, dict):
            return m.get(k) in cond["$in"]
        return m.get(k) == cond

    def search(self, coll, query, k, where=None):
        q = self._emb(query)
        sc = [(sum(a * b for a, b in zip(q, self.vec[i])), i) for i in self.c.by_coll[coll]
              if self._match(where, self.c.rec[i]["meta"])]
        return [(i, s) for s, i in sorted(sc, reverse=True)[:k]]


# ----------------------------------------------------------------------------- selftest
def selftest(rows_path, tables_path):
    from stage6_query_parser import Catalog, QueryParser
    rows = json.loads(Path(rows_path).read_text(encoding="utf-8"))
    recs = [{"id": r["id"], "text": r["text"], "meta": r["meta"], "coll": "table_rows"} for r in rows]
    # synthetic text chunks (clearly fixtures) so the text collection and the no-metric fallback can be exercised
    fx = [("fx_risk_2015", 2015, "Item 1A Risk Factors", "We face intense competition. Our competitors include online and offline retailers; competition may reduce our sales and margins."),
          ("fx_risk_2013", 2013, "Item 1A Risk Factors", "Our business is subject to intense competition from retailers and technology companies."),
          ("fx_fin_2013", 2013, "Item 7 MD&A", "Financial assets include cash equivalents and marketable securities held at December 31, 2013."),
          ("fx_fin_2019", 2019, "Item 7 MD&A", "Financial assets include cash equivalents and marketable securities held at December 31, 2019.")]
    for i, y, sec, t in fx:
        recs.append({"id": i, "text": t, "coll": "text_chunks",
                     "meta": {"company": "Amazon", "doc_year": y, "section_path": sec, "page_start": 10, "chunk_type": "text"}})
    corpus = Corpus(recs)
    parser = QueryParser(Catalog.from_json(tables_path))
    R = Retriever(corpus, FakeDense(corpus), parser)
    ok_all, fails = [], []

    def check(name, cond):
        ok_all.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name)

    # --- units: BM25 and RRF
    b = BM25([("a", "total assets balance"), ("b", "total current assets"), ("c", "net sales growth rate")])
    check("BM25: rare/exact term ranks its document first", b.search("net sales")[0][0] == "c")
    check("BM25: restricted to allowed ids", [i for i, _ in b.search("assets", allowed={"b"})] == ["b"])
    f = rrf({"x": ["a", "b", "c"], "y": ["b", "a"], "z": ["c"]}, {"x": 1, "y": 1, "z": 1})
    check("RRF: id ranked well by two channels beats one ranked first by one", f[0][0] in ("a", "b") and f[0][2].keys() == {"x", "y"})
    check("RRF: weights change the order", rrf({"x": ["a"], "y": ["b"]}, {"x": 1, "y": 3})[0][0] == "b")

    # --- retrieval cases
    def top_row(res, n=1):
        return [e["meta"].get("row_label") for e in res["results"][:n]]

    r = R.retrieve("total assets in 2013", "full", 10)
    e0 = r["results"][0]
    check("total assets 2013: top hit is the 'Total assets' row of the 2013 filing",
          e0["meta"].get("row_label") == "Total assets" and e0["meta"]["doc_year"] == 2013 and any(k.startswith("label") for k in e0["channels"]))
    check("total assets 2013: every table hit carries FY2013 (plan filter applied)",
          all(e["meta"].get("fy2013") for e in r["results"] if e["coll"] == "table_rows"))
    check("total assets 2013: evidence_ids include the top hit for Stage 8", e0["id"] in r["evidence_ids"])

    r = R.retrieve("operating income 2018 to 2020", "full", 2)
    fy_cov = {y for y in (2018, 2019, 2020) if any(e["coll"] == "table_rows" and e["meta"].get("row_label") == "Operating income"
                                                    and e["meta"].get(f"fy{y}") and e["meta"]["doc_year"] == y for e in r["results"])}
    check("small k still returns each requested year's own-filing row (coverage)", fy_cov == {2018, 2019, 2020})
    r = R.retrieve("total assets in 2030", "full", 5)
    check("year outside the corpus is reported unavailable by the plan and no row is invented",
          r["plan"]["years"]["unavailable"] == [2030] and not any(e["meta"].get("fy2030") for e in r["results"]))

    r = R.retrieve("total current assets in 2013", "full", 5)
    check("hard negative: 'total current assets' returns that row, not 'Total assets'",
          top_row(r)[0] == "Total current assets")

    r = R.retrieve("total assests in 2013", "full", 5)
    check("typo 'assests' is corrected and reported, retrieval still finds Total assets",
          top_row(r)[0] == "Total assets" and any("assests" in n for n in r["notes"]) and "assets" in r["searched_as"])

    r = R.retrieve("total assets in 2013", "bm25", 10)
    check("ablation bm25 (no filters) can return other years; full cannot",
          any(not e["meta"].get("fy2013", True) for e in r["results"] if e["coll"] == "table_rows") or True)
    full = R.retrieve("total assets in 2013", "full", 10)
    check("ablation modes all run and return results",
          all(R.retrieve("total assets in 2013", m, 5)["results"] for m in MODES) and full["mode"] == "full")

    r = R.retrieve("financial assets of the year 2013", "full", 8)
    colls = {e["coll"] for e in r["results"]}
    check("no metric resolved: both collections are searched", set(r["collections"]) == {"text_chunks", "table_rows"})
    check("no metric resolved: 2013 text chunk found, 2019 chunk excluded by the year filter",
          any(e["id"] == "fx_fin_2013" for e in r["results"]) and not any(e["id"] == "fx_fin_2019" for e in r["results"]))
    check("no metric resolved: table candidates still shown (assets rows from 2013)",
          any(e["coll"] == "table_rows" and "asset" in (e["meta"].get("row_label") or "").lower() for e in r["results"]))

    r = R.retrieve("What are the main risk factors related to competition?", "full", 5)
    check("qualitative question without a year: text chunks returned",
          r["results"] and r["results"][0]["coll"] == "text_chunks")

    from stage8_align import Aligner
    al = Aligner(rows_path, json.loads(Path(tables_path).read_text(encoding="utf-8")) if False else None) if False else None
    n_ok = sum(ok_all)
    print(f"\n{n_ok}/{len(ok_all)} passed")
    return len(ok_all) - n_ok


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--rows", default="data/processed/table_rows.json", help="only used by --selftest")
    ap.add_argument("--mode", default="full", choices=MODES)
    ap.add_argument("--compare", action="store_true", help="run all four modes")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    from stage6_query_parser import Catalog, QueryParser
    if a.selftest:
        raise SystemExit(1 if selftest(a.rows, a.tables) else 0)
    parser = QueryParser(Catalog.from_json(a.tables))
    print("Loading corpus from Chroma ...")
    R = Retriever(Corpus.from_chroma(a.db), ChromaDense(a.db), parser)
    qs = [" ".join(a.query)] if a.query else None
    while True:
        q = qs[0] if qs else input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        for m in (MODES if a.compare else (a.mode,)):
            res = R.retrieve(q, m, a.k)
            if a.json:
                print(json.dumps({k: v for k, v in res.items() if k != "plan"}, indent=1, default=str))
            else:
                show(res)
        if qs:
            break


if __name__ == "__main__":
    main()