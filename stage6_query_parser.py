"""
Stage 6: deterministic query understanding (no LLM).

Turns a natural-language question into a structured plan that the later stages (hybrid retrieval, year/column
alignment, Python calculation, answer formatting) can act on:

    company, metrics (with the statement and label variants found in the corpus), FINANCIAL years
    (explicit / list / range / before / after / since / consecutive / relative / decade / all), operation
    (difference, pct_change, yoy, cagr, ratio, sum, average, max, min, rank), intent flags, output type,
    and a retrieval plan (which collections, statements and year filters to use).

Company-agnostic and metadata-driven:
  * the available companies, financial years and line-item labels come from the Stage 4 tables
    (data/processed/all_tables.json), so nothing about Amazon is hard-coded;
  * the curated concept list contains only generic financial vocabulary (revenue, operating income, total
    assets, ...), mapped to the labels that really exist in the corpus;
  * "last 5 years", "before 2020", ... are resolved against the financial years actually available.

Usage:
    python stage6_query_parser.py "Compare net sales for 2020 and 2025"
    python stage6_query_parser.py --selftest
    python stage6_query_parser.py            # interactive prompt
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from canon import canonical     # shared with Stage 5: identical labels in the index and in the catalog
from typo import build_vocab, correct as correct_typos

# ----------------------------------------------------------------------------- generic vocabulary
# Curated concepts: generic financial terms only. `forms` are surface phrases a user may type; `label_re`
# is matched against NORMALISED line-item labels found in the corpus (lower-case, punctuation removed).
CONCEPTS = {
    "revenue": dict(stmt="operations", forms=["total net sales", "net sales", "total revenues", "total revenue", "net revenues",
                    "net revenue", "revenues", "revenue", "sales", "turnover", "top line"],
                    label_re=r"^(total )?(net )?(sales|revenues?)$"),
    "cost_of_sales": dict(stmt="operations", forms=["cost of sales", "cost of goods sold", "cost of revenue", "cost of revenues", "cogs"],
                          label_re=r"^cost of (sales|goods sold|revenues?)$"),
    "gross_profit": dict(stmt="operations", forms=["gross profit", "gross income"], label_re=r"^gross (profit|income)$",
                         fallback=dict(formula="revenue - cost_of_sales", components=["revenue", "cost_of_sales"])),
    "operating_expenses": dict(stmt="operations", forms=["total operating expenses", "operating expenses", "opex"],
                               label_re=r"^(total )?operating expenses$"),
    "operating_income": dict(stmt="operations", forms=["operating income loss", "operating income", "income from operations",
                             "operating profit", "operating earnings", "operating loss", "loss from operations"],
                             label_re=r"^(operating (income|profit|earnings|loss)( loss)?|(income|loss)( loss)? from operations)$"),
    "net_income": dict(stmt="operations", forms=["net income loss", "net income", "net earnings", "net profit", "net loss",
                       "profit after tax", "bottom line"], label_re=r"^net (income|earnings|loss|profit)( loss)?$"),
    "pretax_income": dict(stmt="operations", forms=["income before income taxes", "pretax income", "pre tax income", "profit before tax",
                          "earnings before income taxes", "income before taxes"],
                          label_re=r"^(income|earnings|loss)( loss)? before (provision for )?income taxes"),
    "income_tax": dict(stmt="operations", forms=["provision for income taxes", "income tax expense", "income taxes", "tax expense",
                       "income tax provision"], label_re=r"^(provision|benefit)( benefit)? for income taxes|^income tax(es)? (expense|provision)"),
    "interest_expense": dict(stmt="operations", forms=["interest expense"], label_re=r"^interest expense"),
    "eps": dict(stmt="operations", forms=["diluted earnings per share", "basic earnings per share", "earnings per share",
                "net income per share", "loss per share", "diluted eps", "basic eps", "eps"], label_re=r"per share"),
    "total_assets": dict(stmt="balance_sheet", forms=["total assets"], label_re=r"^total assets$"),
    "current_assets": dict(stmt="balance_sheet", forms=["total current assets", "current assets"], label_re=r"^total current assets$"),
    "current_liabilities": dict(stmt="balance_sheet", forms=["total current liabilities", "current liabilities"],
                                label_re=r"^total current liabilities$"),
    "total_liabilities": dict(stmt="balance_sheet", forms=["total liabilities"], label_re=r"^total liabilities$"),
    "equity": dict(stmt="balance_sheet", forms=["total stockholders equity", "total shareholders equity", "stockholders equity",
                   "shareholders equity", "total equity"], label_re=r"^total (stockholders|shareholders|shareowners|owners) (equity|deficit)( (equity|deficit))?$|^total (equity|deficit)$"),
    "cash": dict(stmt="balance_sheet", forms=["cash and cash equivalents", "cash equivalents"], label_re=r"^cash( and cash equivalents)?$"),
    "inventory": dict(stmt="balance_sheet", forms=["inventories", "inventory"], label_re=r"^inventor(y|ies)$"),
    "accounts_receivable": dict(stmt="balance_sheet", forms=["accounts receivable"], label_re=r"^accounts receivable"),
    "accounts_payable": dict(stmt="balance_sheet", forms=["accounts payable"], label_re=r"^accounts payable$"),
    "long_term_debt": dict(stmt="balance_sheet", forms=["long term debt", "long-term debt"], label_re=r"^long term debt"),
    "goodwill": dict(stmt="balance_sheet", forms=["goodwill"], label_re=r"^goodwill$"),
    "ppe": dict(stmt="balance_sheet", forms=["property and equipment", "fixed assets", "property plant and equipment"],
                label_re=r"^(property and equipment|fixed assets|property plant and equipment)( net)?$"),
    "ocf": dict(stmt="cash_flows", forms=["net cash provided by operating activities", "cash provided by operating activities",
                "cash flow from operations", "cash flows from operating activities", "operating cash flows", "operating cash flow",
                "cash from operations"], label_re=r"^net cash .*operating activities"),
    "icf": dict(stmt="cash_flows", forms=["net cash used in investing activities", "investing cash flow", "investing cash flows",
                "cash used in investing"], label_re=r"^net cash .*investing activities"),
    "fincf": dict(stmt="cash_flows", forms=["net cash provided by financing activities", "financing cash flow", "financing cash flows",
                  "cash from financing"], label_re=r"^net cash .*financing activities"),
    "capex": dict(stmt="cash_flows", forms=["purchases of property and equipment", "purchases of fixed assets", "capital expenditures",
                  "capital expenditure", "capex"], label_re=r"^purchases of (property and equipment|fixed assets)|^capital expenditures?"),
    "depreciation": dict(stmt="cash_flows", forms=["depreciation and amortization", "depreciation"], label_re=r"^depreciation"),
    "stock_comp": dict(stmt="cash_flows", forms=["stock based compensation", "share based compensation"], label_re=r"stock based compensation"),
    "rnd": dict(stmt="operations", forms=["research and development", "r and d"], label_re=r"research and development"),
    "sga": dict(stmt="operations", forms=["selling general and administrative", "sg and a"], label_re=r"^selling general and administrative"),
    "marketing": dict(stmt="operations", forms=["sales and marketing", "marketing expense", "marketing"], label_re=r"^(sales and )?marketing"),
    "ga": dict(stmt="operations", forms=["general and administrative"], label_re=r"^general and administrative$"),
}
# Derived metrics are always computed by Python later from base concepts (never read from a table).
DERIVED = {
    "gross_margin": dict(forms=["gross profit margin", "gross margin"], formula="(revenue - cost_of_sales) / revenue * 100",
                         components=["revenue", "cost_of_sales"], unit="percent"),
    "operating_margin": dict(forms=["operating income margin", "operating profit margin", "operating margin"],
                             formula="operating_income / revenue * 100", components=["operating_income", "revenue"], unit="percent"),
    "net_margin": dict(forms=["net profit margin", "net income margin", "net margin", "profit margin"],
                       formula="net_income / revenue * 100", components=["net_income", "revenue"], unit="percent"),
    "free_cash_flow": dict(forms=["free cash flow", "fcf"], formula="ocf - abs(capex)", components=["ocf", "capex"], unit="currency",
                           note="free cash flow = operating cash flow - capital expenditures; capex is printed as a negative outflow, "
                                "so its magnitude is used (definition varies by company)"),
    "current_ratio": dict(forms=["current ratio"], formula="current_assets / current_liabilities",
                          components=["current_assets", "current_liabilities"], unit="ratio"),
    "roe": dict(forms=["return on equity", "roe"], formula="net_income / equity * 100", components=["net_income", "equity"], unit="percent"),
    "roa": dict(forms=["return on assets", "roa"], formula="net_income / total_assets * 100", components=["net_income", "total_assets"], unit="percent"),
    "effective_tax_rate": dict(forms=["effective tax rate"], formula="income_tax / pretax_income * 100",
                               components=["income_tax", "pretax_income"], unit="percent"),
}
STATEMENT_WORDS = {
    "operations": r"income statement|statements? of operations|statements? of income|statements? of earnings|profit and loss|\bp ?and ?l\b",
    "balance_sheet": r"balance sheets?|statements? of financial position",
    "cash_flows": r"cash flow statements?|statements? of cash flows?",
}
NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
STOP_SINGLE = {"basic", "diluted", "other", "total", "revenue", "revenues", "income", "expense", "expenses", "equity", "assets",
               "liabilities", "interest", "taxes", "shares", "sales", "deposits", "common", "preferred", "current", "dividends"}
YR = r"(19[89]\d|20[0-4]\d)"
NUM = r"(\d{1,2}|" + "|".join(NUMW) + r")"
FY_OPT = r"(?:fiscal\s+|financial\s+|full\s+|calendar\s+)?"


def to_int(tok: str) -> int:
    return int(tok) if tok.isdigit() else NUMW[tok]


def tokens(s: str):
    s = s.lower().replace("&", " and ").replace("’", "'").replace("'", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()


def norm_label(label: str) -> str:
    return " ".join(tokens(re.sub(r"(?:\s*[.Ï_]){3,}", " ", label)))


# ----------------------------------------------------------------------------- catalog from Stage 4 tables
class Catalog:
    """Companies, available FINANCIAL years and line-item labels, learned from the validated tables."""

    def __init__(self, tables):
        self.companies = sorted({t["company"] for t in tables})
        self.fy = defaultdict(set)                 # company -> financial years with data
        self.fy_sources = defaultdict(lambda: defaultdict(set))   # company -> fy -> filing years containing it
        self.filings = defaultdict(set)
        self.labels = defaultdict(lambda: defaultdict(Counter))   # company -> statement -> norm label -> n tables
        disp = defaultdict(lambda: defaultdict(Counter))          # company -> norm label -> raw display forms
        self.display = defaultdict(dict)           # company -> norm label -> readable label (most common form)
        for t in tables:
            if any(i.startswith("FAIL") for i in t.get("issues", [])):
                continue                            # tables that failed validation are not indexed
            c = t["company"]
            self.filings[c].add(int(t["doc_year"]))
            for y in t["column_years"]:
                self.fy[c].add(int(y))
                self.fy_sources[c][int(y)].add(int(t["doc_year"]))
            seen = set()
            for r in t["rows"]:
                if r["kind"] == "heading" or not r["values"]:
                    continue
                label, _section, _how = canonical(r)      # same canonical label that Stage 5 indexes
                n = norm_label(label)
                if n and (t["statement"], n) not in seen:
                    seen.add((t["statement"], n))
                    self.labels[c][t["statement"]][n] += 1
                    disp[c][n][label] += 1
        for c in disp:
            for n, forms in disp[c].items():
                self.display[c][n] = forms.most_common(1)[0][0]
        # surface phrases for company-specific line items that no curated concept covers
        self.catalog_phrases = []                   # (token tuple, norm label, statement)
        for c in self.companies:
            best = {}
            for st, cnt in self.labels[c].items():
                for n, k in cnt.items():
                    single_ok = len(n.split()) == 1 and k >= 5 and len(n) >= 6 and n not in STOP_SINGLE
                    if k >= 3 and 2 <= len(n.split()) <= 8 or single_ok:
                        if n not in best or k > best[n][1]:
                            best[n] = (st, k)
            for n, (st, _) in best.items():
                self.catalog_phrases.append((tuple(n.split()), c, n, st))

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def years(self, company):
        return sorted(self.fy[company])

    def labels_for(self, company, concept):
        """Label variants (readable) in the corpus that realise a curated concept, most frequent first."""
        spec = CONCEPTS[concept]
        rx = re.compile(spec["label_re"])
        hits = Counter()
        for st, cnt in self.labels[company].items():
            for n, k in cnt.items():
                if rx.search(n):
                    hits[(st, n)] += k
        out, stmts = [], Counter()
        for (st, n), k in hits.most_common():
            out.append((st, self.display[company][n], k))
            stmts[st] += k
        return out


# ----------------------------------------------------------------------------- year parsing
def parse_years(q: str, avail: list, notes: list):
    """Return dict(requested, resolved, unavailable, mode, spec, flags). `avail` = sorted available financial years."""
    spec, req, flags = [], set(), {"needs_prior": False, "all": False}
    latest = avail[-1] if avail else None
    s = q

    def eat(pattern, fn):
        nonlocal s
        while True:
            m = re.search(pattern, s)
            if not m:
                return
            fn(m)
            s = s[:m.start()] + " " * (m.end() - m.start()) + s[m.end():]

    def add(kind, years, text):
        spec.append({"kind": kind, "text": text.strip(), "years": sorted(set(years))})
        req.update(years)

    # filing-year mentions ("the 2019 10-K"): the filing year is not a financial year, but is used as a hint
    filing_years = []

    def filing(m):
        y = int(next(g for g in m.groups() if g and g.isdigit()))
        filing_years.append(y)
        spec.append({"kind": "filing", "text": m.group(0).strip(), "years": [y]})
        req.add(y)
        notes.append(f"'{m.group(0).strip()}' refers to the {y} filing; the financial year is assumed to be {y} "
                     f"(as originally reported in that filing)")
    eat(rf"\b{YR}\s+(?:form\s+)?(?:10-?k|annual report|filing)\b", filing)
    eat(rf"\b(?:10-?k|annual reports?|filings?)\s+(?:for|of|from|in)\s+{FY_OPT}(?:year\s+)?{YR}\b", filing)

    # N consecutive years from / ending
    eat(rf"\b{NUM}\s+(?:consecutive\s+|successive\s+)?{FY_OPT}years?\s+(?:from|starting|beginning|commencing|since)"
        rf"(?:\s+(?:from|in|with|at))?\s+{YR}\b",
        lambda m: add("consecutive", range(int(m.group(2)), int(m.group(2)) + to_int(m.group(1))), m.group(0)))
    eat(rf"\b{NUM}\s+(?:consecutive\s+|successive\s+)?{FY_OPT}years?\s+(?:ending|ended|through|until|up to|up until|to)"
        rf"(?:\s+(?:in|at))?\s+{YR}\b",
        lambda m: add("consecutive", range(int(m.group(2)) - to_int(m.group(1)) + 1, int(m.group(2)) + 1), m.group(0)))
    eat(rf"\b(?:the\s+)?(?:next|following)\s+{NUM}\s+{FY_OPT}years?\s+(?:after|from|following)\s+{YR}\b",
        lambda m: add("consecutive", range(int(m.group(2)) + 1, int(m.group(2)) + 1 + to_int(m.group(1))), m.group(0)))
    eat(rf"\b(?:the\s+)?{NUM}\s+{FY_OPT}years?\s+(?:before|prior to|preceding)\s+{YR}\b",
        lambda m: add("consecutive", range(int(m.group(2)) - to_int(m.group(1)), int(m.group(2))), m.group(0)))

    # relative to the latest available financial year
    def rel(m):
        n = to_int(m.group(1))
        if not avail:
            return
        add("relative", avail[-n:], m.group(0))
        if len(avail[-n:]) < n:
            notes.append(f"only {len(avail)} financial years are available; asked for {n}")
    eat(rf"\b(?:last|latest|past|recent|trailing|final|most recent)\s+{NUM}\s+{FY_OPT}years?\b", rel)
    eat(rf"\b{NUM}\s+(?:most recent|latest|last)\s+{FY_OPT}years?\b", rel)
    eat(rf"\b{NUM}\s+(?:consecutive|successive)\s+{FY_OPT}years?\b",
        lambda m: (add("relative", avail[-to_int(m.group(1)):], m.group(0)),
                   notes.append("no anchor year given for the consecutive years; using the latest available years")))

    # "from 2017 and 2018", "from 2017, 2019 and 2021": a list of years, NOT an open-ended "from 2017 onwards"
    eat(rf"\bfrom\s+{FY_OPT}{YR}(?:\s*(?:,|and|&|vs\.?|versus|or)\s*{FY_OPT}{YR})+\b",
        lambda m: add("explicit", [int(y) for y in re.findall(r"(?:19|20)\d{2}", m.group(0))], m.group(0)))

    # ranges
    eat(rf"\bbetween\s+{FY_OPT}{YR}\s+and\s+{FY_OPT}{YR}\b",
        lambda m: add("range", range(min(int(m.group(1)), int(m.group(2))), max(int(m.group(1)), int(m.group(2))) + 1), m.group(0)))
    eat(rf"\b(?:from\s+)?{FY_OPT}{YR}\s*(?:to|through|thru|until|till|-)\s*{FY_OPT}{YR}\b",
        lambda m: add("range", range(min(int(m.group(1)), int(m.group(2))), max(int(m.group(1)), int(m.group(2))) + 1), m.group(0)))

    # decades
    eat(r"\b(?:the\s+)?(19[89]0|20[0-4]0)'?s\b", lambda m: add("decade", range(int(m.group(1)), int(m.group(1)) + 10), m.group(0)))

    # one-sided bounds: "before 2005", "after 2020", "since 2015", "up to 2010".
    # "after 2010 and before 2015" is an intersection; "before 2005 and after 2020" is a union.
    bounds = []

    def bound(side, limit_fn):
        def fn(m):
            bounds.append((side, limit_fn(int(m.group(1))), m.group(0).strip()))
        return fn
    eat(rf"\b(?:before|prior to|earlier than|preceding|ahead of)\s+{FY_OPT}{YR}\b", bound("upper", lambda y: y - 1))
    eat(rf"\b(?:after|later than|post)\s+{FY_OPT}{YR}\b", bound("lower", lambda y: y + 1))
    eat(rf"\b(?:until|up to|through|thru|till|up through|up until)\s+{FY_OPT}{YR}\b", bound("upper", lambda y: y))
    eat(rf"\b(?:since|from|starting(?:\s+(?:in|from))?|beginning(?:\s+in)?)\s+{FY_OPT}{YR}\b", bound("lower", lambda y: y))
    eat(rf"\b{FY_OPT}{YR}\s+(?:or earlier|and earlier|and before|or before|or prior)\b", bound("upper", lambda y: y))
    eat(rf"\b{FY_OPT}{YR}\s+(?:onwards?|and later|and after|or later|or after|and beyond|forward|and onward)\b", bound("lower", lambda y: y))
    lows = [b for b in bounds if b[0] == "lower"]
    ups = [b for b in bounds if b[0] == "upper"]
    if len(lows) == 1 and len(ups) == 1 and lows[0][1] <= ups[0][1]:
        add("bound", [a for a in avail if lows[0][1] <= a <= ups[0][1]], f"{lows[0][2]} and {ups[0][2]}")
    else:
        for side, limit, text in bounds:
            ys = [a for a in avail if (a >= limit if side == "lower" else a <= limit)]
            add("bound", ys, text)
            if not ys:
                notes.append(f"no available financial year satisfies '{text}'")

    # words
    if re.search(r"\b(?:previous|prior|preceding)\s+(?:fiscal\s+)?year\b|\byear before\b|\bthe year earlier\b", s):
        flags["needs_prior"] = True
        s = re.sub(r"\b(?:previous|prior|preceding)\s+(?:fiscal\s+)?year\b|\byear before\b|\bthe year earlier\b", " ", s)
    m = re.search(r"\b(?:latest|most recent|current|newest|last)\s+(?:fiscal\s+|financial\s+|reported\s+)?year\b", s)
    if m and latest:
        add("relative", [latest], m.group(0))
        if m.group(0).startswith("last"):
            notes.append(f"'last year' interpreted as the latest available financial year ({latest})")
        s = s[:m.start()] + " " + s[m.end():]
    if re.search(r"\b(?:all|every|each|any)\s+(?:available\s+|fiscal\s+|financial\s+)?years?\b|\bover the years\b|\bhistorical(?:ly)?\b|"
                 r"\bentire (?:period|history)\b|\bacross (?:all )?years\b|\bover time\b|\blong[- ]term trend\b", s):
        flags["all"] = True

    # remaining explicit years: "2020", "2020 and 2025", "2019, 2021"
    for m in re.finditer(rf"(?<![\d.$]){YR}(?!\d|,\d{{3}}(?!\d)|\.\d)", s):
        add("explicit", [int(m.group(1))], m.group(1))

    # previous year relative to the latest requested year (or the latest available)
    if flags["needs_prior"]:
        anchor = max(req) if req else latest
        if anchor:
            add("previous", [anchor - 1], "previous year")
            if not any(sp["kind"] != "previous" for sp in spec):
                add("relative", [anchor], "latest year")
                notes.append(f"'previous year' relative to the latest available year ({anchor})")
    if flags["all"] and not any(sp["kind"] in ("range", "explicit", "relative", "consecutive", "bound",
                                              "decade", "filing") for sp in spec):
        add("all", avail, "all available years")
    availset = set(avail)
    resolved = sorted(y for y in req if y in availset)
    unavailable = sorted(y for y in req if y not in availset)
    if unavailable:
        notes.append(f"financial years not available in the corpus: {unavailable} "
                     f"(available: {avail[0]}-{avail[-1]})" if avail else "no financial years available")
    kinds = {sp["kind"] for sp in spec if sp["kind"] != "previous"}
    mode = "none" if not spec else (next(iter(kinds)) if len(kinds) == 1 else "mixed")
    return {"mode": mode, "spec": spec, "requested": sorted(req), "resolved": resolved, "unavailable": unavailable,
            "filing_years": sorted(set(filing_years))}


# ----------------------------------------------------------------------------- the parser
OPS = [
    ("cagr", r"\bcagr\b|compound(?:ed)? annual growth|compound growth"),
    ("yoy", r"year[- ]over[- ]year|\byoy\b|year on year|annual growth|growth rate each year|growth each year"),
    ("pct_change", r"percent(?:age)?\s+(?:change|increase|decrease|growth|difference|drop|decline)|%\s+(?:change|increase|decrease|growth)|"
                   r"\bgrowth\b|\bgrew\b|\bgrow\b"),
    ("difference", r"\bdifference\b|how much (?:did|has|have|does|do)\b.*\b(?:change|increase|decrease|rise|fall|drop|grow)|"
                   r"\b(?:change|increase|decrease|rise|drop|decline) (?:in|of)\b|\bdelta\b|\bnet change\b|\bchanged?\b"),
    ("ratio", r"\bratio\b|as a (?:percent(?:age)?|share|fraction|proportion) of|\b(?:what|which)\s+(?:percent(?:age)?|share|portion|fraction|proportion)\s+of\b|\bper dollar of\b|\bdivided by\b|\brelative to\b"),
    ("sum", r"\bsum of\b|\bcombined\b|\bcumulative\b|\baggregate\b|\badded together\b|\baltogether\b|\bin total (?:over|across|for)\b"),
    ("average", r"\baverage\b|\bmean\b|\bon average\b"),
    ("max", r"\bhighest\b|\bmost\b(?! recent)|\blargest\b|\bbiggest\b|\bpeak\b|\bmaximum\b|\bgreatest\b|\bbest\b"),
    ("min", r"\blowest\b|\bleast\b|\bsmallest\b|\bminimum\b|\bworst\b"),
]
RANK_RE = r"\brank(?:ed|ing)?\b|\btop\s+(?:\d+|three|five|ten|two|four)\b|\bhighest to lowest\b|\bordered by\b|\bsort(?:ed)? by\b|\bbottom\s+\d+\b"
COMPARE_RE = r"\bcompare[sd]?\b|\bcomparison\b|\bvs\.?\b|\bversus\b|compared (?:to|with)|\bagainst\b|\bdifferen(?:ce|t) between\b"
TREND_RE = r"\btrend(?:s|ing)?\b|\bover time\b|\bevolv\w+|\bover the years\b|\bhistor(?:y|ical|ically)\b|\btrajectory\b|\bgrowth pattern\b"
NARRATIVE_RE = (r"\bwhy\b|\bexplain\b|\bdescribe\b|\bdiscuss\b|\brisks?\b|\bstrateg(?:y|ies)\b|\bcompetit\w+|\bwhat does\b|\bhow does\b|"
                r"\bmanagement\b|\bfactors?\b|\bdrivers?\b|\breasons?\b|\bbusiness model\b|\boutlook\b|\bsummar(?:y|ize|ise)\b|\bpolicy\b|"
                r"\blegal\b|\blitigation\b|\bemployees?\b|\bsegments? (?:are|do|does)\b|\bwhat (?:is|are) the main\b")
NUMERIC_RE = r"\bhow much\b|\bhow many\b|\bwhat (?:was|were|is|are)\b.*\b(?:total|amount|value|number)\b|\bamount\b|\bvalue of\b"
CHART_RE = r"\bchart\b|\bgraph\b|\bplot\b|\bvisuali[sz]e\b|\bdiagram\b"
QUARTER_RE = r"\bq[1-4]\b|\bquarter(?:ly|s)?\b|\bfirst quarter\b|\bsecond quarter\b|\bthird quarter\b|\bfourth quarter\b"
SEGMENT_RE = r"\bsegments?\b|\bby segment\b|\bgeograph\w+|\binternational\b|\bnorth america\b|\bby region\b|\bby product\b|\bby category\b"


class QueryParser:
    def __init__(self, catalog: Catalog):
        self.cat = catalog
        self.phrases = defaultdict(list)       # first token -> [(tokens, kind, key, priority)]
        for cid, spec in CONCEPTS.items():
            for f in spec["forms"]:
                t = tuple(tokens(f))
                self.phrases[t[0]].append((t, "concept", cid, 0))
        for did, spec in DERIVED.items():
            for f in spec["forms"]:
                t = tuple(tokens(f))
                self.phrases[t[0]].append((t, "derived", did, 0))
        for toks, comp, n, st in catalog.catalog_phrases:
            self.phrases[toks[0]].append((toks, "label", (comp, n, st), 1))
        for k in self.phrases:
            self.phrases[k].sort(key=lambda p: (-len(p[0]), p[3]))
        # spelling tolerance: vocabulary = known metric phrases, corpus line-item labels, company names, query grammar
        known = [" ".join(p[0]) for lst in self.phrases.values() for p in lst]
        known += [lab for c in catalog.display for lab in catalog.display[c].values()] + list(catalog.companies)
        known.append(Path(__file__).read_text(encoding="utf-8", errors="ignore"))     # grammar words used in this file
        self.vocab = build_vocab(known)

    # ---- companies
    def find_companies(self, q):
        found = []
        for c in self.cat.companies:
            name = c.lower()
            first = name.split()[0]
            if name in q or (len(first) >= 4 and re.search(rf"\b{re.escape(first)}\b", q)):
                found.append(c)
        return found

    # ---- metrics
    def find_metrics(self, q, company_list, warnings):
        toks = tokens(q)
        used, metrics, i = set(), [], 0
        comp = company_list[0] if company_list else None
        while i < len(toks):
            hit = None
            for t, kind, key, _ in self.phrases.get(toks[i], []):
                if tuple(toks[i:i + len(t)]) == t:
                    hit = (t, kind, key)
                    break
            if hit:
                t, kind, key = hit
                metrics.append(self._metric(" ".join(t), kind, key, comp, warnings))
                i += len(t)
            else:
                i += 1
        # de-duplicate (same concept mentioned twice)
        seen, out = set(), []
        for m in metrics:
            k = (m["concept"], m.get("primary_label"))
            if k not in seen:
                seen.add(k)
                out.append(m)
        return out

    def _base(self, concept, comp, surface, warnings, quiet=False):
        spec = CONCEPTS[concept]
        labels = self.cat.labels_for(comp, concept) if comp else []
        if concept == "eps":                       # per-share rows: keep the statement of the best hits
            labels = [x for x in labels if x[0] == "operations"]
        stmt = spec["stmt"]
        if labels:
            counts = Counter()
            for st, _, k in labels:
                counts[st] += k
            stmt = spec["stmt"] if spec["stmt"] in counts else counts.most_common(1)[0][0]
            labels = [x for x in labels if x[0] == stmt]
        out = {"concept": concept, "surface": surface, "statement": stmt, "derived": False, "formula": None, "components": [],
               "labels": [x[1] for x in labels[:12]], "primary_label": labels[0][1] if labels else None, "source": "curated",
               "value_type": "per_share" if concept == "eps" else "amount"}
        if not labels and "fallback" in spec:
            fb = spec["fallback"]
            out.update(derived=True, formula=fb["formula"], components=list(fb["components"]),
                       note=f"no '{concept}' line in the corpus; computed as {fb['formula']}")
        elif not labels and not quiet:
            warnings.append(f"no line item for '{surface}' ({concept}) was found in the corpus tables")
        return out

    def _metric(self, surface, kind, key, comp, warnings):
        if kind == "concept":
            return self._base(key, comp, surface, warnings)
        if kind == "derived":
            d = DERIVED[key]
            parts = [self._base(c, comp, c, warnings) for c in d["components"]]
            out = {"concept": key, "surface": surface, "derived": True, "formula": d["formula"], "components": d["components"],
                   "unit": d["unit"], "statement": None, "statements": sorted({p["statement"] for p in parts}),
                   "component_labels": {p["concept"]: p["labels"] for p in parts}, "labels": [], "primary_label": None,
                   "source": "curated", "value_type": "derived"}
            for p in parts:
                if p["derived"]:
                    out.setdefault("notes", []).append(p.get("note", ""))
            if d.get("note"):
                out.setdefault("notes", []).append(d["note"])
            if surface == "profit margin":
                out.setdefault("notes", []).append("'profit margin' interpreted as net margin")
            return out
        c, n, st = key                            # corpus-specific line item
        return {"concept": f"label:{n}", "surface": surface, "statement": st, "derived": False, "formula": None, "components": [],
                "labels": [self.cat.display[c][n]], "primary_label": self.cat.display[c][n], "source": "corpus", "value_type": "amount"}

    # ---- main entry
    def parse(self, text: str) -> dict:
        warnings, notes = [], []
        q = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
        q = re.sub(r"\b(?:fy|fiscal(?:\s+year)?|financial\s+year)\s*[-']?\s*(\d{4})\b", r"\1", q)
        q = re.sub(r"\bfy\s*'?(\d{2})\b", lambda m: "20" + m.group(1), q)
        q = re.sub(r"\s+", " ", q).strip()
        q, fixes = correct_typos(q, self.vocab)
        for wrong, right in fixes:
            notes.append(f"spelling: interpreted '{wrong}' as '{right}'")

        companies = self.find_companies(q)
        defaulted = False
        if not companies and len(self.cat.companies) == 1:
            companies, defaulted = list(self.cat.companies), True
        elif not companies:
            companies, defaulted = list(self.cat.companies), True
            warnings.append("no company named; searching all companies in the corpus")
        avail = sorted(set().union(*[self.cat.fy[c] for c in companies])) if companies else []
        if len(companies) > 1:
            notes.append("several companies: financial-year availability is the union across companies")

        yrs = parse_years(q, avail, notes)
        metrics = self.find_metrics(q, companies, warnings)

        # statements
        stmts = []
        for st, rx in STATEMENT_WORDS.items():
            if re.search(rx, q):
                stmts.append(st)
        for m in metrics:
            for st in ([m["statement"]] if m.get("statement") else m.get("statements", [])):
                if st and st not in stmts:
                    stmts.append(st)

        # operations
        ops = [name for name, rx in OPS if re.search(rx, q)]
        rank = bool(re.search(RANK_RE, q))
        if rank:
            ops = ["rank"] + [o for o in ops if o not in ("max", "min")] if not any(o in ops for o in ("max", "min")) else ["rank"] + ops
        if "pct_change" in ops and "difference" in ops and re.search(r"percent|%", q):
            ops.remove("difference")
        if "difference" in ops and "pct_change" not in ops and re.search(r"\bgrow(?:th|n)?\b|\bgrew\b", q):
            ops.append("pct_change")
        if "yoy" in ops and "pct_change" in ops:
            ops.remove("pct_change")
        if "ratio" in ops and any(m["derived"] and m["unit"] in ("percent", "ratio") for m in metrics if m.get("unit")):
            ops.remove("ratio")
        if "difference" in ops and len(yrs["resolved"]) < 2 and not yrs["spec"] and not re.search(r"\bdifference\b", q):
            ops.remove("difference")

        # ratio direction: "X as a percentage of Y", "ratio of X to Y" -> X/Y ; "what percentage of Y was X" -> X/Y
        ratio_spec = None
        if "ratio" in ops and len(metrics) >= 2:
            inverted = bool(re.search(r"\b(?:what|which)\s+(?:percent(?:age)?|share|portion|fraction|proportion)\s+of\b", q))
            num, den = (metrics[1], metrics[0]) if inverted else (metrics[0], metrics[1])
            ratio_spec = {"numerator": num["concept"], "denominator": den["concept"],
                          "unit": "percent" if re.search(r"percent|%|share|portion|fraction|proportion", q) else "ratio"}

        # support years for growth-type calculations
        support = []
        grow = [o for o in ops if o in ("yoy", "pct_change", "difference", "cagr")]
        avail_set = set(avail)
        if grow and yrs["resolved"]:
            lo = min(yrs["resolved"])
            if ("yoy" in grow or len(yrs["resolved"]) == 1) and (lo - 1) in avail_set and (lo - 1) not in yrs["resolved"]:
                support.append(lo - 1)
                notes.append(f"added {lo - 1} so the {'/'.join(grow)} for {lo} can be computed")
        if grow and not yrs["resolved"] and not yrs["spec"]:
            warnings.append("a growth/difference was requested but no years were given")

        # flags and intent
        narrative = bool(re.search(NARRATIVE_RE, q))
        numeric_q = bool(metrics) or bool(re.search(NUMERIC_RE, q)) or bool(grow)
        if re.search(SEGMENT_RE, q):
            warnings.append("segment / geographic breakdowns are in the notes, which Stage 4 does not extract yet; "
                            "only the three primary statements are indexed")
        if re.search(QUARTER_RE, q):
            warnings.append("quarterly data is not indexed; the corpus contains annual 10-K data only")
        n_years = len(yrs["resolved"])
        derived_any = any(m["derived"] for m in metrics)
        base_stmts = set()
        for m in metrics:
            if m["derived"]:
                for c in m["components"]:
                    base_stmts.add(CONCEPTS[c]["stmt"])
            elif m["statement"]:
                base_stmts.add(m["statement"])
        cross = len(base_stmts) >= 2
        compare = bool(re.search(COMPARE_RE, q)) or (n_years == 2 and len(metrics) <= 1 and not ops) or \
            (len(metrics) >= 2 and n_years <= 1)
        trend = bool(re.search(TREND_RE, q))
        rank_over = None
        if "rank" in ops or any(o in ops for o in ("max", "min")):
            rank_over = "years" if (metrics and not re.search(r"\b(?:expense|categor|line item|items|lines|components)\b", q)) else "rows"
            if rank_over == "years" and not yrs["spec"]:
                add_all = {"kind": "all", "text": "all available years", "years": avail}
                yrs["spec"].append(add_all)
                yrs["requested"], yrs["resolved"], yrs["mode"] = list(avail), list(avail), "all"
                n_years = len(avail)
                notes.append("no years given for a ranking: ranking over all available financial years")
        if trend and not yrs["spec"] and metrics:
            yrs["spec"].append({"kind": "all", "text": "all available years", "years": avail})
            yrs["requested"], yrs["resolved"], yrs["mode"] = list(avail), list(avail), "all"
            n_years = len(avail)
            notes.append("trend requested without years: using all available financial years")
        calc_ops = [o for o in ops if o not in ("rank", "max", "min")]
        calc = bool(calc_ops) or derived_any

        if narrative and not metrics and not grow:
            intent = "narrative"
        elif rank_over:
            intent = "ranking"
        elif cross and (calc or len(metrics) >= 2):
            intent = "cross_table"
        elif calc:
            intent = "calculation"
        elif trend:
            intent = "trend"
        elif n_years >= 3:
            intent = "multi_year"
        elif compare:
            intent = "comparison"
        else:
            intent = "fact"
        if narrative and metrics and re.search(r"\bwhy\b|\bexplain\b|\bdiscuss\b|\bdescribe\b|\bdrivers?\b|\breasons?\b|\bfactors?\b", q):
            intent = "narrative"                  # numbers + explanation: table evidence plus text evidence
        needs_text = narrative or intent == "narrative" or not metrics and not numeric_q
        needs_tables = bool(metrics) or (numeric_q and not narrative) or bool(stmts)

        # output type
        chart = bool(re.search(CHART_RE, q))
        if intent == "narrative":
            out_type = "table_plus_explanation" if needs_tables and metrics else "short_text"
        elif intent == "ranking":
            out_type = "ranking_table"
        elif calc:
            out_type = "table_plus_calc"
        elif intent in ("multi_year", "trend") or n_years >= 3:
            out_type = "multi_year_table"
        elif intent in ("comparison", "cross_table") or len(metrics) >= 2 or n_years >= 2:
            out_type = "comparison_table"
        else:
            out_type = "short_text"
        if chart and out_type == "short_text":
            out_type = "comparison_table"

        fetch = sorted(set(yrs["resolved"]) | set(support))
        if not metrics and needs_tables and not stmts:
            warnings.append("no financial metric recognised; the question will rely on text retrieval and table-title matching")
        if not yrs["spec"] and not (rank_over or trend):
            notes.append("no year given; all financial years will be searched (latest preferred)")
        company_key = companies[0] if len(companies) == 1 else None
        fy_sources = {y: sorted(self.cat.fy_sources[company_key][y]) for y in fetch} if company_key else {}
        plan = {
            "collections": (["table_rows"] if needs_tables else []) + (["text_chunks"] if needs_text or intent == "narrative" else []),
            "company": companies,
            "statements": stmts,
            "fy_any": fetch,                       # table_rows metadata flags fy<YEAR>: match any of these years
            "filing_years_hint": yrs["filing_years"],
            "candidate_filings_by_fy": fy_sources,  # which filings contain each financial year (restatement aware)
            "label_variants": {m["concept"]: (m["labels"] or m.get("component_labels")) for m in metrics},
        }
        return {
            "query": text, "normalized": q,
            "companies": companies, "company_defaulted": defaulted,
            "metrics": metrics, "statements": stmts,
            "years": {**yrs, "support_years": support, "fetch_years": fetch,
                      "available_range": [avail[0], avail[-1]] if avail else None},
            "operations": calc_ops + (["rank"] if rank_over else []),
            "rank_over": rank_over,
            "ratio": ratio_spec,
            "intent": intent,
            "flags": {"temporal": bool(yrs["spec"]), "multi_year": n_years >= 3, "comparison": compare, "trend": trend,
                      "cross_table": cross, "calculation": calc, "ranking": bool(rank_over), "narrative": narrative,
                      "needs_tables": needs_tables, "needs_text": needs_text, "chart": chart},
            "output_type": out_type,
            "retrieval_plan": plan,
            "notes": notes, "warnings": warnings,
        }


def summarize(p: dict) -> str:
    y = p["years"]
    lines = [f"Q: {p['query']}",
             f"   company : {', '.join(p['companies'])}{' (default)' if p['company_defaulted'] else ''}",
             "   metrics : " + (", ".join(f"{m['concept']}[{m['primary_label'] or m['formula']}]" for m in p["metrics"]) or "-"),
             f"   years   : {y['mode']} -> {y['resolved']}" + (f"  (+support {y['support_years']})" if y["support_years"] else "")
             + (f"  UNAVAILABLE {y['unavailable']}" if y["unavailable"] else ""),
             f"   ops     : {p['operations'] or '-'} | intent: {p['intent']} | output: {p['output_type']}"
             f" | statements: {p['statements'] or '-'} | collections: {p['retrieval_plan']['collections']}"]
    for n in p["notes"] + p["warnings"]:
        lines.append(f"   note    : {n}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- self-test
def selftest(parser: QueryParser) -> int:
    avail = parser.cat.years(parser.cat.companies[0])
    lo, hi = avail[0], avail[-1]
    last = lambda n: avail[-n:]
    T = [
        # (query, checks)  checks: dict of path -> expected value
        ("What was total net sales in 2022?", {"years": [2022], "intent": "fact", "output_type": "short_text",
                                                "concept": ["revenue"], "ops": []}),
        ("Compare net sales for 2020 and 2025", {"years": [2020, 2025], "intent": "comparison", "output_type": "comparison_table"}),
        ("What was net sales from 2015 to 2020?", {"years": list(range(2015, 2021)), "intent": "multi_year", "output_type": "multi_year_table"}),
        ("Show net sales between 2018 and 2021", {"years": [2018, 2019, 2020, 2021], "output_type": "multi_year_table"}),
        ("net sales 2015-2018", {"years": [2015, 2016, 2017, 2018]}),
        ("Net sales before 2000", {"years": [y for y in avail if y < 2000]}),
        ("Operating income after 2020", {"years": [y for y in avail if y > 2020]}),
        ("net sales since 2023", {"years": [2023, 2024, 2025]}),
        ("net income up to 2000", {"years": [y for y in avail if y <= 2000]}),
        ("What were net sales for 5 consecutive years from 2020?", {"years": [2020, 2021, 2022, 2023, 2024], "output_type": "multi_year_table"}),
        ("net sales for 3 consecutive years ending 2010", {"years": [2008, 2009, 2010]}),
        ("net sales over the last 5 years", {"years": last(5), "output_type": "multi_year_table"}),
        ("show latest 3 years of operating income", {"years": last(3)}),
        ("net sales in the most recent year", {"years": [hi]}),
        ("net sales in 2024 compared to the previous year", {"years": [2023, 2024], "output_type": "comparison_table"}),
        ("YoY growth in operating income for 2022", {"years": [2022], "support": [2021], "ops": ["yoy"], "output_type": "table_plus_calc",
                                                      "intent": "calculation"}),
        ("What is the CAGR of net sales from 2015 to 2020?", {"years": list(range(2015, 2021)), "ops": ["cagr"], "output_type": "table_plus_calc"}),
        ("percentage change in net income between 2019 and 2020", {"years": [2019, 2020], "ops": ["pct_change"]}),
        ("What was the operating margin in 2019 and 2021?", {"years": [2019, 2021], "concept": ["operating_margin"], "output_type": "table_plus_calc",
                                                              "derived": True}),
        ("free cash flow in 2018", {"years": [2018], "concept": ["free_cash_flow"], "derived": True, "output_type": "table_plus_calc"}),
        ("net income as a percentage of total assets for 2023", {"years": [2023], "intent": "cross_table", "stmts": ["operations", "balance_sheet"]}),
        ("Which year had the highest net sales?", {"intent": "ranking", "output_type": "ranking_table", "years": avail}),
        ("Why did operating income decrease in 2022?", {"intent": "narrative", "output_type": "table_plus_explanation", "years": [2022],
                                                         "collections": ["table_rows", "text_chunks"]}),
        ("What are the main risk factors related to competition?", {"intent": "narrative", "collections": ["text_chunks"], "concept": []}),
        ("operating income by segment in 2021", {"years": [2021], "warn": "segment"}),
        ("total assets at the end of 2015", {"years": [2015], "concept": ["total_assets"], "label": "Total assets", "stmts": ["balance_sheet"]}),
        ("revenue in 2030", {"years": [], "unavailable": [2030]}),
        ("net sales in the 2019 10-K", {"years": [2019], "filing": [2019]}),
        ("Q3 2021 net sales", {"warn": "quarterly"}),
        ("net sales in the 2010s", {"years": list(range(2010, 2020))}),
        ("Compare cash and cash equivalents and total assets between 2015 and 2020",
         {"years": list(range(2015, 2021)), "concept": ["cash", "total_assets"], "stmts": ["balance_sheet"]}),
        ("top 3 expense lines in 2022", {"intent": "ranking", "output_type": "ranking_table", "years": [2022], "rank_over": "rows"}),
        ("increase in net sales between 2019 and 2020", {"years": [2019, 2020], "output_type": "table_plus_calc"}),
        ("net sales for fiscal 2018 and FY2020", {"years": [2018, 2020], "output_type": "comparison_table"}),
        ("average operating income over the last 5 years", {"years": last(5), "ops": ["average"], "output_type": "table_plus_calc"}),
        ("net sales in 2020, 2022 and 2024", {"years": [2020, 2022, 2024], "output_type": "multi_year_table"}),
        ("sales and marketing expense in 2022", {"years": [2022], "concept": ["marketing"]}),
        ("cost of sales versus net sales in 2019", {"years": [2019], "concept": ["cost_of_sales", "revenue"], "output_type": "comparison_table"}),
        ("how has net income trended over time?", {"intent": "calculation" if False else "trend", "years": avail, "output_type": "multi_year_table"}),
        ("net cash provided by operating activities in 2019", {"years": [2019], "concept": ["ocf"], "stmts": ["cash_flows"]}),
        ("Amazon total net sales 2021", {"years": [2021], "company": ["Amazon"]}),
        # --- regression cases found by probing
        ("operating income before 2005 and after 2020", {"years": [y for y in avail if y < 2005 or y > 2020]}),
        ("net sales after 2010 and before 2015", {"years": [2011, 2012, 2013, 2014]}),
        ("net sales since 2018 and up to 2020", {"years": [2018, 2019, 2020]}),
        ("2019,2020 net sales", {"years": [2019, 2020]}),
        ("percentage change in net sales from 2017 and 2018", {"years": [2017, 2018], "ops": ["pct_change"]}),
        ("net sales from 2017, 2019 and 2021", {"years": [2017, 2019, 2021]}),
        ("net sales from 2017", {"years": list(range(2017, 2026))}),
        ("net sales from 2017 to 2019", {"years": [2017, 2018, 2019]}),
        ("What percentage of net sales was fulfillment expense in 2022?",
         {"years": [2022], "ratio": ("label:fulfillment", "revenue"), "ops": ["ratio"], "output_type": "table_plus_calc"}),
        ("fulfillment as a percentage of net sales in 2022", {"ratio": ("label:fulfillment", "revenue")}),
        ("ratio of total current assets to total current liabilities in 2022",
         {"ratio": ("current_assets", "current_liabilities"), "years": [2022]}),
        ("net sales in 2021 and 2021", {"years": [2021]}),
        # --- label-era regressions (found by the Stage 8 coverage audit)
        ("total stockholders equity in 2002", {"years": [2002], "concept": ["equity"], "label_has": "deficit"}),
        ("diluted earnings per share in 2004", {"years": [2004], "concept": ["eps"], "label_has": "diluted earnings (loss) per share"}),
        ("cash and cash equivalents in 1997", {"years": [1997], "concept": ["cash"]}),
        ("free cash flow in 2018", {"derived": True, "formula_has": "abs(capex)"}),
    ]
    bad = 0
    for q, chk in T:
        p = parser.parse(q)
        got = {"years": p["years"]["resolved"], "support": p["years"]["support_years"], "intent": p["intent"],
               "output_type": p["output_type"], "ops": [o for o in p["operations"] if o != "rank"] if "ops" in chk else None,
               "concept": [m["concept"] for m in p["metrics"]], "stmts": p["statements"], "derived": any(m["derived"] for m in p["metrics"]),
               "unavailable": p["years"]["unavailable"], "filing": p["years"]["filing_years"], "collections": p["retrieval_plan"]["collections"],
               "rank_over": p["rank_over"], "company": p["companies"],
               "ratio": ((p["ratio"]["numerator"], p["ratio"]["denominator"]) if p["ratio"] else None), "label": (p["metrics"][0]["primary_label"] if p["metrics"] else None)}
        fails = []
        for k, v in chk.items():
            if k == "warn":
                if not any(v in w for w in p["warnings"]):
                    fails.append(f"warning containing '{v}' missing")
            elif k == "ops":
                if got["ops"] != v:
                    fails.append(f"ops {got['ops']} != {v}")
            elif k == "concept" and chk.get("concept") is not None:
                if got["concept"] != v and set(v) - set(got["concept"]):
                    fails.append(f"concepts {got['concept']} != {v}")
            elif k == "stmts":
                if not set(v) <= set(got["stmts"]):
                    fails.append(f"statements {got['stmts']} !>= {v}")
            elif k == "label_has":
                labs = [l.lower() for m in p["metrics"] for l in m["labels"]]
                if not any(v in l for l in labs):
                    fails.append(f"no label variant containing '{v}' in {labs}")
            elif k == "formula_has":
                if not any(v in (m["formula"] or "") for m in p["metrics"]):
                    fails.append(f"formula lacks '{v}'")
            elif k == "label":
                if got["label"] != v:
                    fails.append(f"label {got['label']} != {v}")
            elif got.get(k) != v:
                fails.append(f"{k}: {got.get(k)} != {v}")
        print(("PASS " if not fails else "FAIL ") + q)
        if fails:
            bad += 1
            for f in fails:
                print("       -", f)
            print(summarize(p))
    print(f"\n{len(T) - bad}/{len(T)} passed")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full JSON plan")
    a = ap.parse_args()
    parser = QueryParser(Catalog.from_json(a.tables))
    if a.selftest:
        raise SystemExit(1 if selftest(parser) else 0)
    qs = [" ".join(a.query)] if a.query else None
    while True:
        q = qs[0] if qs else input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        p = parser.parse(q)
        print(json.dumps(p, indent=1) if a.json else summarize(p))
        if qs:
            break


if __name__ == "__main__":
    main()