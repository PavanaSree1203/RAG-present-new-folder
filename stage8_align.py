"""
Stage 8: explicit metric, financial-year, row, column and unit alignment.

The biggest reliability risk in longitudinal filings QA is retrieving the right table and then reading the wrong
year or column. This stage never trusts retrieval rank. For every requested (metric, financial year) it collects ALL
candidate rows from the structured table index (exact label + statement + the financial-year flag), then decides
deterministically which one to use and records WHY:

  * policy "as_reported"  (default): the filing whose latest column is that financial year, i.e. the value as
                          originally reported; if no such filing exists (e.g. the first comparative years) the
                          earliest filing that contains the year.
    policy "latest"       : the most recent filing containing the year (restated value).
    policy "same_filing"  : the latest filing that contains every requested year (a consistent basis).
  * the same financial year printed in several filings is compared on a common scale (thousands vs millions is
    handled; rounding is not a restatement); real differences are reported with the percentage change;
  * value types must match (an amount never matches a per-share or share-count row);
  * derived metrics (margins, free cash flow, ratios) take both components for a year from the SAME filing;
  * the printed column header, column position, units, page and table are kept as provenance.

Company-agnostic: only the plan from Stage 6 and the table index from Stage 5 are used.

Usage:
    python stage8_align.py "Total assets at the end of 2015"
    python stage8_align.py --policy latest "Total assets at the end of 2015"
    python stage8_align.py --test
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def norm_label(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower().replace("&", " and ").replace("'", "").replace("\u2019", "")))


def tolerance(a, b, vtype):
    """Largest difference that is just rounding between two printed values (in standardised units)."""
    if vtype == "per_share":
        return 0.005 + 1e-9
    return 0.5 * max(a["scale"], b["scale"]) + 1e-9


class Aligner:
    def __init__(self, rows_path="data/processed/table_rows.json", concepts=None):
        recs = json.loads(Path(rows_path).read_text(encoding="utf-8"))
        self.rows = [r for r in recs if r["meta"]["kind"] == "row"]
        self.by_label = defaultdict(list)
        for r in self.rows:
            self.by_label[norm_label(r["meta"]["row_label"])].append(r)
        self.concepts = concepts or {}
        years = {int(k[2:]) for r in self.rows for k in r["meta"] if re.fullmatch(r"fy\d{4}", k)}
        self.available_years = sorted(years)

    # ------------------------------------------------------------------ targets
    def targets(self, plan):
        out, seen = [], set()

        def add(concept, labels, statement, vtype, surface, parent):
            if concept not in seen:
                seen.add(concept)
                out.append({"concept": concept, "labels": [l for l in labels if l], "statement": statement,
                            "expected_type": vtype, "surface": surface, "parent": parent})
        for m in plan["metrics"]:
            if m.get("derived") and m.get("components"):
                for c in m["components"]:
                    cs = self.concepts.get(c, {})
                    add(c, (m.get("component_labels") or {}).get(c, []), cs.get("stmt"),
                        "per_share" if c == "eps" else "amount", c.replace("_", " "), m["concept"])
            else:
                add(m["concept"], m.get("labels") or [m.get("primary_label")], m.get("statement"),
                    m.get("value_type", "amount"), m["surface"], None)
        return out

    # ------------------------------------------------------------------ candidates
    def _candidates(self, target, year, qtokens):
        """{filing doc_year: candidate} plus rejected rows; one row per filing."""
        variants = [norm_label(l) for l in target["labels"]]
        per_filing, rejected = defaultdict(list), []
        for vi, v in enumerate(variants):
            for r in self.by_label.get(v, []):
                m = r["meta"]
                if target["statement"] and m["statement"] != target["statement"]:
                    continue
                if not m.get(f"fy{year}"):
                    continue
                if m.get("value_type", "amount") != target["expected_type"]:
                    rejected.append({"filing": m["doc_year"], "label": m["row_label"], "reason":
                                     f"value type {m.get('value_type')} != expected {target['expected_type']}"})
                    continue
                per_filing[m["doc_year"]].append((vi, r))
        by_filing = {}
        for filing, lst in per_filing.items():
            def key(x):                       # prefer labels that share words with the question, then parser order
                vi, r = x
                overlap = len(set(norm_label(r["meta"]["row_label"]).split()) & qtokens)
                return (-overlap, vi, r["meta"]["row_index"])
            lst.sort(key=key)
            vi, r = lst[0]
            m = r["meta"]
            raw, std, vals = json.loads(m["raw_json"]), json.loads(m.get("values_std_json") or "{}"), json.loads(m["values_json"])
            ys = [int(y) for y in m["column_years"].split(",")]
            others = sorted({x[1]["meta"]["row_label"] for x in lst[1:]} - {m["row_label"]})
            by_filing[filing] = {
                "filing": filing, "rec_id": r["id"], "label": m["row_label"], "table_id": m["table_id"], "page": m["page"],
                "value": vals.get(str(year)), "std": std.get(str(year)), "raw": raw.get(str(year)), "units": m["units"],
                "units_known": m.get("units_known", True), "scale": m.get("unit_scale_to_millions", 1.0),
                "std_unit": m.get("std_unit", ""), "value_type": m.get("value_type", "amount"), "latest_fy": m["latest_fy"],
                "column_header": str(year), "column_index": ys.index(year) + 1, "n_columns": len(ys),
                "column_years": ys, "table_status": m["table_status"], "ocr_used": m["ocr_used"],
                "label_alternatives_in_table": others}
        return by_filing, rejected

    @staticmethod
    def _pick(by_filing, year, policy, common=None):
        filings = sorted(by_filing)
        if policy == "latest":
            return filings[-1]
        if policy == "same_filing" and common in by_filing:
            return common
        own = [f for f in filings if by_filing[f]["latest_fy"] == year]
        return own[0] if own else filings[0]

    # ------------------------------------------------------------------ main
    def align(self, plan, policy="as_reported", evidence_ids=None):
        qtokens = set(norm_label(plan["query"]).split())
        yrs = plan["years"]
        requested, support = list(yrs.get("resolved", [])), list(yrs.get("support_years", []))
        fetch = list(yrs.get("fetch_years") or sorted(set(requested) | set(support)))
        targets = self.targets(plan)
        problems, per_target = [], {}
        for t in targets:
            years = fetch
            if not years:                                   # no year in the question: latest financial year of this metric
                allfy = [y for y in self.available_years if self._candidates(t, y, qtokens)[0]]
                years = [max(allfy)] if allfy else []
                if years:
                    problems.append(f"no year given: using the latest financial year with data for '{t['surface']}' ({years[0]})")
            res = {"target": t, "years": {}}
            for y in years:
                by_filing, rejected = self._candidates(t, y, qtokens)
                res["years"][y] = {"by_filing": by_filing, "rejected": rejected}
            per_target[t["concept"]] = res

        # derived metrics and ratios: both components for a year must come from the same filing
        groups = [[c for c in m["components"]] for m in plan["metrics"] if m.get("derived") and m.get("components")]
        if plan.get("ratio"):
            groups.append([plan["ratio"]["numerator"], plan["ratio"]["denominator"]])
        forced = {}
        for g in groups:
            g = [c for c in g if c in per_target]
            if len(g) < 2:
                continue
            for y in per_target[g[0]]["years"]:
                sets = [set(per_target[c]["years"][y]["by_filing"]) for c in g if y in per_target[c]["years"]]
                if len(sets) < len(g) or not all(sets):
                    continue
                common = set.intersection(*sets)
                if common:
                    f = self._pick({k: per_target[g[0]]["years"][y]["by_filing"][k] for k in common}, y, policy)
                    for c in g:
                        forced[(c, y)] = f
                else:
                    problems.append(f"FY{y}: components {g} are not available in one common filing; mixing filings")

        # per-target, per-year decision
        out_metrics = []
        for cid, res in per_target.items():
            t = res["target"]
            ys_all = sorted(res["years"])
            filing_sets = [set(res["years"][y]["by_filing"]) for y in ys_all if res["years"][y]["by_filing"]]
            common_all = max(set.intersection(*filing_sets)) if filing_sets and set.intersection(*filing_sets) and len(filing_sets) == len(ys_all) else None
            cmp_filing = common_all if (policy == "same_filing") else common_all
            m_out = {"concept": cid, "labels": t["labels"], "statement": t["statement"], "expected_type": t["expected_type"],
                     "parent": t["parent"], "comparable_filing": common_all, "years": {}}
            for y in ys_all:
                bf, rejected = res["years"][y]["by_filing"], res["years"][y]["rejected"]
                entry = {"status": "unavailable", "chosen": None, "alternatives": [], "rejected": rejected[:3],
                         "checks": [], "notes": [], "role": "support" if (y in support and y not in requested) else "requested"}
                if not bf:
                    entry["notes"].append(f"no row for FY{y} with a matching label"
                                          + (f" (available financial years: {self.available_years[0]}-{self.available_years[-1]})"
                                             if self.available_years else ""))
                    m_out["years"][str(y)] = entry
                    continue
                filing = forced.get((cid, y)) or self._pick(bf, y, policy, cmp_filing)
                ch = bf[filing]
                entry["chosen"] = ch
                restated = False
                for f in sorted(bf):
                    if f == filing:
                        continue
                    alt = dict(bf[f])
                    diff = None
                    if alt["std"] is not None and ch["std"] is not None:
                        diff = alt["std"] - ch["std"]
                        alt["differs"] = abs(diff) > tolerance(ch, alt, t["expected_type"])
                        alt["pct_diff"] = round(100 * diff / abs(ch["std"]), 3) if ch["std"] else None
                    else:
                        alt["differs"], alt["pct_diff"] = None, None
                    if alt["differs"] and t["expected_type"] in ("per_share", "shares") and ch["std"] and alt["std"]:
                        hi, lo = max(abs(ch["std"]), abs(alt["std"])), min(abs(ch["std"]), abs(alt["std"]))
                        ratio = hi / lo if lo else 0.0
                        f = round(ratio)
                        if f >= 2 and abs(ratio - f) <= 0.02 * f:      # exact N-for-1 factor: a stock split, not a restatement
                            alt["split_factor"] = f
                    restated |= bool(alt["differs"])
                    entry["alternatives"].append(alt)
                diffs = [a for a in entry["alternatives"] if a["differs"]]
                entry["status"] = "ok" if not diffs else (
                    "split_adjusted" if all(a.get("split_factor") for a in diffs) else "restated")
                # --- checks
                if ch["latest_fy"] == y:
                    entry["checks"].append(f"as originally reported: the {ch['filing']} filing's latest column is FY{y}")
                else:
                    entry["notes"].append(f"FY{y} is a comparative column (column {ch['column_index']} of {ch['n_columns']}) "
                                          f"in the {ch['filing']} filing (its latest year is FY{ch['latest_fy']})")
                if not ch["units_known"]:
                    entry["notes"].append("units not stated in the table title: values cannot be standardised")
                if ch["table_status"] == "warn":
                    entry["notes"].append("source table passed validation with warnings")
                if ch["ocr_used"]:
                    entry["notes"].append("value read by OCR")
                if ch["label_alternatives_in_table"]:
                    entry["notes"].append(f"other matching labels in the same table: {ch['label_alternatives_in_table']}; "
                                          f"used '{ch['label']}'")
                if entry["status"] in ("restated", "split_adjusted"):
                    d = [a for a in entry["alternatives"] if a["differs"]]
                    why = ("restated across filings" if entry["status"] == "restated"
                           else "adjusted for a stock split in later filings (the as-reported value is kept)")
                    entry["notes"].append(why + ": " + "; ".join(
                        f"{a['filing']} filing {a['raw']} (" + (f"{a['split_factor']}-for-1 split factor" if a.get("split_factor")
                                                              else f"{a['pct_diff']:+.2f}%") + ")" for a in d))
                same_basis = (cid, y) in forced
                if same_basis:
                    entry["checks"].append("component taken from the same filing as the other component(s)")
                if evidence_ids is not None:
                    entry["retrieved_by_stage7"] = ch["rec_id"] in evidence_ids
                m_out["years"][str(y)] = entry
            out_metrics.append(m_out)
        missing = [(m["concept"], y) for m in out_metrics for y, e in m["years"].items() if e["status"] == "unavailable"]
        for c, y in missing:
            problems.append(f"{c}: FY{y} unavailable")
        return {"query": plan["query"], "policy": policy, "requested_years": requested, "support_years": support,
                "metrics": out_metrics, "problems": problems}


def show(res):
    print(f"\nQ: {res['query']}   [policy: {res['policy']}] requested {res['requested_years']} support {res['support_years']}")
    for m in res["metrics"]:
        print(f"  [{m['concept']}] {m['labels'][:1]} | {m['statement']} | type {m['expected_type']}"
              + (f" | component of {m['parent']}" if m["parent"] else ""))
        for y, e in m["years"].items():
            c = e["chosen"]
            if not c:
                print(f"    FY{y}: UNAVAILABLE - {e['notes'][0] if e['notes'] else ''}")
                continue
            std = f"{c['std']:,.3f} {c['std_unit']}" if c["std"] is not None else "n/a"
            print(f"    FY{y} [{e['status']:8}] {c['raw']:>12} = {std:28} <- {c['filing']} filing p{c['page']} "
                  f"col {c['column_index']}/{c['n_columns']} '{c['column_header']}' units {c['units']}"
                  + (" (support year)" if e["role"] == "support" else ""))
            for n in e["notes"]:
                print(f"           note: {n}")
    for p in res["problems"]:
        print(f"  PROBLEM: {p}")


# ---------------------------------------------------------------------- synthetic self-test
def _rec(company, filing, stmt, label, years, raw, scale, units, vtype="amount", status="ok", ridx=0, ocr=False):
    def num(s):
        t = s.replace(",", "").replace("$", "")
        return -float(t.strip("()")) if t.startswith("(") else float(t)
    vals = {str(y): num(r) for y, r in zip(years, raw)}
    k = 1.0 if vtype == "per_share" else scale
    meta = {"kind": "row", "company": company, "doc_year": filing, "statement": stmt, "table_id": f"{company}_{filing}_{stmt}",
            "page": 40 + ridx, "units": units, "column_years": ",".join(map(str, years)), "latest_fy": max(years),
            "n_years": len(years), "table_status": status, "ocr_used": ocr, "extractor": "t", "units_known": True,
            "unit_scale_to_millions": scale, "section": "", "row_label": label, "row_index": ridx, "row_kind": "line",
            "value_type": vtype, "std_unit": {"amount": "USD millions", "per_share": "USD per share", "shares": "millions of shares"}[vtype],
            "values_std_json": json.dumps({y: v * k for y, v in vals.items()}), "values_json": json.dumps(vals),
            "raw_json": json.dumps({str(y): r for y, r in zip(years, raw)})}
    for y in years:
        meta[f"fy{y}"] = True
    return {"id": f"{company}_{filing}_{stmt}_{label}".replace(" ", "_"), "text": label, "meta": meta}


def selftest():
    import tempfile
    A, M, T = "Co", 1.0, 0.001
    R = [
        _rec(A, 2014, "balance_sheet", "Total assets", [2013, 2014], ["40,159", "54,505"], M, "USD millions"),
        _rec(A, 2015, "balance_sheet", "Total assets", [2014, 2015], ["54,505", "65,444"], M, "USD millions"),
        _rec(A, 2016, "balance_sheet", "Total assets", [2015, 2016], ["64,747", "83,402"], M, "USD millions"),
        _rec(A, 2015, "balance_sheet", "Total current assets", [2014, 2015], ["31,327", "36,474"], M, "USD millions", ridx=1),
        _rec(A, 1997, "operations", "Net sales", [1995, 1996, 1997], ["511", "15,746", "147,787"], T, "USD thousands"),
        _rec(A, 1998, "operations", "Net sales", [1996, 1997, 1998], ["15,746", "147,787", "609,996"], T, "USD thousands"),
        _rec(A, 2004, "operations", "Net sales", [2002, 2003, 2004], ["3,932,936", "5,263,699", "6,921,124"], T, "USD thousands"),
        _rec(A, 2005, "operations", "Net sales", [2003, 2004, 2005], ["5,264", "6,921", "8,490"], M, "USD millions"),
        _rec(A, 2015, "operations", "Total net sales", [2013, 2014, 2015], ["74,452", "88,988", "107,006"], M, "USD millions"),
        _rec(A, 2016, "operations", "Total net sales", [2014, 2015, 2016], ["88,988", "107,006", "135,987"], M, "USD millions"),
        _rec(A, 2015, "operations", "Operating income", [2013, 2014, 2015], ["745", "178", "2,233"], M, "USD millions", ridx=5),
        _rec(A, 2016, "operations", "Operating income", [2014, 2015, 2016], ["178", "2,300", "4,186"], M, "USD millions", ridx=5),  # restated FY2015
        _rec(A, 2015, "operations", "Basic earnings per share", [2013, 2014, 2015], ["0.59", "(0.52)", "1.25"], M, "USD millions", "per_share", ridx=8),
        _rec(A, 2015, "operations", "Diluted earnings per share", [2013, 2014, 2015], ["0.59", "(0.52)", "1.25"], M, "USD millions", "per_share", ridx=9),
        _rec(A, 2021, "operations", "Diluted earnings per share", [2019, 2020, 2021], ["23.46", "42.64", "64.81"], M, "USD millions", "per_share", ridx=11),
        _rec(A, 2022, "operations", "Diluted earnings per share", [2020, 2021, 2022], ["2.09", "3.24", "(0.27)"], M, "USD millions", "per_share", ridx=11),
        _rec(A, 2015, "operations", "Shares used in computation of basic earnings per share", [2013, 2014, 2015], ["462", "462", "467"], M, "USD millions", "shares", ridx=10),
    ]
    f = Path(tempfile.mkdtemp()) / "rows.json"
    f.write_text(json.dumps(R))
    al = Aligner(str(f), concepts={"operating_income": {"stmt": "operations"}, "revenue": {"stmt": "operations"}})

    def plan(q, metrics, years, **kw):
        return {"query": q, "metrics": metrics, "years": {"resolved": years, "support_years": kw.get("support", []),
                "fetch_years": sorted(set(years) | set(kw.get("support", [])))}, "ratio": kw.get("ratio")}
    m = lambda c, labels, stmt, vt="amount": {"concept": c, "labels": labels, "primary_label": labels[0], "statement": stmt,
                                              "surface": c, "derived": False, "value_type": vt}
    TA, NS = m("total_assets", ["Total assets"], "balance_sheet"), m("revenue", ["Total net sales", "Net sales"], "operations")
    checks = []

    def check(name, cond, detail=""):
        checks.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + (f"   [{detail}]" if not cond else ""))

    r = al.align(plan("total assets 2015", [TA], [2015]))
    e = r["metrics"][0]["years"]["2015"]
    check("as_reported picks the 2015 filing's own value", e["chosen"]["filing"] == 2015 and e["chosen"]["value"] == 65444.0, e["chosen"])
    check("restatement vs the 2016 filing is flagged with its percentage", e["status"] == "restated" and
          any(a["filing"] == 2016 and a["differs"] and abs(a["pct_diff"] - (-1.065)) < 0.01 for a in e["alternatives"]), e["alternatives"])
    r = al.align(plan("total assets 2015", [TA], [2015]), policy="latest")
    check("policy latest picks the restated 2016 value", r["metrics"][0]["years"]["2015"]["chosen"]["value"] == 64747.0)
    r = al.align(plan("total assets 2014", [TA], [2014]))
    e = r["metrics"][0]["years"]["2014"]
    check("FY2014 identical in two filings is 'ok', not restated", e["status"] == "ok" and e["chosen"]["filing"] == 2014)
    r = al.align(plan("net sales 1996", [NS], [1996]))
    e = r["metrics"][0]["years"]["1996"]
    check("FY1996 (no own filing) uses the earliest comparative column", e["chosen"]["filing"] == 1997 and e["chosen"]["std"] == 15.746
          and e["chosen"]["column_index"] == 2 and any("comparative" in n for n in e["notes"]), e)
    r = al.align(plan("net sales 2003", [NS], [2003]))
    e = r["metrics"][0]["years"]["2003"]
    check("thousands vs millions: 5,263,699K vs 5,264M is rounding, not a restatement", e["status"] == "ok" and e["chosen"]["filing"] == 2004,
          (e["status"], e["alternatives"]))
    r = al.align(plan("revenue 2030", [NS], [2030]))
    check("unavailable year is reported, never guessed", r["metrics"][0]["years"]["2030"]["status"] == "unavailable" and r["problems"])
    dm = {"concept": "operating_margin", "surface": "operating margin", "derived": True, "components": ["operating_income", "revenue"],
          "component_labels": {"operating_income": ["Operating income"], "revenue": ["Total net sales"]}, "labels": [], "statement": None}
    r = al.align(plan("operating margin 2015", [dm], [2015]))
    ch = {x["concept"]: x["years"]["2015"]["chosen"]["filing"] for x in r["metrics"]}
    check("derived metric: both components from the same filing", len(set(ch.values())) == 1, ch)
    eps = m("eps", ["Basic earnings per share", "Diluted earnings per share", "Shares used in computation of basic earnings per share"],
            "operations", "per_share")
    r = al.align(plan("diluted earnings per share 2015", [eps], [2015]))
    e = r["metrics"][0]["years"]["2015"]
    check("EPS: diluted chosen from the question wording; shares row rejected by value type",
          e["chosen"]["label"] == "Diluted earnings per share" and any(x["label"].startswith("Shares used") for x in e["rejected"]), e["chosen"]["label"])
    r = al.align(plan("diluted earnings per share 2021", [eps], [2021]))
    e = r["metrics"][0]["years"]["2021"]
    check("stock split is reported as a split factor, not a restatement; as-reported 64.81 kept",
          e["status"] == "split_adjusted" and e["chosen"]["value"] == 64.81 and any(a.get("split_factor") == 20 for a in e["alternatives"]),
          (e["status"], e["alternatives"]))
    r = al.align(plan("eps 2015", [eps], [2015]))
    check("EPS without wording: ambiguity reported", bool(r["metrics"][0]["years"]["2015"]["notes"]))
    r = al.align(plan("operating income 2015", [m("operating_income", ["Operating income"], "operations")], [2015]))
    e = r["metrics"][0]["years"]["2015"]
    check("operating income FY2015: 2,233 as reported, restated to 2,300 later", e["chosen"]["value"] == 2233.0 and e["status"] == "restated")
    r = al.align(plan("total assets", [TA], []))
    check("no year in the question: latest financial year with data is used and said so",
          "2016" in r["metrics"][0]["years"] and any("latest financial year" in p for p in r["problems"]), r["problems"])
    for rr in (al.align(plan("total assets 2015 and 2016", [TA], [2015, 2016])), ):
        show(rr)
    print(f"\n{sum(checks)}/{len(checks)} passed")
    return 0 if all(checks) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="*")
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--rows", default="data/processed/table_rows.json")
    ap.add_argument("--policy", default="as_reported", choices=["as_reported", "latest", "same_filing"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    from stage6_query_parser import CONCEPTS, Catalog, QueryParser
    parser = QueryParser(Catalog.from_json(a.tables))
    al = Aligner(a.rows, concepts=CONCEPTS)
    qs = [" ".join(a.query)] if a.query else []
    while True:
        q = qs.pop(0) if qs else input("\nQuestion (blank to quit): ").strip()
        if not q:
            break
        res = al.align(parser.parse(q), policy=a.policy)
        print(json.dumps(res, indent=1, default=str) if a.json else "", end="")
        if not a.json:
            show(res)
        if a.query:
            break


if __name__ == "__main__":
    main()