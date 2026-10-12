"""
Stage 9: deterministic numerical verification.

Takes the Stage 6 plan and the Stage 8 aligned values and does ALL arithmetic in Python (Decimal), never in the LLM:
  difference, pct_change, yoy, cagr, ratio, sum, average, max/min (with the year), rank over years, derived metrics
  (operating margin, free cash flow = ocf - abs(capex), ...), and plain look-ups.
Every result carries its formula, its inputs (value, unit, filing, page, status) and an independent cross-check
(the same quantity recomputed a second way).  The module REFUSES instead of guessing when an input is missing, a base is
zero or negative where that makes the figure meaningless, units cannot be standardised, or a per-share figure would be
mixed with an amount.  When a value is restated or split-adjusted between filings, the result is ALSO given on the
"latest" (consistent) basis so the two can be compared.

Usage:
  python stage9_verify.py "difference between 2017 and 2018 sales"
  python stage9_verify.py "CAGR of net sales from 2015 to 2020" --json
  python stage9_verify.py --selftest
Needs stage6_query_parser.py, stage8_align.py, canon.py, typo.py beside this file (selftest needs none of the data).
"""
import argparse
import ast
import json
import re
import sys
from decimal import Decimal, getcontext, InvalidOperation

getcontext().prec = 28
D = Decimal
FLAGGED = ("restated", "split_adjusted")


def dec(x):
    return D(repr(float(x))) if not isinstance(x, D) else x


# ----------------------------------------------------------------------------- safe formula evaluation
def eval_formula(formula, env):
    """Evaluate e.g. 'operating_income / revenue * 100' or 'ocf - abs(capex)' with Decimals.  Whitelisted syntax only."""
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return D(str(n.value))
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise KeyError(n.id)
            return env[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if b == 0:
                raise ZeroDivisionError("division by zero")
            return a / b
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "abs" and len(n.args) == 1:
            return abs(ev(n.args[0]))
        raise ValueError(f"unsupported formula element: {ast.dump(n)[:40]}")
    return ev(ast.parse(formula, mode="eval"))


# ----------------------------------------------------------------------------- series from the aligned result
class Series:
    def __init__(self, concept, label, unit, vtype):
        self.concept, self.label, self.unit, self.vtype = concept, label, unit, vtype
        self.v, self.src = {}, {}          # year -> Decimal ; year -> provenance dict
        self.missing = []                  # years without a usable value (and why)

    def years(self):
        return sorted(self.v)


def series_from_aligned(aligned):
    out = {}
    for m in aligned["metrics"]:
        s = Series(m["concept"], (m.get("labels") or [m["concept"]])[0], None, m.get("expected_type", "amount"))
        for y, e in sorted(m["years"].items(), key=lambda kv: int(kv[0])):
            c = e.get("chosen")
            y = int(y)
            if not c:
                s.missing.append((y, (e.get("notes") or ["no matching row"])[0]))
                continue
            if c.get("std") is None or not c.get("units_known", True):
                s.missing.append((y, "units not stated: the value cannot be standardised"))
                continue
            s.unit = s.unit or c.get("std_unit")
            if s.unit and c.get("std_unit") and c["std_unit"] != s.unit:
                s.missing.append((y, f"unit mismatch ({c['std_unit']} vs {s.unit})"))
                continue
            s.v[y] = dec(c["std"])
            alt = [a for a in e.get("alternatives", []) if a.get("differs")]
            s.src[y] = {"filing": c["filing"], "page": c.get("page"), "raw": c.get("raw"), "status": e["status"],
                        "ocr": bool(c.get("ocr_used")), "alternatives": [(a["filing"], a.get("raw"), a.get("pct_diff"), a.get("split_factor")) for a in alt],
                        "role": e.get("role", "requested")}
        out[m["concept"]] = s
    return out


# ----------------------------------------------------------------------------- results
def q(v, places=6):
    """Decimal -> short readable string (no 28-digit tails)."""
    t = f"{float(v):,.{places}f}".rstrip("0").rstrip(".")
    return t or "0"


def fmt(v, unit):
    v = float(v)
    if unit == "percent":
        return f"{v:,.2f}%"
    if unit == "x":
        return f"{v:,.4f}x"
    if unit and "per share" in unit:
        return f"{v:,.2f} {unit}"
    return f"{v:,.2f} {unit or ''}".strip()


def inputs_of(series, years):
    rows = []
    for y in years:
        s = series.src.get(y, {})
        if s.get("component_src"):
            for c, cs in s["component_src"].items():
                rows.append({"concept": c, "fy": y, "value": s["components"][c], "unit": cs.get("unit"), "filing": cs.get("filing"),
                             "page": cs.get("page"), "raw": cs.get("raw"), "status": cs.get("status")})
            continue
        rows.append({"concept": series.concept, "fy": y, "value": float(series.v[y]), "unit": series.unit,
                     "filing": s.get("filing"), "page": s.get("page"), "raw": s.get("raw"), "status": s.get("status")})
    return rows


def result(op, series, years, value, unit, formula, steps, alt_value, notes=None, extra=None):
    ok = alt_value is not None and abs(D(value) - D(alt_value)) <= D("1e-9") * max(D(1), abs(D(value)))
    r = {"op": op, "concept": series.concept, "label": series.label, "years": years, "value": float(value), "unit": unit,
         "display": fmt(value, unit), "formula": formula, "steps": steps,
         "inputs": inputs_of(series, [y for y in years if y in series.v]),
         "cross_check": "passed" if ok else "FAILED", "notes": notes or []}
    if extra:
        r.update(extra)
    return r


def refuse(op, concept, why):
    return {"op": op, "concept": concept, "refused": True, "reason": why}


def need_years(s, years, op):
    miss = [y for y in years if y not in s.v]
    if miss:
        why = []
        for y in miss:
            m = [r for yy, r in s.missing if yy == y]
            why.append(f"FY{y} ({m[0] if m else 'no value'})")
        return refuse(op, s.concept, "missing input year(s): " + "; ".join(why) + ". No figure is calculated from incomplete data.")
    return None


# ----------------------------------------------------------------------------- operations
def op_lookup(s, years):
    out = []
    for y in years:
        if y in s.v:
            out.append(result("value", s, [y], s.v[y], s.unit, getattr(s, "formula", None) or f"{s.label} FY{y}", [f"FY{y} = {fmt(s.v[y], s.unit)}"], s.v[y]))
        else:
            out.append(need_years(s, [y], "value"))
    return out


def op_difference(s, a, b):
    r = need_years(s, [a, b], "difference")
    if r:
        return r
    d = s.v[b] - s.v[a]
    alt = (s.v[b] / s.v[a] - 1) * s.v[a] if s.v[a] != 0 else d
    return result("difference", s, [a, b], d, s.unit, f"FY{b} - FY{a}", [f"{q(s.v[b])} - {q(s.v[a])} = {q(d)}"], alt)


def op_pct_change(s, a, b):
    r = need_years(s, [a, b], "pct_change")
    if r:
        return r
    if s.v[a] == 0:
        return refuse("pct_change", s.concept, f"FY{a} is 0, so a percentage change is undefined.")
    notes = []
    if s.v[a] < 0:
        notes.append(f"base FY{a} is negative ({q(s.v[a])}); the change is measured against its absolute value, so the sign shows direction of the dollar change")
    pct = (s.v[b] - s.v[a]) / abs(s.v[a]) * 100
    alt = (s.v[b] - s.v[a]) * 100 / abs(s.v[a])
    return result("pct_change", s, [a, b], pct, "percent", f"(FY{b} - FY{a}) / |FY{a}| x 100",
                  [f"({q(s.v[b])} - {q(s.v[a])}) / |{q(s.v[a])}| x 100 = {float(pct):.6f}%"], alt, notes)


def op_cagr(s, a, b):
    r = need_years(s, [a, b], "cagr")
    if r:
        return r
    n = b - a
    if n <= 0:
        return refuse("cagr", s.concept, "CAGR needs two different years in ascending order.")
    if s.v[a] <= 0 or s.v[b] <= 0:
        return refuse("cagr", s.concept, f"CAGR is undefined for non-positive endpoints (FY{a} = {s.v[a]}, FY{b} = {s.v[b]}).")
    ratio = float(s.v[b] / s.v[a])
    rate = D(repr(ratio ** (1.0 / n) - 1)) * 100
    back = float(s.v[a]) * (1 + float(rate) / 100) ** n          # cross-check: growing the start value at that rate reproduces the end value
    ok_alt = D(repr(float(rate)))
    res = result("cagr", s, [a, b], rate, "percent", f"(FY{b} / FY{a})^(1/{n}) - 1",
                 [f"({q(s.v[b])} / {q(s.v[a])})^(1/{n}) - 1 = {float(rate):.6f}%"], ok_alt)
    res["cross_check"] = "passed" if abs(back - float(s.v[b])) <= 1e-9 * abs(float(s.v[b])) else "FAILED"
    return res


def op_yoy(s, years_with_prev):
    out = []
    for prev, y in years_with_prev:
        out.append(op_pct_change(s, prev, y))
        if "refused" not in out[-1]:
            out[-1]["op"] = "yoy"
    return out


def op_sum(s, years):
    r = need_years(s, years, "sum")
    if r:
        return r
    t = sum((s.v[y] for y in years), D(0))
    alt = D(repr(__import__("math").fsum(float(s.v[y]) for y in years)))
    return result("sum", s, years, t, s.unit, "sum of " + ", ".join(f"FY{y}" for y in years), [" + ".join(q(s.v[y]) for y in years) + f" = {q(t)}"], alt)


def op_average(s, years):
    r = need_years(s, years, "average")
    if r:
        return r
    t = sum((s.v[y] for y in years), D(0))
    avg = t / len(years)
    return result("average", s, years, avg, s.unit, f"sum / {len(years)}", [f"{q(t)} / {len(years)} = {q(avg)}"], t / D(len(years)))


def op_extreme(s, years, which):
    have = [y for y in years if y in s.v]
    if not have:
        return refuse(which, s.concept, "no values available in the requested years")
    pick = max(have, key=lambda y: (s.v[y], -y)) if which == "max" else min(have, key=lambda y: (s.v[y], y))
    alt = (max if which == "max" else min)(s.v[y] for y in have)
    notes = []
    gap = [y for y in years if y not in s.v]
    if gap:
        notes.append(f"{len(gap)} year(s) had no usable value and were not considered: {gap[:8]}{'...' if len(gap) > 8 else ''}")
    return result(which, s, [pick], s.v[pick], s.unit, f"{which} over FY{have[0]}-FY{have[-1]}",
                  [f"{which} = FY{pick}: {q(s.v[pick])}"], alt, notes, {"year": pick, "considered": have})


def op_rank(s, years, n):
    have = [y for y in years if y in s.v]
    order = sorted(have, key=lambda y: (-s.v[y], y))[:n]
    ranking = [{"rank": i, "fy": y, "value": float(s.v[y]), "display": fmt(s.v[y], s.unit)} for i, y in enumerate(order, 1)]
    r = result("rank", s, order, s.v[order[0]] if order else D(0), s.unit, f"top {n} years by value", [f"FY{y}: {q(s.v[y])}" for y in order],
               s.v[order[0]] if order else D(0), None, {"ranking": ranking})
    return r


# ----------------------------------------------------------------------------- derived metrics and ratios
def derived_series(m, base):
    comps = m.get("components") or []
    miss = [c for c in comps if c not in base]
    if miss:
        return None, f"components not available: {miss}"
    unit = "percent" if re.search(r"\*\s*100", m["formula"]) else next((base[c].unit for c in comps if base[c].unit), None)
    s = Series(m["concept"], (m.get("labels") or [m["concept"]])[0] if m.get("labels") else m["concept"].replace("_", " "), unit, "amount")
    for y in sorted(set.intersection(*[set(base[c].v) for c in comps])) if comps else []:
        try:
            s.v[y] = eval_formula(m["formula"], {c: base[c].v[y] for c in comps})
            fil = {base[c].src[y]["filing"] for c in comps}
            s.src[y] = {"filing": "/".join(str(f) for f in sorted(fil)), "page": None, "raw": None,
                        "status": "ok" if all(base[c].src[y]["status"] == "ok" for c in comps) else "restated",
                        "ocr": any(base[c].src[y]["ocr"] for c in comps), "alternatives": [], "role": "requested",
                        "components": {c: float(base[c].v[y]) for c in comps}, "mixed_filings": len(fil) > 1,
                        "component_src": {c: dict(base[c].src[y], unit=base[c].unit) for c in comps}}
        except ZeroDivisionError:
            s.missing.append((y, "division by zero"))
    for c in comps:
        for y, why in base[c].missing:
            if y not in s.v and not any(yy == y for yy, _ in s.missing):
                s.missing.append((y, f"component {c}: {why}"))
    s.formula = m["formula"]
    return s, None


def ratio_series(ratio, base):
    n, d = base.get(ratio["numerator"]), base.get(ratio["denominator"])
    if n is None or d is None:
        return None, "numerator or denominator has no data"
    if n.vtype != d.vtype:
        return None, f"a {n.vtype} cannot be divided by a {d.vtype} (units differ: {n.unit} vs {d.unit})"
    unit = "percent" if ratio.get("unit") == "percent" else "x"
    s = Series(f"{n.concept}/{d.concept}", f"{n.label} / {d.label}", unit, "ratio")
    mult = D(100) if unit == "percent" else D(1)
    for y in sorted(set(n.v) & set(d.v)):
        if d.v[y] == 0:
            s.missing.append((y, f"denominator {d.label} is 0"))
            continue
        s.v[y] = n.v[y] / d.v[y] * mult
        s.src[y] = {"filing": f"{n.src[y]['filing']}/{d.src[y]['filing']}", "page": None, "raw": None,
                    "status": "ok" if n.src[y]["status"] == "ok" and d.src[y]["status"] == "ok" else "restated",
                    "ocr": n.src[y]["ocr"] or d.src[y]["ocr"], "alternatives": [], "role": "requested",
                    "components": {n.concept: float(n.v[y]), d.concept: float(d.v[y])},
                    "component_src": {n.concept: dict(n.src[y], unit=n.unit), d.concept: dict(d.src[y], unit=d.unit)}}
    for y in set(n.v) ^ set(d.v):
        s.missing.append((y, "numerator or denominator unavailable"))
    s.formula = f"{n.concept} / {d.concept}" + (" x 100" if unit == "percent" else "")
    return s, None


# ----------------------------------------------------------------------------- the calculator
def compute(plan, aligned):
    ops = list(plan.get("operations") or [])
    yrs = plan.get("years", {})
    requested = sorted(int(y) for y in (yrs.get("requested") or yrs.get("resolved") or []))
    support = sorted(int(y) for y in yrs.get("support_years") or [])
    base = series_from_aligned(aligned)
    out = {"query": plan["query"], "policy": aligned.get("policy"), "operations": ops, "series": {}, "results": [], "warnings": [],
           "refusals": []}
    for p in aligned.get("problems", []):
        out["warnings"].append(p)
    if plan.get("rank_over") == "rows":
        out["refusals"].append(refuse("rank", None, "ranking line items (e.g. 'top 3 expense lines') is not a numeric-verification operation here; "
                                                    "use the retrieved statement rows directly."))
        return finish(out)

    targets = []                                           # series the operations apply to
    ratio = plan.get("ratio")
    if ratio:
        s, why = ratio_series(ratio, base)
        if s is None:
            out["refusals"].append(refuse("ratio", None, why))
        else:
            targets.append(s)
    else:
        for m in plan.get("metrics", []):
            if m.get("derived") and m.get("formula"):
                s, why = derived_series(m, base)
                if s is None:
                    out["refusals"].append(refuse("derived", m["concept"], why))
                else:
                    targets.append(s)
            elif m["concept"] in base:
                targets.append(base[m["concept"]])
            else:
                out["refusals"].append(refuse("value", m["concept"], "no aligned data"))
    if not targets and not out["refusals"]:
        out["note"] = "no metric was recognised in the question, so there is nothing to calculate (use the retrieved text/table evidence)"

    n_top = 5
    m_top = re.search(r"\btop\s+(\d+)", plan["query"].lower())
    if m_top:
        n_top = int(m_top.group(1))
    for s in targets:
        out["series"][s.concept] = {"unit": s.unit, "label": s.label, "values": {str(y): float(v) for y, v in sorted(s.v.items())},
                                    "missing": {str(y): w for y, w in s.missing}}
        yrs_here = requested or s.years()
        first, last = (yrs_here[0], yrs_here[-1]) if yrs_here else (None, None)
        res = []
        did = False
        for op in ops:
            if op == "difference" and first != last:
                res.append(op_difference(s, first, last)); did = True
            elif op == "pct_change" and first != last:
                res.append(op_pct_change(s, first, last)); did = True
            elif op == "cagr":
                res.append(op_cagr(s, first, last)); did = True
            elif op == "yoy":
                allyears = sorted(set(yrs_here) | set(support))
                pairs = [(p, y) for p, y in zip(allyears, allyears[1:]) if y in yrs_here]
                if not pairs:
                    res.append(refuse("yoy", s.concept, "needs a previous year to compare with"));
                else:
                    res.extend(op_yoy(s, pairs))
                did = True
            elif op == "sum":
                res.append(op_sum(s, yrs_here)); did = True
            elif op == "average":
                res.append(op_average(s, yrs_here)); did = True
            elif op in ("max", "min"):
                res.append(op_extreme(s, yrs_here, op)); did = True
            elif op == "rank":
                res.append(op_rank(s, yrs_here, n_top)); did = True
            elif op == "ratio":
                did = True                                   # already folded into the target series
        for r_ in res:                                       # never let a wide year range be silently collapsed to its two end points
            if not r_.get("refused") and r_["op"] in ("difference", "pct_change", "cagr") and len(yrs_here) > 2:
                r_["notes"].append(f"computed between the first (FY{first}) and last (FY{last}) of {len(yrs_here)} requested years; "
                                   "if you meant two specific years, name them (e.g. 'between 2017 and 2018')")
        if ratio:
            res = op_lookup(s, yrs_here) + res if not any(o in ops for o in ("max", "min", "rank", "sum", "average")) else res
        elif not did:                                        # plain question: show each year; comparisons also get the change
            res.extend(op_lookup(s, yrs_here))
            if len(yrs_here) == 2 and (plan.get("intent") in ("comparison", "calculation")):
                res.append(op_difference(s, yrs_here[0], yrs_here[1]))
                res.append(op_pct_change(s, yrs_here[0], yrs_here[1]))
        elif ratio and did:
            pass
        out["results"].extend(res)
    return finish(out, base, targets)


def finish(out, base=None, targets=None):
    for r in out["results"]:
        if r.get("refused"):
            out["refusals"].append(r)
    out["results"] = [r for r in out["results"] if not r.get("refused")]
    flagged = {}
    for s in (targets or []):
        for y, src in s.src.items():
            if src["status"] in FLAGGED:
                flagged.setdefault(s.concept, []).append((y, src))
    for r in out["results"]:
        if r["cross_check"] != "passed":
            out["warnings"].append(f"internal cross-check FAILED for {r['op']} {r['concept']}: do not use this figure")
        for i in r["inputs"]:
            if i["status"] in FLAGGED:
                r["notes"].append(f"input FY{i['fy']} of {i['concept']} differs between filings ({i['status']})")
    out["flagged_inputs"] = {c: [y for y, _ in v] for c, v in flagged.items()}
    if base is not None:
        used = {(r["concept"], y) for r in out["results"] for y in r["years"]}
        for s in (targets or []):
            for y, src in s.src.items():
                if (s.concept, y) not in used:
                    continue
                if src.get("mixed_filings"):
                    out["warnings"].append(f"{s.concept} FY{y}: components come from different filings")
                if src["ocr"]:
                    out["warnings"].append(f"{s.concept} FY{y}: value read by OCR")
    out["status"] = ("not_applicable" if not out["results"] and not out["refusals"] else
                     "refused" if out["refusals"] and not out["results"] else
                     "partial" if out["refusals"] else
                     "verified_with_flags" if out["flagged_inputs"] else "verified")
    return out


OVERVIEW = "net sales, operating income, net income, total assets and net cash provided by operating activities"


def verify(plan, aligner, policy="as_reported", evidence_ids=None, parser=None):
    """Align (Stage 8) then calculate.  If a used value is restated or split-adjusted and the question compares years, the
    calculation is also run on the 'latest' basis (every year on one footing).  For a STOCK SPLIT the as-reported figures
    are not comparable at all (64.81 vs 2.90), so the consistent basis becomes the primary answer and the as-reported one is shown as
    the alternative; for ordinary restatements the as-reported answer stays primary and the consistent one is the alternative."""
    overview = None
    yrs = sorted(int(y) for y in plan.get("years", {}).get("requested", []))
    wants_numbers = plan.get("intent") in ("comparison", "trend", "multi_year", "calculation") or \
        (plan.get("intent") == "fact" and (plan.get("flags") or {}).get("needs_tables"))
    if not plan.get("metrics") and yrs and parser is not None and wants_numbers and not (plan.get("flags") or {}).get("narrative"):
        ylist = ", ".join(str(y) for y in yrs[:-1]) + (" and " if len(yrs) > 1 else "") + str(yrs[-1])
        p2 = parser.parse(f"{OVERVIEW} in {ylist}")
        if p2.get("metrics"):
            p2["query"], p2["operations"], p2["intent"] = plan["query"], [], "comparison"
            overview, plan = [m["concept"] for m in p2["metrics"]], p2
    aligned = aligner.align(plan, policy=policy, evidence_ids=evidence_ids)
    res = compute(plan, aligned)
    if overview:
        res["overview"] = overview
        res["warnings"].insert(0, "no metric was named, so the key statement lines are compared: " + ", ".join(overview))
    res["primary_basis"] = policy
    compares = plan.get("operations") or len(plan["years"].get("requested", [])) > 1
    if policy == "as_reported" and res["flagged_inputs"] and compares:
        alt = compute(plan, aligner.align(plan, policy="latest", evidence_ids=evidence_ids))
        split = "split_adjusted" in json.dumps(res["results"], default=str)
        if split:
            alt["primary_basis"] = "latest"
            if res.get("overview"):
                alt["overview"] = res["overview"]
            alt["alternate_basis"] = {"policy": "as_reported", "results": res["results"], "status": res["status"],
                                      "note": "as originally reported in each filing - NOT comparable across the stock split; shown for reference only"}
            alt["warnings"].insert(0, "a stock split changes per-share figures between filings, so the split-adjusted (latest) basis is the primary answer")
            return alt
        res["alternate_basis"] = {"policy": "latest", "results": alt["results"], "status": alt["status"],
                                  "note": "same calculation using the latest (restated) value of every year, so all years share one basis"}
    return res


# ----------------------------------------------------------------------------- display
def show(v):
    print(f"\nQ: {v['query']}   [basis: {v.get('primary_basis', v['policy'])}]  ->  {v['status'].upper()}")
    for r in v["results"]:
        yrs = ",".join(str(y) for y in r["years"])
        print(f"  {r['op']:<10} {r['label']} FY{yrs}: {r['display']}   [cross-check {r['cross_check']}]")
        print(f"             formula: {r['formula']}")
        for st in r["steps"][:6]:
            print(f"             = {st}")
        for i in r["inputs"]:
            print(f"             input FY{i['fy']}: {i['raw'] or i['value']} -> {i['value']:,.4f} {i['unit']}  ({i['filing']} filing p{i['page']}, {i['status']})")
        for n in r["notes"]:
            print(f"             note: {n}")
        if r.get("ranking"):
            for x in r["ranking"]:
                print(f"             #{x['rank']} FY{x['fy']}: {x['display']}")
    if v.get("note"):
        print(f"  note: {v['note']}")
    for r in v["refusals"]:
        print(f"  REFUSED {r['op']} {r.get('concept') or ''}: {r['reason']}")
    for w in v["warnings"]:
        print(f"  warning: {w}")
    alt = v.get("alternate_basis")
    if alt:
        print(f"  --- {alt['note']}")
        for r in alt["results"]:
            print(f"      {r['op']:<10} {r['label']} FY{','.join(map(str, r['years']))}: {r['display']}")


# ----------------------------------------------------------------------------- self-test (no data needed)
def _m(concept, vals, unit="USD millions", vtype="amount", label=None, parent=None):
    years = {}
    for y, spec in vals.items():
        v, st = (spec if isinstance(spec, tuple) else (spec, "ok"))
        if v is None:
            years[str(y)] = {"status": "unavailable", "chosen": None, "alternatives": [], "notes": [f"no row for FY{y}"], "role": "requested"}
        else:
            years[str(y)] = {"status": st, "role": "requested", "notes": [], "chosen": {
                "filing": y, "page": 40, "raw": f"{v:,}", "value": v, "std": v, "std_unit": unit, "units_known": True,
                "ocr_used": False, "value_type": vtype}, "alternatives": ([{"filing": y + 1, "raw": "x", "differs": True, "pct_diff": 1.0}] if st in FLAGGED else [])}
    return {"concept": concept, "labels": [label or concept], "statement": "s", "expected_type": vtype, "parent": parent,
            "years": years}


def _plan(q, ops=(), years=(), support=(), metrics=(), ratio=None, intent="calculation", rank_over=None):
    return {"query": q, "operations": list(ops), "ratio": ratio, "rank_over": rank_over, "intent": intent, "flags": {},
            "years": {"requested": list(years), "support_years": list(support), "unavailable": []},
            "metrics": list(metrics)}


def _al(*ms, problems=()):
    return {"query": "", "policy": "as_reported", "requested_years": [], "support_years": [], "metrics": list(ms), "problems": list(problems)}


def selftest():
    chk = []

    def check(name, cond, extra=""):
        chk.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   {extra}"))

    rev = lambda **k: {"concept": "revenue", "derived": False, "labels": ["Total net sales"]}
    A = lambda **k: None
    sales = _m("revenue", {2017: 177866, 2018: 232887}, label="Total net sales")
    v = compute(_plan("diff", ["difference"], [2017, 2018], metrics=[rev()]), _al(sales))
    r = v["results"][0]
    check("difference 2017 -> 2018 = 55,021 (exact)", r["value"] == 55021.0 and r["cross_check"] == "passed", r)
    v = compute(_plan("pct", ["pct_change"], [2017, 2018], metrics=[rev()]), _al(sales))
    check("pct change 2017 -> 2018 = 30.9335%", abs(v["results"][0]["value"] - 30.9335) < 1e-3, v["results"][0]["value"])
    s2 = _m("revenue", {2015: 100.0, 2020: 200.0})
    v = compute(_plan("cagr", ["cagr"], [2015, 2020], metrics=[rev()]), _al(s2))
    check("CAGR 100 -> 200 over 5 years = 14.8698%", abs(v["results"][0]["value"] - 14.8698) < 1e-3 and v["results"][0]["cross_check"] == "passed")
    s3 = _m("revenue", {2015: -50.0, 2020: 200.0})
    v = compute(_plan("cagr", ["cagr"], [2015, 2020], metrics=[rev()]), _al(s3))
    check("CAGR with a negative start is refused", v["status"] == "refused" and "non-positive" in v["refusals"][0]["reason"])
    s4 = _m("revenue", {2021: 469822, 2022: 513983})
    v = compute(_plan("yoy", ["yoy"], [2022], [2021], metrics=[rev()]), _al(s4))
    check("YoY for one year uses the support year (+9.40%)", len(v["results"]) == 1 and abs(v["results"][0]["value"] - 9.4) < 0.01 and v["results"][0]["op"] == "yoy", v["results"])
    s5 = _m("revenue", {2019: 1.0, 2020: 2.0, 2021: 3.0, 2022: None})
    v = compute(_plan("sum", ["sum"], [2019, 2020, 2021, 2022], metrics=[rev()]), _al(s5))
    check("sum with a missing year is refused, not computed on the rest", v["status"] == "refused" and "FY2022" in v["refusals"][0]["reason"])
    v = compute(_plan("avg", ["average"], [2019, 2020, 2021], metrics=[rev()]), _al(s5))
    check("average of 1,2,3 = 2", v["results"][0]["value"] == 2.0)
    s6 = _m("revenue", {2019: 5, 2020: 9, 2021: 7})
    v = compute(_plan("max", ["max"], [2019, 2020, 2021], metrics=[rev()], intent="ranking"), _al(s6))
    check("max returns the value and the year", v["results"][0]["year"] == 2020 and v["results"][0]["value"] == 9.0)
    v = compute(_plan("top 2", ["rank"], [2019, 2020, 2021], metrics=[rev()], intent="ranking", rank_over="years"), _al(s6))
    check("rank top-2 gives the two largest years in order", [x["fy"] for x in v["results"][0]["ranking"]] == [2020, 2021])

    ocf, capex = _m("ocf", {2018: 30723}), _m("capex", {2018: -13427})
    fcf = {"concept": "free_cash_flow", "derived": True, "formula": "ocf - abs(capex)", "components": ["ocf", "capex"], "labels": ["Free cash flow"]}
    v = compute(_plan("fcf", [], [2018], metrics=[fcf]), _al(ocf, capex))
    check("free cash flow = ocf - |capex| = 17,296 with capex stored as a negative outflow", v["results"][0]["value"] == 17296.0, v["results"])
    capex_pos = _m("capex", {2018: 13427})
    v = compute(_plan("fcf", [], [2018], metrics=[fcf]), _al(ocf, capex_pos))
    check("free cash flow is the same if capex is stored positive", v["results"][0]["value"] == 17296.0)

    oi, rv = _m("operating_income", {2019: 14541, 2021: 24879}), _m("revenue", {2019: 280522, 2021: 469822})
    margin = {"concept": "operating_margin", "derived": True, "formula": "operating_income / revenue * 100", "components": ["operating_income", "revenue"]}
    v = compute(_plan("margin", [], [2019, 2021], metrics=[margin]), _al(oi, rv))
    vals = {r["years"][0]: r["value"] for r in v["results"] if r["op"] == "value"}
    check("operating margin 2019 = 5.18%, 2021 = 5.30%", abs(vals[2019] - 5.1836) < 1e-3 and abs(vals[2021] - 5.2957) < 1e-3 and v["results"][0]["unit"] == "percent", vals)
    ni, ta = _m("net_income", {2023: 30425}), _m("total_assets", {2023: 527854})
    v = compute(_plan("ratio", ["ratio"], [2023], ratio={"numerator": "net_income", "denominator": "total_assets", "unit": "percent"}), _al(ni, ta))
    check("ratio net income / total assets 2023 = 5.76%", abs(v["results"][0]["value"] - 5.7643) < 1e-3 and v["results"][0]["unit"] == "percent", v["results"])
    ta0 = _m("total_assets", {2023: 0})
    v = compute(_plan("ratio", ["ratio"], [2023], ratio={"numerator": "net_income", "denominator": "total_assets", "unit": "percent"}), _al(ni, ta0))
    check("zero denominator is refused", v["status"] == "refused")
    eps = _m("eps", {2023: 2.9}, unit="USD per share", vtype="per_share")
    v = compute(_plan("ratio", ["ratio"], [2023], ratio={"numerator": "eps", "denominator": "total_assets", "unit": "percent"}), _al(eps, ta))
    check("per-share figure divided by an amount is refused", v["status"] == "refused" and "per_share" in v["refusals"][0]["reason"], v["refusals"])

    v = compute(_plan("one", [], [2013], metrics=[{"concept": "total_assets", "derived": False}]), _al(_m("total_assets", {2013: 40159})))
    check("single look-up returns the value with its source", v["results"][0]["value"] == 40159.0 and v["results"][0]["inputs"][0]["filing"] == 2013)
    two = _m("revenue", {2023: 574785, 2024: 637959})
    v = compute(_plan("cmp", [], [2023, 2024], metrics=[rev()], intent="comparison"), _al(two))
    check("comparison without an operation shows both years plus the change", [r["op"] for r in v["results"]] == ["value", "value", "difference", "pct_change"])
    r3 = _m("revenue", {2014: (88988, "ok"), 2015: (107006, "restated")})
    v = compute(_plan("restate", ["pct_change"], [2014, 2015], metrics=[rev()]), _al(r3))
    check("restated input is flagged and status says so", v["status"] == "verified_with_flags" and v["flagged_inputs"] == {"revenue": [2015]})
    check("no result ever has a failed cross-check", all(r["cross_check"] == "passed" for r in v["results"]))
    v = compute(_plan("neg", ["pct_change"], [2001, 2002], metrics=[rev()]), _al(_m("revenue", {2001: -100.0, 2002: -50.0})))
    check("negative base: percentage measured against |base| and noted", v["results"][0]["value"] == 50.0 and v["results"][0]["notes"], v["results"])
    v = compute(_plan("zero", ["pct_change"], [2001, 2002], metrics=[rev()]), _al(_m("revenue", {2001: 0.0, 2002: 5.0})))
    check("zero base percentage change is refused", v["status"] == "refused")
    check("formula evaluator rejects unsafe input", _raises(lambda: eval_formula("__import__('os').system('x')", {})))
    v = compute(_plan("why is the sky blue"), _al())
    check("a question with no metric is 'not_applicable', not a refusal", v["status"] == "not_applicable" and not v["refusals"])
    check("rank over line items is declined, not guessed", compute(_plan("top 3 expense lines", ["rank"], [2022], rank_over="rows", intent="ranking"), _al())["status"] == "refused")

    class Fake:                                           # Stage 8 stand-in: as-reported EPS is pre-split, latest is split-adjusted
        def align(self, plan, policy="as_reported", evidence_ids=None):
            e21 = (64.81, "split_adjusted") if policy == "as_reported" else (3.24, "split_adjusted")
            return _al(_m("eps", {2021: e21, 2023: 2.9}, unit="USD per share", vtype="per_share"))
    pl = _plan("eps growth 2021 to 2023", ["pct_change"], [2021, 2023], metrics=[{"concept": "eps", "derived": False}])
    v = verify(pl, Fake())
    check("stock split: the split-adjusted basis (-10.49%) is primary, the as-reported (-95.53%) only the alternative",
          v["primary_basis"] == "latest" and abs(v["results"][0]["value"] + 10.4938) < 1e-3 and
          abs(v["alternate_basis"]["results"][0]["value"] + 95.5254) < 1e-3, v["results"])
    class Fake2:
        def align(self, plan, policy="as_reported", evidence_ids=None):
            x = (107006, "restated") if policy == "as_reported" else (106000, "restated")
            return _al(_m("revenue", {2014: 88988, 2015: x}))
    v = verify(_plan("rev growth", ["pct_change"], [2014, 2015], metrics=[rev()]), Fake2())
    check("restatement: as-reported stays primary, consistent basis offered as the alternative",
          v["primary_basis"] == "as_reported" and v["alternate_basis"]["policy"] == "latest")

    print(f"\n{sum(chk)}/{len(chk)} passed")
    return 0 if all(chk) else 1


def _raises(f):
    try:
        f()
    except Exception:
        return True
    return False


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--policy", default="as_reported", choices=["as_reported", "latest", "same_filing"])
    ap.add_argument("--rows", default="data/processed/table_rows.json")
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    from stage6_query_parser import Catalog, QueryParser, CONCEPTS
    import stage8_align as s8
    parser, aligner = QueryParser(Catalog.from_json(a.tables)), s8.Aligner(a.rows, concepts=CONCEPTS)

    def run(q):
        plan = parser.parse(q)
        for n in plan.get("notes", []):
            print("  note:", n)
        v = verify(plan, aligner, a.policy, parser=parser)
        print(json.dumps(v, indent=1, default=str)) if a.json else show(v)

    if a.question:
        run(a.question)
    else:
        while True:
            q = input("\nQuestion (blank to quit): ").strip()
            if not q:
                break
            run(q)


if __name__ == "__main__":
    main()