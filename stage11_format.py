"""
Stage 11: turn the pipeline result (Stages 6-10) into a display payload: tables, a chart spec, calculation steps, sources,
flags.  Pure formatting - it never calculates; every number comes from the verified series / results of Stage 9.

  payload = format_response(pipeline_result)      # dict, JSON-serialisable, used by the UI (Stage 14)
  print(to_markdown(payload))                      # same content as Markdown (CLI, logs, evaluation reports)
  to_csv(payload["tables"][0])                     # one table as CSV text

  python stage11_format.py --selftest
"""
import csv
import io
import re
import sys

FLAG_NOTE = "* restated or split-adjusted in a later filing; see the alternate basis"


# ----------------------------------------------------------------------------- number formatting
def fmt_value(v, unit):
    if v is None:
        return "-"
    if unit == "percent":
        return f"{v:,.2f}%"
    if unit == "x":
        return f"{v:,.4f}x"
    if unit and "per share" in unit:
        return f"{v:,.2f}"
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}"


def short_unit(unit):
    return {"USD millions": "USD m", "USD thousands": "USD k", "USD per share": "USD/share", "millions of shares": "m shares"}.get(unit, unit or "")


def period(years):
    if not years:
        return ""
    return f"FY{years[0]}" if len(years) == 1 else f"FY{years[0]} to FY{years[-1]}" if len(years) == 2 or years == list(range(years[0], years[-1] + 1)) else \
        ", ".join(f"FY{y}" for y in years[:6]) + ("..." if len(years) > 6 else "")


# ----------------------------------------------------------------------------- tables
def values_table(ver):
    series = ver.get("series") or {}
    years = sorted({int(y) for s in series.values() for y in s["values"]})
    if not years:
        return None
    flagged = {(c, int(y)) for c, ys in (ver.get("flagged_inputs") or {}).items() for y in ys}
    rows = []
    for c, s in series.items():
        cells = []
        for y in years:
            v = s["values"].get(str(y))
            cells.append(fmt_value(v, s["unit"]) + ("*" if v is not None and (c, y) in flagged else ""))
        rows.append([s["label"], short_unit(s["unit"])] + cells)
    note = FLAG_NOTE if flagged else None
    missing = [f"{s['label']} FY{y}: {why}" for s in series.values() for y, why in s["missing"].items()]
    return {"title": "Values by financial year", "columns": ["Line item", "Unit"] + [f"FY{y}" for y in years], "rows": rows,
            "note": note, "missing": missing}


def calc_table(results, title="Calculations"):
    rows = [[r["label"], r["op"].replace("_", " "), period(r["years"]), r["display"], r["formula"]] for r in results if r["op"] != "value"]
    return {"title": title, "columns": ["Line item", "Calculation", "Period", "Result", "Formula"], "rows": rows} if rows else None


# ----------------------------------------------------------------------------- chart
def chart_spec(ver, plan):
    series = ver.get("series") or {}
    years = sorted({int(y) for s in series.values() for y in s["values"]})
    if len(years) < 3:
        return None
    unit = next(iter(series.values()))["unit"]
    same = [(c, s) for c, s in series.items() if s["unit"] == unit and sum(1 for y in years if str(y) in s["values"]) >= 3][:4]
    if not same:
        return None
    return {"kind": "line" if len(years) > 4 else "bar", "title": ", ".join(s["label"] for _, s in same) if len(same) <= 2 else "Selected metrics",
            "unit": short_unit(unit), "x": years,
            "series": [{"name": s["label"], "values": [s["values"].get(str(y)) for y in years]} for _, s in same]}


# ----------------------------------------------------------------------------- sources
def sources_of(ver, answer):
    out, seen = [], set()
    for r in (ver.get("results") or []) + ((ver.get("alternate_basis") or {}).get("results") or []):
        for i in r.get("inputs", []):
            key = (i["filing"], i["page"], i["concept"], i["fy"])
            if key in seen or i.get("page") is None:
                continue
            seen.add(key)
            out.append({"title": f"{i['filing']} 10-K, page {i['page']}", "detail": f"{i['concept'].replace('_', ' ')} FY{i['fy']}: "
                        f"{i['raw'] or fmt_value(i['value'], i['unit'])}" + (f" ({i['status']})" if i.get("status") not in (None, "ok") else "")})
    for c in (answer or {}).get("citations", []):
        if c["kind"] in ("text", "table"):
            m = re.match(r"\[(S\d+)\] \((\d{4}) 10-K, p(\w+)[^)]*\)\s*(.*)", c["text"], re.S)
            if m:
                out.append({"title": f"{m.group(2)} 10-K, page {m.group(3)}", "detail": re.sub(r"\s+", " ", m.group(4))[:280], "tag": m.group(1)})
    return out[:14]


# ----------------------------------------------------------------------------- payload
def format_response(res):
    ver, ans, plan, ret = res.get("verification") or {}, res.get("answer") or {}, res.get("plan") or {}, res.get("retrieval") or {}
    tables = []
    t = values_table(ver) if ver.get("results") or ver.get("series") else None
    if t:
        tables.append(t)
    c = calc_table(ver.get("results") or [])
    if c:
        tables.append(c)
    alt = ver.get("alternate_basis")
    if alt:
        a = calc_table(alt["results"], f"Calculations on the alternate basis ({alt['policy']})")
        if a:
            a["note"] = alt["note"]
            tables.append(a)
    flags = []
    if ver.get("flagged_inputs"):
        flags.append("Some inputs differ between filings (restated or split-adjusted).")
    flags += [w for w in ver.get("warnings", []) if "OCR" not in w][:5]
    flags += [f"Not calculated: {r['op']}{' ' + r['concept'] if r.get('concept') else ''} - {r['reason']}" for r in ver.get("refusals", [])]
    ocr = sorted({y for w in ver.get("warnings", []) if "OCR" in w for y in re.findall(r"FY(\d{4})", w)})
    if ocr:
        flags.append("Values for " + ", ".join(f"FY{y}" for y in ocr) + " were read by OCR from scanned pages; check them against the filing.")
    return {
        "question": res.get("question"), "interpreted_as": res.get("interpreted"),
        "answer": {"text": ans.get("body", ans.get("answer", "")), "full_text": ans.get("answer", ""), "mode": ans.get("mode"),
                   "warnings": ans.get("warnings", []), "llm_status": ans.get("llm_status"), "attempts": ans.get("attempts", [])},
        "status": ver.get("status", "not_applicable"),
        "primary_basis": ver.get("primary_basis"),
        "tables": tables,
        "chart": chart_spec(ver, plan) if ver.get("series") else None,
        "calculations": [{"label": r["label"], "op": r["op"], "period": period(r["years"]), "display": r["display"], "formula": r["formula"],
                          "steps": r["steps"], "cross_check": r["cross_check"], "notes": r["notes"]} for r in ver.get("results", []) if r["op"] != "value"],
        "refusals": ver.get("refusals", []),
        "flags": flags,
        "sources": sources_of(ver, ans),
        "plan": {"intent": plan.get("intent"), "metrics": [m["concept"] for m in plan.get("metrics", [])], "operations": plan.get("operations"),
                 "years": (plan.get("years") or {}).get("requested"), "notes": plan.get("notes", [])},
    }


# ----------------------------------------------------------------------------- exports
def _md_table(t):
    head = "| " + " | ".join(t["columns"]) + " |\n|" + "|".join("---" for _ in t["columns"]) + "|\n"
    return f"**{t['title']}**\n\n" + head + "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in t["rows"]) + \
        (f"\n\n_{t['note']}_" if t.get("note") else "")


def to_markdown(p):
    parts = [f"### {p['question']}", p["answer"]["text"]]
    parts += [_md_table(t) for t in p["tables"]]
    if p["flags"]:
        parts.append("**Flags**\n" + "\n".join(f"- {f}" for f in p["flags"]))
    if p["sources"]:
        parts.append("**Sources**\n" + "\n".join(f"- {s['title']}: {s['detail']}" for s in p["sources"]))
    return "\n\n".join(parts)


def to_csv(table):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(table["columns"])
    w.writerows(table["rows"])
    return buf.getvalue()


# ----------------------------------------------------------------------------- self-test
def _res():
    r1 = {"op": "pct_change", "label": "Total net sales", "years": [2017, 2018], "display": "30.93%", "formula": "(FY2018 - FY2017) / |FY2017| x 100",
          "steps": ["(232,887 - 177,866) / |177,866| x 100 = 30.933962%"], "cross_check": "passed", "notes": [], "value": 30.93, "unit": "percent",
          "inputs": [{"concept": "revenue", "fy": 2017, "value": 177866.0, "unit": "USD millions", "filing": 2017, "page": 48, "raw": "177,866", "status": "ok"},
                     {"concept": "revenue", "fy": 2018, "value": 232887.0, "unit": "USD millions", "filing": 2018, "page": 45, "raw": "232,887", "status": "restated"}]}
    ver = {"status": "verified_with_flags", "results": [r1], "refusals": [], "warnings": ["revenue FY2018: value read by OCR"], "primary_basis": "as_reported",
           "flagged_inputs": {"revenue": [2018]},
           "series": {"revenue": {"unit": "USD millions", "label": "Total net sales", "values": {"2017": 177866.0, "2018": 232887.0}, "missing": {}}},
           "alternate_basis": {"policy": "latest", "note": "latest basis", "results": [dict(r1, display="30.50%")]}}
    ans = {"answer": "Net sales grew 30.93% [R1].\n\nNote: x", "body": "Net sales grew 30.93% [R1].", "mode": "llm", "warnings": [],
           "citations": [{"tag": "S1", "kind": "text", "text": "[S1] (2018 10-K, p30, PART II > Item 7) Net sales increased 31% in 2018."}]}
    return {"question": "pct change", "answer": ans, "verification": ver, "plan": {"intent": "calculation", "metrics": [{"concept": "revenue"}], "operations": ["pct_change"],
                                                                              "years": {"requested": [2017, 2018]}, "notes": []}, "retrieval": {}}


def selftest():
    chk = []

    def check(name, cond, extra=""):
        chk.append(bool(cond))
        print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"   {extra}"))

    check("number formats", fmt_value(177866.0, "USD millions") == "177,866" and fmt_value(30.9339, "percent") == "30.93%" and fmt_value(3.24, "USD per share") == "3.24"
          and fmt_value(-241.0, "USD millions") == "-241" and fmt_value(None, "x") == "-")
    check("period labels", period([2017]) == "FY2017" and period([2017, 2018]) == "FY2017 to FY2018" and period([2015, 2016, 2017, 2018]) == "FY2015 to FY2018")
    p = format_response(_res())
    t0 = p["tables"][0]
    check("values table has one row per series and one column per year", t0["columns"] == ["Line item", "Unit", "FY2017", "FY2018"] and t0["rows"][0] == ["Total net sales", "USD m", "177,866", "232,887*"], t0)
    check("restated cell is starred and a footnote is given", t0["note"] == FLAG_NOTE)
    check("calculation table lists the formula and result", p["tables"][1]["rows"][0][3] == "30.93%" and "x 100" in p["tables"][1]["rows"][0][4])
    check("alternate basis table present and labelled", "alternate basis" in p["tables"][2]["title"])
    check("OCR warnings are merged into one flag", sum("OCR" in f for f in p["flags"]) == 1)
    check("sources combine filing pages and passages", any(s["title"] == "2017 10-K, page 48" for s in p["sources"]) and any(s.get("tag") == "S1" for s in p["sources"]), p["sources"])
    check("two years -> no chart", p["chart"] is None)
    check("UI answer text excludes the trailer (flags/sources are shown separately)", p["answer"]["text"] == "Net sales grew 30.93% [R1]." and "Note: x" in p["answer"]["full_text"])
    r2 = _res(); r2["verification"]["refusals"] = [{"op": "cagr", "concept": "revenue", "reason": "undefined"}]
    check("refusals become visible flags", any(f.startswith("Not calculated: cagr") for f in format_response(r2)["flags"]))
    big = _res()
    big["verification"]["series"]["revenue"]["values"] = {str(y): 100.0 * (y - 2010) for y in range(2011, 2020)}
    c = format_response(big)["chart"]
    check("9 years -> line chart with aligned values", c and c["kind"] == "line" and len(c["x"]) == 9 and len(c["series"][0]["values"]) == 9)
    check("markdown export contains the answer, a table and sources", all(x in to_markdown(p) for x in ("30.93%", "| Line item |", "**Sources**")))
    check("csv export round-trips", list(csv.reader(io.StringIO(to_csv(t0))))[1][2] == "177,866")
    na = format_response({"question": "risks", "answer": {"answer": "text", "mode": "template"}, "verification": {"status": "not_applicable", "results": [], "refusals": [], "warnings": []}, "plan": {}})
    check("text-only question -> no tables, no chart, still valid", na["tables"] == [] and na["chart"] is None and na["status"] == "not_applicable")
    print(f"\n{sum(chk)}/{len(chk)} passed")
    return 0 if all(chk) else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    print(__doc__)