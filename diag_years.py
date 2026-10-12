"""
Trace which financial years each stage uses for ONE question.

    python diag_years.py "percentage change in net sales from 2017 and 2018"
"""
import importlib
import sys
import traceback

q = " ".join(sys.argv[1:]) or "percentage change in net sales from 2017 and 2018"
print("QUESTION:", q)

# ---- Stage 6
p6 = importlib.import_module("stage6_query_parser")
print("\n[Stage 6] file:", p6.__file__)
parser = p6.QueryParser(p6.Catalog.from_json("data/processed/all_tables.json"))
plan = parser.parse(q)
y = plan["years"]
print("[Stage 6] year rules that fired:", [(s["kind"], s["text"], s["years"][:4] + (["..."] if len(s["years"]) > 4 else [])) for s in y["spec"]])
print("[Stage 6] resolved:", y["resolved"], "| support:", y["support_years"], "| fetch:", y["fetch_years"])
print("[Stage 6] operations:", plan["operations"], "| intent:", plan["intent"])
print("[Stage 6] metrics:", [m["concept"] for m in plan["metrics"]])

# ---- Stage 8
try:
    s8 = importlib.import_module("stage8_align")
    print("\n[Stage 8] file:", s8.__file__)
    al = s8.Aligner("data/processed/table_rows.json", concepts=p6.CONCEPTS)
    res = al.align(plan)
    for m in res["metrics"]:
        print(f"[Stage 8] {m['concept']}: " + ", ".join(
            f"FY{yr}={(e['chosen'] or {}).get('raw')}" + ("" if e["chosen"] else "(none)") for yr, e in m["years"].items()))
except Exception:
    traceback.print_exc()
    al = None

# ---- Stage 9
try:
    s9 = importlib.import_module("stage9_verify")
    print("\n[Stage 9] file:", s9.__file__)
    v = s9.verify(plan, al, "as_reported", evidence_ids=None, parser=parser)
    for r in v.get("results", []):
        print(f"[Stage 9] {r['op']} {r['label']} years={r['years']} -> {r['display']}")
        for i in r.get("inputs", []):
            print(f"            input FY{i['fy']} = {i['raw'] or i['value']} ({i['filing']} filing p{i['page']})")
    for r in v.get("refusals", []):
        print("[Stage 9] REFUSED:", r)
    for w in v.get("warnings", []):
        print("[Stage 9] warning:", w)
except Exception:
    traceback.print_exc()
