"""
Stage 5: turn the validated tables from Stage 4 into a canonical semantic form and index them.

Company-agnostic. For every table that passed validation (no FAIL), every numeric row becomes one record:

  <company> | 10-K filing <doc_year> | <table title> | page <p> | units: <u> | section: <s> | line item: <label> |
  FY2020: 58,517; FY2021: 75,111; FY2022: 84,299

so the FINANCIAL YEAR of every value travels with the number. One extra record per table (title, years, line items)
lets whole-table questions find the right table. Records are embedded with BGE (same model as the text index) into a
SEPARATE ChromaDB collection, `table_rows`; the structured values stay in table_rows.json for exact lookup in the
later alignment / verification stages.

Chroma metadata per record: kind (row|table), company, doc_year, statement, table_id, page, units, section,
row_label, column_years, latest_fy, table_status (ok|warn), ocr_used, values_json, raw_json, and one boolean
flag fy<YEAR> = True for every financial year the record covers (so a year filter is a plain metadata filter).

Usage:
    python stage5_index_tables.py              # build records + embed (resumable)
    python stage5_index_tables.py --test       # sanity queries
    python stage5_index_tables.py --rebuild    # drop and rebuild the collection
"""
import argparse
import json
import re
import time
from pathlib import Path

from canon import canonical                      # shared label canonicalisation (same labels as Stage 6)
from stage3_embed_index import MODEL_NAME, MAX_TOKENS, get_client, load_model

COLLECTION = "table_rows"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def table_status(tbl):
    issues = tbl.get("issues", [])
    if any(i.startswith("FAIL") for i in issues):
        return "fail"
    return "warn" if any(i.startswith("WARN") for i in issues) else "ok"


def clean_label(label: str) -> str:
    label = re.sub(r"(?:\s*[.\u00cf_]){3,}", " ", label)
    label = re.sub(r"[\u00fe\u00df]", "", label)               # legacy-font residue (1997 filing)
    toks = label.split()
    if len(toks) >= 4 and sum(len(t) == 1 for t in toks) / len(toks) > 0.6:
        label = "".join(toks)                                    # letter-spaced heading "A S S E T S" -> "ASSETS"
    return re.sub(r"\s+", " ", label).strip(" :")


# ---- unit standardisation: the same financial year is stored in thousands in old filings, millions in new ones
UNIT_SCALE = {"thousands": 0.001, "millions": 1.0, "billions": 1000.0}
SHARES_RE = re.compile(r"shares used|weighted[- ]average shares|number of shares|shares outstanding", re.I)
PER_SHARE_RE = re.compile(r"per (common )?share", re.I)


def unit_scale(tbl):
    m = re.search(r"(thousands|millions|billions)", (tbl.get("units") or "").lower())
    return UNIT_SCALE[m.group(1)] if m else None


def value_type(label: str, section: str, raw=None, scale=None) -> str:
    """amount | shares | per_share. In some years 'per share' appears only in the section heading above the row."""
    ctx = label + " " + section
    if SHARES_RE.search(ctx):
        return "shares"
    if PER_SHARE_RE.search(ctx):
        return "per_share"
    # safety net: dollar amounts in a thousands/millions table never carry decimals in the old filings
    if raw and scale is not None and scale < 1 and all(re.search(r"\d\.\d", v) for v in raw.values() if re.search(r"\d", v)):
        return "per_share"
    return "amount"


def header_text(tbl):
    units = tbl.get("units") or "units not stated"
    return (f"{tbl['company']} | 10-K filing {tbl['doc_year']} | {tbl['title'].title() if tbl['title'].isupper() else tbl['title']}"
            f" | page {tbl['page']} | {units}")


def base_meta(tbl, kind):
    years = [str(y) for y in tbl["column_years"]]
    meta = {"kind": kind, "company": tbl["company"], "doc_year": int(tbl["doc_year"]), "statement": tbl["statement"],
            "table_id": tbl["table_id"], "page": int(tbl["page"]), "units": tbl.get("units") or "",
            "column_years": ",".join(years), "latest_fy": int(max(years)), "n_years": len(years),
            "table_status": table_status(tbl), "ocr_used": bool(tbl.get("ocr_used", False)),
            "extractor": tbl.get("extractor", ""),
            "units_known": unit_scale(tbl) is not None, "unit_scale_to_millions": unit_scale(tbl) or 1.0}
    for y in years:
        meta[f"fy{y}"] = True
    return meta


def row_records(tbl):
    recs = []
    years = [str(y) for y in tbl["column_years"]]
    head = header_text(tbl)
    for i, r in enumerate(tbl["rows"]):
        label, section, how = canonical(r)           # repairs merged headings and empty labels (e.g. net EPS rows)
        if r["kind"] == "heading" or not r["values"] or not label:
            continue
        shown = "; ".join(f"FY{y}: {r['raw'][y]}" for y in years if y in r["raw"])
        text = f"{head} | " + (f"section: {section} | " if section else "") + f"line item: {label} | {shown}"
        meta = base_meta(tbl, "row")
        scale = unit_scale(tbl)
        vtype = value_type(label, section, r["raw"], scale)
        std = {}
        if scale is not None:      # amounts and share counts -> millions; per-share amounts stay as reported
            k = 1.0 if vtype == "per_share" else scale
            std = {y: round(r["values"][y] * k, 6) for y in years if y in r["values"]}
        meta.update({"section": section, "row_label": label, "row_index": i,
                     "row_kind": r["kind"], "value_type": vtype, "label_repair": how,
                     "std_unit": {"amount": "USD millions", "shares": "millions of shares", "per_share": "USD per share"}[vtype] if std else "",
                     "values_std_json": json.dumps(std),
                     "values_json": json.dumps({y: r["values"][y] for y in years if y in r["values"]}),
                     "raw_json": json.dumps({y: r["raw"][y] for y in years if y in r["raw"]})})
        recs.append({"id": f"{tbl['table_id']}_r{i:03d}", "text": text, "meta": meta})
    return recs


def table_record(tbl, model):
    labels = [canonical(r)[0] for r in tbl["rows"] if r["values"] and canonical(r)[0]]
    head = header_text(tbl) + f" | financial years {tbl['column_years'][0]} to {tbl['column_years'][-1]} | line items: "
    while True:   # keep the table-level text inside the embedding model's token limit
        text = head + "; ".join(labels)
        if len(model.tokenizer(text, add_special_tokens=True)["input_ids"]) <= 480 or len(labels) <= 5:
            break
        labels = labels[:-4]
    return {"id": f"{tbl['table_id']}_table", "text": text, "meta": base_meta(tbl, "table")}


def build_records(tables, model):
    records, excluded = [], []
    for tbl in tables:
        if table_status(tbl) == "fail":
            excluded.append({"table_id": tbl["table_id"], "page": tbl["page"],
                             "reasons": [i for i in tbl.get("issues", []) if i.startswith("FAIL")]})
            continue
        records.append(table_record(tbl, model))
        records.extend(row_records(tbl))
    return records, excluded


def build(args):
    tables = json.loads(Path(args.tables).read_text(encoding="utf-8"))
    model = load_model()
    records, excluded = build_records(tables, model)
    Path(args.out).write_text(json.dumps(records, indent=1, ensure_ascii=False), encoding="utf-8")
    Path(args.excluded).write_text(json.dumps(excluded, indent=1, ensure_ascii=False), encoding="utf-8")
    n_rows = sum(1 for r in records if r["meta"]["kind"] == "row")
    print(f"{len(tables)} tables -> {len(tables) - len(excluded)} indexed ({n_rows} row records + "
          f"{len(records) - n_rows} table records); excluded for failed validation: {len(excluded)}")
    for e in excluded:
        print(f"  EXCLUDED {e['table_id']} (page {e['page']}): {e['reasons'][0][:90]}")
    lens = sorted(len(model.tokenizer(r["text"], add_special_tokens=True)["input_ids"]) for r in records)
    over = sum(1 for n in lens if n > MAX_TOKENS)
    print(f"Token lengths: median={lens[len(lens) // 2]} max={lens[-1]} | over {MAX_TOKENS}: {over}")
    if args.stats_only:
        return

    client = get_client(args.db)
    if args.rebuild:
        try:
            client.delete_collection(COLLECTION)
            print("Dropped existing collection.")
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine", "embedding_model": MODEL_NAME})
    by_id = {r["id"]: r for r in records}
    existing = set(col.get(include=[])["ids"])
    ids_list, stale, refresh = list(existing), [], []
    for i in range(0, len(ids_list), 1000):   # resume safety: text changed, or record no longer exists
        got = col.get(ids=ids_list[i:i + 1000], include=["documents", "metadatas"])
        for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            if cid not in by_id or doc != by_id[cid]["text"]:
                stale.append(cid)
            elif any(meta.get(k) != v for k, v in by_id[cid]["meta"].items()):
                refresh.append(cid)           # same text, new/changed metadata -> update in place
    for i in range(0, len(refresh), 1000):
        part = refresh[i:i + 1000]
        col.update(ids=part, metadatas=[by_id[c]["meta"] for c in part])
    if refresh:
        print(f"Refreshed metadata of {len(refresh)} records in place (no re-embedding needed).")
    if stale:
        col.delete(ids=stale)
        existing -= set(stale)
        print(f"Removed {len(stale)} stale entries; they will be re-embedded where still valid.")
    todo = [r for r in records if r["id"] not in existing]
    print(f"Already indexed: {len(existing)} | to embed: {len(todo)}")
    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        part = todo[i:i + args.batch]
        embs = model.encode([r["text"] for r in part], batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        col.add(ids=[r["id"] for r in part], embeddings=embs.tolist(),
                documents=[r["text"] for r in part], metadatas=[r["meta"] for r in part])
        done = i + len(part)
        rate = done / (time.time() - t0)
        print(f"  {done}/{len(todo)} embedded ({rate:.1f}/s, ~{(len(todo) - done) / rate / 60:.1f} min left)", flush=True)
    print(f"Done. Collection '{COLLECTION}' now has {col.count()} records -> {args.db}")


def search(col, model, query, k=5, where=None):
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    res = col.query(query_embeddings=q.tolist(), n_results=k, where=where)
    return [{"id": res["ids"][0][i], "sim": 1 - res["distances"][0][i], "meta": res["metadatas"][0][i],
             "text": res["documents"][0][i]} for i in range(len(res["ids"][0]))]


def run_tests(args):
    model = load_model()
    col = get_client(args.db).get_collection(COLLECTION)
    print(f"Collection '{COLLECTION}' has {col.count()} records\n")
    tests = [
        ("What was net sales in 2018?", None),
        ("What was net sales in 2018?", {"$and": [{"kind": "row"}, {"fy2018": True}]}),
        ("What was operating income in 2021?", {"$and": [{"kind": "row"}, {"fy2021": True}]}),
        ("Total assets at the end of 2015", {"$and": [{"kind": "row"}, {"fy2015": True}]}),
        ("Net cash provided by operating activities in 2019", None),
        ("income statement", {"kind": "table"}),
    ]
    for q, where in tests:
        print("=" * 100)
        print(f"Q: {q}   [where={json.dumps(where)}]")
        for r in search(col, model, q, 5, where):
            m = r["meta"]
            label = m.get("row_label") or "(whole table)"
            vals = json.loads(m.get("values_std_json") or m.get("values_json") or "{}")
            print(f"  {r['sim']:.3f} | filing {m['doc_year']} | {m['statement']:13} p{m['page']} | {label[:40]:40} | {vals}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="data/processed/all_tables.json")
    ap.add_argument("--out", default="data/processed/table_rows.json")
    ap.add_argument("--excluded", default="data/processed/tables_excluded.json")
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    run_tests(args) if args.test else build(args)


if __name__ == "__main__":
    main()