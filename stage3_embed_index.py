"""
Stage 3: embed text chunks with BAAI/bge-base-en-v1.5 and store them in a persistent ChromaDB collection.

Usage:
    python stage3_embed_index.py --stats-only      # token-length check, no embedding
    python stage3_embed_index.py                   # build/resume the index
    python stage3_embed_index.py --test            # run sanity queries against the index
    python stage3_embed_index.py --rebuild         # drop and rebuild the collection

The script can be stopped and restarted: chunks already in the collection are skipped.
"""
import argparse
import json
import time
from pathlib import Path

import chromadb
from chromadb.config import Settings

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # recommended for BGE queries
COLLECTION = "text_chunks"
MAX_TOKENS = 512  # BGE-base limit; longer inputs are silently truncated


def load_model():
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)
    model.max_seq_length = MAX_TOKENS
    return model


def to_metadata(c: dict) -> dict:
    """Chroma metadata must be str/int/float/bool (no None, no lists)."""
    return {
        "chunk_type": c["chunk_type"],
        "company": c["company"],
        "doc_year": int(c["doc_year"]),
        "filing_fiscal_year": int(c["filing_fiscal_year"]),
        "fy_mentioned": ",".join(str(y) for y in c["financial_years_mentioned"]),
        "part": c["part"] or "",
        "item": c["item"] or "",
        "item_title": c["item_title"] or "",
        "section_path": c["section_path"],
        "page_start": int(c["page_start"]),
        "page_end": int(c["page_end"]),
        "source_file": c["source_file"],
        "ocr_used": bool(c.get("ocr_used", False)),
        "numeric_ratio": float(c["numeric_ratio"]),
        "word_count": int(c["word_count"]),
    }


def get_client(db_path: str):
    return chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False))


def token_stats(model, chunks):
    lens = [len(model.tokenizer(c["embedding_text"], add_special_tokens=True)["input_ids"]) for c in chunks]
    over = [n for n in lens if n > MAX_TOKENS]
    lens_sorted = sorted(lens)
    print(f"Token lengths: median={lens_sorted[len(lens)//2]}  p95={lens_sorted[int(len(lens)*0.95)]}  max={lens_sorted[-1]}")
    print(f"Chunks over {MAX_TOKENS} tokens (will be truncated): {len(over)} of {len(lens)} ({100*len(over)/len(lens):.1f}%)")
    return lens


def build(args):
    chunks = json.loads(Path(args.chunks).read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")
    model = load_model()
    token_stats(model, chunks)
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
    existing = set(col.get(include=[])["ids"])
    # Resume safety: a stored chunk is stale if its text differs from the current corpus (or no longer exists).
    by_id = {c["chunk_id"]: c for c in chunks}
    ids_list, stale = list(existing), []
    for i in range(0, len(ids_list), 1000):
        got = col.get(ids=ids_list[i:i + 1000], include=["documents"])
        for cid, doc in zip(got["ids"], got["documents"]):
            if cid not in by_id or doc != by_id[cid]["text"]:
                stale.append(cid)
    if stale:
        col.delete(ids=stale)
        existing -= set(stale)
        print(f"Removed {len(stale)} stale entries (text changed since they were embedded); they will be re-embedded.")
    todo = [c for c in chunks if c["chunk_id"] not in existing]
    print(f"Already indexed: {len(existing)} | to embed: {len(todo)}")

    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        part = todo[i:i + args.batch]
        embs = model.encode([c["embedding_text"] for c in part], batch_size=32,
                            normalize_embeddings=True, show_progress_bar=False)
        col.add(ids=[c["chunk_id"] for c in part], embeddings=embs.tolist(),
                documents=[c["text"] for c in part], metadatas=[to_metadata(c) for c in part])
        done = i + len(part)
        rate = done / (time.time() - t0)
        eta = (len(todo) - done) / rate if rate else 0
        print(f"  {done}/{len(todo)} embedded  ({rate:.1f} chunks/s, ~{eta/60:.1f} min left)", flush=True)
    print(f"Done. Collection '{COLLECTION}' now has {col.count()} chunks -> {args.db}")


def search(col, model, query, k=5, doc_year=None):
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
    where = {"doc_year": doc_year} if doc_year else None
    res = col.query(query_embeddings=q.tolist(), n_results=k, where=where)
    out = []
    for i in range(len(res["ids"][0])):
        out.append({"id": res["ids"][0][i], "sim": 1 - res["distances"][0][i],
                    "meta": res["metadatas"][0][i], "text": res["documents"][0][i]})
    return out


def run_tests(args):
    model = load_model()
    col = get_client(args.db).get_collection(COLLECTION)
    print(f"Collection has {col.count()} chunks\n")
    tests = [
        ("What was total net sales?", 2019),
        ("What was total net sales?", 2020),
        ("What was operating income for the AWS segment?", 2021),
        ("What are the risks related to competition?", None),
        ("How much free cash flow did the company generate?", None),
        ("What was net sales in 2018?", None),   # no filter: should the 2018 filing win over others?
    ]
    for q, yr in tests:
        print("=" * 100)
        print(f"Q: {q}   [filter doc_year={yr}]")
        for r in search(col, model, q, k=5, doc_year=yr):
            m = r["meta"]
            print(f"  {r['sim']:.3f} | {m['doc_year']} | p{m['page_start']}-{m['page_end']} | {m['section_path'][:70]}")
            print(f"        {r['text'][:130].replace(chr(10), ' ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default="data/processed/all_chunks.json")
    ap.add_argument("--db", default="data/chroma")
    ap.add_argument("--batch", type=int, default=128, help="chunks per Chroma write")
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_tests(args)
    else:
        build(args)


if __name__ == "__main__":
    main()