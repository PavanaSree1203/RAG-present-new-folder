"""

Stage 14: local web UI for the 10-K question-answering system (standard library only - no Flask/Streamlit needed).



  python stage14_ui.py                      # full pipeline: Chroma + BGE + Ollama (llama3.2:3b), http://127.0.0.1:8000

  python stage14_ui.py --no-llm             # deterministic answers only (no Ollama)

  python stage14_ui.py --offline --no-llm   # no Chroma, no BGE: reads the JSON files (BM25 + label channels)

  python stage14_ui.py --port 8080



Needs stage6_query_parser.py, stage7_*.py, stage8_align.py, stage9_verify.py, stage10_answer.py, stage11_format.py,

stage12_ocr_question.py (image questions; optional), canon.py, typo.py in the same folder and the data/ folder from the earlier stages.

"""

import argparse

import json

import threading

import traceback

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer



EXAMPLES = [

    {"group": "Figures", "items": ["What was total net sales in 2022?", "Net income in 2019", "Total assets in 2023"]},

    {"group": "Calculations", "items": ["Percentage change in net sales from 2017 to 2018", "Free cash flow in 2018",

                                        "CAGR of net sales from 2015 to 2020", "Operating margin in 2019 and 2021"]},

    {"group": "Comparisons", "items": ["Compare 2017 and 2018", "What was the difference in net sales between 2017 and 2018?"]},

    {"group": "What the filings say", "items": ["Why did net sales increase in 2013?", "What are the main risks related to competition?",

                                                "What are the discussions in the year 2017?"]},

]





def suggest(res, payload):

    """Follow-up questions built from what was just asked (no LLM needed): different year, related metric, next analysis."""

    p = payload.get("plan") or {}

    ys = sorted({int(y) for y in (p.get("years") or []) if str(y).isdigit()})

    ms = p.get("metrics") or []

    name = lambda c: {"revenue": "net sales"}.get(c, c.replace("_", " "))

    out = []

    if ms and ys:

        m, y = name(ms[0]), ys[-1]

        if len(ys) == 1:

            out += [f"And {m} in {y - 1}?", f"Percentage change in {m} from {y - 1} to {y}", f"Why did {m} change in {y}?"]

        else:

            out += [f"Show a chart of {m} from {ys[0]} to {ys[-1]}", f"CAGR of {m} from {ys[0]} to {ys[-1]}", f"What drove the change in {m} in {ys[-1]}?"]

        if len(ms) > 1:

            out.append(f"Compare {name(ms[1])} for {ys[0]} and {ys[-1]}")

        else:

            out.append(f"Operating margin in {y}")

    elif ys:

        y = ys[-1]

        out += [f"What are the main risks discussed in {y}?", f"Compare {y - 1} and {y}", f"Net sales and net income in {y}", f"What did management say about growth in {y}?"]

    elif ms:

        out += [f"{name(ms[0]).capitalize()} from 2015 to 2020", f"CAGR of {name(ms[0])} from 2015 to 2020"]

    else:

        out += ["What are the discussions in the year 2017?", "Compare 2017 and 2018", "What are the main risks related to competition?"]

    seen, uniq = set(), []

    for q in out:

        if q.lower() not in seen and q.lower() != (res.get("question") or "").lower():

            seen.add(q.lower())

            uniq.append(q)

    return uniq[:4]





def retrieval_summary(res):

    out = []

    for e in (res.get("retrieval") or {}).get("results", [])[:12]:

        m = e["meta"]

        out.append({"kind": "table" if e["coll"] == "table_rows" else "text", "filing": m.get("doc_year"),

                    "page": m.get("page") or m.get("page_start"),

                    "what": (m.get("row_label") or "(whole table)") if e["coll"] == "table_rows" else str(m.get("section_path", ""))[-80:],

                    "statement": m.get("statement"), "channels": e["channels"], "score": e["score"]})

    return out





class App:

    def __init__(self, pipeline):

        self.pipeline, self.lock = pipeline, threading.Lock()



    def ask(self, question, policy="as_reported", mode="full", history=None):

        import stage11_format as s11

        with self.lock:                                       # the models/indexes are not thread-safe

            res = self.pipeline.ask(question, mode=mode, policy=policy, history=history)

        payload = s11.format_response(res)

        payload["retrieval"] = retrieval_summary(res)

        payload["answer_context"] = res["answer"].get("context", "")

        payload["markdown"] = s11.to_markdown(payload)

        payload["suggestions"] = suggest(res, payload)

        return payload



    def info(self):

        p = self.pipeline

        llm = getattr(p, "llm", None)

        dense = type(getattr(p.retriever, "dense", None)).__name__

        offline = dense in ("_NoDense", "FakeDense", "NoneType")

        return {"models": [

            {"role": "Language model (answers)", "name": getattr(llm, "model", None) or "not used (template answers)", "how": "Ollama, local. Writes the answer only from the retrieved evidence; numbers and [R#]/[S#] citations are checked, then one retry, then a template fallback."},

            {"role": "Embeddings (meaning search)", "name": "BAAI/bge-base-en-v1.5" if not offline else "off", "how": "Chunks and table rows embedded and stored in ChromaDB (text_chunks, table_rows)."},

            {"role": "Keyword search", "name": "BM25 (built in)", "how": "Exact words, years and line-item names."},

            {"role": "Line-item label match", "name": "canonical label index", "how": "Finds table rows by their normalised label, e.g. 'Total net sales'."},

            {"role": "Fusion", "name": "weighted Reciprocal Rank Fusion", "how": "k=60, weights BM25 1, dense 1, label 1.5. Then year coverage, per-section diversity, near-duplicate removal."},

            {"role": "Query understanding", "name": "rule-based parser + spell correction", "how": "Metrics, years, ranges, intent; fixes typos with difflib."},

            {"role": "Year alignment", "name": "restatement and split detection", "how": "as reported / latest / same filing; stock-split factors (e.g. 20:1)."},

            {"role": "Arithmetic", "name": "Python Decimal + safe evaluator", "how": "Every calculated number has a formula, steps and an independent cross-check. The LLM never calculates."},

            {"role": "Extraction", "name": "PyMuPDF, pdfplumber, Tesseract OCR", "how": "Scanned filings (2019, 2020) are read by OCR and flagged."},

        ]}





def make_handler(app):

    class H(BaseHTTPRequestHandler):

        def log_message(self, fmt, *args):

            pass



        def _send(self, code, body, ctype):

            data = body if isinstance(body, bytes) else body.encode("utf-8")

            self.send_response(code)

            self.send_header("Content-Type", ctype)

            self.send_header("Content-Length", str(len(data)))

            self.end_headers()

            self.wfile.write(data)



        def do_GET(self):

            if self.path in ("/", "/index.html"):

                self._send(200, PAGE.replace("__EXAMPLES__", json.dumps(EXAMPLES)), "text/html; charset=utf-8")

            elif self.path == "/api/info":

                self._send(200, json.dumps(app.info(), default=str), "application/json")

            elif self.path == "/api/health":

                self._send(200, json.dumps({"ok": True}), "application/json")

            else:

                self._send(404, "not found", "text/plain")



        def do_POST(self):

            if self.path == "/api/ocr":

                try:

                    import base64

                    import stage12_ocr_question as s12

                    n = int(self.headers.get("Content-Length", 0))

                    img = json.loads(self.rfile.read(n).decode("utf-8")).get("image", "")

                    raw = base64.b64decode(img.split(",", 1)[-1])

                    with app.lock:

                        out = s12.read_question(raw)

                    return self._send(200, json.dumps(out), "application/json")

                except ImportError:

                    return self._send(500, json.dumps({"error": "stage12_ocr_question.py is not in this folder"}), "application/json")

                except Exception as e:

                    traceback.print_exc()

                    return self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")

            if self.path != "/api/ask":

                return self._send(404, "not found", "text/plain")

            try:

                n = int(self.headers.get("Content-Length", 0))

                body = json.loads(self.rfile.read(n).decode("utf-8"))

                q = (body.get("question") or "").strip()

                if not q:

                    return self._send(400, json.dumps({"error": "empty question"}), "application/json")

                out = app.ask(q, body.get("policy", "as_reported"), body.get("mode", "full"), body.get("history") or [])

                self._send(200, json.dumps(out, default=str), "application/json")

            except Exception as e:                            # show the problem in the page instead of a blank screen

                traceback.print_exc()

                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")

    return H





def bind_server(host, port, handler, tries=30):

    """Windows can refuse a port (WinError 10013: in use, firewall, or reserved by Hyper-V/WSL); try the next ones."""

    last = None

    for p in range(port, port + tries):

        try:

            srv = ThreadingHTTPServer((host, p), handler)

            if p != port:

                print(f"Port {port} is not available on this machine ({last}); using port {p} instead.")

            return srv

        except OSError as e:

            last = f"{type(e).__name__}: {e}".split("\n")[0][:90]

    raise SystemExit(f"No free port found in {port}-{port + tries - 1}: {last}. Try --port 50000.")





def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--port", type=int, default=8000)

    ap.add_argument("--host", default="127.0.0.1")

    ap.add_argument("--db", default="data/chroma")

    ap.add_argument("--rows", default="data/processed/table_rows.json")

    ap.add_argument("--tables", default="data/processed/all_tables.json")

    ap.add_argument("--chunks", default="data/processed/all_chunks.json")

    ap.add_argument("--model", default="llama3.2:3b")

    ap.add_argument("--ollama-url", default="http://localhost:11434")

    ap.add_argument("--no-llm", action="store_true")

    ap.add_argument("--no-dense", action="store_true")

    ap.add_argument("--offline", action="store_true")

    a = ap.parse_args()

    from stage10_answer import Pipeline

    print("Loading corpus and models (first start takes a moment) ...")

    pipe = Pipeline(a.db, a.rows, a.tables, a.model, a.ollama_url, not a.no_llm, not a.no_dense, a.chunks, a.offline)

    srv = bind_server(a.host, a.port, make_handler(App(pipe)))

    print(f"Ready: http://{a.host}:{srv.server_address[1]}   (Ctrl+C to stop)")

    try:

        srv.serve_forever()

    except KeyboardInterrupt:

        print("bye")





PAGE = r"""<!doctype html>

<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">

<title>Amazon 10-K Research Workspace</title>

<style>
:root{--bg:#f3f5f8;--card:#fff;--ink:#172335;--mute:#69778a;--line:#e1e7ef;--accent:#2157a6;--accent-ink:#fff;--user:#eaf1fc;--ok:#137547;--warn:#9a6700;--bad:#b42318;--chip:#f1f4f8;--c1:#2157a6;--c2:#0f8b8d;--c3:#8b5cf6;--c4:#d97706;--shadow:0 5px 20px rgba(23,35,53,.045)}
@media(prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#10151d;--card:#171f2b;--ink:#edf2f8;--mute:#a0adbd;--line:#2b3747;--accent:#8ab4f8;--accent-ink:#10213c;--user:#202f47;--ok:#5bd49a;--warn:#f4c76b;--bad:#ff938b;--chip:#202b39;--c1:#8ab4f8;--c2:#5bd4ce;--c3:#b6a0ff;--c4:#ffbd75;--shadow:0 8px 28px rgba(0,0,0,.18)}}
:root[data-theme=dark]{--bg:#10151d;--card:#171f2b;--ink:#edf2f8;--mute:#a0adbd;--line:#2b3747;--accent:#8ab4f8;--accent-ink:#10213c;--user:#202f47;--ok:#5bd49a;--warn:#f4c76b;--bad:#ff938b;--chip:#202b39;--c1:#8ab4f8;--c2:#5bd4ce;--c3:#b6a0ff;--c4:#ffbd75;--shadow:0 8px 28px rgba(0,0,0,.18)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.6 Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;letter-spacing:-.01em}
.app-shell{display:grid;grid-template-columns:250px minmax(420px,1fr) 292px;min-height:100vh;max-width:1900px;margin:0 auto}.sidebar{min-width:0;background:var(--card);padding:23px 17px;border-right:1px solid var(--line);height:100vh;position:sticky;top:0;overflow-y:auto}.rightbar{border-right:0;border-left:1px solid var(--line);padding:23px 17px}.center-column{min-width:0;min-height:100vh;padding:23px clamp(18px,2.4vw,38px) 220px}.brand{display:flex;align-items:center;gap:11px;padding:2px 4px 22px;border-bottom:1px solid var(--line);margin-bottom:22px}.brand-mark{width:37px;height:37px;display:grid;place-items:center;border-radius:11px;background:#e8effb;color:#174ea6;font-weight:850;font-size:16px}.brand-name{font-size:14px;font-weight:800;line-height:1.25}.brand-caption{font-size:11px;color:var(--mute);margin-top:3px}.side-label{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--mute);font-weight:800;margin:18px 5px 9px}.side-action{width:100%;display:flex;align-items:center;gap:9px;text-align:left;padding:10px 11px;margin:4px 0;border-radius:9px;font-size:12.5px}.side-action.active{background:var(--chip);border-color:var(--line);font-weight:700}.side-note{font-size:11.5px;color:var(--mute);padding:10px 5px;line-height:1.55}.side-divider{height:1px;background:var(--line);margin:18px 0}.side-status{display:flex;gap:8px;align-items:flex-start;padding:10px 11px;background:var(--chip);border:1px solid var(--line);border-radius:10px;font-size:11.5px;color:var(--mute)}.status-dot{width:7px;height:7px;border-radius:50%;background:var(--ok);margin-top:6px;flex:none}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:15px;flex-wrap:wrap;margin:0 0 22px;padding:0 0 18px;border-bottom:1px solid var(--line)}header>div:first-child{max-width:690px}h1{font-size:clamp(22px,2vw,29px);line-height:1.2;letter-spacing:-.04em;font-weight:780;margin:0 0 7px}.sub{color:var(--mute);font-size:12px;line-height:1.55}header .row{gap:7px}button,select{font:inherit}button{border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);padding:8px 11px;cursor:pointer;transition:background .15s,border-color .15s,transform .15s}button:hover{border-color:var(--accent);background:var(--chip)}button:active{transform:translateY(1px)}button.primary{background:var(--accent);color:var(--accent-ink);border-color:var(--accent);font-weight:700;padding:9px 17px}button.primary:hover{filter:brightness(1.06)}button:disabled{opacity:.55;cursor:wait}select{width:100%;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:8px;padding:8px 9px;margin-top:5px}.setting{margin:14px 0}.setting label{display:block;color:var(--mute);font-size:11px;font-weight:700;letter-spacing:.02em}.setting-help{font-size:10.5px;color:var(--mute);margin-top:5px}.right-heading{font-size:13px;font-weight:800;margin:0 0 5px}.right-intro{font-size:11.5px;color:var(--mute);margin-bottom:16px}.side-button{width:100%;text-align:left;font-size:12px;margin:6px 0}
#aboutbox{box-shadow:none}.msg{margin:22px 0}.me{display:flex;justify-content:flex-end}.me .bubble{background:var(--user);border:1px solid color-mix(in srgb,var(--accent) 13%,var(--line));border-radius:15px 15px 4px 15px;padding:12px 15px;max-width:min(82%,760px);overflow-wrap:anywhere}.bot{background:var(--card);border:1px solid var(--line);border-radius:5px 14px 14px 14px;padding:20px 22px;box-shadow:var(--shadow);min-width:0}.interp{color:var(--mute);font-size:12px;margin:-4px 0 12px}.pills{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:15px}.pill{font-size:10.5px;font-weight:700;padding:4px 8px;border-radius:6px;background:var(--chip);color:var(--mute);border:1px solid var(--line)}.pill.ok{color:var(--ok)}.pill.warn{color:var(--warn)}.pill.bad{color:var(--bad)}.answer p{margin:0 0 12px}.answer ul{margin:6px 0 14px;padding-left:22px}.answer li{margin:5px 0}.answer{font-size:14.5px;line-height:1.8;overflow-wrap:anywhere}.answer b{font-weight:750}.tag{font-size:10px;color:var(--accent);vertical-align:super;margin-left:2px;font-weight:700}.flags{margin:14px 0 0;padding:12px 14px;list-style:none;background:color-mix(in srgb,var(--warn) 8%,var(--card));border:1px solid color-mix(in srgb,var(--warn) 22%,var(--line));border-radius:10px}.flags li{color:var(--warn);font-size:12px;padding-left:16px;position:relative;margin:3px 0}.flags li:before{content:"!";position:absolute;left:2px;font-weight:800}.more{margin-top:18px;padding-top:15px;border-top:1px solid var(--line)}.more>button{font-size:12px;font-weight:700}.tabs{display:flex;gap:5px;flex-wrap:wrap;border-bottom:1px solid var(--line);margin-top:18px}.tab{border:0;background:transparent;border-bottom:2px solid transparent;border-radius:0;padding:10px 12px;color:var(--mute)}.tab:hover{background:var(--chip)}.tab.on{color:var(--accent);border-bottom-color:var(--accent);font-weight:750}.pane{display:none;padding-top:16px;min-width:0}.pane.on{display:block}table{border-collapse:separate;border-spacing:0;width:100%;font-size:12px}th,td{text-align:right;padding:9px 10px;border-bottom:1px solid var(--line);white-space:nowrap}th:first-child,td:first-child{text-align:left;white-space:normal}th{color:var(--mute);font-weight:750;font-size:10px;text-transform:uppercase;letter-spacing:.04em;background:var(--chip)}th:first-child{border-radius:7px 0 0 0}th:last-child{border-radius:0 7px 0 0}.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:9px}.tt{display:flex;justify-content:space-between;align-items:center;gap:10px;margin:18px 0 8px}.tt b{font-size:12.5px}.note{color:var(--mute);font-size:11px;margin-top:7px}.src{padding:12px 0;border-bottom:1px solid var(--line);overflow-wrap:anywhere}.src:last-child{border-bottom:0}.src b{font-size:12px}.src div{color:var(--mute);font-size:11.5px}.step{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:11px;background:var(--chip);padding:8px 10px;border-radius:7px;margin:6px 0;overflow-x:auto}pre{background:var(--chip);border:1px solid var(--line);padding:13px;border-radius:9px;overflow:auto;font-size:11px;white-space:pre-wrap;overflow-wrap:anywhere}.empty{color:var(--mute);padding:10px 0}.err{color:var(--bad)}.dots span{display:inline-block;width:7px;height:7px;margin-right:4px;border-radius:50%;background:var(--accent);animation:b 1s infinite}.dots span:nth-child(2){animation-delay:.15s}.dots span:nth-child(3){animation-delay:.3s}@keyframes b{0%,60%,100%{opacity:.25}30%{opacity:1}}
.dock{position:fixed;z-index:5;left:250px;right:292px;bottom:0;background:linear-gradient(transparent,var(--bg) 20%);padding:30px clamp(14px,2.4vw,36px) 14px}.dockin{max-width:1100px;margin:0 auto;background:var(--card);border:1px solid var(--line);border-radius:14px;padding:11px 13px 10px;box-shadow:0 10px 30px rgba(16,24,40,.12)}textarea{width:100%;border:0;outline:0;resize:vertical;min-height:45px;max-height:140px;background:transparent;color:var(--ink);font:inherit;padding:4px 2px}textarea::placeholder{color:var(--mute)}.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.spacer{flex:1}.chips{display:flex;gap:7px;flex-wrap:wrap;margin:8px 0 0}.chip{font-size:11.5px;padding:8px 9px;border-radius:8px;text-align:left;background:var(--card);max-width:100%;white-space:normal;line-height:1.4}.hello{margin-top:24px;background:var(--card);border:1px solid var(--line);border-radius:15px;padding:24px;box-shadow:var(--shadow)}.hello h2{font-size:20px;letter-spacing:-.03em;margin:0 0 6px}.hello>.sub{max-width:700px}.aboutbox{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:13px 14px;margin:0 0 14px;font-size:12px}.aboutbox .m{padding:10px 0;border-bottom:1px solid var(--line)}.aboutbox .m:last-child{border:0}.aboutbox b{font-weight:750}.aboutbox .sub{display:block;margin-top:3px}.sugg{margin-top:20px;padding-top:16px;border-top:1px solid var(--line)}.sugg .sub{margin-bottom:7px;font-weight:700}.grp{margin:15px 0 6px;font-size:9.5px;color:var(--mute);text-transform:uppercase;letter-spacing:.1em;font-weight:800}.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;margin-top:8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:5px}svg text{fill:var(--mute);font-size:11px}.sidebar .chips{display:grid;grid-template-columns:1fr}.sidebar .chip{width:100%;font-size:11px;padding:8px 9px}.sidebar .grp:first-child{margin-top:0}.sidebar .grp{margin-top:18px}.sidebar .chips .grp{grid-column:1}.sidebar .side-examples{margin-top:4px}.sidebar .chip:hover{border-color:var(--accent)}
@media(max-width:1050px){.app-shell{grid-template-columns:220px minmax(0,1fr)}.rightbar{grid-column:1/-1;position:static;height:auto;border-top:1px solid var(--line);border-left:0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.rightbar>.side-divider{display:none}.dock{left:220px;right:0}.center-column{padding-bottom:210px}}
@media(max-width:760px){.app-shell{display:flex;flex-direction:column}.sidebar{position:static;height:auto;overflow:visible;border-right:0;border-bottom:1px solid var(--line);padding:14px 15px}.brand{padding-bottom:12px;margin-bottom:10px}.sidebar .side-label{margin-top:10px}.sidebar .side-examples{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px}.sidebar .side-examples .grp{grid-column:1/-1}.sidebar .side-examples .chips{display:contents}.sidebar .chip{height:100%}.sidebar .side-note,.sidebar .side-divider,.sidebar .side-status{display:none}.center-column{padding:17px 14px 220px}.rightbar{display:block;padding:16px;border-top:1px solid var(--line)}.dock{left:0;right:0;padding:26px 9px 9px}.dockin{padding:10px;border-radius:13px}.dockin .row{gap:7px}.dockin .row .sub{font-size:10px}.dockin select{max-width:100%}.dock .row label{width:calc(50% - 6px)}.dock .row label select{width:100%}.bot{padding:16px 14px}.hello{padding:18px}.me .bubble{max-width:92%}.answer{font-size:14px}.tabs{gap:0}.tab{padding:9px 10px;font-size:11px}header .row{width:100%}header .row button{flex:1}}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{scroll-behavior:auto!important;animation:none!important;transition:none!important}}
</style></head><body><div class="app-shell">
<aside class="sidebar">
  <div class="brand"><div class="brand-mark">10K</div><div><div class="brand-name">Amazon Research</div><div class="brand-caption">Annual filing intelligence</div></div></div>
  <div class="side-label">Workspace</div>
  <button class="side-action active" id="clear"><span>＋</span> New conversation</button>
  <button class="side-action" id="jumpchat"><span>↗</span> Back to conversation</button>
  <div class="side-divider"></div>
  <div class="side-label">Explore questions</div>
  <div id="sidebar-examples" class="side-examples"></div>
  <div class="side-divider"></div>
  <div class="side-label">How to use</div>
  <div class="side-note">Ask about annual figures, year-over-year comparisons, trends, calculations, or management discussion. You can also upload, paste, or drop an image containing a question.</div>
  <div class="side-status"><span class="status-dot"></span><span><b>Evidence-first answers</b><br>Responses are grounded in retrieved filing passages, with checks for calculations and citations.</span></div>
</aside>
<section class="center-column" id="conversation">
<header><div><h1>10-K Research Workspace</h1><div class="sub">Explore Amazon annual filings with evidence-backed answers, transparent calculations, and filing-level sources.</div></div>
<div class="row"><button id="about">Models &amp; methods</button><button id="theme">Theme</button></div></header>
<div id="chat"></div>
</section>
<aside class="sidebar rightbar">
  <div><div class="right-heading">Query settings</div><div class="right-intro">Choose how answers should interpret filings and retrieve evidence.</div>
    <div class="setting"><label for="policy">Financial basis</label><select id="policy"><option value="as_reported">As reported</option><option value="latest">Latest (restated)</option><option value="same_filing">Same filing</option></select><div class="setting-help">Controls which reported or restated values are used.</div></div>
    <div class="setting"><label for="mode">Retrieval strategy</label><select id="mode"><option value="full">Full hybrid</option><option value="hybrid">BM25 + dense</option><option value="bm25">BM25 only</option><option value="dense">Dense only</option></select><div class="setting-help">Select hybrid, keyword, or semantic retrieval.</div></div>
    <button class="side-button" id="about-side">View model and method details</button>
  </div>
  <div class="side-divider"></div>
  <div><div class="right-heading">Research workflow</div><div class="right-intro">Each answer can expose its supporting material.</div>
    <div class="side-status"><span class="status-dot"></span><span><b>Answer verification</b><br>Calculation formulas, checks, and warning flags are included when relevant.</span></div>
    <div class="side-note">Use <b>Show details</b> on an answer to inspect tables, charts, sources, retrieved evidence, and computation steps.</div>
  </div>
  <div class="side-divider"></div>
  <div><div class="right-heading">Models &amp; methods</div><div class="right-intro">Technical details for this local research pipeline.</div><div id="aboutbox" class="aboutbox" hidden></div></div>
</aside>
</div>
<div class="dock"><div class="dockin">
<textarea id="q" rows="1" placeholder="Ask a question, e.g. 'Compare 2017 and 2018' or 'What are the discussions in the year 2017?'"></textarea>
<div id="ocrnote" class="sub" hidden></div>
<div class="row"><button class="primary" id="go">Send</button><button id="img" title="Ask from an image (or paste / drop one)">Image</button><input type="file" id="file" accept="image/*" hidden><span class="sub">Enter to send · Shift+Enter for a new line · Image, paste or drop reads a question from a picture.</span></div></div></div>

<script>

const EX=__EXAMPLES__, $=s=>document.querySelector(s);

const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

let turns=[], busy=false;

function store(k,v){try{if(v===undefined)return localStorage.getItem(k);localStorage.setItem(k,v)}catch(e){return null}}

if(store("theme"))document.documentElement.dataset.theme=store("theme");

$("#theme").onclick=()=>{const r=document.documentElement;const dark=(r.dataset.theme||(matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light"))==="dark";r.dataset.theme=dark?"light":"dark";store("theme",r.dataset.theme)};

$("#clear").onclick=()=>{turns=[];hello()};

let aboutLoaded=false;

async function toggleAbout(){const bx=$("#aboutbox");bx.hidden=!bx.hidden;
  if(!bx.hidden&&!aboutLoaded){try{const r=await (await fetch("/api/info")).json();bx.innerHTML="<b>Models and methods used</b>"+r.models.map(m=>'<div class="m"><b>'+esc(m.role)+":</b> "+esc(m.name)+'<span class="sub">'+esc(m.how)+"</span></div>").join("");aboutLoaded=true}catch(e){bx.textContent="Could not load: "+e.message}}}
$("#about").onclick=toggleAbout;$("#about-side").onclick=toggleAbout;
$("#jumpchat").onclick=()=>$("#conversation").scrollIntoView({behavior:"smooth",block:"start"});

$("#go").onclick=()=>send();

$("#q").addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send()}});

$("#q").addEventListener("input",e=>{e.target.style.height="auto";e.target.style.height=Math.min(e.target.scrollHeight,140)+"px"});

function ocr(file){

  if(!file)return;const note=$("#ocrnote");note.hidden=false;note.textContent="Reading the image...";

  const fr=new FileReader();fr.onload=async()=>{

    try{const r=await fetch("/api/ocr",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({image:fr.result})});

      const d=await r.json();if(!r.ok)throw new Error(d.error||r.statusText);

      const q=(d.questions&&d.questions[0])||d.text||"";if(q){$("#q").value=q;$("#q").style.height="auto";$("#q").style.height=Math.min($("#q").scrollHeight,140)+"px"}

      note.textContent=(q?"Read from image (confidence "+d.confidence+"%). Check the text, then press Send. ":"")+(d.warnings||[]).join(" | ");note.style.color=d.needs_review?"var(--warn)":"var(--mute)";

      $("#q").focus();

    }catch(e){note.textContent="Could not read the image: "+e.message;note.style.color="var(--bad)"}};

  fr.readAsDataURL(file);

}

$("#img").onclick=()=>$("#file").click();

$("#file").onchange=e=>{ocr(e.target.files[0]);e.target.value=""};

$("#q").addEventListener("paste",e=>{const it=[...(e.clipboardData||{}).items||[]].find(i=>i.type.startsWith("image/"));if(it){e.preventDefault();ocr(it.getAsFile())}});

document.addEventListener("dragover",e=>e.preventDefault());

document.addEventListener("drop",e=>{const f=(e.dataTransfer||{}).files&&e.dataTransfer.files[0];if(f&&f.type.startsWith("image/")){e.preventDefault();ocr(f)}});

function hello(){
  $("#chat").innerHTML='<div class="hello"><h2>What would you like to know?</h2><div class="sub">Ask about figures, comparisons, trends, or what the filings say. Follow-up questions like "and 2019?" work too. Select any example from the left panel to start.</div><div class="row" style="margin-top:18px;gap:8px"><span class="pill">Financial figures</span><span class="pill">Year comparisons</span><span class="pill">Calculations</span><span class="pill">Filing evidence</span></div></div>';
  const root=$("#sidebar-examples");root.innerHTML="";
  EX.forEach(g=>{const t=document.createElement("div");t.className="grp";t.textContent=g.group;root.appendChild(t);
    const w=document.createElement("div");w.className="chips";g.items.forEach(e=>{const b=document.createElement("button");b.className="chip";b.textContent=e;b.onclick=()=>send(e);w.appendChild(b)});root.appendChild(w)});
}

hello();

function md(t){

  const lines=esc(t).split("\n");let h="",ul=false;

  const inl=x=>x.replace(/\*\*(.+?)\*\*/g,"<b>$1</b>").replace(/\[((?:R|S)\d+)\]/g,'<span class="tag">[$1]</span>');

  for(const l of lines){

    if(/^\s*[-*] /.test(l)){if(!ul){h+="<ul>";ul=true}h+="<li>"+inl(l.replace(/^\s*[-*] /,""))+"</li>"}

    else{if(ul){h+="</ul>";ul=false}if(l.trim())h+="<p>"+inl(l)+"</p>"}

  }

  return h+(ul?"</ul>":"");

}

function pill(t,c){return '<span class="pill '+(c||"")+'">'+esc(t)+'</span>'}

async function send(text){

  if(busy)return;const q=(text||$("#q").value).trim();if(!q)return;

  if(!turns.length)$("#chat").innerHTML="";

  $("#q").value="";$("#q").style.height="auto";

  const history=turns.slice(-4).map(t=>({q:t.q,a:t.a}));

  $("#chat").insertAdjacentHTML("beforeend",'<div class="msg me"><div class="bubble">'+esc(q)+'</div></div>');

  const id="m"+turns.length;

  $("#chat").insertAdjacentHTML("beforeend",'<div class="msg" id="'+id+'"><div class="bot"><span class="dots"><span></span><span></span><span></span></span> <span class="sub">Retrieving evidence, checking the numbers, writing the answer...</span></div></div>');

  window.scrollTo(0,document.body.scrollHeight);busy=true;$("#go").disabled=true;

  try{

    const r=await fetch("/api/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q,policy:$("#policy").value,mode:$("#mode").value,history})});

    const d=await r.json();if(!r.ok)throw new Error(d.error||r.statusText);

    turns.push({q,a:d.answer.text,d});render(id,d,turns.length-1);

  }catch(e){$("#"+id).innerHTML='<div class="bot err">Something went wrong: '+esc(e.message)+'</div>'}

  busy=false;$("#go").disabled=false;window.scrollTo(0,document.body.scrollHeight);$("#q").focus();

}

function render(id,d,n){

  const st={verified:["verified","ok"],verified_with_flags:["verified, with flags","warn"],partial:["partly calculated","warn"],refused:["not calculated","bad"],not_applicable:["from the filings' text",""]}[d.status]||[d.status,""];

  const llm=d.answer.mode==="llm";

  let h='<div class="bot">'+(d.interpreted_as?'<div class="interp">Understood as: '+esc(d.interpreted_as)+"</div>":"");

  h+='<div class="pills">'+pill(st[0],st[1])+pill(llm?"written by Llama, grounding-checked":"template answer - LLM "+(d.answer.llm_status||"not used"),llm?"ok":"warn")+(d.primary_basis?pill("basis: "+d.primary_basis):"")+"</div>";

  h+='<div class="answer">'+md(d.answer.text)+"</div>";

  const fl=[...d.flags,...d.answer.warnings];if(fl.length)h+='<ul class="flags">'+fl.map(f=>"<li>"+esc(f)+"</li>").join("")+"</ul>";

  h+='<div class="more"><button data-more="'+n+'">Show details (tables, chart, sources)</button><div id="det'+n+'" hidden>'+details(d,n)+"</div></div>";

  if((d.suggestions||[]).length)h+='<div class="sugg"><div class="sub">You could ask next</div><div class="chips" data-sugg></div></div>';

  h+="</div>";

  $("#"+id).innerHTML=h;

  const sg=$("#"+id).querySelector("[data-sugg]");if(sg)d.suggestions.forEach(q=>{const b=document.createElement("button");b.className="chip";b.textContent=q;b.onclick=()=>send(q);sg.appendChild(b)});

  $("#"+id).querySelector("[data-more]").onclick=e=>{const el=$("#det"+n);el.hidden=!el.hidden;e.target.textContent=el.hidden?"Show details (tables, chart, sources)":"Hide details"};

  $("#"+id).querySelectorAll(".tab").forEach(t=>t.onclick=()=>{const root=$("#det"+n);root.querySelectorAll(".tab").forEach(x=>x.classList.toggle("on",x===t));root.querySelectorAll(".pane").forEach(p=>p.classList.toggle("on",p.dataset.p===t.dataset.p))});

  $("#"+id).querySelectorAll("[data-csv]").forEach(b=>b.onclick=()=>csv(n,+b.dataset.csv));

}

function details(d,n){

  return '<div class="tabs"><button class="tab on" data-p="t">Tables</button><button class="tab" data-p="c">Chart</button><button class="tab" data-p="s">Sources</button><button class="tab" data-p="h">How it was computed</button></div>'+

   '<div class="pane on" data-p="t">'+tables(d)+'</div><div class="pane" data-p="c">'+chart(d)+'</div><div class="pane" data-p="s">'+sources(d)+'</div><div class="pane" data-p="h">'+how(d)+"</div>";

}

function tables(d){

  if(!d.tables.length)return '<div class="empty">No figures were needed for this question. See Sources for the passages used.</div>';

  return d.tables.map((t,i)=>'<div class="tt"><b>'+esc(t.title)+'</b><button data-csv="'+i+'">CSV</button></div><div class="scroll"><table><tr>'+t.columns.map(c=>"<th>"+esc(c)+"</th>").join("")+"</tr>"+

    t.rows.map(r=>"<tr>"+r.map(c=>"<td>"+esc(c)+"</td>").join("")+"</tr>").join("")+"</table></div>"+(t.note?'<div class="note">'+esc(t.note)+"</div>":"")+

    ((t.missing&&t.missing.length)?'<div class="note">Missing: '+esc(t.missing.join("; "))+"</div>":"")).join("");

}

function csv(n,i){const t=turns[n].d.tables[i];const q=v=>'"'+String(v).replace(/"/g,'""')+'"';const s=[t.columns,...t.rows].map(r=>r.map(q).join(",")).join("\n");

  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([s],{type:"text/csv"}));a.download=t.title.replace(/\W+/g,"_")+".csv";a.click()}

function sources(d){

  let h=d.sources.length?d.sources.map(s=>'<div class="src"><b>'+esc(s.title)+(s.tag?" ["+esc(s.tag)+"]":"")+"</b><div>"+esc(s.detail)+"</div></div>").join(""):'<div class="empty">No sources.</div>';

  if(d.retrieval.length){h+='<div class="tt"><b>Retrieved evidence (ranked)</b></div><div class="scroll"><table><tr><th>What</th><th>Filing</th><th>Page</th><th>Channels</th></tr>'+

    d.retrieval.map(r=>"<tr><td>"+esc(r.what)+(r.statement?" <span class='sub'>("+esc(r.statement)+")</span>":"")+"</td><td>"+esc(r.filing)+"</td><td>"+esc(r.page)+"</td><td>"+Object.entries(r.channels).map(([k,v])=>esc(k.split(":")[0])+"#"+v).join(" ")+"</td></tr>").join("")+"</table></div>"}

  return h;

}

function how(d){

  let h="";

  if(d.calculations.length)h+=d.calculations.map(c=>'<div class="src"><b>'+esc(c.label)+" - "+esc(c.op.replace("_"," "))+" "+esc(c.period)+" = "+esc(c.display)+'</b><div>Formula: '+esc(c.formula)+" &middot; independent cross-check: "+esc(c.cross_check)+"</div>"+c.steps.map(s=>'<div class="step">'+esc(s)+"</div>").join("")+c.notes.map(n=>'<div class="note">'+esc(n)+"</div>").join("")+"</div>").join("");

  else h+='<div class="empty">No calculation was needed.</div>';

  if(d.refusals.length)h+='<div class="tt"><b>Not calculated</b></div>'+d.refusals.map(r=>'<div class="err">'+esc(r.op)+": "+esc(r.reason)+"</div>").join("");

  h+='<div class="tt"><b>Understood as</b></div><pre>'+esc(JSON.stringify(d.plan,null,1))+"</pre>";

  h+='<div class="tt"><b>Language model</b></div><div class="note">Status: '+esc(d.answer.llm_status||"-")+"</div>";

  if((d.answer.attempts||[]).length)h+=d.answer.attempts.map((a,i)=>'<div class="note">Model output, attempt '+(i+1)+":</div><pre>"+esc(a)+"</pre>").join("");

  h+='<div class="tt"><b>Context given to the language model</b></div><pre>'+esc(d.answer_context)+"</pre>";

  return h;

}

function chart(d){

  const c=d.chart;if(!c)return '<div class="empty">A chart appears when three or more years of the same unit are available.</div>';

  const W=800,H=280,m={l:64,r:16,t:14,b:30},xs=c.x,all=c.series.flatMap(s=>s.values).filter(v=>v!=null);

  let lo=Math.min(0,...all),hi=Math.max(...all);if(hi===lo)hi=lo+1;

  const X=i=>m.l+(xs.length===1?0:i*(W-m.l-m.r)/(xs.length-1)),Y=v=>H-m.b-(v-lo)/(hi-lo)*(H-m.t-m.b);

  const col=["var(--c1)","var(--c2)","var(--c3)","var(--c4)"];

  const f=v=>Math.abs(v)>=1e6?(v/1e6).toFixed(1)+"M":Math.abs(v)>=1e3?(v/1e3).toFixed(0)+"k":v.toFixed(Math.abs(v)<10?1:0);

  let s='<svg viewBox="0 0 '+W+" "+H+'" width="100%" role="img" aria-label="'+esc(c.title)+'">';

  for(let k=0;k<=4;k++){const v=lo+(hi-lo)*k/4,y=Y(v);s+='<line x1="'+m.l+'" x2="'+(W-m.r)+'" y1="'+y+'" y2="'+y+'" stroke="var(--line)"/><text x="'+(m.l-8)+'" y="'+(y+4)+'" text-anchor="end">'+f(v)+"</text>"}

  xs.forEach((x,i)=>{if(xs.length<=12||i%Math.ceil(xs.length/10)===0)s+='<text x="'+X(i)+'" y="'+(H-8)+'" text-anchor="middle">'+x+"</text>"});

  c.series.forEach((se,j)=>{const pts=se.values.map((v,i)=>v==null?null:[X(i),Y(v),v,xs[i]]).filter(Boolean);

    if(c.kind==="line"){s+='<polyline fill="none" stroke="'+col[j]+'" stroke-width="2.2" points="'+pts.map(p=>p[0]+","+p[1]).join(" ")+'"/>';pts.forEach(p=>s+='<circle cx="'+p[0]+'" cy="'+p[1]+'" r="3" fill="'+col[j]+'"><title>FY'+p[3]+": "+p[2].toLocaleString()+"</title></circle>")}

    else{const bw=(W-m.l-m.r)/xs.length/(c.series.length+1);pts.forEach(p=>{const x=p[0]-bw*c.series.length/2+j*bw;s+='<rect x="'+x+'" y="'+Math.min(p[1],Y(0))+'" width="'+bw*.9+'" height="'+Math.abs(Y(0)-p[1])+'" fill="'+col[j]+'"><title>FY'+p[3]+": "+p[2].toLocaleString()+"</title></rect>"})}});

  s+="</svg>";

  return '<div class="sub">'+esc(c.title)+" ("+esc(c.unit)+")</div>"+s+'<div class="legend">'+c.series.map((se,j)=>'<span><i style="background:'+col[j]+'"></i>'+esc(se.name)+"</span>").join("")+"</div>";

}

</script></body></html>

"""



if __name__ == "__main__":

    main()
