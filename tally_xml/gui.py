"""Local GUI for the tally-llm project — Python standard library only.

Run:  python3 -m tally_xml.gui [--port 7860]
Then open http://127.0.0.1:7860

Tabs:
  Validate — paste TallyPrime XML, see the four structural checks.
  Samples  — the two surviving fine-tuned model-generated XML samples.
  Generate — DISABLED (fail-closed). Generation stays disabled even with
             adapter files present: inference is not wired into this build.
             Nothing here produces template XML and calls it model output.

No data leaves this machine; there is no automatic import into live Tally
company books. Anything you paste is validated locally and never uploaded.
"""
import argparse
import html
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from .generate import adapter_status, generate_voucher, AdapterMissingError
from .validate import check

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_samples():
    with open(os.path.join(ROOT, "data", "model_generated_samples.json")) as f:
        samples = json.load(f)
    for s in samples:
        s["validation"] = check(s["model_output"])
    return samples


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tally-llm — TallyPrime XML assistant</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d21;--mut:#5b6470;--line:#dfe3e8;--red:#b3261e;--redbg:#fdecea;--green:#137333;--greenbg:#e6f4ea;--blue:#1a56db}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{background:#111827;color:#fff;padding:20px 24px}header h1{margin:0;font-size:22px}header p{margin:6px 0 0;color:#cbd5e1;font-size:14px}
main{max-width:900px;margin:0 auto;padding:20px 16px 60px}
.banner{border-radius:10px;padding:14px 16px;font-size:14px;line-height:1.5;border:1px solid var(--line)}
.banner.bad{background:var(--redbg);border-color:#f5c6c2}.banner.good{background:var(--greenbg);border-color:#a3d9b1}
.banner strong{color:var(--red)}.banner.good strong{color:var(--green)}
.tabs{display:flex;gap:8px;margin:18px 0 0}.tabs button{border:1px solid var(--line);background:var(--card);padding:10px 18px;border-radius:8px 8px 0 0;cursor:pointer;font-size:15px}
.tabs button.active{background:var(--card);border-bottom-color:var(--card);font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:0 10px 10px 10px;padding:18px}
textarea,input[type=text]{width:100%;padding:10px;border:1px solid var(--line);border-radius:8px;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace}
button.primary{background:var(--blue);color:#fff;border:0;padding:10px 20px;border-radius:8px;cursor:pointer;font-size:15px}
button.primary:disabled{background:#9aa3af;cursor:not-allowed}
pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;font-size:12.5px;white-space:pre-wrap;word-break:break-word}
.checks{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:12px 0}.checks div{border:1px solid var(--line);border-radius:8px;padding:10px;text-align:center}
.checks .v{font-size:22px;font-weight:700}.ok{color:var(--green)}.bad{color:var(--red)}
.muted{color:var(--mut);font-size:13px}.sample{border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.error{background:var(--redbg);border:1px solid #f5c6c2;border-radius:8px;padding:12px;color:var(--red)}
a{color:var(--blue)}
@media(max-width:600px){.checks{grid-template-columns:repeat(2,1fr)}header h1{font-size:19px}}
</style></head><body>
<header><h1>tally-llm — TallyPrime XML assistant</h1>
<p>Plain-English instruction &rarr; TallyPrime XML import message. Validated locally. No live-book import.</p></header>
<main>
<div id="status" class="banner">Loading status&hellip;</div>
<div class="tabs"><button data-tab="validate" class="active">Validate XML</button>
<button data-tab="samples">Model samples</button>
<button data-tab="generate">Generate (disabled)</button></div>
<div id="tab-validate" class="card">
<p>Paste a TallyPrime <code>&lt;ENVELOPE&gt;</code> import message. It is checked locally against a structural subset of the Tally schema (well-formed XML, expected envelope/voucher shape, balanced debits/credits, field consistency) — this is <strong>not</strong> a complete Tally schema validator.</p>
<textarea id="xml" rows="12" placeholder="&lt;ENVELOPE&gt;...&lt;/ENVELOPE&gt;"></textarea><br><br>
<button class="primary" onclick="validate()">Validate</button>
<div id="vout"></div></div>
<div id="tab-samples" class="card" hidden><p class="muted">The only two surviving fine-tuned model-generated samples (Phase 2 run; the full output file is missing). Each is labelled as model output, not a gold answer.</p><div id="sout">Loading&hellip;</div></div>
<div id="tab-generate" class="card" hidden>
<p><strong>Generation is disabled.</strong> The fine-tuned LoRA adapter is missing (it was saved only to the volatile Colab runtime). This app fails closed: it will not generate template XML and present it as trained-model output.</p>
<p>Restoring the adapter files (<code>adapter_config.json</code> + <code>adapter_model.safetensors</code> under <code>models/adapter/</code>) does <strong>not</strong> enable generation on its own: model inference is not wired into this build, so generation stays disabled until inference is actually implemented and verified.</p>
<textarea id="desc" rows="3" placeholder="e.g. Record a receipt of Rs 5,000 from Iyer Foods into HDFC Bank on 2025-07-21."></textarea><br><br>
<button class="primary" onclick="generate()">Generate XML</button>
<div id="gout"></div></div>
<p class="muted" style="margin-top:16px">Status: benchmark scores are keyword scores on synthetic data, not live accounting accuracy. Nothing here has been tested in live Tally Prime. See README for full caveats.</p>
</main>
<script>
const $=id=>document.getElementById(id);
document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active')); b.classList.add('active');
  ['validate','samples','generate'].forEach(t=>$('tab-'+t).hidden=(t!==b.dataset.tab));
  if(b.dataset.tab==='samples') loadSamples();
});
async function loadStatus(){const r=await fetch('/api/status');const s=await r.json();
  const el=$('status');
  let head, cls;
  if(s.generation_enabled){head='Adapter found — generation enabled.';cls='good';}
  else if(s.adapter_present){head='Adapter files found — generation STILL DISABLED (inference not wired in).';cls='bad';}
  else if(s.config_present){head='Only adapter_config.json found — weights missing, NOT a usable adapter. Generation disabled.';cls='bad';}
  else{head='ADAPTER MISSING — generation disabled (fail-closed).';cls='bad';}
  el.className='banner '+cls;
  el.innerHTML='<strong>'+head+'</strong><br>'+s.message;}
async function validate(){const xml=$('xml').value;
  let v;
  try{
    const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({xml})});
    v=await r.json();
  }catch(e){ $('vout').innerHTML='<div class="error">Could not reach the validator: '+e.message+'</div>'; return; }
  if(v.error){ $('vout').innerHTML='<div class="error">'+v.error+'</div>'; return; }
  const chip=(k)=>'<div><div class="v '+(v[k]?'ok':'bad')+'">'+(v[k]?'&#10003;':'&#10007;')+'</div>'+k+'</div>';
  $('vout').innerHTML='<div class="checks">'+['wellformed','schema','balanced','fields'].map(chip).join('')+'</div>'
    +(xml.trim()==='' ? '<div class="error">Paste an ENVELOPE XML message first.</div>':'');}
async function loadSamples(){const r=await fetch('/api/samples');const s=await r.json();
  $('sout').innerHTML=s.map((x,i)=>'<div class="sample"><p><strong>Sample '+(i+1)+'</strong> — '+x.instruction+'</p><pre>'+x.model_output.replace(/</g,'&lt;')+'</pre><p class="muted">Validation: wellformed '+x.validation.wellformed+', schema '+x.validation.schema+', balanced '+x.validation.balanced+', fields '+x.validation.fields+' &middot; <a href="/api/download/sample?n='+i+'">Download XML</a></p></div>').join('');}
async function generate(){const r=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({description:$('desc').value})});
  const j=await r.json(); $('gout').innerHTML='<div class="error" style="margin-top:12px">'+j.error+'</div>';}
loadStatus();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep stdout clean
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None  # caller reports a 400, never crashes on bad input

    def _body_or_400(self):
        body = self._json()
        if body is None:
            self._send(400, json.dumps({"error": "Request body is not valid JSON."}))
            return None
        if not isinstance(body, dict):
            self._send(400, json.dumps({"error": "Request body must be a JSON object."}))
            return None
        return body

    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif p.path == "/api/status":
            self._send(200, json.dumps(adapter_status()))
        elif p.path == "/api/samples":
            self._send(200, json.dumps(load_samples()))
        elif p.path == "/api/download/sample":
            try:
                n = int(parse_qs(p.query).get("n", ["0"])[0])
                s = load_samples()[n]
            except (ValueError, IndexError):
                self._send(404, json.dumps({"error": "sample not found"}))
                return
            data = s["model_output"].encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/xml")
            self.send_header("Content-Disposition", f"attachment; filename=sample_{n+1}.xml")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path == "/api/validate":
            body = self._body_or_400()
            if body is None:
                return
            xml = body.get("xml", "")
            if not isinstance(xml, str) or not xml.strip():
                self._send(400, json.dumps({"error": "Paste an ENVELOPE XML message first."}))
                return
            self._send(200, json.dumps(check(xml)))
        elif self.path == "/api/generate":
            body = self._body_or_400()
            if body is None:
                return
            try:
                out = generate_voucher(body.get("description", ""))
                self._send(200, json.dumps({"xml": out}))
            except AdapterMissingError as e:
                self._send(503, json.dumps({"error": str(e)}))
            except NotImplementedError as e:
                self._send(501, json.dumps({"error": str(e)}))
        else:
            self._send(404, json.dumps({"error": "not found"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print(f"tally-llm GUI at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
