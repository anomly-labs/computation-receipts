#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""cr_verify_service.py — a small local web verifier for Computation Receipts.

Paste a receipt, get a verdict. Runs entirely on this repository's reference
implementation (stdlib + numpy, no framework), so hosting it anywhere IS the
third-party experience: if this service can verify a receipt, so can anyone.

Endpoints (all JSON in/out):
  GET  /                 the single-page UI
  GET  /vectors          the published conformance vectors
  POST /wellformed       {receipt} -> {status, reason}
  POST /verify           {claimed, recomputed} -> {status, reason}
  POST /verify-sampled   {receipt, rows} -> {status, reason, indices}
                         rows = nested JSON array, the verifier's OWN re-execution
  POST /beacon-audit     {transcript} -> {ok, failures} — re-fetches the pinned
                         drand round from the LIVE public beacon (network required)

Run: python3 cr_verify_service.py [port]     (default 8321)

Deliberately NOT a prover: this service never computes receipts for you, because a
verifier that also produces what it checks proves nothing. It also holds no state —
every request carries everything the verdict depends on.
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cr.receipt import (  # noqa: E402
    Receipt,
    ReceiptError,
    check_wellformed,
    sample_indices_of,
    verify,
    verify_sampled,
)

VECTORS = Path(__file__).resolve().parents[1] / "spec" / "CR-v0.1-conformance-vectors.json"


def _receipt(obj) -> Receipt:
    return Receipt(obj["manifest"], obj["certificate"], obj.get("meta", {}))


def handle(path: str, body: dict) -> dict:
    if path == "/wellformed":
        v = check_wellformed(_receipt(body["receipt"]))
        return {"status": v.status, "reason": v.reason}
    if path == "/verify":
        v = verify(_receipt(body["claimed"]), _receipt(body["recomputed"]))
        return {"status": v.status, "reason": v.reason}
    if path == "/verify-sampled":
        r = _receipt(body["receipt"])
        idx = sample_indices_of(r)
        # tensor digests are dtype-sensitive and JSON is not, so the caller states
        # the dtype of their re-executed rows explicitly — guessing here would turn
        # honest rows into a REJECT and teach exactly the wrong lesson
        if "dtype" not in body:
            raise ReceiptError('"dtype" is required (e.g. "uint16"): tensor digests '
                               "are dtype-sensitive and JSON does not carry dtype")
        rows = np.asarray(body["rows"], dtype=np.dtype(body["dtype"]))
        v = verify_sampled(r, rows)
        return {"status": v.status, "reason": v.reason,
                "indices": [int(i) for i in idx]}
    if path == "/beacon-audit":
        from cr.beacon import audit
        t = body["transcript"]
        failures = audit(t.get("beacon", t),
                         commit_time=t.get("commit_received_unix"))
        return {"ok": not failures, "failures": failures}
    raise ReceiptError(f"unknown endpoint {path}")


PAGE = """<!doctype html><meta charset="utf-8">
<title>CR Verifier</title>
<style>body{font:14px/1.5 system-ui;max-width:880px;margin:2rem auto;padding:0 1rem;color:#1d2129}
h1{font-size:1.4rem} textarea{width:100%;height:10rem;font:12px monospace}
button{padding:.4rem 1rem;margin:.4rem .4rem 0 0} pre{background:#f2f6fa;padding:.8rem;white-space:pre-wrap}
.ACCEPT{color:#1a7a3a;font-weight:bold}.REJECT{color:#b02a1e;font-weight:bold}
.MALFORMED,.UNVERIFIABLE{color:#a06000;font-weight:bold}
small{color:#667}</style>
<h1>Computation Receipt verifier</h1>
<p><small>Runs the open reference implementation, locally. A verdict means what the spec
says it means — this page never trusts the prover, and neither should you.</small></p>
<p><b>Verify:</b> paste the CLAIMED receipt and YOUR OWN re-executed receipt.</p>
<textarea id=a placeholder='claimed receipt JSON'></textarea>
<textarea id=b placeholder='your re-executed receipt JSON (leave empty to only check well-formedness)'></textarea>
<p><b>Or sampled:</b> paste a sampled receipt above (first box) and your re-executed rows below.</p>
<textarea id=rows placeholder='re-executed rows as a JSON array (only for sampled receipts)'></textarea>
<p><b>Or beacon transcript:</b></p>
<textarea id=tr placeholder='protocol transcript JSON (audited against the LIVE drand beacon)'></textarea>
<div>
<button onclick="go()">Verify</button>
<button onclick="goSampled()">Verify sampled</button>
<button onclick="goBeacon()">Audit beacon</button>
<a href="/vectors" style="margin-left:1rem">conformance vectors</a></div>
<pre id=out>verdicts appear here</pre>
<script>
async function post(p, body){const r = await fetch(p,{method:'POST',body:JSON.stringify(body)});
  return await r.json();}
function show(o){const e=document.getElementById('out');
  e.innerHTML=(o.status?('<span class="'+o.status+'">'+o.status+'</span> — '+o.reason)
   :JSON.stringify(o,null,1)) + (o.indices?('\\nindices: '+JSON.stringify(o.indices)):'');}
async function go(){try{const a=JSON.parse(document.getElementById('a').value);
  const bt=document.getElementById('b').value.trim();
  show(bt?await post('/verify',{claimed:a,recomputed:JSON.parse(bt)})
        :await post('/wellformed',{receipt:a}));}catch(e){show({error:''+e})}}
async function goSampled(){try{show(await post('/verify-sampled',
  {receipt:JSON.parse(document.getElementById('a').value),
   rows:JSON.parse(document.getElementById('rows').value)}));}catch(e){show({error:''+e})}}
async function goBeacon(){try{show(await post('/beacon-audit',
  {transcript:JSON.parse(document.getElementById('tr').value)}));}catch(e){show({error:''+e})}}
</script>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/vectors":
            self._send(200, VECTORS.read_bytes())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            self._send(200, handle(self.path, body))
        except (ReceiptError, KeyError, ValueError, TypeError) as e:
            self._send(400, {"error": str(e)})
        except Exception as e:  # network failures on beacon audit, etc.
            self._send(502, {"error": str(e)})

    def log_message(self, *a):  # quiet
        pass


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8321
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"CR verifier at http://127.0.0.1:{port}/  (Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
