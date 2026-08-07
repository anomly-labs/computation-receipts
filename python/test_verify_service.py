#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""test_verify_service.py — the verifier service accepts honestly and refuses correctly.

Starts the service in-process on a scratch port and drives it over real HTTP: an
honest verify (two independently built receipts of the same computation) must ACCEPT;
a tampered output must REJECT; a sampled receipt verifies from the verifier's own rows
and refuses tampered rows; malformed input gets a 400, not a stack trace.
"""
import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cr.receipt import build_receipt, build_sampled_receipt, sample_indices_of  # noqa: E402
from cr_verify_service import H  # noqa: E402

PORT = 8329


def post(path, body, expect=200):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            assert r.status == expect, (path, r.status)
            return json.load(r)
    except urllib.error.HTTPError as e:
        assert e.code == expect, (path, e.code, e.read()[:200])
        return json.load(e)


def rjson(r):
    return {"manifest": r.manifest, "certificate": r.certificate, "meta": r.meta}


def main() -> int:
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    rng = np.random.default_rng(7)
    Y = rng.integers(1, 0x7FFF, size=(32, 8), dtype=np.uint16)
    kw = dict(profile="bposit16-quire256", computation="example.service")
    claimed = build_receipt(output=Y, **kw)
    recomputed = build_receipt(output=Y, **kw)
    Yt = Y.copy()
    Yt[0, 0] ^= 1
    tampered = build_receipt(output=Yt, **kw)

    ok = {}
    ok["ui_serves"] = b"Computation Receipt verifier" in urllib.request.urlopen(
        f"http://127.0.0.1:{PORT}/").read()
    ok["vectors_serve"] = len(json.load(urllib.request.urlopen(
        f"http://127.0.0.1:{PORT}/vectors"))) == 13
    ok["honest_accept"] = post("/verify", {"claimed": rjson(claimed),
                                           "recomputed": rjson(recomputed)})["status"] == "ACCEPT"
    ok["tamper_reject"] = post("/verify", {"claimed": rjson(claimed),
                                           "recomputed": rjson(tampered)})["status"] == "REJECT"
    ok["wellformed"] = post("/wellformed", {"receipt": rjson(claimed)})["status"] == "ACCEPT"

    sr = build_sampled_receipt(output=Y, sample_size=4, challenge=b"svc", **kw)
    idx = sample_indices_of(sr)
    res = post("/verify-sampled", {"receipt": rjson(sr), "rows": Y[idx].tolist(), "dtype": "uint16"})
    ok["sampled_accept"] = res["status"] == "ACCEPT" and res["indices"] == [int(i) for i in idx]
    bad = Y[idx].copy()
    bad[0, 0] ^= 1
    ok["sampled_reject"] = post("/verify-sampled", {"receipt": rjson(sr),
                                                    "rows": bad.tolist(), "dtype": "uint16"})["status"] == "REJECT"
    ok["garbage_is_400"] = "error" in post("/verify", {"claimed": {"manifest": {}}},
                                           expect=400)

    srv.shutdown()
    for k, v in ok.items():
        print(f"  {k:<18} {'ok' if v else 'FAIL'}")
    failed = [k for k, v in ok.items() if not v]
    if failed:
        print(f"FAIL: {failed}")
        return 1
    print("PASS: the verifier service accepts honest receipts, refuses tampered ones "
          "(full and sampled), serves the vectors and UI, and fails cleanly on garbage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
