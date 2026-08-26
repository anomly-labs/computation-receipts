# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""in_toto_statement_demo.py — carry a Computation Receipt inside an in-toto v1 Statement.

SELF-CONTAINED and offline: NumPy + stdlib + this repo's reference `cr` package. See
INTEROP.md for the mapping rules this demo exercises. It shows the *carriage* layer —
how a CR claim rides the supply-chain stack — not new verification semantics: the
receipt here names the order-dependent `float64` profile, so re-execution verdicts
stay exactly as honest as `hpc_reduction_demo.py` explains (no ACCEPT is claimed).

    python3 examples/python/in_toto_statement_demo.py

Five steps, one PASS/FAIL line each:

  1. PRODUCER  — build a real receipt for a float64 reduction, wrap it verbatim as the
                 `predicate` of an in-toto v1 Statement whose subject is the certified
                 output (subject digest = manifest.output.digest, per INTEROP.md rule 2).
  2. DSSE      — compute the DSSE pre-authentication encoding (PAE) for
                 payloadType application/vnd.in-toto+json: the exact bytes a signer
                 (Sigstore, KMS, offline key) would sign. No key is used here —
                 signing is the carrier's job, not CR's (INTEROP.md rule 3).
  3. CONSUMER  — parse the Statement back: predicateType is CR's, and the subject
                 digest equals the embedded manifest's output digest (the join
                 between the two trust systems).
  4. INTEGRITY — recompute sha256(canonical(manifest)) from the carried predicate and
                 match it against the carried `certificate`. Arithmetic-free, so any
                 in-toto consumer can run this check with no numerics at all.
  5. TAMPER    — flip one field inside the carried predicate (the claimed output
                 digest) and watch step-4's check refuse it. The carrier's signature
                 is NOT needed to catch this: the claim is self-binding.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

# make the reference `cr` package importable whether run from repo root or examples/python/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from cr import build_receipt  # noqa: E402
from cr.receipt import canonical_bytes, certificate_of  # noqa: E402

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://github.com/anomly-labs/computation-receipts/blob/main/spec/CR-v0.1.md"
PAYLOAD_TYPE = "application/vnd.in-toto+json"


def _line(ok: bool, label: str, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def dsse_pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE v1 pre-authentication encoding: PAE(type, payload) =
    b"DSSEv1 " + len(type) + b" " + type + b" " + len(payload) + b" " + payload."""
    t = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def main() -> int:
    ok = True

    # -- 1. PRODUCER ------------------------------------------------------------------
    rng = np.random.default_rng(20260826)
    terms = rng.standard_normal(4096)
    total = np.asarray([float(np.sum(terms))])
    receipt = build_receipt(
        profile="float64",
        computation="hpc.allreduce.global-sum",
        inputs={"terms": terms},
        output=total,
    )
    r = json.loads(receipt.to_json())
    out_digest = r["manifest"]["output"]["digest"]          # "sha256:<hex>"
    statement = {
        "_type": STATEMENT_TYPE,
        "subject": [{
            "name": "global-sum.result",
            "digest": {"sha256": out_digest.split(":", 1)[1]},
        }],
        "predicateType": PREDICATE_TYPE,
        "predicate": r,                                      # the receipt, verbatim
    }
    ok &= _line(statement["predicate"]["certificate"] == r["certificate"],
                "PRODUCER", f"Statement wraps receipt {r['certificate'][:23]}… verbatim")

    # -- 2. DSSE ----------------------------------------------------------------------
    payload = json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()
    pae = dsse_pae(PAYLOAD_TYPE, payload)
    ok &= _line(pae.startswith(b"DSSEv1 ") and payload in pae,
                "DSSE", f"PAE is {len(pae)} bytes — hand these to your signer "
                        f"(sha256 {hashlib.sha256(pae).hexdigest()[:16]}…)")

    # -- 3. CONSUMER ------------------------------------------------------------------
    parsed = json.loads(payload)
    pred = parsed["predicate"]
    subject_hex = parsed["subject"][0]["digest"]["sha256"]
    join_ok = (parsed["predicateType"] == PREDICATE_TYPE
               and f"sha256:{subject_hex}" == pred["manifest"]["output"]["digest"])
    ok &= _line(join_ok, "CONSUMER",
                "predicateType is CR's and subject digest == manifest.output.digest")

    # -- 4. INTEGRITY (arithmetic-free) ------------------------------------------------
    recomputed = certificate_of(pred["manifest"])
    ok &= _line(recomputed == pred["certificate"], "INTEGRITY",
                "sha256(canonical(manifest)) recomputed from the carried bytes matches "
                "the carried certificate")

    # -- 5. TAMPER --------------------------------------------------------------------
    tampered = json.loads(canonical_bytes(pred))             # deep copy via canonical JSON
    tampered["manifest"]["output"]["digest"] = "sha256:" + "0" * 64
    caught = certificate_of(tampered["manifest"]) != tampered["certificate"]
    ok &= _line(caught, "TAMPER",
                "one edited field inside the carried predicate → certificate mismatch → "
                "refused, no signature needed")

    print(f"\n{'PASS' if ok else 'FAIL'}: CR rides an in-toto Statement round-trip "
          f"(carriage only — verification semantics unchanged; see INTEROP.md)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
