#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""check_conformance.py — verify this implementation against the PUBLISHED vectors.

Two directions, both required:

  1. The shipped implementation regenerates every published vector byte-for-byte —
     if it cannot, the code and the spec have diverged and NEITHER should be trusted
     until reconciled.
  2. The negative vectors' required refusals are enforced — a verifier that cannot
     say no is worthless, so the refusals are the part of conformance that matters
     most.

A third-party implementation conforms when it reproduces the same vectors from the
spec document alone. If you are that third party and a vector is ambiguous in the
spec, that ambiguity is a spec bug: please report it.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cr.receipt import Receipt, check_wellformed, conformance_vectors  # noqa: E402

SPEC_VECTORS = Path(__file__).resolve().parents[1] / "spec" / "CR-v0.1-conformance-vectors.json"


def main() -> int:
    published = {v["name"]: v for v in json.loads(SPEC_VECTORS.read_text(encoding="utf-8"))}
    regenerated = {v["name"]: v for v in conformance_vectors()}

    ok = True
    if set(published) != set(regenerated):
        print(f"FAIL: vector sets differ: {set(published) ^ set(regenerated)}")
        ok = False
    for name in sorted(published):
        same = published.get(name) == regenerated.get(name)
        print(f"  {'ok  ' if same else 'FAIL'} {name}")
        ok = ok and same

    # the refusals: a conforming verifier MUST return exactly these verdicts
    from cr.receipt import verify
    for name, v in sorted(published.items()):
        if "expect_verdict" not in v:
            continue
        r = Receipt(json.loads(v["manifest_canonical"]), v["certificate"], {})
        got = check_wellformed(r)
        verdict = got.status
        if verdict == "ACCEPT":  # well-formed — refusal must come from assessability
            verdict = verify(r, None).status
        good = verdict == v["expect_verdict"]
        print(f"  {'ok  ' if good else 'FAIL'} {name}: required {v['expect_verdict']}, got {verdict}")
        ok = ok and good

    print("PASS: implementation matches the published vectors, refusals included."
          if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
