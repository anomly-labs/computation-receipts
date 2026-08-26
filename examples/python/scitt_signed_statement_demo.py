# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""scitt_signed_statement_demo.py — a CR claim as a real SCITT-style Signed Statement.

Offline. NumPy + stdlib + `cryptography` (for Ed25519 only; the script SKIPs cleanly
with exit 0 if it isn't installed). No COSE library is used: COSE_Sign1 is a small,
fixed CBOR structure (RFC 9052), so the ~50 lines of CBOR this demo needs are written
out here — every byte on the wire is inspectable, which is the point of an interop
example. See INTEROP.md §2 for the mapping this exercises.

What it is NOT: a full SCITT deployment. There is no Transparency Service here, so no
registration and no Receipt (IETF sense: inclusion proof, RFC 9942). The demo produces
and verifies the ISSUER half — the Signed Statement — which is the part that carries
the CR claim. A real deployment signs with its own keys and its Transparency Service's
COSE profile; the demo key is derived from a fixed seed so the output is deterministic,
and must never be used for anything real.

    python3 examples/python/scitt_signed_statement_demo.py

Four steps, one PASS/FAIL line each:

  1. ISSUER   — build a real receipt; payload = canonical CR manifest bytes
                (canonical BEFORE enveloping, so the payload is byte-stable across
                implementations); protected headers: alg EdDSA, content type
                application/vnd.anomly.cr+json, CWT claims {issuer, subject=the
                receipt's certificate}. Sign per RFC 9052 Sig_structure.
  2. WIRE     — the Signed Statement is a tagged COSE_Sign1; print its size + prefix.
  3. VERIFIER — decode the CBOR back, check the headers, verify the Ed25519
                signature, then recompute sha256(canonical(payload manifest)) and
                match it against the CWT subject claim. Signature says WHO asserted;
                only CR re-execution says the claim is TRUE (not done here — this
                demo is carriage, exactly like the in-toto one).
  4. TAMPER   — flip one payload byte and watch signature verification refuse it.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from cr import build_receipt  # noqa: E402
from cr.receipt import canonical_bytes, certificate_of  # noqa: E402

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
    )
    from cryptography.exceptions import InvalidSignature
except ImportError:  # pragma: no cover
    print("  [SKIP] `cryptography` not installed — SCITT demo needs Ed25519. "
          "pip install cryptography")
    sys.exit(0)

CONTENT_TYPE = "application/vnd.anomly.cr+json"
ISSUER = "did:web:provider.example"
ALG_EDDSA = -8            # COSE algorithm registry
H_ALG, H_CTY, H_CWT = 1, 3, 15   # COSE header labels (15 = CWT claims, RFC 9597)
CWT_ISS, CWT_SUB = 1, 2


# ---------------------------------------------------------------- minimal CBOR
def _head(major: int, n: int) -> bytes:
    if n < 24:
        return bytes([(major << 5) | n])
    for bits, code, fmt in ((8, 24, "B"), (16, 25, ">H"), (32, 26, ">I"), (64, 27, ">Q")):
        if n < (1 << bits):
            return bytes([(major << 5) | code]) + struct.pack(fmt, n)
    raise ValueError("length too large")


def cbor(obj) -> bytes:
    """Encode the small subset COSE_Sign1 needs: int, bytes, str, list, dict, tag."""
    if isinstance(obj, Tag):
        return _head(6, obj.n) + cbor(obj.value)
    if isinstance(obj, bool):
        raise TypeError("bool not needed here")
    if isinstance(obj, int):
        return _head(0, obj) if obj >= 0 else _head(1, -1 - obj)
    if isinstance(obj, bytes):
        return _head(2, len(obj)) + obj
    if isinstance(obj, str):
        b = obj.encode()
        return _head(3, len(b)) + b
    if isinstance(obj, list):
        return _head(4, len(obj)) + b"".join(cbor(x) for x in obj)
    if isinstance(obj, dict):
        # definite-length map, keys in insertion order (fixed by this script)
        return _head(5, len(obj)) + b"".join(cbor(k) + cbor(v) for k, v in obj.items())
    raise TypeError(type(obj))


class Tag:
    def __init__(self, n: int, value):
        self.n, self.value = n, value


def cbor_decode(b: bytes):
    """Decode the same subset; returns (value, rest)."""
    ib, rest = b[0], b[1:]
    major, info = ib >> 5, ib & 0x1F
    if info < 24:
        n = info
    else:
        size = {24: 1, 25: 2, 26: 4, 27: 8}[info]
        n = int.from_bytes(rest[:size], "big")
        rest = rest[size:]
    if major == 0:
        return n, rest
    if major == 1:
        return -1 - n, rest
    if major == 2:
        return rest[:n], rest[n:]
    if major == 3:
        return rest[:n].decode(), rest[n:]
    if major == 4:
        out = []
        for _ in range(n):
            v, rest = cbor_decode(rest)
            out.append(v)
        return out, rest
    if major == 5:
        out = {}
        for _ in range(n):
            k, rest = cbor_decode(rest)
            v, rest = cbor_decode(rest)
            out[k] = v
        return out, rest
    if major == 6:
        v, rest = cbor_decode(rest)
        return Tag(n, v), rest
    raise ValueError(f"major {major} not in demo subset")


def _line(ok: bool, label: str, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def main() -> int:
    ok = True

    # -- 1. ISSUER --------------------------------------------------------------------
    rng = np.random.default_rng(20260826)
    terms = rng.standard_normal(4096)
    receipt = build_receipt(profile="float64",
                            computation="hpc.allreduce.global-sum",
                            inputs={"terms": terms},
                            output=np.asarray([float(np.sum(terms))]))
    r = json.loads(receipt.to_json())
    payload = canonical_bytes(r["manifest"])         # canonical BEFORE enveloping
    protected_map = {
        H_ALG: ALG_EDDSA,
        H_CTY: CONTENT_TYPE,
        H_CWT: {CWT_ISS: ISSUER, CWT_SUB: r["certificate"]},
    }
    protected = cbor(protected_map)
    # DEMO KEY ONLY: fixed seed for deterministic output; never sign anything real.
    key = Ed25519PrivateKey.from_private_bytes(b"\x01" * 32)
    sig_structure = cbor(["Signature1", protected, b"", payload])
    signature = key.sign(sig_structure)
    signed_statement = cbor(Tag(18, [protected, {}, payload, signature]))
    ok &= _line(True, "ISSUER",
                f"Signed Statement over certificate {r['certificate'][:23]}… "
                f"(payload = {len(payload)}-byte canonical manifest)")

    # -- 2. WIRE ----------------------------------------------------------------------
    ok &= _line(signed_statement[0] == 0xD2, "WIRE",
                f"tagged COSE_Sign1 (0xd2…), {len(signed_statement)} bytes, "
                f"sha256 {hashlib.sha256(signed_statement).hexdigest()[:16]}…")

    # -- 3. VERIFIER ------------------------------------------------------------------
    tag, rest = cbor_decode(signed_statement)
    prot_b, _unprot, payload2, sig2 = tag.value
    hdr, _ = cbor_decode(prot_b)
    hdr_ok = (tag.n == 18 and hdr[H_ALG] == ALG_EDDSA and hdr[H_CTY] == CONTENT_TYPE
              and hdr[H_CWT][CWT_ISS] == ISSUER)
    try:
        key.public_key().verify(sig2, cbor(["Signature1", prot_b, b"", payload2]))
        sig_ok = True
    except InvalidSignature:
        sig_ok = False
    manifest2 = json.loads(payload2)
    bind_ok = certificate_of(manifest2) == hdr[H_CWT][CWT_SUB] == r["certificate"]
    ok &= _line(hdr_ok and sig_ok and bind_ok, "VERIFIER",
                "headers + Ed25519 signature verify; sha256(canonical(payload)) == "
                "CWT subject == the claimed certificate")

    # -- 4. TAMPER --------------------------------------------------------------------
    bad = bytearray(payload2)
    bad[len(bad) // 2] ^= 0x01
    try:
        key.public_key().verify(sig2, cbor(["Signature1", prot_b, b"", bytes(bad)]))
        caught = False
    except InvalidSignature:
        caught = True
    ok &= _line(caught, "TAMPER",
                "one flipped payload byte → signature verification refuses")

    print(f"\n{'PASS' if ok else 'FAIL'}: CR claim carried as a SCITT-style Signed "
          f"Statement (issuer half; registration/Receipt is a Transparency Service's "
          f"job — see INTEROP.md §2)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
