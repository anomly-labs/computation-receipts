# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""hpc_reduction_demo.py — run a real HPC reduction through Computation Receipts, yourself.

This is a SELF-CONTAINED, publicly runnable demo. It uses only NumPy, the Python standard
library, and this repository's reference `cr` package — no b-posit arithmetic, no special
hardware, nothing from Anomly's private tree. An external evaluator can execute it and watch the
CR verifier reach its verdicts on a real, ill-conditioned reduction (the `MPI_Allreduce`
reproducibility problem: summing across N ranks vs M ranks happens in a different order, so
floating point disagrees — for an ill-conditioned sum, in the FIRST bits).

    cd examples/python && python3 hpc_reduction_demo.py
    #   (or from the repo root: python3 examples/python/hpc_reduction_demo.py)

It shows four things, each REAL and reproducible on your machine:

  1. PAIN — an ill-conditioned float64 reduction summed in a 64-rank order vs a 1024-rank order
     gives DIFFERENT numbers (and both are wrong). Two receipts, verdict UNVERIFIABLE:
     "outputs differ, which is expected and proves nothing."

  2. HONESTY — a well-conditioned float64 reduction gives the SAME number in both orders, yet CR
     STILL says UNVERIFIABLE: "outputs happened to match, but agreement is not evidence." CR never
     sells a coincidence as a proof — the profile is order-dependent in general, so agreement here
     is luck, not a certificate.

  3. A CHEAT GOES UNFLAGGED — a dishonest prover claims a WRONG total over the same inputs. The
     binding can't catch it (inputs are byte-identical) and the verdict is UNVERIFIABLE, not REJECT:
     under order-dependent arithmetic a wrong number is indistinguishable from a legitimate
     re-execution on a different rank count. Float cannot flag the cheat. Exact arithmetic is exactly
     the gap-closer that turns this into a REJECT.

  4. WHY EXACT FIXES IT — the same ill-conditioned sum, accumulated exactly with Python's
     `fractions.Fraction`, is byte-identical in BOTH orders (order-independent by construction). That
     is the property a receipt needs to reach ACCEPT.

The one thing this public demo does NOT do is emit an ACCEPTing receipt, and the reason is stated
honestly below: ACCEPT requires the receipt to name an *order-independent arithmetic profile* the
verifier can assess, and the registered ones today are the b-posit / 256-bit-quire family, whose
implementation lives in Anomly's research tree (see ../../REAL-WORKLOADS.md). A generic public
"exact-accumulation" profile would let a pure-Python evaluator close the loop to ACCEPT; adding one
is a spec/registry decision, flagged at the end — not taken here.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np

# make the reference `cr` package importable whether run from repo root or examples/python/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
from cr import REJECT, UNVERIFIABLE, build_receipt, verify  # noqa: E402

COMPUTATION = "hpc.allreduce.global-sum"


def ill_conditioned_terms(k: int = 4096, scale: float = 1e9, seed: int = 20260810):
    """A global sum whose terms nearly cancel: true total O(1), terms O(1e9) → condition ~1e9, so
    float loses every significant bit and the loss depends on the summation (rank) order."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(k // 2) * scale
    terms = np.concatenate([u, -u])                       # cancels to 0 exactly
    rng.shuffle(terms)
    terms = np.concatenate([terms, [1.0, 0.5, -0.25]])    # known small true remainder = +1.25
    return terms.astype(np.float64)


def well_conditioned_terms(k: int = 4096, seed: int = 7):
    """Small integer-valued terms: float64 sums integers < 2^53 exactly, so BOTH orders agree to the
    last bit — the setup for showing CR refuses to certify even a genuine coincidence."""
    rng = np.random.default_rng(seed)
    return rng.integers(-1000, 1000, size=k).astype(np.float64)


def rank_order(n: int, ranks: int) -> np.ndarray:
    """A permutation standing in for how `ranks` MPI ranks would partition and combine the sum."""
    return np.random.default_rng(ranks).permutation(n)


def float_reduce(terms: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Sequential float64 accumulation in `perm` order (a specific rank decomposition's order)."""
    acc = np.float64(0.0)
    for i in perm:
        acc += terms[i]
    return np.array([acc], dtype=np.float64)


def exact_reduce(terms: np.ndarray, perm: np.ndarray) -> Fraction:
    """Exact rational accumulation — order-independent by construction (the reference truth)."""
    total = Fraction(0)
    for i in perm:
        total += Fraction(float(terms[i]))
    return total


def _receipt(terms: np.ndarray, total: np.ndarray, *, profile: str):
    return build_receipt(profile=profile, computation=COMPUTATION,
                         inputs={"terms": terms}, output=total)


def main() -> int:
    line = "=" * 84
    print(line)
    print("Run a real HPC reduction through Computation Receipts — publicly, on your machine")
    print(line)

    # ---- 1. PAIN: ill-conditioned float64, two rank orders disagree ----
    terms = ill_conditioned_terms()
    n = terms.shape[0]
    p64, p1024 = rank_order(n, 64), rank_order(n, 1024)
    f64 = float_reduce(terms, p64)
    f1024 = float_reduce(terms, p1024)
    exact = exact_reduce(terms, p64)
    r_prover = _receipt(terms, f64, profile="float64")
    r_verifier = _receipt(terms, f1024, profile="float64")
    v = verify(r_prover, r_verifier)
    print(f"\n1. ILL-CONDITIONED float64 sum ({n:,} terms, condition ~1e9, exact total {float(exact):+.4f}):")
    print(f"     64-rank order   : {f64[0]:+.6f}")
    print(f"     1024-rank order : {f1024[0]:+.6f}   (different core count -> different answer, both wrong)")
    print(f"     CR verify() -> {v.status}: {v.reason}")
    assert v.status == UNVERIFIABLE

    # ---- 2. HONESTY: well-conditioned float64, both orders agree, still UNVERIFIABLE ----
    wterms = well_conditioned_terms()
    wn = wterms.shape[0]
    w_a = float_reduce(wterms, rank_order(wn, 64))
    w_b = float_reduce(wterms, rank_order(wn, 1024))
    rw_a = _receipt(wterms, w_a, profile="float64")
    rw_b = _receipt(wterms, w_b, profile="float64")
    vw = verify(rw_a, rw_b)
    print(f"\n2. WELL-CONDITIONED float64 sum ({wn:,} small integer terms):")
    print(f"     64-rank order   : {w_a[0]:+.1f}")
    print(f"     1024-rank order : {w_b[0]:+.1f}   (identical to the last bit)")
    print(f"     outputs byte-identical? {rw_a.manifest['output']['digest'] == rw_b.manifest['output']['digest']}")
    print(f"     CR verify() -> {vw.status}: {vw.reason}")
    print("     ^ CR refuses to certify a coincidence: float is order-dependent in general, so")
    print("       agreement here is luck, not proof.")
    assert vw.status == UNVERIFIABLE

    # ---- 3. A CHEAT ON THE OUTPUT GOES UNFLAGGED: same inputs, a wrong claimed sum ----
    # A dishonest prover claims a different total over the SAME terms (a wrong/cheated reduction).
    # Inputs are byte-identical, so the binding does not catch it; the arithmetic must — and float
    # cannot, because a wrong number is indistinguishable from an honest different-rank re-execution.
    r_cheat = _receipt(terms, f64 + 100.0, profile="float64")   # claims total +101.25, same inputs
    vt = verify(r_cheat, r_verifier)                            # verifier re-runs honestly (+1.2501)
    print("\n3. A CHEATED OUTPUT over the SAME inputs (prover claims a wrong sum):")
    print(f"     prover claims   : {(f64 + 100.0)[0]:+.6f}   (a fabricated total, terms unchanged)")
    print(f"     verifier re-runs: {f1024[0]:+.6f}")
    print(f"     CR verify() -> {vt.status}: {vt.reason}")
    print("     ^ NOT REJECT: the binding can't catch it (inputs match) and float's order-dependence")
    print("       means a wrong number is indistinguishable from an honest different-rank re-execution.")
    print("       Exact arithmetic is precisely what turns this into a REJECT (see REAL-WORKLOADS.md).")
    assert vt.status == UNVERIFIABLE

    # ---- 4. WHY EXACT FIXES IT: exact accumulation is order-independent ----
    e64 = exact_reduce(terms, p64)
    e1024 = exact_reduce(terms, p1024)
    print("\n4. EXACT accumulation of the SAME ill-conditioned sum (fractions.Fraction):")
    print(f"     64-rank order   : {float(e64):+.6f}")
    print(f"     1024-rank order : {float(e1024):+.6f}")
    print(f"     byte-identical across rank orders? {e64 == e1024}   <- order-independent, for ANY inputs")
    print("     This is the property a receipt needs to reach ACCEPT.")
    assert e64 == e1024

    print("\n" + line)
    print("What you just verified yourself:")
    print("  * float's honest CR verdict is UNVERIFIABLE — whether it disagrees, agrees, OR is")
    print("    tampered. That is the format telling the truth about order-dependent arithmetic.")
    print("  * exact accumulation is order-independent, which is what an ACCEPT needs.")
    print()
    print("Why this public demo stops short of an ACCEPTing receipt (stated honestly):")
    print("  verify() returns ACCEPT only when the receipt names an ORDER-INDEPENDENT arithmetic")
    print("  profile it can assess. The registered ones today are the b-posit / 256-bit-quire")
    print("  family, whose implementation lives in Anomly's research tree (see ../../REAL-WORKLOADS.md")
    print("  for that half run on real Llama weights and reductions). A generic public")
    print("  'exact-accumulation' profile would let this pure-Python demo close the loop to ACCEPT;")
    print("  a review-ready proposal (a verified, no-spec-change option) is in")
    print("  docs/proposals/exact-accumulation-profile.md. Adding it is a deliberate decision, not")
    print("  taken here.")
    print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
