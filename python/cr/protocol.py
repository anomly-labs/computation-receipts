# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""protocol.py — the verifier side of the CR §10 challenge protocol.

A sampled receipt with an empty (or prover-chosen) challenge is evidence against faults,
not against an adaptive prover: the prover can grind — recompute a tampered output,
derive the indices, retry until the tampered rows fall outside the sample. The defence
is temporal: the prover COMMITS to the full output first (a full receipt), then a
challenge the prover could not predict arrives, and the sample indices derive from the
committed manifest plus that challenge.

The challenge can come from `issue_nonce()` on the verifier's own machine (the
counterparty must then trust the two of you did not collude) or from a public
randomness beacon via `cr.beacon` (nobody chose it, and anyone can re-fetch the round
forever). The verifier trusts nothing from the prover beyond the two receipts: it
re-derives the indices itself and re-executes the sampled rows with its own
implementation on its own hardware.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Mapping

import numpy as np

from .receipt import (
    ACCEPT,
    MALFORMED,
    REJECT,
    Receipt,
    Verdict,
    check_wellformed,
    sample_indices_of,
    verify_sampled,
)

NONCE_BYTES = 32


def issue_nonce() -> bytes:
    """A fresh challenge. Issue it only AFTER the commitment receipt is in hand —
    a nonce the prover sees before fixing its output protects nothing."""
    return os.urandom(NONCE_BYTES)


def base_manifest(r: Receipt) -> Mapping[str, Any]:
    """The manifest minus the sample section and version tag — what commit and reveal
    must agree on for the reveal to be about the committed computation."""
    return {k: v for k, v in r.manifest.items() if k not in ("sample", "cr")}


def check_reveal(
    commit: Receipt,
    reveal: Receipt,
    nonce: bytes,
    reexecute_rows: Callable[[np.ndarray], np.ndarray],
) -> Verdict:
    """Protocol verdict on a reveal against a commitment and the verifier's own nonce.

    `reexecute_rows(indices)` must return the verifier's OWN re-execution of those output
    rows from the operands it holds — never rows the prover supplied.
    """
    wf = check_wellformed(commit)
    if not wf:
        return Verdict(MALFORMED, f"commitment not well-formed: {wf.reason}")
    if "sample" in commit.manifest:
        return Verdict(MALFORMED,
                       "commitment must be a FULL receipt: a sampled commitment lets the "
                       "prover grind before the challenge exists")
    if not nonce:
        return Verdict(MALFORMED, "empty challenge — that is the fixed-string mode this "
                                  "protocol exists to replace")
    if base_manifest(commit) != base_manifest(reveal):
        return Verdict(REJECT,
                       "reveal is not about the committed computation — the prover "
                       "changed operands, arithmetic or output after the challenge was "
                       "issued")
    sample = reveal.manifest.get("sample")
    if not isinstance(sample, Mapping):
        return Verdict(MALFORMED, "reveal carries no sample section")
    if sample.get("challenge") != nonce.hex():
        return Verdict(REJECT,
                       "reveal challenge is not the issued one — a prover-chosen "
                       "challenge is grindable and proves nothing")
    idx = sample_indices_of(reveal)  # re-derived, never trusted from the prover
    return verify_sampled(reveal, np.asarray(reexecute_rows(idx)))


def _selftest() -> int:
    """Protocol-logic gate, arithmetic-free: honest ACCEPT and one refusal per lie.

    Uses arbitrary code arrays as the 'computation' — this tests the PROTOCOL, not any
    number system, so it runs identically for every implementer of the spec.
    """
    from .receipt import build_receipt, build_sampled_receipt

    rng = np.random.default_rng(20260806)
    Yc = rng.integers(1, 0x7FFF, size=(48, 24), dtype=np.uint16)
    kw = dict(profile="bposit16-quire256", computation="example.selftest")

    def receipts(out, challenge):
        return (build_receipt(output=out, **kw),
                build_sampled_receipt(output=out, sample_size=6, challenge=challenge, **kw))

    honest_rows = lambda idx: Yc[idx]  # noqa: E731 — the verifier "re-executes" honestly
    ok = {}
    nonce = issue_nonce()
    commit, reveal = receipts(Yc, nonce)
    ok["honest_accept"] = check_reveal(commit, reveal, nonce, honest_rows).status == ACCEPT

    _, ground = receipts(Yc, b"prover-chosen")
    ok["prover_chosen_challenge_reject"] = (
        check_reveal(commit, ground, nonce, honest_rows).status == REJECT)

    Yt = Yc.copy()
    Yt[0, 0] ^= 1
    _, swapped = receipts(Yt, nonce)
    ok["post_challenge_swap_reject"] = (
        check_reveal(commit, swapped, nonce, honest_rows).status == REJECT)

    idx = sample_indices_of(reveal)
    bad = Yc[idx].copy()
    bad[0, 0] ^= 1
    ok["sampled_row_tamper_reject"] = (
        check_reveal(commit, reveal, nonce, lambda i: bad).status == REJECT)

    ok["sampled_commitment_refused"] = (
        check_reveal(reveal, reveal, nonce, honest_rows).status == MALFORMED)

    _, a = receipts(Yc, b"\x01" * NONCE_BYTES)
    _, b = receipts(Yc, b"\x02" * NONCE_BYTES)
    ok["indices_depend_on_challenge"] = (
        sample_indices_of(a).tolist() != sample_indices_of(b).tolist())

    for k, v in ok.items():
        print(f"  {k:<34} {'ok' if v else 'FAIL'}")
    failed = [k for k, v in ok.items() if not v]
    if failed:
        print(f"FAIL: {failed}")
        return 1
    print("PASS: protocol accepts an honest prover and refuses a prover-chosen "
          "challenge, a post-challenge output swap, a tampered sampled row, and a "
          "grindable sampled commitment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
