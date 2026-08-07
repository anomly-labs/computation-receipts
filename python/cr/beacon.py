# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""beacon.py — public-randomness challenges for the §10 protocol (drand).

The 2026-08-05 on-silicon protocol run issued its nonce from the verifier's own
`os.urandom`, and the research note says what that leaves open: a third party must
trust that prover and verifier did not collude on the nonce. This module closes that
gap with drand (the League of Entropy beacon): the challenge becomes the published
randomness of a specific beacon ROUND — a value neither party chose, that did not
exist before its round time, and that anyone can re-fetch from the public network
years later to audit the transcript.

Round discipline is what makes it sound. The round is not picked by the prover (that
would be grindable at one attempt per 30 s): the verifier pins R = the first round
whose time is strictly after it received the commitment, waits for R to be published,
and uses its randomness. The transcript records the round number, so an auditor
checks three things: the commitment predates round R's publish time, the challenge
equals round R's randomness (re-fetched from the beacon, not from the transcript),
and the sample indices derive from that challenge.

Honest scope: `audit()` checks the transcript against the LIVE public beacon — chain
hash, round, randomness. It does not re-verify the BLS signature chain itself (the
signature is recorded in the provenance for a third party who wants to do that with a
pairing library; drand's own verifiers exist for exactly this). And a beacon is only
as available as the network: `beacon_challenge` blocks and raises if the round never
publishes, and `audit` returns a network-error FAILURE rather than raising — but neither
ever substitutes local randomness, because a locally-generated "beacon" is the collusion
this module exists to remove.
"""
from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

DRAND_URL = "https://api.drand.sh"
# the drand mainnet "default" chain (30 s period, chained BLS); pinned so a transcript
# can never silently switch to a chain the auditor did not expect
CHAIN_HASH = "8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce"
PERIOD = 30
GENESIS = 1595431050


def _get(path: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(f"{DRAND_URL}{path}", timeout=timeout) as r:
        return json.load(r)


def chain_info() -> dict[str, Any]:
    return _get("/info")


def fetch_round(round_no: int | None = None) -> dict[str, Any]:
    """One published beacon round (or the latest). Raises on network failure."""
    return _get(f"/public/{round_no}" if round_no else "/public/latest")


def round_at(unix_time: float) -> int:
    """The round whose randomness is published at or before `unix_time`."""
    return max(1, int((unix_time - GENESIS) // PERIOD) + 1)


def round_time(round_no: int) -> int:
    """Publish time of a round — what an auditor compares the commitment time against."""
    return GENESIS + (round_no - 1) * PERIOD


def first_round_after(unix_time: float) -> int:
    """The verifier's round pin: the first round strictly after the commitment."""
    return round_at(unix_time) + 1


def beacon_challenge(commit_time: float, *, max_wait: float = 90.0) -> tuple[bytes, dict[str, Any]]:
    """The §10 challenge from the first beacon round after `commit_time`.

    Blocks until that round is published (at most one period plus skew). Returns the
    32-byte randomness and a provenance dict for the transcript. The caller supplies
    `commit_time` = when it received the prover's commitment; passing a later time is
    safe (a later round), passing an earlier one defeats the purpose.
    """
    r = first_round_after(commit_time)
    deadline = time.time() + max_wait
    while True:
        try:
            pulse = fetch_round(r)
            break
        except Exception:
            if time.time() > deadline:
                raise TimeoutError(f"drand round {r} not available within {max_wait}s")
            time.sleep(3.0)
    provenance = {
        "source": "drand", "chain_hash": CHAIN_HASH, "round": r,
        "round_time": round_time(r), "period": PERIOD,
        "randomness": pulse["randomness"], "signature": pulse.get("signature", ""),
    }
    return bytes.fromhex(pulse["randomness"]), provenance


def audit(provenance: dict[str, Any], *, commit_time: float | None = None) -> list[str]:
    """Third-party audit of a transcript's beacon provenance against the LIVE beacon.

    Returns a list of failures (empty = audit passed). Never trusts the transcript's
    randomness: re-fetches the pinned round from the public network and compares.
    """
    failures = []
    if provenance.get("source") != "drand":
        return [f"unknown beacon source {provenance.get('source')!r}"]
    if provenance.get("chain_hash") != CHAIN_HASH:
        failures.append("transcript pins a different drand chain than expected")
    # A public auditing tool must return a verdict, not raise, on a hostile transcript:
    # a missing/invalid round or an unreachable beacon is an audit FAILURE, not a crash.
    try:
        r = int(provenance["round"])
        if r < 1:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        return failures + [f"transcript round is missing or invalid: {provenance.get('round')!r}"]
    try:
        pulse = fetch_round(r)
    except Exception as e:
        return failures + [f"could not re-fetch round {r} from the beacon to verify: {e}"]
    if pulse.get("randomness") != provenance.get("randomness"):
        failures.append(f"round {r} randomness does not match the public beacon")
    if provenance.get("round_time") != round_time(r):
        failures.append("transcript round_time inconsistent with chain parameters")
    if commit_time is not None and round_time(r) <= commit_time:
        failures.append("beacon round is not after the commitment — challenge was "
                        "predictable when the output was fixed")
    return failures
