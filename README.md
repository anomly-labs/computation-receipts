# Computation Receipts (CR)

**A receipt for a numeric computation that anyone can check by re-executing it — on any
hardware, with any conforming implementation, with no trusted machine anywhere.**

Ordinary floating-point arithmetic rounds inside its sums, so honest machines disagree
in the last bits and "is this result correct?" has no checkable answer. Computed with
**exact accumulation** (e.g. a 256-bit quire), the answer is bit-identical on every
honest execution — so a result can carry a *certificate*: a canonical manifest binding
the inputs, model, computation graph, arithmetic profile, and output by digest. A
verifier re-executes and compares. Identical → the unique correct answer. Anything
tampered — one output bit, one substituted weight — → REJECT.

This repository is the format: the specification, the published conformance vectors
(including the refusals a verifier MUST issue), a reference implementation in Python,
and an independently written C emitter.

## Verify everything in two minutes

```bash
cd python
python3 run_checks.py            # ← runs all five checks below, one PASS/FAIL line each

# or run them individually:
python3 check_conformance.py     # implementation vs the 17 published vectors, refusals included
python3 -m cr.protocol           # the §10 challenge-protocol logic, arithmetic-free
python3 test_verify_service.py   # the local verifier service's own gate (honest ACCEPT, tamper REJECT)
python3 conformance_runner.py --demo   # the third-party self-cert runner, grading the reference

# the local web verifier (paste a receipt, get a verdict) — serves until Ctrl-C:
python3 cr_verify_service.py
```

A from-scratch **JavaScript** implementation (emitter *and* verifier) proves the format is
implementable without this repository (spec §9): it reproduces **all 17 vectors** — every
digest, certificate, sampled-index set, chain digest and refusal verdict — graded by the same
runner a third party uses:

```bash
node examples/js/cr.mjs | python3 python/conformance_runner.py /dev/stdin   # PASS: 17/17
```

Requires Python 3.10+ and numpy. No other dependencies, no account, no hardware.
The C emitter (`c/cr_receipt.h`) is a single self-contained header — `#include` it and
compile with any C11 compiler; it needs only libc.

## Run a real workload yourself

Beyond the tiny conformance vectors, a self-contained demo runs a real, ill-conditioned HPC
reduction (the `MPI_Allreduce` reproducibility problem) through the verifier — numpy + stdlib
only, nothing from a private tree:

```bash
python3 examples/python/hpc_reduction_demo.py
```

It shows float's honest CR verdict — UNVERIFIABLE whether the two rank orders **disagree**,
**agree** (a coincidence CR refuses to certify), or a **cheated output** is claimed (which float
cannot flag) — and then that exact accumulation is order-independent, the property an ACCEPT needs.
See [`REAL-WORKLOADS.md`](REAL-WORKLOADS.md) for the exact-arithmetic half (real Llama weights and
reductions) that reaches ACCEPT/REJECT.

## What is in the box

| path | what it is |
|---|---|
| `spec/CR-v0.1.md` | the specification — the authority; everything else serves it |
| `spec/CR-v0.1-conformance-vectors.json` | 12 published vectors: canonicalisation, tensor digests, one receipt of each kind, and **three negative vectors** pinning required refusals |
| `python/cr/receipt.py` | reference implementation: build/verify for full (v0.1), sampled (v0.1.1) and chained (v0.1.2) receipts |
| `python/cr/protocol.py` | the verifier side of the commit → challenge → reveal protocol (§10) |
| `python/cr/beacon.py` | public-randomness challenges (drand): pin the first round after the commitment; audit a transcript against the live beacon forever |
| `c/cr_receipt.h` | dependency-free C emitter, written against the spec rather than the reference source; produces byte-identical certificates (differential-fuzzed vs the Python reference, 4,000/4,000 across full/sampled/chunked) |
| `examples/python/hpc_reduction_demo.py` | self-contained, publicly runnable: a real ill-conditioned HPC reduction through the verifier (numpy + stdlib only) — the format doing its job on a real workload |
| `docs/how-computation-receipts-work.pdf` | a 3-page plain-language explainer |
| `IMPLEMENTERS.md` | the third-party implementer's guide: order of attack + every known pitfall |

## What CR deliberately does not do

No confidentiality (the verifier must hold the operands — secrecy is the domain of
other tools, which compose with receipts). No machine identity (machine metadata is
recorded but never certified — the machine's irrelevance is the point). No timing
claims. Stating the boundary is what makes ACCEPT mean something.

## Status, honestly

- The format has **two implementations, both written at Anomly** (Python reference +
  the C emitter). Byte-identical output between them is measured and gated — but two
  in-house implementations are the minimum bar, not ecosystem proof. **A third-party
  implementation built from the spec alone is the thing this repository exists to
  invite.** If a vector is ambiguous, that is a spec bug: please open an issue.
- The protocol has run end to end against physical hardware: an FPGA prover committed,
  a challenge arrived from a public drand beacon round pinned strictly after the
  commitment, and the challenge-derived sample re-executed bit-exactly on x86 — a
  different architecture and implementation. Full LLM forward passes and greedily
  decoded text have been attested as receipt chains at full model width.
- Emitting `order_independent: true` receipts requires exact arithmetic (the profile
  registry in the spec). An implementation that only emits `order_independent: false`
  receipts is still fully conforming — the format's job is to make the distinction
  visible and checkable, not to exclude anyone.

## Implementing CR without this repository

The intended path (spec §9–§10): reproduce the conformance vectors in your language —
this validates canonicalisation and digesting and needs **no arithmetic**; then emit a
receipt from your own computation; then, if your arithmetic qualifies, demonstrate
order-independence by permuting contraction order and showing the output digest is
unchanged.

---
Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
Contact: [anomly.com/contact](https://anomly.com/contact) · Licensing: Apache-2.0 (code) / CC-BY-4.0 (spec & docs) — see `LICENSES.md`.
