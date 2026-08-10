<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Computation Receipts — a brief for the skeptical evaluator

**One sentence:** a receipt for a numeric computation that anyone can check by re-executing it, on
any hardware, with any conforming implementation, with no trusted machine anywhere — and you can
verify every claim on this page yourself in an afternoon.

## The problem it solves

Floating-point arithmetic rounds *inside* its sums, so two honest machines disagree in the last
bits. That means "is this result correct?" has no checkable answer today: a different GPU, a
different kernel, a different shard count all produce different bits, so a mismatch can't
distinguish *cheated* from *ran somewhere else*. Every current answer to this pays a heavy price —
a zero-knowledge prover (100–1000× overhead), or a trusted-hardware enclave (you must trust the
chip vendor), or pinning the exact hardware and kernels (surrendering the heterogeneity that makes
compute cheap).

## What CR does

Compute with **exact accumulation** (a 256-bit quire over b-posit values) and the result is
bit-identical on every honest execution. A result can then carry a **certificate**: a canonical
manifest binding the inputs, model, computation graph, arithmetic profile, and output — by digest.
A verifier re-executes on its *own* independent hardware and compares.

- Identical → the unique correct answer. **ACCEPT.**
- One output bit flipped, one substituted weight, a smaller model swapped in → **REJECT.**
- Float / order-dependent arithmetic → **UNVERIFIABLE**, stated honestly, because agreement would
  be luck and disagreement is expected. CR never pretends a coincidence is a proof.

## The honest scope — read this first

CR delivers a real ACCEPT/REJECT **only for order-independent (exact-quire) arithmetic.** On the
float-on-GPU stack that runs most of the industry today, the honest verdict is UNVERIFIABLE. So the
fit is specifically the cases where exact reproducibility is worth an arithmetic cost:

- regulated or audited inference (a checkable record of *what* ran),
- dispute resolution between a compute provider and a client,
- detecting model substitution (charging for a big model, serving a cheap one),
- cross-vendor result agreement where "it matched on their box too" has to mean something.

If your workload is best-effort float where nobody will ever dispute the bits, CR is not for you,
and we would rather tell you that now.

See [`REAL-WORKLOADS.md`](REAL-WORKLOADS.md) for CR run on real workloads in exactly these classes —
an ill-conditioned HPC reduction (float gives a different, wrong answer at each core count; the quire
is bit-identical), a distributed-training gradient all-reduce on real Llama-3.2-1B weights (float is
not bit-reproducible across worker counts; the quire is), and a real pretrained forward pass
(ACCEPT across execution plans, REJECT on a one-ULP tamper, UNVERIFIABLE in float).

## Where it sits versus what you already know

| Approach | Cost / trust | CR's difference |
|---|---|---|
| Zero-knowledge ML | ~100–1000× prover overhead | Verification is just re-execution — no proving tax |
| Hardware / enclave attestation | Trust the chip vendor's root of trust | The **math** is the root of trust; no trusted machine anywhere |
| Deterministic replay / pinned kernels | Surrender hardware heterogeneity | An honest re-execution on *different* hardware still matches bit-for-bit |

The wedge is **cross-hardware bit-identity that also catches wrong answers**, without a trusted
party and without a prover tax. That is the one claim worth pressure-testing.

## Verify every claim above yourself — in an afternoon

Nothing here asks you to trust us. The repository is the format, the reference, and the tools to
grade a *second* implementation:

1. **Reproduce the reference gate.** `cd python && python3 run_checks.py` → 5/5: the reference
   reproduces the 17 published conformance vectors, enforces the required refusals, runs the
   challenge protocol, and the C emitter compiles standalone with only libc.
2. **See a second implementation reproduce it.**
   `node examples/js/cr.mjs | python3 python/conformance_runner.py /dev/stdin` → 17/17. That JS file
   is a from-scratch implementation (Node stdlib only, no reference code) — emitter *and* verifier —
   reproducing every vector from the spec text alone. It even verifies receipts the Python reference
   emits, which is the whole point.
3. **The real test — implement it yourself.** Start with the canonicalisation layer (spec §2–§3):
   a few hours, no arithmetic required. `python3 python/conformance_runner.py --emit` prints a
   fill-in template; grade yourself against the published vectors. **If you hit anything the spec
   leaves ambiguous, that is a spec bug — and the most valuable thing you can send us.** We found
   seven such ambiguities ourselves this way and pinned them (spec §2 rules 6–7, §3.1–§3.2, §10,
   §12); the eighth is yours to find.

`IMPLEMENTERS.md` walks the path with worked examples and every pitfall we know about.

## What we're actually asking

Either — (a) implement from the spec and tell us where it is unclear, or (b) run a real workload
you care about through it and tell us whether the verdict is useful. **Both are worth more to us
than agreement.** An implementation we did not write, or a "this ambiguity would have burned me," is
the thing that turns a well-specified format into a standard.

## Status, stated plainly

This is a **v0.1 draft** — hardened, independently re-derived, and passing its own conformance
gate, but **not yet adopted by anyone outside Anomly.** That is exactly why this brief exists. The
arithmetic it relies on (b-posit, the 256-bit quire) is published prior work (Gustafson & Yonemoto
2017; Posit Standard 2022); CR's contribution is the **verification layer** on top — the receipt
format, the canonical form, the conformance vectors, and the refusal rules.

---
Start here: [`README.md`](README.md) · Spec: [`spec/CR-v0.1.md`](spec/CR-v0.1.md) · Implementing:
[`IMPLEMENTERS.md`](IMPLEMENTERS.md) · Integrating: [`INTEGRATION.md`](INTEGRATION.md) · Maturity &
open questions: [`STATUS.md`](STATUS.md) · Contact: [anomly.com/contact](https://anomly.com/contact)
