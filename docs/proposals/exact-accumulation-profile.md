<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Proposal: a generic public *exact-accumulation* arithmetic profile

**Status:** PROPOSAL — awaiting a yes/no decision. Nothing in the spec or the registry has been
changed. This document only describes an addition and the reasoning behind it.

## Why this exists

The publicly runnable demo `examples/python/hpc_reduction_demo.py` lets anyone run a real
ill-conditioned reduction through the CR verifier with numpy + stdlib only. It reaches every honest
float verdict — UNVERIFIABLE when two rank orders disagree, when they agree by luck, and when a
cheated total is claimed. What it *cannot* do publicly is reach an **ACCEPT**, and the reason is
precise:

- `verify()` returns ACCEPT only when the receipt names an **order-independent** arithmetic profile
  the verifier can assess (`_assessable` returns UNVERIFIABLE for any unregistered profile; the
  registry — not the prover — is authoritative for `order_independent`).
- The only registered `order_independent: true` profiles today are the **b-posit / 256-bit-quire**
  family, whose arithmetic implementation lives in Anomly's research tree, not in this repository.

So a pure-Python evaluator has no order-independent profile to name, and is stuck at UNVERIFIABLE.
A single generic, publicly reproducible exact-accumulation profile closes that gap: it lets the
public demo — and any third party's own reduction/training/inference reproducibility test — reach a
real ACCEPT with nothing but stdlib.

## The one real constraint

CR binds an output through `digest_tensor`, which hashes a numpy array's dtype + shape + canonical
little-endian bytes and **refuses non-numeric kinds** (object/void). An *exact rational* result has
no numpy dtype, so "just emit the exact value" is not a registry change — it is a spec-level change
to how outputs are encoded and digested. That constraint is what splits this proposal into two
options: one that lives entirely inside the existing output encoding, and one that does not.

---

## Option A — `exact-real-f64` (RECOMMENDED: no spec change)

Accumulate **every reduction exactly in ℚ** (no intermediate rounding, so the result is independent
of summation/rank/worker order), then round **each output element once** to IEEE-754 binary64 with
`roundTiesToEven`. The delivered output is therefore an ordinary `float64` tensor — `digest_tensor`
handles it unchanged, and no new output-encoding clause is needed.

This is exactly the **Kulisch long-accumulator / ReproBLAS "reproducible, correctly-rounded
reduction"** guarantee, expressed as a standard double. (The b-posit 256-bit quire is the
hardware-native sibling of the same idea; `exact-real-f64` is the same order-independence guarantee
delivered as a correctly-rounded IEEE double, reproducible by anyone in software.)

### Proposed registry entry

```python
# additive; the unqualified b-posit profiles keep their current meaning
"exact-real-f64": {
    "accumulation": "exact",           # exact real (rational) accumulation, no intermediate rounding
    "order_independent": True,         # the exact sum is unique; one final correct rounding is unique
    "params": {
        "accumulate": "exact-real",    # operands read as exact reals; sums taken in ℚ
        "deliver": "binary64",         # final result rounded to IEEE-754 double
        "rounding": "roundTiesToEven", # the single, final rounding mode (IEEE default)
        "reduce_axis": "declared-by-computation",
    },
},
```

### Verifier reproducibility semantics (what a conformer must produce)

For each output element `y_j` that is a reduction `Σ_i a_ij` (dot product, matmul entry, gradient
sum, global sum):

1. Read every operand `a_ij` as its **exact real value**. Each IEEE-754 double is exactly a dyadic
   rational, so this is unambiguous and lossless.
2. Compute the reduction **exactly in ℚ** (arbitrary-precision rationals). The result is a single
   rational, independent of the order in which terms were combined — this is the whole point.
3. Round that rational **once** to `binary64` under `roundTiesToEven`. This yields one, and only one,
   double for a given exact value.
4. Emit the output tensor as `float64`; digest it with the existing `digest_tensor`.

Two independent conformers that follow steps 1–4 produce **byte-identical** output tensors, so the
existing `verify()` path reaches ACCEPT with no changes to it.

**Reference implementation is trivial and stdlib-only** (this is the emitter side; the registry
entry alone is what makes `verify()` ACCEPT matching receipts):

```python
from fractions import Fraction
def exact_real_reduce(terms):                     # terms: 1-D float64
    return float(sum((Fraction(float(t)) for t in terms), Fraction(0)))
    # CPython float(Fraction) is correctly rounded, round-half-to-even — verified below
```

### Verified, not assumed

Measured on this machine (scratch check, not committed to the registry), the ill-conditioned
reduction from the public demo, 4099 terms, condition ~1e9:

| method | 64-rank order | 1024-rank order | equal? |
|---|---|---|---|
| naive float64 | `1.250002384185791` | `1.2500786483287811` | **no** (order-dependent) |
| `exact-real-f64` | `1.25` | `1.25` | **yes — byte-identical** |

`float(Fraction)` correct-rounding / ties-to-even confirmed directly: `float(1+2⁻⁶⁰)=1.0`,
`float(1+2⁻⁵³)=1.0` (tie → even), `float(1+3·2⁻⁵⁴)=1.0000000000000002` (>½ ulp → up). The proposal
rests on a checked property, not a hope. Per-element rounding means it generalizes beyond scalar
sums to matmul / gradient outputs (each output element is an independent exact reduction rounded
once), so it covers all three real-workload pillars, not just the reduction.

### Honesty caveats (must ship in the profile's prose)

- It certifies **order-independence + correct final rounding**, NOT b-posit hardware numerics. The
  delivered double equals the exact real up to ≤ ½ ulp — but it is the **same** double for every
  honest implementation, which is what makes it re-verifiable. Accuracy is a bonus; reproducibility
  is the claim (the same honest framing already used for the HPC pillar).
- It is a **different** profile from `bposit16-quire256`, not a replacement. b-posit certifies a
  specific low-precision *datapath* (posit16 rounding of each operand + a 256-bit quire);
  `exact-real-f64` certifies exact-real accumulation delivered as a correctly-rounded double. A
  receipt naming one is not interchangeable with the other, and `verify()` already REJECTs a profile
  mismatch between prover and verifier.
- Producing an `exact-real-f64` receipt requires software that actually does exact-accumulate-then-
  round (a long accumulator / ReproBLAS-style kernel, or quire hardware) — naive float hardware does
  not qualify and would fail to match, which is correct.

### Interop / conformance-vector impact

Purely additive. No existing conformance vector changes; no existing code path changes except adding
one dict entry to `PROFILES`. The C emitter and JS implementation are unaffected (they never compute
the arithmetic — they digest a supplied output). If desired, one new positive conformance vector
(an `exact-real-f64` receipt over a small reduction) and its ACCEPT could be added to lock the wiring.

---

## Option B — `exact-rational` (full exactness; needs a spec §-level change)

Deliver the **exact rational** result itself, with no final rounding. This is strictly more
information than Option A, but it **cannot** be done as a registry entry alone, because the output is
not a numpy numeric tensor. It requires a new canonical output-encoding clause in the spec (the
`output` binding, spec §4–§5), because `digest_tensor` is numeric-dtype-only by design.

Two candidate canonical encodings, both language-independent and pinnable:

- **B1 — fixed-point on a declared grid.** The profile declares a scale `2⁻F` and range `±2^R`
  (exactly what a Kulisch/quire accumulator is). The exact result is an integer multiple of `2⁻F`;
  encode that integer as a fixed-width big-endian two's-complement byte string (e.g. a 256-bit =
  32-byte value for a quire-sized grid). Fits `digest_tensor` if expressed as a `uint8`/`int32`
  array of fixed shape, but the *meaning* of those bytes (a scaled integer, not a tensor) is new and
  must be specified.
- **B2 — canonical num/den.** Encode the reduced (lowest-terms, sign-normalized) numerator and
  denominator as length-prefixed big-endian two's-complement integers. Fully exact and unambiguous,
  but definitively a new output section, and unbounded in size (a dot product of many doubles has a
  denominator up to `2^1074` and a large numerator).

**Honest finding:** Option B is viable but is a **spec decision, not a registry decision** — it adds
a new way to encode and digest outputs. It should not be hacked into `digest_tensor` (which correctly
refuses non-numeric kinds today). Recommend deferring B unless a concrete workload needs exact
rational *delivery* rather than reproducible correctly-rounded delivery. For the demo's purpose —
close the loop to a real public ACCEPT — Option A is sufficient and cleaner.

---

## Recommendation

Adopt **Option A (`exact-real-f64`)**: it is a one-entry, additive registry change that rests on a
verified property, needs no spec surgery, is reproducible by anyone with stdlib, maps to an
established HPC guarantee (ReproBLAS/Kulisch), and lets the public demo reach a real ACCEPT. Defer
**Option B** to a deliberate spec cycle if exact-rational *delivery* is ever required.

If Option A is approved, the follow-up is small and mechanical: add the `PROFILES` entry, extend the
public demo with an `exact-real-f64` prover/verifier pair (Fraction accumulate → round once) showing
ACCEPT across two rank orders and REJECT on a cheated output, and (optionally) add one positive
conformance vector.

## Ry decision

- **Option A (`exact-real-f64`, no spec change): yes / no** — ______
- **Option B (`exact-rational`, needs a spec §4–§5 output-encoding clause): yes / no / defer** — ______
- If A is yes: also add a positive conformance vector for it? **yes / no** — ______
