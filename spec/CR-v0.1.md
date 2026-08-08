<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Computation Receipt (CR) — version 0.1

**Status:** draft, published for implementation and comment.
**Media type:** `application/cr+json` · **Reference implementation:** `python/cr/receipt.py`
(this repository) · **Conformance check:** `python/check_conformance.py` against
`spec/CR-v0.1-conformance-vectors.json`

---

## 0. What this is, and what it is not

A **computation receipt** is a small, self-describing document that binds an output to the model,
the inputs, the computation, and the **arithmetic** that produced it, in a canonical form any
implementation can reproduce byte for byte.

It is not a proof system. A CR verifier **re-executes** the computation and compares. There is no
succinctness claim and no weight-hiding: the verifier needs the model. What a receipt provides
that a log line does not is a *decidable* question — did this exact computation produce this exact
output — and, crucially, an honest answer about whether that question is decidable at all for the
arithmetic in use.

**The format deliberately describes non-reproducible computations too.** A receipt over float
accumulation is a valid CR receipt; it simply carries `order_independent: false`, and a conforming
verifier must return `UNVERIFIABLE` for it rather than a verdict. This is not a courtesy to
floating point. It is what lets CR be adopted by anyone who computes anything, while making
visible — mechanically, in the receipt — which computations can actually be attested and which
cannot.

## 1. Why arithmetic is in the manifest

Two runs of the same model on the same inputs agree bitwise only if the accumulation is
order-independent. Floating-point accumulation is not associative, so tiling, shard count, kernel
version and hardware all change the result. An honest provider re-running on a different GPU
legitimately produces different bits.

That single fact is why "just hash the output" does not work, and it is the reason the
verifiable-ML literature treats FP non-associativity as *the* obstacle. The three known responses
each pay a tax: succinct proofs (ZKML) cost 100–1000× prover overhead and work in fixed point;
deterministic replay must pin one reduction order, surrendering hardware heterogeneity and
performance; optimistic/tolerance schemes give up exactness.

Exact accumulation into a wide fixed-point accumulator (a *quire*) is order-independent by
construction. Bit-identity then holds across heterogeneous hardware **for free**, and a mismatch
means, unambiguously, that a different computation was run.

CR does not mandate any particular arithmetic. It requires that the receipt **declare** it, so a
verifier can tell a real verdict from a meaningless one.

## 2. Canonical form

All digests are computed over a canonical serialization. Implementations MUST produce identical
bytes for equal manifests.

1. UTF-8 encoded JSON.
2. Object keys sorted by Unicode code point, ascending.
3. No insignificant whitespace: separators are exactly `,` and `:`.
4. `NaN`, `Infinity`, `-Infinity` MUST NOT appear; a manifest containing them is malformed.
5. Non-ASCII characters are emitted literally (not `\u`-escaped).
6. **String escaping is exactly this and nothing else.** Within a string, escape `"` as `\"`
   and `\` as `\\`; escape the C0 controls U+0000–U+001F as `\b` `\t` `\n` `\f` `\r` where
   those two-character forms are defined and as `\u00xx` (with **lowercase** hex digits)
   otherwise. The solidus `/` is **NOT** escaped. No other character is escaped.
7. **Numbers are integers only.** A conformant manifest contains no floating-point *values*
   (shapes and parameters are integers, digests are strings); integers are written in base 10
   with no leading zeros and a leading `-` for negatives. Floating-point serialization is
   therefore out of scope for v0.1.

> Rationale: this is the minimum that makes two independent implementations agree. Anything
> weaker — pretty-printed JSON, insertion-ordered keys, a language's default float repr — makes
> the digest an artifact of the emitter rather than of the computation. Rules 6–7 were added
> after an independent from-scratch canonicaliser showed that rules 1–5 alone left the string
> escaping under-pinned: RFC 8259 permits `\/`, `	`-style control escapes, and uppercase
> `\u` hex, so two honest emitters could produce different bytes — and therefore different
> certificates — for the *same* computation id (ids routinely contain `/`). The reference
> implementations (Python and C) already emit exactly rule 6; this pins the spec to them.

## 3. Digests

A digest is the string `"<alg>:<lowercase-hex>"`. `alg` is named explicitly; implementations MUST
support `sha256` and MAY support others (`sha512` is registered). Never assume the algorithm.

### 3.1 Tensor digest

For a single array, over the chosen `alg`:

```
H( canonical({"dtype": <dtype-string>, "shape": [<dims>]}) || <raw bytes, little-endian, C order> )
```

Byte order is **forced to little-endian**, not inherited from the host. `dtype-string` is the
numpy-style code with byte-order prefix stripped (`f8`, `i4`, …).

> Rationale: a receipt produced on a big-endian machine must hash identically to one produced on a
> little-endian machine. Inheriting host byte order would break cross-hardware verification, which
> is the entire premise.

### 3.2 Named tensor collection

For a set of named tensors (a model's weights, or a set of inputs), iterate names in sorted order
and update the hash with `canonical({"name": <name>, "digest": <tensor digest>})` for each. An
empty collection is permitted and yields the digest of the empty hash state.

> Rationale: dictionary iteration order is not part of the model.

## 4. The manifest

```json
{
  "cr": "0.1",
  "digest_alg": "sha256",
  "arithmetic": {
    "profile": "bposit16-quire256",
    "accumulation": "exact",
    "order_independent": true,
    "params": {"n": 16, "es": 3, "quire_bits": 256, "frac_bits": 96}
  },
  "computation": {"id": "…", "version": "1"},
  "model":  {"digest": "sha256:…", "n_tensors": 6},
  "input":  {"digest": "sha256:…", "n_tensors": 1},
  "output": {"digest": "sha256:…", "shape": [16, 32]}
}
```

| Field | Meaning |
|---|---|
| `cr` | Format version. A verifier MUST refuse a version it does not implement rather than guess. |
| `digest_alg` | Algorithm for every digest in this receipt. |
| `arithmetic.profile` | Registered profile name (§7). |
| `arithmetic.order_independent` | **The decisive field.** `true` asserts an honest re-execution anywhere reproduces these bits. |
| `computation.id` / `.version` | Identifies what was run. v0.1 uses a name; a full graph identifier is a v0.2 candidate (§8). |
| `model` / `input` | Named-collection digests (§3.2). |
| `output` | Tensor digest and shape. |

## 5. The receipt

```json
{"manifest": { … }, "certificate": "sha256:…", "meta": { … }}
```

The **certificate** is the digest of the canonical manifest:

```
certificate = digest( canonical(manifest) )
```

So it binds every field at once: change the model, the inputs, the computation, the arithmetic or
the output, and the certificate changes. A bare output hash cannot do this — the same output
produced by a different model would be indistinguishable.

`meta` is **untrusted provenance** (operator, timestamp, host, hardware) and is **excluded from
the certificate**.

> Rationale: including provenance would make a receipt fail to verify on any other machine, which
> is precisely the property being sold. `meta` is for humans and logs; it is never evidence.

## 6. Verification

A verifier is given a claimed receipt and independently re-executes the computation, producing its
own receipt. It returns exactly one of:

| Verdict | Condition |
|---|---|
| `MALFORMED` | Unknown version, missing field, unsupported algorithm, or `certificate ≠ digest(canonical(manifest))`. |
| `REJECT` | Well-formed, `order_independent: true`, and the model, input, computation, profile or output digest differs. |
| `UNVERIFIABLE` | Well-formed but `order_independent: false` — **including when the outputs happen to match.** |
| `ACCEPT` | Well-formed, `order_independent: true`, and the re-execution reproduced the certificate. |

`UNVERIFIABLE` is not a soft `REJECT`. Under order-dependent accumulation, agreement is luck and
disagreement is expected; neither is evidence. A verifier that returned `ACCEPT` because two float
runs coincidentally matched would be asserting something it cannot know, and implementations MUST
NOT do this.

### 6.1 The order-independence claim MUST be checked, never trusted

`arithmetic.profile` and `arithmetic.order_independent` are both written by the prover, and the
second is what gates `ACCEPT`. Two rules close that (added 2026-08-04 after the audit in §13
found the first exploitable):

1. **For a registered profile (§7), the declared `accumulation`, `order_independent` and
   `params` MUST match the registry, or the receipt is `MALFORMED`.** The registry is
   authoritative, not the receipt. Without this a receipt could name `float64` while asserting
   `order_independent: true` and obtain an `ACCEPT` — forging precisely the verdict this
   section reserves for exact arithmetic.
2. **For a profile the verifier does not have in its registry, the verdict MUST be
   `UNVERIFIABLE`.** The registry is deliberately open (§7), so verifiers will meet profiles
   they do not implement; there is then nothing to check the claim against, and returning
   `ACCEPT` would be trusting the prover rather than verifying. This is the same honest answer
   the format already gives for float, for the same reason.

## 7. Arithmetic profile registry

| Profile | Accumulation | Order-independent | Params |
|---|---|---|---|
| `bposit16-quire256` | exact | **yes** | `n=16, es=3, quire_bits=256, frac_bits=96` |
| `bposit16-quire256-b48` | exact | **yes** | as above **+ `regime_bound=48`** (canonical / silicon-validated) |
| `bposit16-quire256-b112` | exact | **yes** | as above **+ `regime_bound=112`** (this SDK's current codec) |
| `float64` | rounded | no | `bits=64` |
| `float32` | rounded | no | `bits=32` |

Registering a profile requires: a precise definition of the number system and accumulator, and —
for any profile claiming `order_independent: true` — a demonstration that permuting contraction
order leaves the output bit-identical. The registry is intended to be **open**: any arithmetic
that can meet the order-independence bar belongs in it, including ones Anomly did not design.

> This is deliberate. A registry with one vendor's entry is a product; a registry others can join
> is infrastructure. Anomly's position is not that others cannot qualify — it is that on
> exact-quire silicon, qualifying costs nothing.

> **Implementer caution — the regime bound of the base profile is an open v1 decision.**
> Implementations of b-posit16 exist under two conventions: regime-bounded at ±48 (the
> silicon-validated envelope) and unbounded to ±112 (the current reference codec). Inside
> `|total exponent| ≤ 48` the two agree bit-for-bit; outside it they decode the same code to
> **different values**, so two conforming-looking implementations would REJECT each other's
> receipts. Until the v1 revision pins the base profile, implementers SHOULD (a) stay within
> the ±48 envelope (snap sub-2^-20 magnitudes to zero before encoding, as the reference does),
> or (b) declare the explicit sub-profile (`-b48` / `-b112`) rather than the base. A verifier
> that cannot tell which convention a receipt used is looking at exactly the ambiguity this
> registry exists to eliminate — report any receipt that hits it.

> **Registry correction, 2026-08-04.** `bposit16-quire256` previously declared
> `es=2, frac_bits=128`. Both were wrong about the format the profile names: b-posit16 has a
> three-bit exponent field (maxpos 2^112, value grid identical to standard posit16 es=3) and
> its quire is a 2^-96 fixed-point grid. A registry that misdescribes its own arithmetic
> defeats the purpose of binding arithmetic into the receipt, so it was corrected.
>
> Consequence, stated plainly: **receipts issued before the correction are now `MALFORMED`**
> under §6.1 rule 1, because their declared params contradict the registry. That is the rule
> working as intended — those receipts really did declare the wrong arithmetic — but it does
> invalidate previously-issued certificates for this profile, including the 2026-08-04
> on-silicon artifact. The bit-level evidence that artifact carries (identical operand, model
> and output digests across FPGA and x86) is independent of the params field and unaffected.

> **Regime-explicit profiles, 2026-08-04.** `bposit16-quire256` does not state a regime
> bound, and implementations disagree about it: this SDK's codec admits |exponent| ≤ 112
> while the canonical/forge/CUDA implementations bound at 48. A value in (2^48, 2^112]
> therefore decodes to different bits under two implementations that both name that profile
> — an honest prover REJECTed for a cause neither side can see, which is the failure this
> format exists to prevent.
>
> The two `-bNN` profiles above close that by **naming the bound**. Implementations with
> different conventions now disagree on the *profile*, so a verifier REJECTs citing the
> profile rather than silently comparing incomparable outputs. `spacetime.fastquire.encode`
> takes a matching `regime_bound=` argument so one codec can honour either convention.
>
> These are **additive**: the unqualified profile keeps its meaning and remains the default.
> Narrowing the default is a spec decision pending in
> `bposit-v1-spec-rev-2026-08-04.md`; what shipped here is the mechanism and its gate.

## 8. Known gaps in v0.1

Stated plainly so implementers are not surprised, and so v0.2 has an agenda:

- **Computation identity is a name, not a graph.** Two parties must agree out-of-band what
  `computation.id` means. A content-addressed graph (ONNX/StableHLO digest) is the obvious v0.2
  successor and is the single biggest gap.
- **No signatures.** A receipt proves what was computed, not who computed it. Detached signatures
  over the certificate compose cleanly and are deliberately out of scope here.
- **No streaming/partial receipts** for computations too large to re-execute whole. ~~Sampled and
  chunked attestation is the expected v0.2 work~~ → **sampled receipts shipped as the v0.1.1
  addendum (§10)**; streaming/chunked receipts remain open.
- **No canonical numeric encoding for `params`** beyond JSON integers; float-valued parameters in
  a future profile would need a rule.

## 9. Conformance

An implementation conforms to CR v0.1 if it reproduces every published conformance vector exactly
and implements the verdict table in §6.

**Negative vectors are part of conformance, and are the part that matters most.** Vectors carrying
an `expect_verdict` field publish a manifest a conforming verifier MUST *refuse*, together with the
required verdict. A suite of positive vectors alone cannot establish soundness: an implementation
that accepted everything would reproduce every positive digest and still be worthless. Both
soundness bugs found on 2026-08-04 (§13, §13.1) were of exactly that shape — *accepting what must
be rejected* — and neither would have been caught by certifying against the original suite. The
three published refusals are:

| vector | required verdict | what it prevents |
|---|---|---|
| `refuse/profile-contradicts-registry` | `MALFORMED` | forging the `ACCEPT` reserved for exact arithmetic by asserting `order_independent` under a rounded profile (§6.1) |
| `refuse/sample-space-shrunk` | `MALFORMED` | claiming full coverage while most units go unattested (§10) |
| `refuse/unknown-profile-not-accepted` | `UNVERIFIABLE` | accepting an order-independence claim the verifier cannot assess (§6.1) |
| `refuse/output-digest-missing` | `MALFORMED` | a sampled receipt with no full-output digest would ACCEPT its sample while never committing to the unsampled units (§10) |

The sampled vector additionally publishes its derived `sample_indices`, because §10 established
that a matching `sample.digest` does not by itself prove an implementation used the specified
index rule. The vectors pin canonicalisation, tensor digesting and one
full receipt — deliberately **not** the arithmetic, so an implementation in any language on any
platform can be checked without reproducing any particular number system. (The
`bposit16-quire256` profile itself is a bounded configuration of posit arithmetic —
Gustafson & Yonemoto 2017, Posit Standard 2022 — not an Anomly invention; the format
deliberately builds on published, vendor-neutral mathematics.)

**Published vectors:** [`CR-v0.1-conformance-vectors.json`](CR-v0.1-conformance-vectors.json)
(**13 vectors**: canonicalisation, tensor and collection digests, one receipt of each kind —
full, sampled, chunked — and **four negative vectors**). The file is checked against the reference implementation by the `test_receipt` gate,
so it cannot silently drift — a standard whose conformance file has diverged from its reference
implementation is worse than none, because third parties would certify against the wrong bytes.

### 9.1 Independent implementation — measured interoperability

A specification is only a standard if someone else's code produces the same bytes. There is now
one independent implementation and a gate that proves it agrees:

- **The implementation:** `cr_receipt.h` in the FPGA host of the F2 `cl_bposit` design
  (`chip-design/fpga/f2/cl_bposit/software/`). Dependency-free C, written against this document
  rather than the reference source, sharing no code with `spacetime.receipt`. It emits receipts
  for exact-quire MAC results read back off the card.
- **The gate:** `cross_check_receipt.py` runs the host, parses the receipts it prints, rebuilds
  each one with the Python reference, and asserts the **certificates and the canonical manifests
  are byte-identical**. Result: **6/6 identical** across all three receipt kinds — full (v0.1), sampled (v0.1.1, including the derived PRF index sets), and chunked (v0.1.2, including the chain digest).
- **The gate is falsifiable, which is the part that matters.** Perturbing a single digit of the
  arithmetic parameters (`es` 2→3) must make it fail, and does — reporting 0/2 with a manifest
  diff. (That control initially passed for the wrong reason: a missing header prerequisite in
  the host Makefile meant it ran a stale binary. Anyone re-deriving these results should confirm
  their own control fails before trusting a pass.)

Two implementations agreeing is weak evidence compared to five, and both were written inside
Anomly. It is offered as the minimum bar a format should clear before being called a standard,
not as proof of ecosystem adoption.

**Tooling** — deliberately usable without importing our library:

```bash
space-time receipt check   <receipt.json>                    # well-formedness only
space-time receipt verify  <claimed.json> --against <replay.json>
space-time receipt vectors --out vectors.json                # regenerate the published set
```

Exit codes: `0` only on ACCEPT, `1` on any negative verdict (REJECT / UNVERIFIABLE / MALFORMED),
`2` if a file could not be read — so the verifier composes into CI and audit pipelines without
parsing its output.

## 10. Implementing CR without this repository

The intended path for a third party, in order: reproduce §9's vectors in your language (this
validates canonicalisation and digesting, and needs no arithmetic); emit a receipt from your own
computation using §4–§5; check it with `space-time receipt check`; then, if your arithmetic
qualifies for `order_independent: true` under §7, demonstrate it by permuting contraction order
and showing the output digest is unchanged.

An implementation that only ever emits `order_independent: false` receipts is still a conforming
CR implementation. That is intentional: the format's job is to make the distinction visible and
checkable, not to exclude anyone from expressing it.

## 10. Sampled receipts — the v0.1.1 addendum

For computations too large to re-execute whole, `cr: "0.1.1"` adds one manifest section and
lets a verifier re-execute only a **sample** of output units (rows along axis 0 of the output):

```json
"sample": {
  "rule": "sha256-ctr-reject-v1",
  "n_units": 128256,
  "size": 512,
  "challenge": "<hex, may be empty>",
  "digest": "sha256:..."
}
```

**Index derivation (normative).** Let `base` be the manifest with the `sample` key removed
(note `base` still contains the FULL output digest). Then

```
seed    = SHA-256( canonical(base) || challenge_bytes )
indices = first `size` distinct values of ( u64_be(SHA-256(seed || ctr_be8)[0:8]) mod n_units ),
          ctr = 0,1,2,... , rejecting draws >= floor(2^64/n_units)*n_units (modulo-bias
          rejection) (for a power-of-two n_units this bound is 2^64 — nothing is rejected; implementing
          it as (2^64-1)/n*n is a bug that wrongly rejects the top n draws), returned sorted ascending
```

`sample.digest` is the §3 tensor digest of `output[indices]`. The verifier derives the indices
itself — a conforming verifier MUST NOT accept indices listed by the prover.

> **Why that MUST is not a formality** (found 2026-08-04 by a negative control against the
> independent C implementation, §9.1): a wrong index rule can still produce a **matching
> certificate**. Perturbing the PRF counter to little-endian made the C emitter select units
> `[0,1,4]` where the reference selects `[0,1,6]`, yet the certificates agreed — because on
> that vector both units held the same value, so digesting the wrong rows produced identical
> bytes. Agreement of `sample.digest` therefore does **not** by itself establish that the
> prover used the specified index rule. Conformance testing between implementations MUST
> compare the derived index sets directly, not only the certificates; and a verifier that
> derives its own indices (as required) is unaffected by the ambiguity.

**Why the prover cannot cherry-pick.** The seed commits to everything else in the manifest,
*including the digest of the complete output*, before the indices exist. Choosing which units
get checked after seeing the outputs would require changing the output digest, which moves the
sample.

**`n_units` MUST equal the output's leading dimension** (normative; added 2026-08-04 after the
audit below found this exploitable). A verifier MUST reject a receipt where
`sample.n_units != output.shape[0]`.

> **The gap this closes.** Nothing previously tied the declared sampling space to the actual
> output. A prover holding a 128-unit output could declare `n_units: 8`, recompute
> `sample.digest` over that prefix, and emit a receipt that is well-formed, internally
> consistent, and **ACCEPTs — while 120 of 128 units are never sampled and never attested.**
> Coverage claimed in the ACCEPT reason would be "8/8, 100%" when the true coverage is 6%.
> This is a soundness bug, not a documentation gap: sampling is only meaningful if the
> denominator is bound.

**Verdicts.** As §6, applied to the sampled slice: ACCEPT means the re-executed sampled rows
reproduce `sample.digest` bit-exactly under order-independent arithmetic; order-dependent
profiles are UNVERIFIABLE exactly as for full receipts; a `cr: "0.1"` receipt carrying a
`sample` section, or a `cr: "0.1.1"` receipt lacking one, is MALFORMED.

**Security boundary (normative, and the honest part).** Sampling weakens the claim, in two
stated ways, and implementations MUST NOT present a sampled ACCEPT as a full one:

1. *Unsampled units are unattested.* ACCEPT means "`size`/`n_units` re-executed bit-exactly and
   the full output digest is committed" — never "every unit was checked". A non-adaptive tamper
   of `k` units escapes with probability `C(n-k, s)/C(n, s) <= (1 - s/n)^k`.
2. *An adaptive prover can grind an empty challenge.* Recompute a tampered output, derive the
   indices, retry until the tampered units escape the sample (~`n/s` attempts for `k = 1`). The
   defence is the `challenge`: supplied by the verifier or a public randomness beacon **after**
   the prover has fixed its output, grinding requires predicting it. A sampled receipt with an
   empty challenge is evidence against non-adaptive faults (bit flips, wrong weights, silent
   drift) only — not against an adversarial prover.

**Verifier flow** (`space-time receipt`): `sample-indices <receipt.json>` prints the units to
re-execute; `verify-sampled <receipt.json> --rows <rows.npy>` compares the verifier's
re-executed `output[indices]` and prints the verdict with its coverage stated.

## 11. Content-addressed computation identity — v0.2 DRAFT, published for comment

§8 names computation identity as the biggest gap: `computation.id` is a *name*, so two parties
must agree out-of-band what it means. This section drafts the successor. It is implemented in
the reference implementation as an OPTIONAL field and does not change v0.1/v0.1.1 semantics:
a receipt without it is exactly as before.

**CRG v0 — canonical computation graph.** A computation is described as a JSON operator DAG:

```json
{
  "crg": "0",
  "inputs":  ["W", "H"],
  "nodes":   [{"op": "gemm", "attrs": {}, "args": ["in:W", "in:H"]}],
  "outputs": ["node:0"]
}
```

- `inputs` are the graph's formal tensor names; receipts bind actual tensors to these names
  through the §3.2 model/input collections.
- `nodes` MUST be in topological order; node k may reference `"in:<name>"` or `"node:<j>"`
  with `j < k`. `attrs` is a canonical-JSON-safe object (no floats containing NaN/Inf, §2).
- `outputs` reference nodes (or inputs, for identity graphs).
- The graph digest is `digest(canonical(graph))` (§2, §3). Two computations are the same
  exactly when their canonical graphs hash identically.

**Binding.** The manifest's computation section gains an optional field:

```json
"computation": {"id": "attest.gemm", "version": "1", "graph_digest": "sha256:..."}
```

`id` remains for human naming; `graph_digest` is what a verifier compares. Because the
certificate already binds the whole computation section, a different graph is REJECTed by the
existing §6 rule with no verifier changes.

**Op vocabulary** — CRG v0 pins *how a graph hashes*, not what ops exist. §11.1 registers
the vocabulary.

### 11.1 Op registry (v0.2 draft)

Structure is universal; vocabulary is a registry that grows. A graph can be structurally
valid and hashable while using ops a given implementation does not model, so **op checking is
a separate step from graph validation** — implementations MUST NOT reject a graph as
malformed merely because they do not know an op.

| op | arity | attributes | meaning |
|---|---|---|---|
| `gemm` | 2 | `transpose_a`, `transpose_b` | matrix multiply; the computation this format was built to attest |
| `dot` | 2 | — | vector dot product |
| `reduce_sum` | 1 | `axis` | summation along an axis |
| `relu` | 1 | — | elementwise max(0, x) |
| `external_ir` | 0 | `ir`, `digest` | **escape hatch** (below) |

The registry is deliberately small. Every entry is a compatibility promise, so registering
ops nobody emits is a liability rather than a courtesy. Ops are added when something actually
emits them.

**The `external_ir` escape hatch.** A computation whose graph CR does not model is carried as
a single node naming a foreign content-addressed IR and its digest:

```json
{"crg": "0", "inputs": [], "outputs": ["node:0"],
 "nodes": [{"op": "external_ir", "attrs": {"ir": "onnx", "digest": "sha256:…"}, "args": []}]}
```

This closes the "same name, different computation" ambiguity for ONNX or StableHLO pipelines
without pretending to understand them: **CR identifies the computation by that digest and
makes no claim about the IR's internals.** An implementation that can verify the IR itself is
free to do more; the format does not require it to.

*Unchanged from §11:* a graph digest identifies the computation **as declared**. Neither the
registry nor the escape hatch makes it evidence that the declared graph is what executed —
only re-execution does that.

**Honesty note.** A graph digest identifies the computation *as declared*. It does not prove
the prover executed that graph — only re-execution (§6) does. The graph closes the "same name,
different computation" ambiguity; it is not an execution trace.

## 12. Chunked receipts / receipt chains — v0.1.2 addendum

For computations that stream — token-by-token generation, long reductions — `cr: "0.1.2"`
attests incrementally. Each **chunk receipt** is a complete CR receipt over that chunk's
inputs and output, plus one section chaining it to its predecessor:

```json
"chunk": {"index": 3, "prev_certificate": "sha256:...", "closing": false}
```

- Chunk 0 has `prev_certificate: ""`; chunk k > 0 carries chunk k-1's **certificate**, so the
  chain fixes ordering and membership as it grows — a reordered, dropped or foreign chunk
  breaks the linkage and is REJECTed.
- A **closing receipt** terminates the chain: `{"index": n, "prev_certificate": <cert of
  chunk n-1>, "closing": true, "n_chunks": n, "chain_digest": <digest over every certificate
  in order>}`, with the receipt's `output` section binding the digest of the TOTAL output.
- Verification: each chunk verifies as an ordinary §6 receipt against the verifier's own
  re-execution of that chunk; the linkage and the closing receipt verify structurally. A
  verifier may check any prefix incrementally as chunks arrive. Sampling composes: §10 can be
  applied per chunk, or across chunks by sampling which chunks to re-execute.

**Honest boundaries (normative).**

1. *A chain proves commitment order, not wall-clock timing.* A prover can compute everything
   first and emit the chain afterwards; nothing here timestamps computation.
2. *An unclosed chain does not attest totality.* Without the closing receipt, truncation is
   undetectable — a conforming verifier MUST refuse an unclosed chain unless the caller
   explicitly requests prefix verification, and MUST state in an ACCEPT of a prefix that
   totality is unattested.
3. *The closing receipt is the only defence against truncation.* Its `n_chunks` and
   `chain_digest` bind the exact membership; a closing receipt over a shorter chain REJECTs.

**Verifier flow** (`space-time receipt`): `verify-chain <chain.json> --against
<reexecuted-chain.json>`, where each file is a JSON array of receipt objects, the claimed
chain optionally ending in its closing receipt; `--allow-open` verifies an explicit prefix.

## 13. Digest-agreement audit (2026-08-04)

The §10 index finding — that a matching `sample.digest` does not establish the prover used the
specified index rule — suggested a general question worth asking of the whole format: **where
else does agreement of a digest fail to establish the property it appears to establish?** Each
case below is a permanent regression test in `test_receipt`, including the ones that came back
clean, because "we checked and it holds" is the useful record.

| probe | result |
|---|---|
| Can a prover declare a sampling space smaller than the real output? | **YES — soundness bug, now fixed** (§10 `n_units` rule). Was well-formed and ACCEPTed with 6% true coverage reported as 100%. |
| Can two different name/tensor splits collide in a named-collection digest? | No. The canonical per-entry encoding keeps `{"ab","c"}` and `{"a","bc"}` distinct. |
| Does `chain_digest` bind chunk order, or only membership? | Binds order; reversing a chain changes it. |
| Do two graphs computing the same function share a `graph_digest`? | No — identity is **syntactic**. An unreferenced (dead) node changes the digest. Not a soundness hole, but implementations must not expect semantically equivalent graphs to match. |
| Does an empty `model` digest distinguish "no model" from "model not bound"? | **No, and it cannot.** `model=None` and `model={}` produce identical certificates, both with `n_tensors: 0`. A receipt attests *what it binds*; an empty collection means nothing was bound, never that nothing existed. Re-execution still catches a withheld model (the verifier's own binding differs → REJECT), so this is an interpretation rule, not a hole: **do not read `n_tensors: 0` as evidence a computation is model-free.** |

### 13.1 Verifier-side audit (same lens, applied to the verdict logic)

The format-side audit asked what is not bound to something external. The same question of the
verifier: **can an `ACCEPT` be obtained without the property actually holding?**

| probe | result |
|---|---|
| Can a prover assert `order_independent: true` under a rounded profile? | **YES — soundness bug, now fixed** (§6.1 rule 1). A `float64` receipt claiming order-independence obtained an `ACCEPT`: the exact verdict the format exists to withhold from float. |
| Can an unregistered profile obtain `ACCEPT` on the prover's word? | **YES — now fixed** (§6.1 rule 2). Unknown profiles are `UNVERIFIABLE`; the verifier has nothing to check the claim against. |
| Can a receipt under one `digest_alg` verify against a re-execution under another? | No. Algorithm is bound in the manifest; the certificates differ and the comparison rejects. |
| Can a chunk receipt be replayed into a different chain? | No. Each chunk binds its computation and its predecessor's certificate, so a foreign chunk breaks §12 linkage. |
| Is a claimed `graph_digest` tied to a real graph? | No — a prover can bind any digest. Re-execution still catches it (the verifier's own computation section differs → `REJECT`), so this is the same interpretation rule as the empty `model`: **a receipt attests what it binds, and `graph_digest` identifies the computation *as declared*.** |
| Does hostile JSON produce a verdict, or a crash? | **Was a crash** — a non-object `manifest` raised `AttributeError` inside `check_wellformed` instead of returning `MALFORMED`. Now type-validated in both `from_json` and `check_wellformed`: a verifier ingests untrusted documents, and crashing on one is its own kind of wrong answer. |
| Would the published vectors have caught either soundness bug? | **No — the largest gap of the three passes.** The suite was entirely positive, so an implementation accepting everything would have passed it. Fixed by publishing three negative vectors with required verdicts (§9). |
| Does an `allow_open` prefix ACCEPT overstate what was verified? | Not if the reason is read: the verdict string says totality is UNATTESTED (§12 requires it). The risk is a caller who checks only the status — which is why §12 makes the statement mandatory rather than advisory. |

The lesson generalises beyond this format: a self-consistent document proves only that its
author was consistent. Every field a verdict depends on must be *bound to something external*
— the output's real shape, the verifier's own re-execution, a challenge the prover cannot
predict — or it is the prover's word restated as a hash.
