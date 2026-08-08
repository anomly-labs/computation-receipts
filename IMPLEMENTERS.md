<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Implementing CR in your language — the practical guide

This is the path we would take in your seat, plus every pitfall we know about —
including the ones we found by hitting them. If the spec and this guide disagree, the
spec wins and the disagreement is a bug: please report it.

## Order of attack (mirrors spec §9-§10)

1. **Canonicalisation first — no arithmetic needed.** Reproduce the `canonical/*`
   vectors: UTF-8 JSON, keys sorted **by Unicode code point**, separators `,` and `:`
   with no whitespace, non-ASCII passed through as UTF-8 (not `\u`-escaped), and
   **NaN/Infinity are errors**, never emitted. **String escaping is pinned exactly**
   (spec §2 rule 6, and the first thing an independent implementation gets subtly
   wrong): escape only `"`→`\"`, `\`→`\\`, and the C0 controls U+0000–U+001F as
   `\b \t \n \f \r` where defined else `\u00xx` with **lowercase** hex; the solidus
   `/` is **not** escaped. The `canonical/escaping` vector exercises `/`, a tab and
   `\x1f` precisely to catch a JSON encoder that escapes `\/` or emits `	`.
2. **Tensor digests.** `sha256( canonical({"dtype":…,"shape":…}) ‖ raw-bytes )` with
   the raw bytes **forced little-endian, C-order contiguous**. The dtype string is the
   numpy code with the leading byte-order/alignment character stripped — and that
   includes `|`: `<f8`→`f8`, `<i4`→`i4`, and for **single-byte types** (int8, uint8,
   bool) `|i1`→`i1`, `|u1`→`u1`, `|b1`→`b1` (spec §3.1). Getting this wrong on int8 —
   the quantized-weight case — is the difference between a matching and a diverging
   certificate; the `tensor/int8-6` vector exists to catch it. Named collections
   iterate names **sorted by Unicode code point** (§3.2); reproduce the `tensor/*` and
   `tensors/*` vectors, and prove the collection digest is order-independent.
3. **One full receipt.** Build the manifest exactly as §4 lays it out, hash it, match
   the published receipt vector byte-for-byte (`manifest_canonical` is in the vector
   precisely so you can diff bytes, not vibes).
4. **The verdicts — the part that matters most.** Implement all four, then run the
   three `refuse/*` vectors. A verifier that cannot refuse is worthless; the two
   soundness bugs we found in our own code were both accept-what-must-be-rejected,
   and only negative vectors catch that class.
5. **Sampled + chained receipts** (§10, §12), where the exact byte-level rules matter
   most:
   - **Sampled PRF (§10).** `seed = sha256(canonical(base) ‖ challenge_bytes)` where
     `challenge_bytes` is the **hex-decode** of the `challenge` string, *not* its UTF-8
     bytes (this is the single easiest thing to get wrong — it diverges on every
     non-empty challenge). Draw `u64_be(sha256(seed ‖ counter_be8)[0:8]) mod n_units`
     for counter 0,1,2,…, **rejecting draws ≥ ⌊2⁶⁴/n⌋·n** (do *not* implement the bound
     as `(2⁶⁴−1)/n·n` — that wrongly rejects the top `n` draws), take the first `size`
     distinct values, return sorted; `1 ≤ size ≤ n_units`. The `receipt/sampled` vector
     publishes the derived indices (challenge `"beef"`) so a matching *digest* alone
     cannot hide a wrong index rule.
   - **Chains (§12).** `chain_digest` is over one hash, updated with
     `canonical({"certificate": c})` for each **chunk** certificate in index order
     (`0..n-1`) — **not** the closing receipt's own certificate, **not** raw-concatenated
     strings. The `chain/closing-2chunk` vector pins it. Prev-certificate linkage,
     the closing receipt's `n_chunks`/`chain_digest`, and the unclosed-chain truncation
     refusal are all recomputed, never trusted.
   - **Section exclusivity (§10).** Each version carries exactly its own optional
     section: a `0.1.1` receipt MUST NOT carry a `chunk` section, a `0.1.2` MUST NOT
     carry a `sample` section, a `0.1` neither. Any mix is `MALFORMED`
     (`refuse/cross-version-section`).
6. **Only then, arithmetic.** Everything above needs no number system. If your
   arithmetic can demonstrate order-independence (permute the contraction order,
   show the output digest unchanged), you can emit `order_independent: true`
   receipts; if not, you are still a fully conforming implementation — the format's
   job is to make the distinction visible, not to exclude you.

## Pitfalls, in the order you will meet them

- **JSON does not carry dtype.** Tensor digests are dtype-sensitive; the moment rows
  or tensors cross a JSON boundary (APIs, files), the dtype must travel explicitly or
  an honest re-execution digests differently and you will chase a phantom REJECT.
  We shipped this bug in our own verifier service the same night we wrote it.
- **The regime-bound caution (spec, PROFILES section).** b-posit16 implementations
  exist bounded at ±48 and unbounded to ±112; they agree bit-for-bit only inside
  `|total exponent| ≤ 48`. Until v1 pins the base profile, stay inside the envelope
  or declare the `-b48`/`-b112` sub-profile explicitly.
- **Number formatting in canonical JSON.** Integers must serialize without a decimal
  point; if your language renders `1.0` where Python renders `1`, your certificates
  will differ on manifests containing float-valued metadata. (CR v0.1 keeps `params`
  integer-valued deliberately; treat any float in a manifest as a red flag.)
- **Endianness is forced, not inherited.** If you are on big-endian hardware and
  your digests differ, you inherited native byte order somewhere.
- **Meta is not certified.** Do not include `meta` in the certificate preimage, and
  do not be surprised that two receipts with different `meta` share a certificate —
  that is the design: machine details are provenance, never evidence.
- **Verify means recompute.** `sample_indices_of` re-derives indices from the
  manifest + challenge; never read indices from anything the prover sent. Same for
  the chain: position, linkage, and the closing digest are recomputed, not trusted.

## A worked second-language example

`examples/js/cr_canonical.mjs` is a from-scratch **JavaScript** implementation of the
canonicalisation layer (§2 canonical form + §3 tensor / collection digests), written from
the spec text with Node's standard library only — no CR code, no reading of the reference.
It emits the eight canonicalisation-layer vectors and passes the runner **8/8**:

```bash
node examples/js/cr_canonical.mjs | python3 python/conformance_runner.py /dev/stdin
```

It exists to prove the point of §9 concretely — that the pinned rules are enough for a
second implementation in a *different language* to reproduce the vectors, escaping, int8
dtype and code-point sort included — and to give you a short worked reference to read
alongside the spec.

`examples/js/cr.mjs` goes further: a from-scratch JavaScript **emitter** for the whole
receipt layer — manifest + certificate (§4–§5), the sampled-index PRF (§10, hex-decoded
challenge and all), and the chain digest (§12). It reproduces **12 of the 17 vectors** —
every canonicalisation and every positive receipt / sampled / chain vector — using only
canonicalisation + sha256 + the PRF, no b-posit arithmetic (a receipt's *values* need the
number system; its *bytes* do not). The remaining five are `refuse/*`, which assert a
verdict a *verifier* must return — the verifier half, left as the well-specified next step.

## Self-certify with the conformance runner

You do not have to eyeball diffs. Emit your conformance vectors in the published schema
(`python/conformance_runner.py --emit` prints a fill-in template with the inputs and the
value fields blanked), then grade yourself:

```bash
python3 python/conformance_runner.py your-vectors.json
```

It compares every pinned value — canonical bytes, digests, certificate,
`manifest_canonical`, sampled indices, `chain_digest`, and the refuse-verdicts — against
the published vectors and prints a per-vector PASS/FAIL with a precise field-level diff,
plus coverage per layer (canonicalisation, which needs no arithmetic, vs receipt/verdict).
`--demo` grades the reference against itself so you can see a full-green run first. The
runner depends only on the published vectors file, so it is a neutral referee, not the
reference grading itself.

## Getting listed

When your implementation reproduces the 17 vectors (refusals included) — i.e. the runner
prints `PASS: 17/17` — open an issue with your results and a pointer to the code. We list
independent implementations in the README with no further ceremony — an implementation we
did not write is the most valuable thing this repository can accumulate, and the first
spec ambiguity you hit is worth more to us than the listing is to you.
