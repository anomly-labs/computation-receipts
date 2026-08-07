<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Implementing CR in your language — the practical guide

This is the path we would take in your seat, plus every pitfall we know about —
including the ones we found by hitting them. If the spec and this guide disagree, the
spec wins and the disagreement is a bug: please report it.

## Order of attack (mirrors spec §9-§10)

1. **Canonicalisation first — no arithmetic needed.** Reproduce the `canonical/*`
   vectors: UTF-8 JSON, keys sorted, separators `,` and `:` with no whitespace,
   non-ASCII passed through as UTF-8 (not `\u`-escaped), and **NaN/Infinity are
   errors**, never emitted. Three vectors, minutes of work, and they catch most
   language-default JSON encoders doing something else.
2. **Tensor digests.** `sha256( canonical({"dtype":…,"shape":…}) ‖ raw-bytes )` with
   the raw bytes **forced little-endian, C-order contiguous**. The dtype string is
   numpy convention *without* the byte-order prefix (`u2`, `f8`, `i4`). Reproduce the
   `tensor/*` and `tensors/*` vectors — the named-collection digest is
   order-independent over names by construction; prove yours is.
3. **One full receipt.** Build the manifest exactly as §4 lays it out, hash it, match
   the published receipt vector byte-for-byte (`manifest_canonical` is in the vector
   precisely so you can diff bytes, not vibes).
4. **The verdicts — the part that matters most.** Implement all four, then run the
   three `refuse/*` vectors. A verifier that cannot refuse is worthless; the two
   soundness bugs we found in our own code were both accept-what-must-be-rejected,
   and only negative vectors catch that class.
5. **Sampled + chained receipts** (§10, §12): the PRF index derivation
   (`sha256(seed ‖ counter_be8)` with modulo-bias rejection), and the chain rules
   (prev-certificate linkage, closing receipt, the truncation refusal).
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

## Getting listed

When your implementation reproduces the 12 vectors (refusals included), open an issue
with your results and a pointer to the code. We list independent implementations in
the README with no further ceremony — an implementation we did not write is the most
valuable thing this repository can accumulate, and the first spec ambiguity you hit
is worth more to us than the listing is to you.
