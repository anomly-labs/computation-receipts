<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Carrying a Computation Receipt inside the supply-chain trust stack

CR answers a question the existing stack doesn't: *was the number right, and can anyone
re-derive it?* The existing stack answers everything around it: what artifact is this
(signing, SLSA), what environment ran it (TEE attestation), and was this statement
registered when they say it was (transparency logs). These compose — a CR is a **claim**,
and the supply-chain stack is plumbing for shipping, signing, and logging claims. This
page shows the two standard carriages: an **in-toto attestation** and a **SCITT Signed
Statement**. Nothing here changes the CR format; `spec/CR-v0.1.md` remains the authority
for the bytes being carried.

**Naming discipline, up front.** In IETF usage (SCITT, RFC 9942) a *"Receipt"* is a
transparency log's proof that a statement was registered. A Computation Receipt is not
that — it is the *claim inside* the statement: a correctness assertion checkable by
re-execution. When writing for an IETF audience, say "CR claim" or "computation
certificate" for our object and reserve bare "receipt" for the log's inclusion proof.
The two are complementary: the log proves *when and by whom the claim was registered*;
re-executing the CR proves *the claim is true*.

## 1. As an in-toto attestation (v1 Statement)

An [in-toto Statement](https://github.com/in-toto/attestation) binds a predicate to
subject artifacts by digest. The natural mapping:

| in-toto field | CR mapping |
|---|---|
| `subject[0].digest.sha256` | the receipt's `manifest.output.digest` (hex, no `sha256:` prefix) — the Statement is *about* the certified output |
| `subject[0].name` | your artifact name for the output (result file, tensor id, response id) |
| `predicateType` | `https://github.com/anomly-labs/computation-receipts/blob/main/spec/CR-v0.1.md` |
| `predicate` | the full receipt object, verbatim (`certificate` + `manifest` + `meta`) |

Rules that keep the composition sound:

1. **The receipt travels verbatim.** The predicate is the exact receipt JSON object —
   re-canonicalize per CR §canonicalization before hashing, never per the carrier's
   pretty-printer. A verifier extracts `predicate`, recomputes
   `sha256(canonical(manifest))`, and proceeds exactly as with a bare receipt.
2. **Subject digest MUST equal the manifest's output digest.** That's the join between
   the two trust systems: in-toto consumers match the subject against the artifact they
   hold; CR verifiers re-execute against the same digest. If they disagree, the
   attestation is malformed — refuse it.
3. **Signing is the carrier's job, not CR's.** Wrap the Statement in a DSSE envelope
   (`payloadType: application/vnd.in-toto+json`) and sign with whatever key
   infrastructure you already run (Sigstore, KMS, offline keys). A signature says *who
   asserts this claim*; only re-execution says the claim is true. An unsigned CR is
   still fully verifiable — signature and verification are independent layers.

Runnable, offline, numpy + stdlib only:

```bash
python3 examples/python/in_toto_statement_demo.py    # emits a Statement + DSSE PAE, then verifies the embedded receipt round-trip
```

The demo builds a real receipt for a small reduction, wraps it in a v1 Statement,
computes the DSSE pre-authentication encoding (what your signer would actually sign),
then plays the *consumer*: extracts the predicate, checks subject↔output digest
equality, and re-verifies the receipt. One PASS/FAIL line per step.

## 2. As a SCITT Signed Statement

In [SCITT](https://datatracker.ietf.org/wg/scitt/about/) terms: an **Issuer** (the
compute provider) makes a **Signed Statement** whose payload is the CR claim, registers
it with a **Transparency Service**, and gets back a **Receipt** (IETF sense: inclusion
proof, RFC 9942). Mapping onto the COSE_Sign1 envelope:

| COSE / SCITT element | CR mapping |
|---|---|
| payload | the canonical CR manifest bytes (CR §canonicalization) — canonical *before* enveloping, so the payload is byte-stable across implementations |
| protected header: content type (label 3) | `application/vnd.anomly.cr+json` |
| protected header: CWT claims — issuer | the compute provider's identifier |
| protected header: CWT claims — subject | the receipt's `certificate` string (`sha256:…`) — the claim's stable name |
| signature | the Issuer's key, per their SCITT profile |

What each layer buys, kept honest:

- The **Signed Statement** proves the provider asserted this certificate, non-repudiably.
- The **Transparency Service receipt** proves the assertion was registered at a point in
  time — drift and back-dating become detectable.
- **Neither proves the number is right.** Only re-execution per CR §verification does.
  A SCITT deployment carrying CR claims gets all three properties; a SCITT deployment
  carrying plain logs gets only the first two. That third property is the point of CR.

Runnable, offline (numpy + stdlib + `cryptography` for Ed25519; SKIPs cleanly without it):

```bash
python3 examples/python/scitt_signed_statement_demo.py   # issuer builds + signs, verifier decodes + checks the binding, tamper refused
```

The demo emits a genuine tagged COSE_Sign1 (the CBOR is written out inline — ~50
lines — so every wire byte is inspectable) and covers the ISSUER half only: there is
no Transparency Service here, hence no registration and no Receipt (IETF sense). A
real deployment signs with its own keys under its Transparency Service's COSE
profile — the demo key is a fixed-seed demo artifact. If you are wiring CR into an
actual SCITT deployment and want conformance guidance, open an issue — this is
exactly the kind of external implementation the format wants:
<https://github.com/anomly-labs/computation-receipts/issues>.

## 3. What not to do

- **Don't re-serialize the manifest inside a carrier.** Canonicalization is CR's, not
  JSON-LD's, not the DSSE tooling's. Carry bytes; verify per spec.
- **Don't treat a signature or an inclusion proof as verification.** They are
  provenance. Verification is re-execution (§verification; verdicts ACCEPT / REJECT /
  UNVERIFIABLE / MALFORMED).
- **Don't put big tensors in the carrier.** CR binds model/inputs/output *by digest* on
  purpose; carriers carry claims, artifact stores carry artifacts.
