# Status & maturity

**This is a v0.1 draft under active hardening — not a frozen standard.** Share it,
review it, build on it, but expect the format to still move at the edges.

## What has been done

The reference implementation and the C emitter have been through an independent
adversarial testing campaign (2026-08-08):

- ~18k-case soundness property fuzzer over the verifier
- hypothesis property-based testing (round-trip, certificate binding, refusal, sampled
  invariants, canonicalisation stability)
- ~48k-case protocol + beacon-audit fuzzer
- ~8k-case **C ↔ Python byte-for-byte differential** (manifest emitters + sample-index PRF)
- whole-C-surface **AddressSanitizer + UBSan** sweep; multi-compiler (gcc + clang)
  strict-warning build (`-Wconversion -Wsign-conversion …`)
- two LLM code reviews (function-level + spec-vs-code conformance), every finding
  weaponized against the real code
- sound-objective **evolutionary** attack search against the certificate binding
- a hostile-input fuzzer over `Receipt.from_json` (the untrusted entry point), an
  exotic-numpy-dtype fuzzer over the tensor-digest layer (collision / reproducibility),
  a chain/chunked-receipt verify fuzzer, a cross-version-section weaponization of the
  challenge-response protocol, a network-mocked `beacon.audit` hostile-transcript fuzzer,
  a profile-registry fuzzer, and an HTTP-level hostile-body sweep of the verifier service

**Fourteen real defects were found and fixed** in the process (two would have been release
blockers). No forgery of the certificate binding has been found. All shipped self-checks
currently pass — run them yourself:

```bash
cd python && python3 run_checks.py
```

## What is deliberately still open

- **The b-posit16 regime bound (±48 vs ±112).** Two conventions exist; they agree inside
  the ±48 envelope. This is a v1 decision, flagged as an implementer caution in the spec.
- **The profile registry is intentionally open** — any arithmetic that can demonstrate
  order-independence belongs in it, not only Anomly's.
- **The arithmetic itself** (b-posit, the 256-bit quire) is published prior work
  (Gustafson & Yonemoto 2017; Posit Standard 2022) — this project's contribution is the
  *verification layer* (the receipt format and protocol), not the number system.

## What this is not (yet)

A finished, frozen, independently-adopted standard. The single most valuable next step is
exactly what a second reader provides: an **independent implementation built from the spec
alone** (`spec/CR-v0.1.md`) that reproduces the conformance vectors. If a vector is
ambiguous, that is a spec bug worth reporting.
