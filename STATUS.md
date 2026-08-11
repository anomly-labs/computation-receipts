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
  a profile-registry fuzzer, an HTTP-level hostile-body sweep of the verifier service,
  a canonicalisation (non-finite-value) fuzzer, a C↔Python special-character emitter
  differential, and a graph-digest / C-print hostile-input probe

**Eighteen code defects were found and fixed** in the shipped surface (two would have been
release blockers), plus three spec-level clarifications (S1–S3) pinned in `spec/CR-v0.1.md`
after an independent implementation built from the spec text alone found cross-implementation
ambiguities. No forgery of the certificate binding has been found. All shipped self-checks
currently pass — run them yourself:

```bash
cd python && python3 run_checks.py
```

## Recent (2026-08-10/11)

- **Real-workload demonstrations** ([REAL-WORKLOADS.md](REAL-WORKLOADS.md)): HPC reduction,
  distributed-training all-reduce, real-Llama inference (incl. measured bf16 decision flips and
  generation forks), a CG solver — and **§5: one certificate re-verified across different
  silicon** (RTX 5090 / RTX 3090 / CPU bit-identical; cuBLAS FP32 deterministic per-GPU yet
  different across the two architectures).
- **A second exact profile registered**: `bposit8-imma-int32` (spec §7) — the INT8-tensor-core
  W8A8 path, with the cross-GPU receipts as its order-independence demonstration. The open
  registry is no longer hypothetical.
- **`examples/python/cross_gpu_receipt_demo.py`**: pair it with the mosyne-bposit 60-second GPU
  demo to run the receipt flow on your own hardware's codes.

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
