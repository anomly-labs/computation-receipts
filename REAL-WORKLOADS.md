<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Computation Receipts on real workloads — HPC, training, inference

The 17 published conformance vectors pin the *format*; they are deliberately tiny. This page is the
format doing its actual job on the three workload classes where cross-machine bit-for-bit
reproducibility is a real, funded, unsolved problem — and where a receipt therefore buys something
float-on-a-GPU cannot. Each result below was run on real data (real pretrained weights, real
ill-conditioned reductions), and each states its honest scope.

The pattern is the same in all three: a **prover** computes under one execution decomposition (one
tiling, one rank count, one worker count) and emits a CR receipt; a **verifier** re-executes under a
*different* decomposition — what a different machine would do — and the same reference verifier that
grades the 17 vectors returns the verdict.

## 1. HPC — the reduction / `MPI_Allreduce` problem

Summing a value across N ranks and across M ranks happens in a different order, so in floating point
the results disagree in the last bits — and for an ill-conditioned reduction they disagree in the
*first* bits. "Run the same simulation on a different core count, get a different number" is the pain
that Intel's Conditional Numerical Reproducibility and the ReproBLAS line of work exist to address.

**Ran:** an ill-conditioned global sum — 4,099 terms, condition number ~1e9 (terms O(1e9), true
total O(1)) — summed as a 64-rank job vs a 1024-rank job.

| | 64-rank order | 1024-rank order | verdict |
|---|---|---|---|
| float64 | `+3328.000000` | `+5888.000000` | **UNVERIFIABLE** — not reproducible, both wrong vs exact `+1.25` |
| exact quire | `+1.250000` | `+1.250000` | **ACCEPT** — bit-identical across rank counts |

A reduction result that re-verifies on any machine at any core count.

## 2. Training — the DDP gradient all-reduce

Data-parallel training splits a batch across workers, sums the local gradient shards with an
all-reduce, and steps the weights. That sum is order-dependent, so the *same* step on 8 GPUs and on
64 GPUs produces gradients that differ in the last bits — and over thousands of steps that compounds.
A multi-million-dollar training run is, as a result, **not bit-reproducible**, and "did the provider
run the step I paid for, on the model I specified" has no checkable answer.

**Ran:** the weight gradient `X^T @ dY` of a **real Llama-3.2-1B MLP weight** (the reduction the
all-reduce performs), batch 256, summed as 8 workers vs 64 workers.

- **float32:** the reduced gradient is *not* bit-identical across worker counts (max |Δ| ≈ 3.8e-6) →
  **UNVERIFIABLE**.
- **exact quire:** the reduced gradient — and therefore the updated weights — is bit-identical across
  worker counts → **ACCEPT**. A corrupted shard (one worker's contribution perturbed) → **REJECT**.

Bit-reproducible distributed training, checkably.

## 3. Inference — a real pretrained forward pass

**Ran:** a **Llama-3.2-1B layer-0** attention+MLP sub-block on its real trained weights, under exact
quire. A prover computes it under one contraction order; a verifier re-executes under a completely
different order (a different chip's tiling).

- Output **bit-identical** across the two independent plans → the certificate matches → **ACCEPT**.
- One weight perturbed by a single ULP in the re-execution → **REJECT** ("re-execution used a
  different model").
- The same block in float64 → the two plans disagree → **UNVERIFIABLE** (agreement would be luck,
  disagreement is expected — CR never sells a coincidence as a proof).

## Honest scope (read this)

- **Real values, tractable width.** The 256-bit quire is emulated in Python here, which is too slow
  at full model width, so the neural pillars run a real *sub-block* (d=48) of the real weight
  matrices, not the full SwiGLU/RoPE forward. The reproducibility property is a property of the
  **accumulation**, so a sub-block is a faithful witness — but these are witnesses, not throughput
  benchmarks.
- **Cross-implementation, not yet cross-silicon here.** The decompositions above are different
  execution *plans* on one CPU, standing in for different machines. Re-verification on genuinely
  different silicon is the stronger proof; Anomly has run b-posit/quire GEMM bit-exact on real FPGA
  and Tenstorrent hardware, and that is the direction this composes toward.
- **The certifiable property is order-independence, not infinite precision.** The quire sums the
  b-posit-rounded operands *exactly and identically across decompositions*, for any inputs — that is
  what a receipt certifies. When operands fit the format it also equals the true value; that is a
  bonus, not the claim.

## What you can reproduce yourself — with no arithmetic and no special hardware

The **format** is the part that needs nothing but Python: the 17 conformance vectors, the reference
verifier, and a from-scratch second implementation that reproduces every vector (see
[`README.md`](README.md) and [`EVALUATORS.md`](EVALUATORS.md), "verify every claim in an afternoon").
The real-workload results above combine that format with exact-quire **arithmetic** — b-posit and the
256-bit quire, which are published prior work (Gustafson & Yonemoto 2017; Posit Standard 2022), not
part of this repository. CR's contribution is the verification layer that turns their order-independence
into a checkable certificate.

If your workload is a large reduction, a distributed training run, or exact-arithmetic inference where
reproducibility has to be *checkable*, that is exactly the fit — and the most useful thing you can send
back is either a spec ambiguity you hit implementing it, or a real workload of your own we should try.

---
Back to: [`EVALUATORS.md`](EVALUATORS.md) · [`README.md`](README.md) · Contact:
[anomly.com/contact](https://anomly.com/contact)
