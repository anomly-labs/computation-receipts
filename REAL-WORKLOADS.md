<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Computation Receipts on real workloads — HPC, training, inference

The 17 published conformance vectors pin the *format*; they are deliberately tiny. This page is the
format doing its actual job on the workload classes where cross-machine bit-for-bit reproducibility
is a real, funded, unsolved problem — HPC reductions, distributed training, inference, and iterative
solvers — where a receipt therefore buys something float-on-a-GPU cannot. Each result below was run
on real data (real pretrained weights, real ill-conditioned systems), and each states its honest scope.

The pattern is the same throughout: a **prover** computes under one execution decomposition (one
tiling, one rank count, one worker/core count) and emits a CR receipt; a **verifier** re-executes
under a *different* decomposition — what a different machine would do — and the same reference verifier
that grades the 17 vectors returns the verdict.

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

**Multi-step: the divergence compounds.** One step's last-bit difference is not the whole story —
it feeds back through the weights (a changed weight changes the next forward, hence the next
gradient), so over a training trajectory the two worker-count runs drift apart and keep drifting.
Ran a real 40-step full-batch SGD of `y = X·W1` on that same Llama-3.2-1B weight (loss genuinely
descends, `0.01001 → 0.00724`):

| | step 1 | step 40 | verdict on final weights |
|---|---|---|---|
| float32, `max\|Δweight\|` between 8- and 64-worker runs | `1.40e-9` | `5.12e-9` (grows) | **UNVERIFIABLE** — two worker counts, two different models |
| exact quire | `0` | `0` (bit-identical every step) | **ACCEPT** — the run re-verifies at any worker count |

That compounding is why a long, expensive training run is not bit-reproducible in floating point;
exact accumulation removes it at the source. (Honest scope as above: real weight values, a tractable
emulated-quire sub-block — a witness of the accumulation property, not a throughput benchmark.)

## 3. Inference — a real pretrained forward pass

**Ran:** a **Llama-3.2-1B layer-0** attention+MLP sub-block on its real trained weights, under exact
quire. A prover computes it under one contraction order; a verifier re-executes under a completely
different order (a different chip's tiling).

- Output **bit-identical** across the two independent plans → the certificate matches → **ACCEPT**.
- One weight perturbed by a single ULP in the re-execution → **REJECT** ("re-execution used a
  different model").
- The same block in float64 → the two plans disagree → **UNVERIFIABLE** (agreement would be luck,
  disagreement is expected — CR never sells a coincidence as a proof).

**Does the non-reproducibility change the answer, or just the low bits?** At **bf16 — the precision
production inference actually runs — yes.** On the full real Llama-3.2-1B, computing the final LM-head
reduction (over the 2048-wide hidden dimension) in several different summation orders — the same
arithmetic, only the tiling changes — flips real next-token argmax decisions. Measured across **8
diverse texts** (narrative, news, code, dialogue, technical, Q&A, list, philosophical), 6 tilings
each: **aggregate 9.6% of decisions order-sensitive (31 of 324 positions), per-text 3.8%–19.4%
(median 7.6%)**, concentrated at genuinely ambiguous positions (top-1/top-2 margins ~4e-2). The same
measurement in fp32 flips **0% on every text** (fp32's error sits below the margins). The rate grows
with the number of tilings compared and with text ambiguity, but the invariant holds: bf16 decode is
not reproducible across GPU tilings/hardware — a different accelerator can emit a different token — so
CR's verdict on that decision is **UNVERIFIABLE**. Exact-quire accumulation is order-invariant, so
those decisions are the same on every re-execution → **ACCEPT**. (Honest scope: this certifies
*reproducibility* — the same token on any honest hardware — not fp64 *correctness*; b-posit16 operands
are 16-bit like bf16. The value is a bit-reproducible, re-verifiable decode.)

**And the flips compound.** Greedy-decoding the same prompt while varying *only* the bf16 LM-head
reduction order (a shared fp64 backbone isolates the effect) forks the output into **different text**
on all three prompts tried — the sequences share a prefix, then a single order-induced token flip
changes the context and they diverge into different continuations. So the non-reproducibility does not
wash out at the sequence level: same model, same prompt, a different tiling, a different story — and a
generation nobody can certify. Nor is this an artifact of the isolation: repeating the experiment with
*every* linear in the model (113 of them) computing its reduction in a per-run order, bf16 throughout,
also forks all three prompts (two within 40 generated tokens, the third at token 44). Exact-quire
accumulation removes it at the source.

## 4. HPC — an iterative solver (conjugate gradient)

Conjugate gradient — the workhorse iterative solver for large sparse SPD systems in CFD, FEM and
PDE-constrained problems — computes two **dot products** per iteration (`α = rᵀr / pᵀAp`,
`β = r_newᵀr_new / rᵀr`). A dot product's floating-point value depends on how its vector is
partitioned across cores, so the *same* solve on a different core count takes a slightly different α
at step 1, a different search direction at step 2, and the trajectories diverge — frequently reaching
the tolerance in a **different number of iterations**. This is a documented HPC reproducibility pain
(the reproducible-BLAS / reproducible-CG line of work exists for it).

**Ran:** CG on a real ill-conditioned SPD system (n = 64, condition number ~300), under a 64-core vs
a 1024-core partition order.

| | iterations to tol | trajectory `max‖x₆₄ − x₁₀₂₄‖` | verdict on final `x` |
|---|---|---|---|
| float32 | **50 vs 51** (different count) | `1.1` (final `x` differs by `4.9e-3`) | **UNVERIFIABLE** |
| exact quire | 57 vs 57 | `0` (bit-identical every iteration) | **ACCEPT** |

A corrupted `b` shard → **REJECT**. (Honest scope: operands are b-posit16, so the accuracy floor is
that precision — the true residual settles near `6e-3`; the receipt certifies the **reproducibility**
of the trajectory across core counts, not tighter accuracy. Emulated-quire witness, not a throughput
benchmark.)

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

## What it costs — the honest verification tax

The question after "does it work" is "what does it cost me." There are two separable costs, and
conflating them is the usual sleight of hand:

**1. Compute cost** — running the workload in exact b-posit/quire arithmetic instead of float.
Measured on this CPU, one exact-quire transformer forward costs **roughly 15–20× a float forward**
(up to ~22× at the deepest stack) across model depth (1→8 blocks) — and the exact multiplier is
wall-clock and varies run to run, so treat it as an order-of-magnitude figure, not a constant. That ratio is a *software-emulation*
artefact: the 256-bit quire is emulated in NumPy here. On native b-posit silicon the quire lives in the MAC datapath — it
accumulates every cycle, so exact accumulation is inherent, not an add-on — and b-posit/quire GEMM
has run **bit-exact on real FPGA and Tenstorrent hardware at hardware throughput**. The number to
quote is the silicon cost (≈ native); the CPU ratio is the emulator's tax, not the idea's.

**2. Verification cost** — the extra work to *check* a result: re-execute and compare a 32-byte
digest. That is ~1 extra forward pass plus a hash. This is the cost that actually distinguishes CR,
and it is small by construction:

| approach | verification cost | trust assumption |
|---|---|---|
| **CR** (re-execute + compare digest) | **~1× forward pass + a hash** | **none — the math is the root of trust** |
| Zero-knowledge ML | ~100–1000×+ prover overhead | none, but you pay the prover tax |
| Trusted-hardware enclave | ~1× (near-native) | trust the chip vendor's root of trust |
| Deterministic replay / pinned kernels | ~1× | none, but you surrender hardware heterogeneity |

A receipt is a **checksum/ECC/TLS-style tax**: a bounded, defined overhead you pay *only when you
choose to verify*, to turn a result into one anybody can re-check — with no prover and no trusted
machine. CR buys the enclave's ~1× verification cost *without* the enclave's trust assumption, and
ZKML's zero-trust *without* ZKML's prover tax. The price of that trade is that the compute must be
exact-quire (more than float in emulation, ≈ native on b-posit silicon) and float workloads stay
UNVERIFIABLE — which is exactly the honest scope above.

## What you can reproduce yourself — with no arithmetic and no special hardware

The **format** is the part that needs nothing but Python: the 17 conformance vectors, the reference
verifier, and a from-scratch second implementation that reproduces every vector (see
[`README.md`](README.md) and [`EVALUATORS.md`](EVALUATORS.md), "verify every claim in an afternoon").

And you can run the **HPC reduction above yourself**, publicly, with numpy + stdlib only —
`python3 examples/python/hpc_reduction_demo.py`. It reproduces float's honest UNVERIFIABLE verdict
(whether the rank orders disagree, agree by luck, or a cheated total is claimed) and shows exact
accumulation is order-independent. The one thing it stops short of is emitting an *ACCEPTing* receipt:
that needs the receipt to name an order-independent arithmetic profile the verifier can assess, and
the registered ones today are the b-posit / 256-bit-quire family (the arithmetic below). A generic
public "exact-accumulation" profile would let a pure-Python evaluator close that loop to a real
ACCEPT — see the review-ready proposal in
[`docs/proposals/exact-accumulation-profile.md`](docs/proposals/exact-accumulation-profile.md)
(a verified, no-spec-change option is recommended there; the decision is deliberate, not taken
unilaterally).
The real-workload results above combine that format with exact-quire **arithmetic** — b-posit and the
256-bit quire, which are published prior work (Gustafson & Yonemoto 2017; Posit Standard 2022), not
part of this repository. CR's contribution is the verification layer that turns their order-independence
into a checkable certificate.

**Reproducing every number on this page.** The four pillar results above are produced by one command
in Anomly's research tree (`python -m spacetime.cr_pillars`), which runs all pillars, prints the
summary, and **archives each prover's receipt as JSON**. A receipt is what a second machine re-checks:
when a computation is re-run on different hardware — a different CPU, an FPGA, a Tenstorrent card — the
verifier recomputes and calls `verify(archived_receipt, its_own_receipt)`, and bit-identical
exact-quire output yields ACCEPT across the hardware boundary. Those archived certificates are the
setup for that cross-silicon proof; the arithmetic itself (b-posit / quire) is the research-tree part,
per the scope above.

If your workload is a large reduction, a distributed training run, or exact-arithmetic inference where
reproducibility has to be *checkable*, that is exactly the fit — and the most useful thing you can send
back is either a spec ambiguity you hit implementing it, or a real workload of your own we should try.

---
Back to: [`EVALUATORS.md`](EVALUATORS.md) · [`README.md`](README.md) · Contact:
[anomly.com/contact](https://anomly.com/contact)
