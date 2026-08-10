<!-- Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs). -->
# Integrating Computation Receipts into a real pipeline

For the two sides of a real deployment: a **provider** that runs a workload and wants to hand back a
checkable result, and a **client / auditor** that wants to verify one. This is the *wiring* guide —
`EVALUATORS.md` is why you'd care, `IMPLEMENTERS.md` is how to reimplement the format in another
language. Here we assume the reference `cr` package and show the minimal changes to an existing
HPC / training / inference pipeline.

## The 30-second model

```
provider:  y = compute(...)                       # your existing reduction / step / forward pass
           r = build_receipt(profile=…, computation=…, model=…, inputs=…, output=y)
           ship (y, r.to_json())                  # the certificate travels with the result

client:    y2 = recompute(...)                     # on the client's OWN, different hardware
           r2 = build_receipt(… output=y2 …)       # same bindings
           verdict = verify(r_from_provider, r2)    # ACCEPT / REJECT / UNVERIFIABLE / MALFORMED
```

The certificate is `sha256(canonical(manifest))`; the manifest binds the output, model, inputs,
computation id, and arithmetic profile *by digest*. Verification is re-execution plus a comparison —
no prover, no trusted machine.

## Emit side — the minimal diff

`build_receipt` is keyword-only; bind whatever the result should be *about*. `model`/`inputs` are
maps of named numpy arrays (digested, not stored); `output` is the numpy array you're certifying.

**HPC reduction** (a global sum / `MPI_Allreduce` result):
```python
from cr import build_receipt
total = allreduce(local_terms)                                  # your existing reduction
r = build_receipt(profile="bposit16-quire256",                  # the arithmetic you actually ran
                  computation="hpc.allreduce.global-sum",
                  inputs={"terms": terms},                       # bind the summands
                  output=np.asarray([total]))
emit(total, r.to_json())
```

**Training step** (a DDP gradient all-reduce / weight update):
```python
grad = all_reduce(local_grads)                                  # Σ shards over the worker dim
W_next = W - lr * grad
r = build_receipt(profile="bposit16-quire256",
                  computation="training.ddp.allreduce-gradient.linear",
                  computation_version=f"step={step},lr={lr}",
                  model={"W": W},                                # pre-step weights = the model
                  inputs={"X": X, "dY": dY},                     # the batch
                  output=grad)                                   # (or W_next — certify what matters)
emit(W_next, r.to_json())
```

**Inference** (a forward pass):
```python
y = model.forward(x)
r = build_receipt(profile="bposit16-quire256",
                  computation="llama-3.2-1b.layer0.attn-mlp",
                  model={f"w{i}": w for i, w in enumerate(weights)},
                  inputs={"x": x},
                  output=y)
serve(y, r.to_json())
```

That is the whole emit path: one call, attach `r.to_json()` to the response. Optionally pass a
`graph=` (a CRG computation graph) to also bind *how* the output was computed, closing the
"same name, different computation" gap — see spec §11.

## Verify side — re-execute and read the verdict

```python
from cr import Receipt, verify
claimed = Receipt.from_json(received_json)
y2 = recompute_on_my_hardware(...)                              # independent re-execution
mine = build_receipt(profile=claimed.manifest["arithmetic"]["profile"],
                     computation=claimed.manifest["computation"]["id"],
                     model=…, inputs=…, output=y2)
print(verify(claimed, mine))
```

The four verdicts are operationally distinct — treat them differently:

| verdict | meaning | what to do |
|---|---|---|
| **ACCEPT** | re-execution reproduced the certificate bit-for-bit | trust the result; keep the receipt as the record |
| **REJECT** | a bound section differs — different model, inputs, computation, profile, or (under order-independent arithmetic) a different output | do **not** trust it; you have a concrete, attributable mismatch |
| **UNVERIFIABLE** | the arithmetic is order-dependent (e.g. float), so agreement would be luck and disagreement is expected — no verdict is possible | the result is not checkable *as arithmetic*; see the prerequisite below |
| **MALFORMED** | the receipt is structurally invalid or its certificate doesn't match its manifest | reject the document itself; it was tampered or miscomputed |

`verify(claimed, None)` returns UNVERIFIABLE ("no re-execution supplied") — a re-execution is
required for any real verdict.

## The honest prerequisite — read this before you integrate

**A real ACCEPT requires order-independent (exact-quire) arithmetic.** If your pipeline runs
ordinary float32/float64, two honest machines disagree in the last bits, so CR's honest verdict is
**UNVERIFIABLE** — by design, not as a failure. CR will not tell you a float coincidence is a proof.

So CR fits precisely where you either already run exact accumulation (a Kulisch/quire long
accumulator, ReproBLAS-style correctly-rounded reductions, b-posit/quire hardware) or can afford to:
regulated/audited inference, provider↔client dispute resolution, detecting model substitution,
cross-vendor result agreement. If your workload is best-effort float nobody will dispute, CR's
verdict will be UNVERIFIABLE and it is not the tool for you — we would rather say so here.

(A generic *public* profile that a pure-software pipeline can use to reach ACCEPT —
`exact-real-f64`, accumulate exactly then round once — is proposed and POC-verified in
[`docs/proposals/exact-accumulation-profile.md`](docs/proposals/exact-accumulation-profile.md); it
is not yet in the registry.)

## Outputs too big to re-execute whole — sampled receipts

When re-executing the entire output is impractical, a **sampled receipt** commits to the full output
digest but lets the verifier re-execute only a fraction of the output rows (spec §10):

```python
from cr import build_sampled_receipt, sample_indices_of, verify_sampled
r = build_sampled_receipt(profile=…, computation=…, model=…, inputs=…,
                          output=Y, sample_size=64, challenge=chal)   # verify cost ≈ 64/n_rows
idx = sample_indices_of(r)                                            # deterministic, prover can't pick
verdict = verify_sampled(r, recompute_rows(idx))                     # re-execute only those rows
```

Two boundaries to integrate correctly:
- The sample indices derive from the manifest (which already binds the *full* output digest) plus a
  `challenge`. With an **empty** challenge a sampled receipt is evidence only against
  non-adaptive faults (bit flips, wrong weights, drift) — an adaptive prover can grind.
- Against an adversarial prover, supply a `challenge` the prover cannot predict: a verifier nonce, or
  a public-randomness beacon fixed *after* the output (`cr.beacon.beacon_challenge`, and
  `cr.beacon.audit` to re-check a transcript against the live beacon later). ACCEPT means "the
  sampled rows reproduced and the full output is committed", never "every row was checked".

For a long streaming run, chunked receipts (`build_chunk_receipt` → `build_closing_receipt` →
`verify_chain`) bind a sequence into one closing certificate.

## What NOT to trust a receipt for

- **Confidentiality** — the verifier must hold the operands to re-execute; a receipt hides nothing.
  Compose with other tools if secrecy is required.
- **Timing / performance** — a receipt says *what* was computed, never how fast or how cheaply.
- **Machine identity** — machine metadata is recorded but never certified; the machine's irrelevance
  is the point. A receipt does not attest *who* ran it, only that the computation reproduces.

---
Start here: [`README.md`](README.md) · Why it matters: [`EVALUATORS.md`](EVALUATORS.md) · On real
workloads: [`REAL-WORKLOADS.md`](REAL-WORKLOADS.md) · Spec: [`spec/CR-v0.1.md`](spec/CR-v0.1.md) ·
Contact: [anomly.com/contact](https://anomly.com/contact)
