# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""cross_gpu_receipt_demo.py — turn a real cross-GPU run into verified Computation Receipts.

The companion repo `mosyne-bposit` (github.com/anomly-labs/mosyne-bposit) ships a 60-second GPU
demo (`scripts/repro_bench/min_demo && make demo`) that sums 65,536 log-uniform values on every
NVIDIA GPU in your machine two ways: IEEE FP32 with `atomicAdd` (different bit pattern nearly
every run) and bposit16 + a 256-bit quire (one bit pattern, every run, every GPU). This example
closes the loop with the RECEIPT layer: it takes the codes that demo prints and produces CR
receipts that the reference verifier grades — the same verifier that grades the published
conformance vectors.

    # after running the mosyne-bposit demo on a machine with 1+ NVIDIA GPUs:
    python3 cross_gpu_receipt_demo.py --gpu-a 0x7627 --gpu-b 0x7627

    # no GPU handy? the defaults are the codes from our published RTX 5090 + RTX 3090 run,
    # so running it bare shows the flow (clearly labeled as our run, not yours):
    python3 cross_gpu_receipt_demo.py

What it does, honestly:
  * regenerates the demo's exact input vector locally (the kernel's LCG is pinned: seed
    0x5EED1234, multiplier 1664525, increment 1013904223, 65,536 log-uniform values on
    [1e-3, 1e3]) and binds that vector into both receipts, so they attest one shared input
    (the demo kernel builds its summands from the same generator);
  * builds one receipt per GPU under the registered `bposit16-quire256` profile, output = the
    16-bit sum code each GPU produced, and runs `verify()` on the pair: matching codes on an
    order-independent profile → ACCEPT — one certificate, two different pieces of silicon;
  * demonstrates the refusals: a mismatched code (e.g. a typo, a broken card, a dishonest
    prover) → REJECT with the profile's teeth, and the FP32 atomicAdd results under the
    `float32` profile → UNVERIFIABLE even when they happen to agree;
  * scope, stated plainly: the receipt certifies that both machines produced the SAME exact
    accumulation over these bound inputs — reproducibility, not fp64 accuracy, and this script
    trusts you to type the codes your hardware printed (the demo binary is what computed them).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from cr.receipt import build_receipt, verify  # noqa: E402

COMPUTATION = "mosyne-bposit.min-demo.quire256-sum.loguniform-65536"
N = 65536

# The codes printed by our published run (RTX 5090 Blackwell + RTX 3090 Ampere, 2026-08-10),
# used only as documented defaults so the flow is demonstrable without a GPU.
PUBLISHED_BPOSIT = "0x7627"
# The DISTINCT FP32 atomicAdd bit patterns each GPU produced across 5 runs (the demo prints the
# deduplicated set: 5 unique on the 5090, 4 unique on the 3090, 8 distinct across both).
PUBLISHED_FP32_A = ["0x4a93ec13", "0x4a93ec41", "0x4a93ec46", "0x4a93ec68", "0x4a93ec6b"]
PUBLISHED_FP32_B = ["0x4a93ec5c", "0x4a93ec68", "0x4a93ec70", "0x4a93ec78"]


def demo_inputs() -> np.ndarray:
    """The min_demo kernel's input vector, regenerated bit-for-bit (same LCG, same mapping)."""
    seed = np.uint32(0x5EED1234)
    vals = np.empty(N, dtype=np.float64)
    s = int(seed)
    for i in range(N):
        s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
        u = (s >> 1) / 0x7FFFFFFF
        vals[i] = 10.0 ** (-3.0 + 6.0 * u)
    return vals


def receipt_for(code: int, substrate: str, inputs: np.ndarray):
    return build_receipt(profile="bposit16-quire256", computation=COMPUTATION,
                         inputs={"x": inputs}, output=np.array([code], dtype=np.uint16),
                         meta={"substrate": substrate})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-a", default=None, help="bposit16 sum code printed by GPU A, e.g. 0x7627")
    ap.add_argument("--gpu-b", default=None, help="bposit16 sum code printed by GPU B")
    args = ap.parse_args()
    using_published = args.gpu_a is None and args.gpu_b is None
    code_a = int(args.gpu_a or PUBLISHED_BPOSIT, 16)
    code_b = int(args.gpu_b or PUBLISHED_BPOSIT, 16)

    line = "=" * 86
    print(line)
    print("Cross-GPU Computation Receipts — one exact sum, two pieces of silicon, one certificate")
    print(line)
    src = ("our published RTX 5090 + RTX 3090 run (defaults — run the mosyne-bposit demo to "
           "use YOUR hardware's codes)" if using_published else "your hardware's codes")
    print(f"  codes from: {src}")

    x = demo_inputs()
    print(f"  inputs: regenerated the demo's {N} log-uniform values locally and bound them "
          f"into every receipt\n")

    ra = receipt_for(code_a, "GPU A", x)
    rb = receipt_for(code_b, "GPU B", x)
    v = verify(ra, rb)
    print(f"  GPU A bposit16 sum {code_a:#06x}  |  GPU B bposit16 sum {code_b:#06x}")
    print(f"  verify(GPU A receipt, GPU B receipt) -> {v.status}")
    if v.status == "ACCEPT":
        print("     one certificate, re-verified on different silicon — the receipt's whole point\n")
    else:
        print("     the two machines did NOT produce the same exact accumulation — investigate\n")

    r_bad = receipt_for(code_a ^ 1, "GPU B (corrupted)", x)
    print(f"  a mismatched code ({code_a ^ 1:#06x}) -> verify -> {verify(ra, r_bad).status}"
          "   <- wrong answers are caught, not averaged away")

    fa = build_receipt(profile="float32", computation=COMPUTATION + ".fp32-atomicadd",
                       inputs={"x": x.astype(np.float32)},
                       output=np.array([int(h, 16) for h in PUBLISHED_FP32_A], dtype=np.uint32))
    fb = build_receipt(profile="float32", computation=COMPUTATION + ".fp32-atomicadd",
                       inputs={"x": x.astype(np.float32)},
                       output=np.array([int(h, 16) for h in PUBLISHED_FP32_B], dtype=np.uint32))
    print(f"  the FP32 atomicAdd runs (published: {len(set(PUBLISHED_FP32_A + PUBLISHED_FP32_B))} "
          f"distinct patterns in 10 runs) -> verify -> {verify(fa, fb).status}")
    print("     float cannot be certified even when runs agree — agreement is luck, not evidence")

    print("\n  Scope: this certifies that both machines produced the SAME exact accumulation over")
    print("  the bound inputs (reproducibility, not fp64 accuracy), and it trusts the codes you")
    print("  typed — the mosyne-bposit demo binary is what computed them on your GPUs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
