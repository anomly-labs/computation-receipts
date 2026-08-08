#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""run_checks.py — verify the whole CR repository in one command.

For a third party (or a CI job) deciding whether to trust this implementation: this runs
every self-check the repo ships and prints one PASS/FAIL line each, exiting non-zero if
any fails. No account, no network required for the core checks (the beacon audit is
network-dependent and is therefore not part of this gate).

  1. conformance   — the reference implementation reproduces the published vectors,
                     refusals included (the load-bearing check).
  2. protocol      — the §10 challenge-protocol logic (arithmetic-free).
  3. service       — the local verifier service accepts honest / refuses tampered.
  4. c-standalone  — the dependency-free C emitter compiles and links with only libc.
  5. runner        — the third-party conformance runner grades the reference against
                     itself (the tool a second implementer uses to self-certify, §9).

Run:  python3 run_checks.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _run_py(argv, needle):
    r = subprocess.run([sys.executable, *argv], cwd=HERE, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()
    ok = r.returncode == 0 and any(needle in ln for ln in tail)
    return ok, (tail[-1] if tail else "(no output)")


def _c_standalone():
    """The C header must compile+link with nothing but libc — the 'dependency-free' claim."""
    src = ('#include "cr_receipt.h"\n'
           'int main(void){char b[4096],c[72];'
           'cr_build_manifest(b,sizeof b,"x","1",'
           '"sha256:0000000000000000000000000000000000000000000000000000000000000000",0,'
           '"sha256:0000000000000000000000000000000000000000000000000000000000000000",1,c);'
           'return c[0]?0:0;}\n')
    exe = Path(tempfile.gettempdir()) / "cr_run_checks_standalone"
    r = subprocess.run(["cc", "-std=c11", "-I", str(ROOT / "c"), "-x", "c", "-",
                        "-o", str(exe)], input=src, capture_output=True, text=True)
    if r.returncode != 0:
        return False, "C did not compile/link standalone: " + r.stderr.strip()[:120]
    return True, "compiles and links with libc only"


def main() -> int:
    checks = [
        ("conformance", lambda: _run_py(["check_conformance.py"], "PASS")),
        ("protocol", lambda: _run_py(["-m", "cr.protocol"], "PASS")),
        ("service", lambda: _run_py(["test_verify_service.py"], "PASS")),
        ("c-standalone", _c_standalone),
        ("runner", lambda: _run_py(["conformance_runner.py", "--demo"], "PASS:")),
    ]
    print("# Computation Receipts — full repository self-check\n")
    all_ok = True
    for name, fn in checks:
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"raised: {e}"
        all_ok = all_ok and ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<13} {detail}")
    print()
    if all_ok:
        print("ALL CHECKS PASSED — this implementation reproduces the published vectors, "
              "enforces the required refusals, runs the protocol, and ships a "
              "standalone-compilable C emitter.")
        return 0
    print("SOME CHECKS FAILED — do not trust this build until reconciled.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
