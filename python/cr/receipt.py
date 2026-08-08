# Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
"""receipt.py — Computation Receipt (CR) v0.1: the reference implementation.

A *certificate* is a hash of an output. A **receipt** is the thing you can act on: it binds the
output digest to the model, the inputs, the computation, and — decisively — the ARITHMETIC that
produced it, in a canonical form any implementation can reproduce byte for byte.

WHY THE ARITHMETIC IS IN THE MANIFEST
    Two runs of the same model on the same inputs are only expected to agree if the accumulation
    is order-independent. Float accumulation is not, so a float receipt is a *record* but not a
    *proof*: an honest re-execution on other hardware legitimately disagrees. Exact-quire
    accumulation is order-independent by construction, so a mismatch means, unambiguously, that a
    different computation was run. The receipt states which regime it is in, and the verifier
    reports UNVERIFIABLE rather than REJECT when the arithmetic cannot support a verdict.

    That distinction is the whole point of the format, and it is why the format deliberately
    describes float computations too. A receipt format that only works on one vendor's arithmetic
    is a vendor artifact; one that describes any computation and makes visible which are
    attestable is infrastructure. Anomly's advantage is not that others cannot emit receipts — it
    is that on exact-quire arithmetic the receipts actually verify, and in silicon that costs
    nothing.

DESIGN RULES (all load-bearing for interoperability)
    * Canonical serialization: UTF-8 JSON, keys sorted, no insignificant whitespace, no NaN/Inf.
      Two conforming implementations must produce identical bytes for the same manifest.
    * Digest agility: the algorithm is a field, never assumed. v0.1 requires sha256 support.
    * Tensor digests pin dtype, shape, and BYTE ORDER (little-endian) explicitly — otherwise the
      same tensor hashes differently on different machines, which would defeat the purpose.
    * `meta` is untrusted provenance (who ran it, when, on what) and is EXCLUDED from the
      certificate. Including it would make receipts fail to verify across machines, which is
      exactly the property being sold.

Spec: docs/spec/CR-v0.1.md. Conformance vectors: conformance_vectors(). Gate: test_receipt.py.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

CR_VERSION = "0.1"
CR_SAMPLED_VERSION = "0.1.1"      # v0.1 + a sample section; spec §10
CR_CHUNKED_VERSION = "0.1.2"      # v0.1 + a chunk section (receipt chains); spec §12
SAMPLE_RULE = "sha256-ctr-reject-v1"
MEDIA_TYPE = "application/cr+json"

_HASHES = {"sha256": hashlib.sha256, "sha512": hashlib.sha512}

# Arithmetic profiles this reference implementation knows how to describe. `order_independent`
# is the field a verifier keys off: it is the claim that an honest re-execution anywhere agrees.
#
# CHANGING ANYTHING HERE CHANGES EVERY CERTIFICATE, and the independent implementations do not
# find out on their own — an emitter-only implementation (the F2 host's cr_receipt.h) keeps
# producing the old bytes and cannot detect the drift. On 2026-08-04 correcting es/frac_bits
# here silently took cross-language conformance from 6/6 to 0/6 while the design-partner brief
# still quoted 6/6. After ANY edit to PROFILES run:
#     chip-design/fpga/f2/cl_bposit/software/cross_check_receipt.py
# and regenerate the published vectors (`space-time receipt vectors --out ...`).
PROFILES: dict[str, dict[str, Any]] = {
    # CORRECTED 2026-08-04: this registry previously declared es=2, frac_bits=128. Both
    # were wrong about the format it names. b-posit16 has a THREE-bit exponent field
    # (fastquire: `e = total_e & 7`, maxpos 2^112, value grid identical to standard posit16
    # es=3) and its quire is a 2^-96 fixed-point grid (golden_vectors: quire_frac_bits 96;
    # fastquire: `total_e = 16*h + b - 1 - 96`). A registry that misdescribes the arithmetic
    # defeats the point of binding arithmetic into the receipt.
    "bposit16-quire256": {
        "accumulation": "exact",
        "order_independent": True,
        "params": {"n": 16, "es": 3, "quire_bits": 256, "frac_bits": 96},
    },
    # REGIME-EXPLICIT PROFILES (added 2026-08-04 to close a latent interop hazard).
    #
    # `bposit16-quire256` above does not state its regime bound, and the implementations
    # disagree: this SDK's codec admits |exponent| <= 112 while the canonical/forge/CUDA
    # ones bound at 48. A value in (2^48, 2^112] therefore decodes to different bits under
    # two implementations that both name that profile — an honest prover REJECTed for a
    # reason neither side can see. Naming the bound makes the disagreement a REJECT on the
    # PROFILE (a stated cause) instead of a silent divergence in the output.
    #
    # These are additive. The unqualified profile keeps its current meaning and remains the
    # default, because narrowing it is Ry's call (docs/spec/bposit-v1-spec-rev-2026-08-04.md);
    # what ships tonight is the mechanism and its gate, not a behaviour change.
    "bposit16-quire256-b48": {
        "accumulation": "exact",
        "order_independent": True,
        "params": {"n": 16, "es": 3, "quire_bits": 256, "frac_bits": 96,
                   "regime_bound": 48},
    },
    "bposit16-quire256-b112": {
        "accumulation": "exact",
        "order_independent": True,
        "params": {"n": 16, "es": 3, "quire_bits": 256, "frac_bits": 96,
                   "regime_bound": 112},
    },
    "float64": {
        "accumulation": "rounded",
        "order_independent": False,
        "params": {"bits": 64},
    },
    "float32": {
        "accumulation": "rounded",
        "order_independent": False,
        "params": {"bits": 32},
    },
}


class ReceiptError(ValueError):
    """Malformed receipt, or one this implementation cannot interpret."""


# --------------------------------------------------------------------------------------------
# canonical form
# --------------------------------------------------------------------------------------------

def canonical_bytes(obj: Any) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, tight separators, no NaN/Inf.

    `allow_nan=False` matters — Python would otherwise emit bare `NaN`, which is not JSON and
    which other implementations would reject or parse differently.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _hash(alg: str):
    try:
        return _HASHES[alg]()
    except KeyError:
        raise ReceiptError(f"unsupported digest algorithm {alg!r}") from None


def digest_bytes(data: bytes, alg: str = "sha256") -> str:
    h = _hash(alg)
    h.update(data)
    return f"{alg}:{h.hexdigest()}"


_DIGEST_HEXLEN = {"sha256": 64, "sha512": 128}


def _is_digest(s: Any, alg: str) -> bool:
    """A well-formed `<alg>:<lowercase-hex>` digest of the length `alg` produces.

    Used by wellformedness checking: a bound section that names no valid digest binds
    nothing, so the format must reject it rather than treat absence as vacuously fine.
    """
    if not isinstance(s, str) or ":" not in s:
        return False
    prefix, _, hexpart = s.partition(":")
    n = _DIGEST_HEXLEN.get(prefix)
    return (n is not None and len(hexpart) == n
            and all(c in "0123456789abcdef" for c in hexpart))


def digest_tensor(a: np.ndarray, alg: str = "sha256") -> str:
    """Digest one array, binding dtype, shape and little-endian raw bytes.

    Byte order is forced rather than inherited: a receipt produced on a big-endian machine must
    hash identically to one produced on a little-endian machine, or cross-hardware verification —
    the entire premise — silently breaks.
    """
    arr = np.ascontiguousarray(a)
    # SOUNDNESS (found 2026-08-08, exotic-dtype fuzz): the digest binds dtype as the numpy
    # type string, which is lossy for the object ('O') and void/structured ('V') kinds.
    # Every structured dtype of a given itemsize stringifies to the same "V<n>", so two
    # logically distinct structured tensors with identical raw bytes share a digest — a
    # binding collision. Object arrays are worse: their bytes are process-local pointers, so
    # such a receipt could never re-verify (and no error would say why). A CR tensor is a
    # numeric array; every other kind (biufcMmSU) is byte-reproducible and fully determined
    # by its dtype string. Refuse O/V with a clean error rather than emit an unsound digest.
    if arr.dtype.kind in "OV":
        raise ReceiptError(
            f"cannot digest tensor of dtype {arr.dtype!r}: object and structured arrays "
            "have no reproducible, collision-free byte encoding"
        )
    le = arr.astype(arr.dtype.newbyteorder("<"), copy=False)
    h = _hash(alg)
    h.update(canonical_bytes({"dtype": np.dtype(arr.dtype).str.lstrip("<>|="),
                              "shape": list(arr.shape)}))
    h.update(le.tobytes(order="C"))
    return f"{alg}:{h.hexdigest()}"


def digest_tensors(named: Mapping[str, np.ndarray], alg: str = "sha256") -> str:
    """Digest a named tensor collection (a model's weights), order-independently.

    Sorted by name so that dict iteration order — which is not part of the model — cannot change
    the digest.
    """
    h = _hash(alg)
    for name in sorted(named):
        h.update(canonical_bytes({"name": name, "digest": digest_tensor(named[name], alg)}))
    return f"{alg}:{h.hexdigest()}"


# --------------------------------------------------------------------------------------------
# the receipt
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Receipt:
    manifest: dict[str, Any]
    certificate: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(
            {"manifest": self.manifest, "certificate": self.certificate, "meta": self.meta},
            sort_keys=True, indent=indent, ensure_ascii=False, allow_nan=False,
        )

    @staticmethod
    def from_json(text: str) -> Receipt:
        try:
            obj = json.loads(text)
        # ROBUSTNESS (found 2026-08-08, from_json fuzz): json.loads raises more than
        # JSONDecodeError on a hostile document — RecursionError on deeply-nested JSON,
        # and ValueError on an over-long integer literal (Python's int-string limit). A
        # verifier ingests untrusted text and must answer, not crash: convert every
        # parse-time failure to a clean ReceiptError. (JSONDecodeError is a ValueError.)
        except (ValueError, RecursionError) as e:
            raise ReceiptError(f"not valid or well-bounded JSON: {e}") from None
        if not isinstance(obj, dict):
            raise ReceiptError("receipt must be a JSON object")
        for k in ("manifest", "certificate"):
            if k not in obj:
                raise ReceiptError(f"receipt missing required key {k!r}")
        # Types are validated HERE so hostile input produces a clean error rather than an
        # AttributeError deeper in check_wellformed. A verifier ingests untrusted documents;
        # crashing on a malformed one is its own kind of wrong answer.
        if not isinstance(obj["manifest"], dict):
            raise ReceiptError("manifest must be a JSON object")
        if not isinstance(obj["certificate"], str):
            raise ReceiptError("certificate must be a string")
        meta = obj.get("meta", {})
        if not isinstance(meta, dict):
            raise ReceiptError("meta must be a JSON object")
        return Receipt(obj["manifest"], obj["certificate"], meta)

    @property
    def order_independent(self) -> bool:
        return bool(self.manifest.get("arithmetic", {}).get("order_independent", False))


def certificate_of(manifest: Mapping[str, Any]) -> str:
    """The certificate is the digest OF THE MANIFEST, not of the output alone.

    So it binds model, input, computation, arithmetic and output together: altering any one of
    them changes the certificate. A bare output hash — what `verifiable.certificate` produces —
    cannot distinguish "same output, different model" and so cannot carry a claim.
    """
    alg = manifest.get("digest_alg", "sha256")
    return digest_bytes(canonical_bytes(manifest), alg)


# --------------------------------------------------------------------------------------------
# CRG v0 — content-addressed computation identity (spec §11, v0.2 draft)
# --------------------------------------------------------------------------------------------

CRG_VERSION = "0"


# CRG v0.2 draft op vocabulary (spec §11.1). CRG v0 left `op` opaque, which makes two
# graphs comparable only if both sides spell ops identically by luck. This registry fixes
# the names and arities we actually use, and the attribute keys each op may carry.
#
# `None` for attrs means "any attributes permitted" — used by the escape hatch below.
# Registration is deliberately small: an op vocabulary that guesses at ops nobody emits is
# a liability, because every entry is a compatibility promise.
OPS: dict[str, dict[str, Any]] = {
    # exact-quire matrix multiply: the computation this whole product line attests
    "gemm":       {"arity": 2, "attrs": {"transpose_a", "transpose_b"}},
    "dot":        {"arity": 2, "attrs": set()},
    "reduce_sum": {"arity": 1, "attrs": {"axis"}},
    "relu":       {"arity": 1, "attrs": set()},
    # ESCAPE HATCH: carry a foreign content-addressed IR (ONNX, StableHLO) as one opaque
    # node. `ir` names the format and `digest` its content hash. This keeps CRG useful for
    # computations whose graph we do not model, without pretending to understand them —
    # the digest is the identity, and CR makes no claim about the IR's internals.
    "external_ir": {"arity": 0, "attrs": {"ir", "digest"}},
}


def validate_ops(graph: Mapping[str, Any]) -> None:
    """Check a graph against the registered op vocabulary (spec §11.1).

    Separate from validate_graph() on purpose: a graph can be structurally valid and
    hashable while using ops this implementation does not know. Structure is universal;
    vocabulary is a registry that grows. Callers that need both call both.
    """
    for k, nd in enumerate(graph.get("nodes", [])):
        spec = OPS.get(nd["op"])
        if spec is None:
            raise ReceiptError(f"node {k}: op {nd['op']!r} is not in the registered "
                               "vocabulary (§11.1); use 'external_ir' to carry an "
                               "unmodelled computation by digest")
        if len(nd["args"]) != spec["arity"]:
            raise ReceiptError(f"node {k}: op {nd['op']!r} takes {spec['arity']} argument(s), "
                               f"got {len(nd['args'])}")
        if spec["attrs"] is not None:
            extra = set(nd["attrs"]) - spec["attrs"]
            if extra:
                raise ReceiptError(f"node {k}: op {nd['op']!r} does not define "
                                   f"attribute(s) {sorted(extra)}")


def external_ir_graph(ir: str, digest: str) -> dict[str, Any]:
    """A one-node graph carrying a foreign IR by content digest (ONNX, StableHLO, ...).

    The honest position: CR identifies the computation by that digest and says nothing
    about what the IR contains. It closes the "same name, different computation" gap
    without overclaiming that we understand the graph.
    """
    return {"crg": CRG_VERSION, "inputs": [],
            "nodes": [{"op": "external_ir", "attrs": {"digest": digest, "ir": ir},
                       "args": []}],
            "outputs": ["node:0"]}


def validate_graph(graph: Mapping[str, Any]) -> None:
    """Structural validation of a CRG v0 graph. Raises ReceiptError with the defect named.

    Pins the invariants two implementations must share for graph digests to be comparable:
    version, key set, topological node order, and reference discipline (`in:<name>` /
    `node:<j>` with j strictly before the referencing node).
    """
    if graph.get("crg") != CRG_VERSION:
        raise ReceiptError(f"unsupported graph version {graph.get('crg')!r}")
    if set(graph) != {"crg", "inputs", "nodes", "outputs"}:
        raise ReceiptError(f"graph keys must be exactly crg/inputs/nodes/outputs, got {sorted(graph)}")
    ins = graph["inputs"]
    if not isinstance(ins, list) or not all(isinstance(n, str) for n in ins) \
            or len(set(ins)) != len(ins):
        raise ReceiptError("graph inputs must be a list of distinct names")
    nodes = graph["nodes"]
    if not isinstance(nodes, list):
        raise ReceiptError("graph nodes must be a list")

    def check_ref(ref: str, at: int) -> None:
        if isinstance(ref, str) and ref.startswith("in:") and ref[3:] in ins:
            return
        if isinstance(ref, str) and ref.startswith("node:"):
            j = ref[5:]
            if j.isdigit() and int(j) < at:
                return
            raise ReceiptError(f"node {at}: reference {ref!r} is not to an earlier node "
                               "(nodes must be topologically ordered)")
        raise ReceiptError(f"node {at}: bad reference {ref!r}")

    for k, nd in enumerate(nodes):
        if not isinstance(nd, dict) or set(nd) != {"op", "attrs", "args"}:
            raise ReceiptError(f"node {k}: keys must be exactly op/attrs/args")
        if not isinstance(nd["op"], str) or not nd["op"]:
            raise ReceiptError(f"node {k}: op must be a non-empty string")
        if not isinstance(nd["attrs"], dict):
            raise ReceiptError(f"node {k}: attrs must be an object")
        for ref in nd["args"]:
            check_ref(ref, k)
    outs = graph["outputs"]
    if not isinstance(outs, list) or not outs:
        raise ReceiptError("graph outputs must be a non-empty list")
    for ref in outs:
        check_ref(ref, len(nodes))


def canonical_graph_digest(graph: Mapping[str, Any], alg: str = "sha256") -> str:
    """digest(canonical(graph)) after structural validation — the computation's identity."""
    validate_graph(graph)
    return digest_bytes(canonical_bytes(graph), alg)


def gemm_graph() -> dict[str, Any]:
    """The CRG v0 graph for the attested GEMM (fastquire / U200 / silicon path) — the
    reference computation this whole product line attests."""
    return {"crg": CRG_VERSION, "inputs": ["W", "H"],
            "nodes": [{"op": "gemm", "attrs": {}, "args": ["in:W", "in:H"]}],
            "outputs": ["node:0"]}


def build_receipt(
    *,
    profile: str,
    computation: str,
    computation_version: str = "1",
    model: Mapping[str, np.ndarray] | None = None,
    inputs: Mapping[str, np.ndarray] | None = None,
    output: np.ndarray,
    graph: Mapping[str, Any] | None = None,
    digest_alg: str = "sha256",
    meta: Mapping[str, Any] | None = None,
) -> Receipt:
    """Assemble a CR v0.1 receipt. `meta` is recorded but never certified.

    `graph` (optional, spec §11 / v0.2 draft): a CRG v0 computation graph; its canonical
    digest is bound into the computation section, closing the "same name, different
    computation" ambiguity. Receipts without it behave exactly as before.
    """
    if profile not in PROFILES:
        raise ReceiptError(f"unknown arithmetic profile {profile!r}")
    p = PROFILES[profile]
    comp: dict[str, Any] = {"id": computation, "version": computation_version}
    if graph is not None:
        comp["graph_digest"] = canonical_graph_digest(graph, digest_alg)
    manifest: dict[str, Any] = {
        "cr": CR_VERSION,
        "digest_alg": digest_alg,
        "arithmetic": {
            "profile": profile,
            "accumulation": p["accumulation"],
            "order_independent": p["order_independent"],
            "params": p["params"],
        },
        "computation": comp,
        "model": {"digest": digest_tensors(model or {}, digest_alg),
                  "n_tensors": len(model or {})},
        "input": {"digest": digest_tensors(inputs or {}, digest_alg),
                  "n_tensors": len(inputs or {})},
        "output": {"digest": digest_tensor(output, digest_alg),
                   "shape": list(np.asarray(output).shape)},
    }
    return Receipt(manifest, certificate_of(manifest), dict(meta or {}))


# --------------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------------

ACCEPT, REJECT, UNVERIFIABLE, MALFORMED = "ACCEPT", "REJECT", "UNVERIFIABLE", "MALFORMED"


@dataclass(frozen=True)
class Verdict:
    status: str
    reason: str

    def __bool__(self) -> bool:      # truthy only on ACCEPT
        return self.status == ACCEPT


def check_wellformed(r: Receipt) -> Verdict:
    """Structural validation, independent of re-execution."""
    m = r.manifest
    # A Receipt can be constructed directly, not only via from_json, so validate here too:
    # a verifier must answer MALFORMED for a hostile document, never raise.
    if not isinstance(m, dict) or not isinstance(r.certificate, str):
        return Verdict(MALFORMED, "manifest must be an object and certificate a string")
    for _k in ("arithmetic", "computation", "model", "input", "output"):
        if _k in m and not isinstance(m[_k], dict):
            return Verdict(MALFORMED, f"manifest.{_k} must be an object")
    if m.get("cr") not in (CR_VERSION, CR_SAMPLED_VERSION, CR_CHUNKED_VERSION):
        return Verdict(MALFORMED, f"unsupported receipt version {m.get('cr')!r}")
    for k in ("digest_alg", "arithmetic", "computation", "model", "input", "output"):
        if k not in m:
            return Verdict(MALFORMED, f"manifest missing {k!r}")
    if m["digest_alg"] not in _HASHES:
        return Verdict(MALFORMED, f"unsupported digest algorithm {m['digest_alg']!r}")
    # SOUNDNESS (found 2026-08-08, open-cr pre-release audit): the binding is only as
    # strong as the digests it names, so every bound section MUST actually carry a
    # well-formed digest. Without this a manifest could omit `output.digest` entirely
    # and still be "well-formed" — and a SAMPLED receipt, whose verify() checks only the
    # sample digest, would then ACCEPT while the FULL output was never committed to,
    # defeating the §10 pre-commitment the whole sampling argument rests on.
    _alg = m["digest_alg"]
    for _sec in ("model", "input", "output"):
        if not _is_digest(m[_sec].get("digest"), _alg):
            return Verdict(MALFORMED,
                           f"{_sec}.digest is missing or not a well-formed {_alg} digest")
    if not (isinstance(m["output"].get("shape"), list) and m["output"]["shape"]):
        return Verdict(MALFORMED, "output.shape must be a non-empty list")
    # SOUNDNESS (found 2026-08-04): profile and order_independent are separate fields the
    # prover writes, and order_independent is what gates ACCEPT. Without this check a
    # receipt could name a ROUNDED profile (float64) while asserting order_independent:
    # true, and verify() would return ACCEPT — forging exactly the verdict §6 exists to
    # withhold for order-dependent arithmetic. For a REGISTERED profile the declared
    # fields must match the registry; the registry, not the prover, is authoritative.
    a = m["arithmetic"]
    if not isinstance(a, dict) or "profile" not in a:
        return Verdict(MALFORMED, "arithmetic section missing profile")
    reg = PROFILES.get(a["profile"])
    if reg is not None:
        for field in ("accumulation", "order_independent", "params"):
            if a.get(field) != reg[field]:
                return Verdict(MALFORMED,
                               f"arithmetic.{field} {a.get(field)!r} contradicts the registered "
                               f"profile {a['profile']!r} ({reg[field]!r}); the registry is "
                               "authoritative, not the receipt")
    # Version and section travel together or not at all: each version carries EXACTLY its
    # own optional section and no other. A v0.1 receipt must not smuggle a sample or chunk
    # section; and — HARDENING (found 2026-08-08, cross-section fuzz) — a sampled (0.1.1)
    # receipt must not carry a chunk section, nor a chunked (0.1.2) receipt a sample
    # section. The verdict paths already resist a cross-section (verify rebinds via the
    # certificate; verify_sampled/verify_chain gate on `cr`), so this is defence in depth
    # against downgrade/confusion, and it removes the smuggle surface outright.
    if m["cr"] == CR_VERSION and "sample" in m:
        return Verdict(MALFORMED, "v0.1 receipt carries a sample section; sampled receipts are cr 0.1.1")
    if m["cr"] == CR_VERSION and "chunk" in m:
        return Verdict(MALFORMED, "v0.1 receipt carries a chunk section; chunked receipts are cr 0.1.2")
    if m["cr"] == CR_SAMPLED_VERSION and "chunk" in m:
        return Verdict(MALFORMED, "sampled (0.1.1) receipt carries a chunk section; chunked receipts are cr 0.1.2")
    if m["cr"] == CR_CHUNKED_VERSION and "sample" in m:
        return Verdict(MALFORMED, "chunked (0.1.2) receipt carries a sample section; sampled receipts are cr 0.1.1")
    if m["cr"] == CR_CHUNKED_VERSION:
        c = m.get("chunk")
        if not isinstance(c, dict):
            return Verdict(MALFORMED, "cr 0.1.2 receipt missing chunk section")
        if not (isinstance(c.get("index"), int) and c["index"] >= 0):
            return Verdict(MALFORMED, f"chunk index invalid: {c.get('index')!r}")
        if not isinstance(c.get("prev_certificate"), str):
            return Verdict(MALFORMED, "chunk section missing prev_certificate")
        if c["index"] == 0 and c["prev_certificate"] != "":
            return Verdict(MALFORMED, "chunk 0 must have an empty prev_certificate")
        if c["index"] > 0 and not c["prev_certificate"]:
            return Verdict(MALFORMED, f"chunk {c['index']} has no prev_certificate")
        if not isinstance(c.get("closing"), bool):
            return Verdict(MALFORMED, "chunk section missing closing flag")
        if c["closing"] and not (isinstance(c.get("n_chunks"), int)
                                 and c["n_chunks"] == c["index"]
                                 and isinstance(c.get("chain_digest"), str)):
            return Verdict(MALFORMED, "closing receipt needs n_chunks == index and a chain_digest")
    if m["cr"] == CR_SAMPLED_VERSION:
        s = m.get("sample")
        if not isinstance(s, dict):
            return Verdict(MALFORMED, "cr 0.1.1 receipt missing sample section")
        if s.get("rule") != SAMPLE_RULE:
            return Verdict(MALFORMED, f"unknown sample rule {s.get('rule')!r}")
        n, size = s.get("n_units"), s.get("size")
        if not (isinstance(n, int) and isinstance(size, int) and 1 <= size <= n):
            return Verdict(MALFORMED, f"sample size/n_units invalid: {size!r}/{n!r}")
        ch = s.get("challenge")
        if not isinstance(ch, str) or not isinstance(s.get("digest"), str):
            return Verdict(MALFORMED, "sample section missing challenge or digest")
        # ROBUSTNESS (found 2026-08-08, sampled-verify fuzz): the challenge must be valid
        # hex. sample_indices_of feeds it to bytes.fromhex(), which raises ValueError on
        # non-hex or odd-length input — so a receipt that passes every other structural
        # check would crash the verifier here instead of getting a MALFORMED verdict. The
        # emitter always writes canonical even-length hex; require it (empty is allowed).
        if len(ch) % 2 or any(c not in "0123456789abcdefABCDEF" for c in ch):
            return Verdict(MALFORMED, f"sample.challenge is not valid hex: {ch!r}")
        # SOUNDNESS (found 2026-08-04): n_units MUST equal the output's leading
        # dimension. Without this a prover can declare a sampling space smaller
        # than the real output — the verifier then draws indices only from that
        # prefix, every unit beyond it goes unchecked, and the receipt is still
        # self-consistent and ACCEPTs. Binding the count is what makes coverage
        # mean what it says.
        oshape = m["output"].get("shape")
        if not (isinstance(oshape, list) and oshape and oshape[0] == n):
            return Verdict(MALFORMED,
                           f"sample.n_units {n!r} does not match output leading dimension "
                           f"{(oshape[0] if isinstance(oshape, list) and oshape else None)!r} "
                           "(a shrunken sampling space would leave units unattested)")
    if certificate_of(m) != r.certificate:
        return Verdict(MALFORMED, "certificate does not match manifest (tampered or miscomputed)")
    return Verdict(ACCEPT, "well-formed")


def _assessable(r: Receipt) -> Verdict | None:
    """UNVERIFIABLE if this verifier cannot assess the receipt's order-independence claim.

    §7's profile registry is deliberately open, so a verifier will meet profiles it does
    not implement. `order_independent` is written by the prover; for an unregistered
    profile there is nothing to check it against, and returning ACCEPT would be trusting
    the claim rather than verifying it. Saying UNVERIFIABLE is the honest answer — the same
    answer the format already gives for float.
    """
    prof = r.manifest.get("arithmetic", {}).get("profile")
    if prof not in PROFILES:
        return Verdict(UNVERIFIABLE,
                       f"arithmetic profile {prof!r} is not in this verifier's registry: its "
                       "order-independence claim cannot be assessed, so no verdict is possible")
    return None


def verify(r: Receipt, recomputed: Receipt | None) -> Verdict:
    """Compare a claimed receipt against an independent re-execution.

    The four outcomes are deliberately distinct. UNVERIFIABLE is not a soft REJECT: it is the
    verifier stating that the arithmetic named in the receipt cannot support a verdict at all,
    which is the honest answer for a float receipt and the thing that makes the exact-arithmetic
    case worth paying for.
    """
    wf = check_wellformed(r)
    if not wf:
        return wf
    if recomputed is None:
        return Verdict(UNVERIFIABLE, "no re-execution supplied")
    wf2 = check_wellformed(recomputed)
    if not wf2:
        return Verdict(MALFORMED, f"re-execution receipt malformed: {wf2.reason}")

    a, b = r.manifest, recomputed.manifest
    for k in ("model", "input", "computation"):
        if a[k] != b[k]:
            return Verdict(REJECT, f"re-execution used a different {k}")
    if a["arithmetic"]["profile"] != b["arithmetic"]["profile"]:
        return Verdict(REJECT, "re-execution used a different arithmetic profile")

    unknown = _assessable(r)
    if unknown is not None:
        return unknown
    if not r.order_independent:
        same = a["output"]["digest"] == b["output"]["digest"]
        return Verdict(
            UNVERIFIABLE,
            "arithmetic is order-dependent: outputs "
            + ("happened to match, but agreement is not evidence" if same
               else "differ, which is expected and proves nothing"),
        )

    if a["output"]["digest"] != b["output"]["digest"]:
        return Verdict(REJECT, "output digest differs under order-independent arithmetic")
    if r.certificate != recomputed.certificate:
        return Verdict(REJECT, "certificate differs")
    return Verdict(ACCEPT, "re-execution reproduced the certificate")


# --------------------------------------------------------------------------------------------
# sampled receipts (CR v0.1.1) — spec §10
# --------------------------------------------------------------------------------------------
# For computations too large to re-execute whole, the verifier re-executes only a sample of
# output units (rows along axis 0). The sample indices are NOT chosen by the prover: they are a
# deterministic function of the certificate-minus-sample (which already binds the FULL output
# digest) and an optional external challenge. So the prover commits to every output unit before
# learning which units will be checked.
#
# What this does and does not buy — stated here because it is a security boundary, not a detail:
#   * A NON-ADAPTIVE tamper of k of the N units escapes detection with probability
#     C(N-k, s)/C(N, s)  <=  (1 - s/N)^k  (s = sample size).
#   * An ADAPTIVE prover can GRIND: recompute the tampered output, derive the indices, and
#     retry until the tampered units fall outside the sample (~N/s attempts for k=1). The
#     defence is the `challenge`: when it is supplied by the verifier or a public beacon AFTER
#     the prover has fixed the output, grinding requires predicting the challenge. A sampled
#     receipt with an empty challenge is therefore only evidence against non-adaptive faults
#     (bit flips, wrong weights, drift) — not against an adversarial prover.
#   * Unsampled units are UNATTESTED. ACCEPT means "the sampled slice reproduced bit-exactly
#     and the full output digest is committed", never "every unit was checked".


def sampled_indices(n_units: int, size: int, seed: bytes) -> np.ndarray:
    """Deterministic sample-without-replacement per SAMPLE_RULE.

    Draw 8-byte big-endian integers from sha256(seed || counter_be8), counter 0,1,2,...;
    reject values >= floor(2^64 / n) * n (modulo-bias rejection); map v % n; skip repeats;
    stop at `size` distinct indices; return them sorted ascending. Any conforming
    implementation reproduces the same indices from the same seed.
    """
    if not 1 <= size <= n_units:
        raise ReceiptError(f"sample size {size} out of range for {n_units} units")
    limit = (2 ** 64 // n_units) * n_units
    out: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(out) < size:
        v = int.from_bytes(hashlib.sha256(seed + counter.to_bytes(8, "big")).digest()[:8], "big")
        counter += 1
        if v >= limit:
            continue
        i = v % n_units
        if i not in seen:
            seen.add(i)
            out.append(i)
    return np.array(sorted(out), dtype=np.int64)


def _sample_seed(manifest_without_sample: Mapping[str, Any], challenge: bytes) -> bytes:
    return hashlib.sha256(canonical_bytes(manifest_without_sample) + challenge).digest()


def build_sampled_receipt(
    *,
    profile: str,
    computation: str,
    computation_version: str = "1",
    model: Mapping[str, np.ndarray] | None = None,
    inputs: Mapping[str, np.ndarray] | None = None,
    output: np.ndarray,
    sample_size: int,
    challenge: bytes = b"",
    graph: Mapping[str, Any] | None = None,
    digest_alg: str = "sha256",
    meta: Mapping[str, Any] | None = None,
) -> Receipt:
    """A CR v0.1.1 receipt whose verification cost is `sample_size/n_units` of re-execution.

    The full output digest is still bound; the sample indices derive from everything ELSE in
    the manifest plus the challenge, so they cannot be chosen after the fact (see the grinding
    caveat above for what "cannot" means without an external challenge).
    """
    out_arr = np.asarray(output)
    if out_arr.ndim < 1 or out_arr.shape[0] < 1:
        raise ReceiptError("sampled receipts need an output with at least one unit on axis 0")
    full = build_receipt(profile=profile, computation=computation,
                         computation_version=computation_version, model=model, inputs=inputs,
                         output=out_arr, graph=graph, digest_alg=digest_alg)
    base = dict(full.manifest)
    base["cr"] = CR_SAMPLED_VERSION
    idx = sampled_indices(out_arr.shape[0], sample_size, _sample_seed(base, challenge))
    manifest = dict(base)
    manifest["sample"] = {
        "rule": SAMPLE_RULE,
        "n_units": int(out_arr.shape[0]),
        "size": int(sample_size),
        "challenge": challenge.hex(),
        "digest": digest_tensor(out_arr[idx], digest_alg),
    }
    return Receipt(manifest, certificate_of(manifest), dict(meta or {}))


def sample_indices_of(r: Receipt) -> np.ndarray:
    """The indices a verifier must re-execute — recomputed, never trusted from the prover."""
    wf = check_wellformed(r)
    if not wf:
        raise ReceiptError(f"not a well-formed sampled receipt: {wf.reason}")
    if r.manifest.get("cr") != CR_SAMPLED_VERSION:
        raise ReceiptError("not a sampled receipt (cr != 0.1.1)")
    s = r.manifest["sample"]
    base = {k: v for k, v in r.manifest.items() if k != "sample"}
    seed = _sample_seed(base, bytes.fromhex(s["challenge"]))
    return sampled_indices(s["n_units"], s["size"], seed)


def verify_sampled(r: Receipt, reexecuted_rows: np.ndarray) -> Verdict:
    """Verdict on a sampled receipt given the verifier's OWN re-execution of the sampled units.

    `reexecuted_rows` must be output[indices] as re-executed by the verifier, same dtype and
    per-row shape as the claimed output, indices from `sample_indices_of(r)`.
    """
    wf = check_wellformed(r)
    if not wf:
        return wf
    if r.manifest.get("cr") != CR_SAMPLED_VERSION:
        return Verdict(MALFORMED, "not a sampled receipt (cr != 0.1.1)")
    unknown = _assessable(r)
    if unknown is not None:
        return unknown
    if not r.order_independent:
        return Verdict(UNVERIFIABLE,
                       "arithmetic is order-dependent: a sampled re-execution proves nothing")
    s = r.manifest["sample"]
    rows = np.asarray(reexecuted_rows)
    if rows.shape[0] != s["size"]:
        return Verdict(MALFORMED,
                       f"re-executed rows count {rows.shape[0]} != sample size {s['size']}")
    if digest_tensor(rows, r.manifest["digest_alg"]) != s["digest"]:
        return Verdict(REJECT, "sampled slice does not reproduce")
    n, size = s["n_units"], s["size"]
    return Verdict(ACCEPT,
                   f"sampled slice reproduced bit-exactly ({size}/{n} units, "
                   f"{100.0 * size / n:.2f}% coverage; a non-adaptive tamper of k units "
                   f"escapes with probability <= (1-{size}/{n})^k; unsampled units are "
                   "unattested)")


# --------------------------------------------------------------------------------------------
# chunked receipts / receipt chains (CR v0.1.2) — spec §12
# --------------------------------------------------------------------------------------------
# For computations that stream (token-by-token generation, long reductions), a receipt CHAIN
# attests incrementally: chunk k's manifest carries chunk k-1's certificate, so the chain
# fixes ordering and membership as it grows, and a CLOSING receipt binds the chunk count and
# the digest of every certificate in the chain — without it, truncation is undetectable.
#
# What a chain proves: each chunk's binding (as a full CR receipt) plus the order they were
# committed in. What it does NOT prove: wall-clock timing (a prover can compute everything
# first and emit a chain after), and — for an unclosed chain — totality: an honest-looking
# prefix says nothing about how many chunks were supposed to follow.


def build_chunk_receipt(
    *,
    profile: str,
    computation: str,
    computation_version: str = "1",
    model: Mapping[str, np.ndarray] | None = None,
    inputs: Mapping[str, np.ndarray] | None = None,
    output: np.ndarray,
    chunk_index: int,
    prev_certificate: str = "",
    graph: Mapping[str, Any] | None = None,
    digest_alg: str = "sha256",
    meta: Mapping[str, Any] | None = None,
) -> Receipt:
    """One link of a receipt chain: a full CR receipt over THIS chunk's inputs/output, plus
    the chunk section that chains it to its predecessor."""
    if chunk_index == 0 and prev_certificate:
        raise ReceiptError("chunk 0 takes no prev_certificate")
    if chunk_index > 0 and not prev_certificate:
        raise ReceiptError(f"chunk {chunk_index} needs prev_certificate")
    full = build_receipt(profile=profile, computation=computation,
                         computation_version=computation_version, model=model, inputs=inputs,
                         output=output, graph=graph, digest_alg=digest_alg)
    manifest = dict(full.manifest)
    manifest["cr"] = CR_CHUNKED_VERSION
    manifest["chunk"] = {"index": int(chunk_index), "prev_certificate": prev_certificate,
                         "closing": False}
    return Receipt(manifest, certificate_of(manifest), dict(meta or {}))


def chain_digest_of(chain: Sequence[Receipt], alg: str = "sha256") -> str:
    """Digest over every certificate in order — what the closing receipt commits to."""
    h = _hash(alg)
    for r in chain:
        h.update(canonical_bytes({"certificate": r.certificate}))
    return f"{alg}:{h.hexdigest()}"


def build_closing_receipt(
    *,
    chain: Sequence[Receipt],
    output_total: np.ndarray,
    digest_alg: str = "sha256",
    meta: Mapping[str, Any] | None = None,
) -> Receipt:
    """The chain's terminator: binds the chunk count, the digest of every certificate, and
    the digest of the TOTAL output. Without it a chain can be silently truncated."""
    if not chain:
        raise ReceiptError("cannot close an empty chain")
    head = chain[0].manifest
    profile = head["arithmetic"]["profile"]
    comp = head["computation"]
    full = build_receipt(profile=profile, computation=comp["id"],
                         computation_version=comp["version"], model=None, inputs=None,
                         output=output_total, digest_alg=digest_alg)
    manifest = dict(full.manifest)
    manifest["cr"] = CR_CHUNKED_VERSION
    manifest["computation"] = dict(comp)          # carry graph_digest through if present
    manifest["model"] = dict(head["model"])       # the chain's model, bound once more
    manifest["chunk"] = {"index": len(chain), "prev_certificate": chain[-1].certificate,
                         "closing": True, "n_chunks": len(chain),
                         "chain_digest": chain_digest_of(chain, digest_alg)}
    return Receipt(manifest, certificate_of(manifest), dict(meta or {}))


def _is_closing_receipt(r: Receipt) -> bool:
    """True only for a structurally-plausible closing receipt.

    ROBUSTNESS (found 2026-08-08, chain-verify fuzz): verify_chain must peek at the last
    receipt to decide whether it is a closing terminator BEFORE it can run check_wellformed
    on the body — and a hostile receipt's `manifest` or `chunk` may be any JSON value, not a
    dict. Calling `.get()` on a non-dict raised AttributeError (reachable straight through
    from_json, which only guarantees manifest is a dict), crashing the verifier instead of
    returning MALFORMED. Answer False for anything that is not a dict-shaped closing chunk;
    the per-chunk check_wellformed below then produces the proper MALFORMED verdict.
    """
    m = getattr(r, "manifest", None)
    if not isinstance(m, dict):
        return False
    c = m.get("chunk")
    return isinstance(c, dict) and c.get("closing") is True


def verify_chain(claimed: Sequence[Receipt], recomputed: Sequence[Receipt] | None,
                 *, allow_open: bool = False) -> Verdict:
    """Verify a receipt chain against the verifier's own re-executed chain.

    `recomputed` is built by the verifier with the same builders from its own re-executions
    (mirroring `verify`). The last claimed receipt may be a closing receipt; with
    `allow_open=True` a chain without one verifies as an explicit OPEN PREFIX — ACCEPT, but
    the reason says totality is unattested.
    """
    if not claimed:
        return Verdict(MALFORMED, "empty chain")
    body = list(claimed)
    closing = None
    if _is_closing_receipt(body[-1]):
        closing = body[-1]
        body = body[:-1]
    if not body:
        return Verdict(MALFORMED, "chain has a closing receipt but no chunks")
    for k, r in enumerate(body):
        wf = check_wellformed(r)
        if not wf:
            return Verdict(MALFORMED, f"chunk {k}: {wf.reason}")
        c = r.manifest.get("chunk")
        # `c` may be a non-dict here for a valid non-chunk receipt (e.g. a 0.1.1 sampled
        # receipt that carries a rogue `chunk` field check_wellformed does not police);
        # isinstance guards the .get so a smuggled non-dict chunk yields MALFORMED, not a raise.
        if not isinstance(c, dict) or c.get("closing"):
            return Verdict(MALFORMED, f"chunk {k}: not a chunk receipt")
        if c["index"] != k:
            return Verdict(REJECT, f"chunk order broken: position {k} carries index {c['index']}")
        want_prev = "" if k == 0 else body[k - 1].certificate
        if c["prev_certificate"] != want_prev:
            return Verdict(REJECT, f"chunk {k}: prev_certificate does not match chunk {k - 1} "
                                   "(reordered, dropped or foreign chunk)")
    if closing is not None:
        wf = check_wellformed(closing)
        if not wf:
            return Verdict(MALFORMED, f"closing receipt: {wf.reason}")
        cc = closing.manifest["chunk"]
        if cc["n_chunks"] != len(body):
            return Verdict(REJECT, f"closing receipt claims {cc['n_chunks']} chunks, chain has {len(body)}")
        if cc["prev_certificate"] != body[-1].certificate:
            return Verdict(REJECT, "closing receipt does not chain from the last chunk")
        if cc["chain_digest"] != chain_digest_of(body, closing.manifest["digest_alg"]):
            return Verdict(REJECT, "closing chain_digest does not match the chain")
    elif not allow_open:
        return Verdict(REJECT, "chain is not closed: truncation would be undetectable "
                               "(pass allow_open=True to verify an explicit prefix)")
    if recomputed is None:
        return Verdict(UNVERIFIABLE, "no re-executed chain supplied")
    if len(recomputed) != len(body):
        return Verdict(MALFORMED, f"re-executed chain has {len(recomputed)} chunks, claimed body has {len(body)}")
    for k, (a, b) in enumerate(zip(body, recomputed)):
        v = verify(a, b)
        if v.status != ACCEPT:
            return Verdict(v.status, f"chunk {k}: {v.reason}")
    state = (f"closed chain of {len(body)} chunks, totality bound by closing receipt"
             if closing is not None else
             f"OPEN PREFIX of {len(body)} chunks: ordering and content verified, "
             "totality UNATTESTED (no closing receipt)")
    return Verdict(ACCEPT, f"every chunk re-executed to its certificate; {state}")


# --------------------------------------------------------------------------------------------
# conformance
# --------------------------------------------------------------------------------------------

def conformance_vectors() -> list[dict[str, Any]]:
    """Fixed vectors a third-party implementation must reproduce exactly to claim CR v0.1.

    These pin the canonicalisation, not the arithmetic: any language, any platform, same bytes.
    This is what makes CR a standard someone else can implement rather than a description of
    what our code happens to do.
    """
    out: list[dict[str, Any]] = []

    out.append({"name": "canonical/empty-object", "input": {},
                "canonical": canonical_bytes({}).decode(), "digest": digest_bytes(canonical_bytes({}))})
    nested = {"b": 1, "a": {"d": [1, 2, 3], "c": "x"}}
    out.append({"name": "canonical/key-order", "input": nested,
                "canonical": canonical_bytes(nested).decode(),
                "digest": digest_bytes(canonical_bytes(nested))})
    uni = {"k": "café-é中"}
    out.append({"name": "canonical/utf8", "input": uni,
                "canonical": canonical_bytes(uni).decode(),
                "digest": digest_bytes(canonical_bytes(uni))})

    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    out.append({"name": "tensor/float64-2x3", "digest": digest_tensor(a)})
    b = np.arange(4, dtype=np.int32)
    out.append({"name": "tensor/int32-4", "digest": digest_tensor(b)})
    out.append({"name": "tensors/named-order-independent",
                "digest": digest_tensors({"w2": b, "w1": a})})

    r = build_receipt(profile="bposit16-quire256", computation="conformance.identity",
                      model={"w": a}, inputs={"x": b}, output=a)
    out.append({"name": "receipt/exact-profile", "certificate": r.certificate,
                "manifest_canonical": canonical_bytes(r.manifest).decode()})

    # ---- the other two receipt kinds, so a third party can certify all of them ----
    y = np.arange(64, dtype=np.float64).reshape(16, 4)
    sr = build_sampled_receipt(profile="bposit16-quire256", computation="conformance.sampled",
                               inputs={"x": b}, output=y, sample_size=4,
                               challenge=bytes.fromhex("beef"))
    out.append({"name": "receipt/sampled",
                "certificate": sr.certificate,
                "manifest_canonical": canonical_bytes(sr.manifest).decode(),
                # the PRF result is pinned explicitly: a matching sample digest does NOT
                # prove an implementation derived the same indices (§10), so the indices
                # are published as their own conformance value.
                "sample_indices": [int(i) for i in sample_indices_of(sr)]})

    kr = build_chunk_receipt(profile="bposit16-quire256", computation="conformance.chunked",
                             inputs={"x": b}, output=a, chunk_index=0)
    out.append({"name": "receipt/chunk-0", "certificate": kr.certificate,
                "manifest_canonical": canonical_bytes(kr.manifest).decode(),
                "chain_digest": chain_digest_of([kr])})

    # ---- NEGATIVE vectors ----------------------------------------------------------
    # Both soundness bugs found on 2026-08-04 were of the form "accepts what it must
    # reject", and a suite of positive vectors alone cannot catch that class: an
    # implementation that accepts everything passes every positive vector. These pin the
    # required REFUSALS. `expect_verdict` is the value a conforming verifier must return.
    _bad_prof = dict(r.manifest)
    _bad_prof["arithmetic"] = dict(_bad_prof["arithmetic"])
    _bad_prof["arithmetic"]["profile"] = "float64"        # rounded profile...
    _bad_prof["arithmetic"]["order_independent"] = True   # ...contradicted by the flag
    out.append({"name": "refuse/profile-contradicts-registry",
                "manifest_canonical": canonical_bytes(_bad_prof).decode(),
                "certificate": certificate_of(_bad_prof),
                "expect_verdict": MALFORMED,
                "why": "arithmetic fields must match the registered profile (§6.1 rule 1); "
                       "accepting this forges the ACCEPT reserved for exact arithmetic"})

    _shrunk = dict(sr.manifest)
    _shrunk["sample"] = dict(_shrunk["sample"])
    _shrunk["sample"]["n_units"] = 4                      # the real output has 16 units
    out.append({"name": "refuse/sample-space-shrunk",
                "manifest_canonical": canonical_bytes(_shrunk).decode(),
                "certificate": certificate_of(_shrunk),
                "expect_verdict": MALFORMED,
                "why": "sample.n_units must equal output.shape[0] (§10); accepting this "
                       "reports full coverage while most units go unattested"})

    _unknown = dict(r.manifest)
    _unknown["arithmetic"] = dict(_unknown["arithmetic"])
    _unknown["arithmetic"]["profile"] = "vendor-unregistered-v0"
    out.append({"name": "refuse/unknown-profile-not-accepted",
                "manifest_canonical": canonical_bytes(_unknown).decode(),
                "certificate": certificate_of(_unknown),
                "expect_verdict": UNVERIFIABLE,
                "why": "a verifier cannot assess an order-independence claim for a profile "
                       "it does not implement (§6.1 rule 2); ACCEPT would trust the prover"})

    _nodig = dict(sr.manifest)                            # a SAMPLED receipt...
    _nodig["output"] = {"shape": _nodig["output"]["shape"]}  # ...with no output digest
    out.append({"name": "refuse/output-digest-missing",
                "manifest_canonical": canonical_bytes(_nodig).decode(),
                "certificate": certificate_of(_nodig),
                "expect_verdict": MALFORMED,
                "why": "every bound section must carry a well-formed digest; a sampled "
                       "receipt with no full-output digest would ACCEPT its sampled slice "
                       "while never committing to the unsampled units, defeating §10"})
    return out


def selftest() -> dict[str, Any]:
    """Exercise the four verdicts and the binding property. Mirrored by test_receipt.py."""
    rng = np.random.default_rng(7)
    W = {"w1": rng.standard_normal((4, 4)), "w2": rng.standard_normal((4, 4))}
    X = {"x": rng.standard_normal((3, 4))}
    y = X["x"] @ W["w1"] @ W["w2"]

    mk = lambda prof, out, model=W: build_receipt(
        profile=prof, computation="selftest.chain", model=model, inputs=X, output=out)

    exact = mk("bposit16-quire256", y)
    same = mk("bposit16-quire256", y)
    perturbed = mk("bposit16-quire256", y + 1e-15)
    other_model = mk("bposit16-quire256", y, model={"w1": W["w1"], "w2": W["w2"] * 1.0000001})
    floaty = mk("float32", y)
    floaty2 = mk("float32", y + 1e-15)

    tampered = Receipt({**exact.manifest, "computation": {"id": "other", "version": "1"}},
                       exact.certificate)

    return {
        "accept": verify(exact, same),
        "reject_output": verify(exact, perturbed),
        "reject_model": verify(exact, other_model),
        "unverifiable_float_match": verify(floaty, floaty),
        "unverifiable_float_differ": verify(floaty, floaty2),
        "malformed_tampered": check_wellformed(tampered),
        "roundtrip_ok": Receipt.from_json(exact.to_json()).certificate == exact.certificate,
        "meta_not_certified": build_receipt(
            profile="bposit16-quire256", computation="selftest.chain", model=W, inputs=X,
            output=y, meta={"host": "somewhere", "when": "then"}).certificate == exact.certificate,
    }


if __name__ == "__main__":  # pragma: no cover
    for k, v in selftest().items():
        print(f"{k:26s} {v}")
    print(f"\n{len(conformance_vectors())} conformance vectors")
