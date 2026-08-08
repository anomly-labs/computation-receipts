// Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
//
// cr.mjs — a fuller from-scratch CR implementation in JavaScript (Node stdlib only).
//
// Extends cr_canonical.mjs from the canonicalisation layer to the RECEIPT layer: it builds the
// manifests, certificates, sampled-index PRF (§10) and chain digest (§12) for the receipt/*
// and chain/* vectors, all from the spec text — no CR code, no reading of the reference
// vectors, and (deliberately) no b-posit arithmetic: a receipt's *values* need the number
// system, but a receipt's *bytes* need only canonicalisation + sha256 digests + the PRF.
//
//   node examples/js/cr.mjs | python3 python/conformance_runner.py /dev/stdin
//
// The refuse/* vectors are intentionally NOT emitted here: they assert a VERDICT a verifier
// must return, which needs the §6/§6.1 wellformedness logic, not an emitter. This file is the
// emitter half of a second implementation; verifying is left as the (well-specified) next step.

import { createHash } from 'node:crypto';

// ---- §2 canonical form -------------------------------------------------------------------
function escStr(s) {
  let out = '"';
  for (const ch of s) {
    const cp = ch.codePointAt(0);
    if (cp === 0x22) out += '\\"';
    else if (cp === 0x5c) out += '\\\\';
    else if (cp === 0x08) out += '\\b';
    else if (cp === 0x09) out += '\\t';
    else if (cp === 0x0a) out += '\\n';
    else if (cp === 0x0c) out += '\\f';
    else if (cp === 0x0d) out += '\\r';
    else if (cp < 0x20) out += '\\u' + cp.toString(16).padStart(4, '0');
    else out += ch;
  }
  return out + '"';
}
function cmpCp(a, b) {
  const A = [...a], B = [...b];
  for (let i = 0; i < Math.min(A.length, B.length); i++) {
    const d = A[i].codePointAt(0) - B[i].codePointAt(0);
    if (d !== 0) return d;
  }
  return A.length - B.length;
}
function canon(v) {
  if (v === null) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) throw new Error('§2 rule 4: NaN/Infinity');
    if (!Number.isInteger(v)) throw new Error('§2 rule 7: integers only');
    return String(v);
  }
  if (typeof v === 'string') return escStr(v);
  if (Array.isArray(v)) return '[' + v.map(canon).join(',') + ']';
  if (typeof v === 'object') {
    return '{' + Object.keys(v).sort(cmpCp).map(k => escStr(k) + ':' + canon(v[k])).join(',') + '}';
  }
  throw new Error('unserialisable');
}
const sha256 = buf => createHash('sha256').update(buf).digest();
const digestBytes = str => 'sha256:' + createHash('sha256').update(Buffer.from(str, 'utf8')).digest('hex');
const EMPTY_COLLECTION = 'sha256:' + createHash('sha256').digest('hex');   // §3.2 empty-hash-state

// ---- §3 digests --------------------------------------------------------------------------
function tensorDigest(code, shape, raw) {
  const h = createHash('sha256');
  h.update(Buffer.from(canon({ dtype: code, shape }), 'utf8'));
  h.update(raw);
  return 'sha256:' + h.digest('hex');
}
function namedDigest(named) {
  const h = createHash('sha256');
  for (const name of Object.keys(named).sort(cmpCp)) {
    h.update(Buffer.from(canon({ name, digest: named[name] }), 'utf8'));
  }
  return 'sha256:' + h.digest('hex');
}

// ---- §5 certificate ----------------------------------------------------------------------
const certificateOf = manifest => digestBytes(canon(manifest));

// ---- §7 the one registered profile used by the vectors -----------------------------------
const ARITH = {
  accumulation: 'exact', order_independent: true,
  params: { es: 3, frac_bits: 96, n: 16, quire_bits: 256 },
  profile: 'bposit16-quire256',
};

// ---- §10 sampled-index PRF (sha256-ctr-reject-v1) -----------------------------------------
function sampledIndices(base, challengeHex, nUnits, size) {
  if (!(1 <= size && size <= nUnits)) throw new Error('size out of range');
  const challengeBytes = Buffer.from(challengeHex, 'hex');            // S4: hex-decode, not UTF-8
  const seed = sha256(Buffer.concat([Buffer.from(canon(base), 'utf8'), challengeBytes]));
  const limit = (2n ** 64n / BigInt(nUnits)) * BigInt(nUnits);        // ⌊2^64/n⌋·n
  const out = [], seen = new Set();
  for (let ctr = 0n; out.length < size; ctr++) {
    const ctrBe = Buffer.alloc(8);
    ctrBe.writeBigUInt64BE(ctr);
    const v = sha256(Buffer.concat([seed, ctrBe])).subarray(0, 8).readBigUInt64BE();
    if (v >= limit) continue;
    const i = Number(v % BigInt(nUnits));
    if (!seen.has(i)) { seen.add(i); out.push(i); }
  }
  return out.sort((a, b) => a - b);
}

// ---- §12 chain digest --------------------------------------------------------------------
function chainDigest(certs) {
  const h = createHash('sha256');
  for (const c of certs) h.update(Buffer.from(canon({ certificate: c }), 'utf8'));   // chunk certs, in order
  return 'sha256:' + h.digest('hex');
}

// ---- the reference tensors (reconstructed from the arange convention) ---------------------
const arange = (n, f = x => x) => Array.from({ length: n }, (_, i) => f(i));
const f64le = v => { const b = Buffer.allocUnsafe(8 * v.length); v.forEach((x, i) => b.writeDoubleLE(x, 8 * i)); return b; };
const i32le = v => { const b = Buffer.allocUnsafe(4 * v.length); v.forEach((x, i) => b.writeInt32LE(x, 4 * i)); return b; };
const i8b   = v => { const b = Buffer.allocUnsafe(v.length);     v.forEach((x, i) => b.writeInt8(x, i)); return b; };

const A_f64 = arange(6);                                  // np.arange(6,f8).reshape(2,3)
const B_i32 = arange(4);                                  // np.arange(4,i4)
const Y_f64 = arange(64);                                 // np.arange(64,f8).reshape(16,4)
const C_i8  = [-3, -2, -1, 0, 1, 2];                      // np.arange(-3,3,i1)

const D_A = tensorDigest('f8', [2, 3], f64le(A_f64));
const D_B = tensorDigest('i4', [4], i32le(B_i32));
const D_Y = tensorDigest('f8', [16, 4], f64le(Y_f64));
const D_I8 = tensorDigest('i1', [6], i8b(C_i8));
const MODEL_A = namedDigest({ w: D_A });                  // model {"w": A}
const INPUT_B = namedDigest({ x: D_B });                  // input {"x": B}

// ---- receipt manifests -------------------------------------------------------------------
// receipt/exact-profile (§4–§5): a full v0.1 receipt binding model|input|computation|arith|output.
const exact = {
  arithmetic: ARITH, computation: { id: 'conformance.identity', version: '1' },
  cr: '0.1', digest_alg: 'sha256',
  input: { digest: INPUT_B, n_tensors: 1 },
  model: { digest: MODEL_A, n_tensors: 1 },
  output: { digest: D_A, shape: [2, 3] },
};

// receipt/sampled (§10): challenge "beef"; indices derived, sample.digest over the sampled rows.
const sampledBase = {
  arithmetic: ARITH, computation: { id: 'conformance.sampled', version: '1' },
  cr: '0.1.1', digest_alg: 'sha256',
  input: { digest: INPUT_B, n_tensors: 1 },
  model: { digest: EMPTY_COLLECTION, n_tensors: 0 },
  output: { digest: D_Y, shape: [16, 4] },
};
const sIdx = sampledIndices(sampledBase, 'beef', 16, 4);
const sampledRows = sIdx.flatMap(i => Y_f64.slice(i * 4, i * 4 + 4));   // rows of the 16x4 output
const sampleDigest = tensorDigest('f8', [sIdx.length, 4], f64le(sampledRows));
const sampled = { ...sampledBase, sample: {
  challenge: 'beef', digest: sampleDigest, n_units: 16, rule: 'sha256-ctr-reject-v1', size: 4,
} };

// receipt/chunk-0 (§12): first link of a chain; chain_digest over just this chunk's certificate.
const chunk0 = {
  arithmetic: ARITH, chunk: { closing: false, index: 0, prev_certificate: '' },
  computation: { id: 'conformance.chunked', version: '1' }, cr: '0.1.2', digest_alg: 'sha256',
  input: { digest: INPUT_B, n_tensors: 1 },
  model: { digest: EMPTY_COLLECTION, n_tensors: 0 },
  output: { digest: D_A, shape: [2, 3] },
};
const chunk0Cert = certificateOf(chunk0);

// chain/closing-2chunk: chunk0, chunk1 (chained to chunk0), and the closing receipt over both.
const chunk1 = { ...chunk0, chunk: { closing: false, index: 1, prev_certificate: chunk0Cert } };
const chunk1Cert = certificateOf(chunk1);
const twoChunkDigest = chainDigest([chunk0Cert, chunk1Cert]);   // over the two CHUNK certs, in order
const closing = {
  arithmetic: ARITH,
  chunk: { chain_digest: twoChunkDigest, closing: true, index: 2, n_chunks: 2, prev_certificate: chunk1Cert },
  computation: { id: 'conformance.chunked', version: '1' }, cr: '0.1.2', digest_alg: 'sha256',
  input: { digest: EMPTY_COLLECTION, n_tensors: 0 },      // closing receipt: inputs=None, model=None
  model: { digest: EMPTY_COLLECTION, n_tensors: 0 },
  output: { digest: D_A, shape: [2, 3] },
};

// ---- emit every vector this emitter can reproduce (canonicalisation + receipt layers) -----
const canonVec = (name, input) => ({ name, input, canonical: canon(input), digest: digestBytes(canon(input)) });
const receiptVec = (name, m, extra = {}) =>
  ({ name, certificate: certificateOf(m), manifest_canonical: canon(m), ...extra });

const vectors = [
  canonVec('canonical/empty-object', {}),
  canonVec('canonical/key-order', { b: 1, a: { d: [1, 2, 3], c: 'x' } }),
  canonVec('canonical/utf8', { k: 'café-é中' }),
  canonVec('canonical/escaping', { path: 'model/forward', ctl: 'a\tb\nc\x1fd', q: 'he said "hi"\\done' }),
  { name: 'tensor/float64-2x3', digest: D_A },
  { name: 'tensor/int32-4', digest: D_B },
  { name: 'tensor/int8-6', digest: D_I8 },
  { name: 'tensors/named-order-independent', digest: namedDigest({ w2: D_B, w1: D_A }) },
  receiptVec('receipt/exact-profile', exact),
  receiptVec('receipt/sampled', sampled, { sample_indices: sIdx }),
  receiptVec('receipt/chunk-0', chunk0, { chain_digest: chainDigest([chunk0Cert]) }),
  receiptVec('chain/closing-2chunk', closing, { chain_digest: twoChunkDigest }),
];

process.stdout.write(JSON.stringify(vectors, null, 2) + '\n');
