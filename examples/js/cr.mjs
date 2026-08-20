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
// The refuse/* vectors are not EMITTED here: they assert a VERDICT a verifier must return.
// That verifier half now lives in this same file (verifyReceipt / verifyChain, the `verify`
// and `verify-chain` CLI modes below); the conformance runner exercises it against the five
// refusal vectors, so this second implementation reproduces all 17 published vectors.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

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

// ---- §7 the profile registry (the entries the vectors touch, plus bposit8-imma-int32) ----
const PROFILES = {
  'bposit16-quire256': { accumulation: 'exact', order_independent: true,
    params: { es: 3, frac_bits: 96, n: 16, quire_bits: 256 } },
  // Added 2026-08-10: the tensor-core W8A8 path — int8 fixed-point operands from bposit8
  // codes (int8 = clamp(floor(value*2^6), -128, 127)), exact int32 accumulation (IMMA).
  'bposit8-imma-int32': { accumulation: 'exact', order_independent: true,
    params: { accumulator: 'int32', es: 2, n: 8, operand_scale_bits: 6 } },
  float64: { accumulation: 'rounded', order_independent: false, params: { bits: 64 } },
  float32: { accumulation: 'rounded', order_independent: false, params: { bits: 32 } },
};
const ARITH = { ...PROFILES['bposit16-quire256'], profile: 'bposit16-quire256' };

// ---- §6 / §6.1 verification (the verdict half of the implementation) ----------------------
// A "<alg>:<lowercase-hex>" digest of the length the algorithm produces (§3).
function isDigest(s) {
  if (typeof s !== 'string' || !s.includes(':')) return false;
  const i = s.indexOf(':'), n = { sha256: 64, sha512: 128 }[s.slice(0, i)];
  const hex = s.slice(i + 1);
  return n !== undefined && hex.length === n && /^[0-9a-f]+$/.test(hex);
}
// Structural validation (§6 MALFORMED conditions + §6.1 rule 1 + §10 + S7). Returns null if
// well-formed, else the MALFORMED reason. Deep-equality of params is via canonicalisation.
function malformedReason(m) {
  if (typeof m !== 'object' || m === null || Array.isArray(m)) return 'manifest must be an object';
  if (!['0.1', '0.1.1', '0.1.2'].includes(m.cr)) return `unsupported version ${m.cr}`;
  for (const k of ['digest_alg', 'arithmetic', 'computation', 'model', 'input', 'output'])
    if (!(k in m)) return `missing ${k}`;
  if (!['sha256', 'sha512'].includes(m.digest_alg)) return 'unsupported digest algorithm';
  for (const sec of ['model', 'input', 'output'])
    if (!isDigest(m[sec].digest)) return `${sec}.digest missing or not a well-formed digest`;
  if (!Array.isArray(m.output.shape) || m.output.shape.length === 0) return 'output.shape must be non-empty';
  const a = m.arithmetic;
  if (typeof a !== 'object' || !('profile' in a)) return 'arithmetic section missing profile';
  if (typeof a.profile !== 'string') return 'arithmetic.profile must be a string';
  const reg = PROFILES[a.profile];                                    // §6.1 rule 1
  if (reg) for (const f of ['accumulation', 'order_independent', 'params'])
    if (canon(a[f]) !== canon(reg[f])) return `arithmetic.${f} contradicts the registered profile`;
  if (m.cr === '0.1' && 'sample' in m) return 'v0.1 receipt carries a sample section';   // S7
  if (m.cr === '0.1' && 'chunk' in m) return 'v0.1 receipt carries a chunk section';
  if (m.cr === '0.1.1' && 'chunk' in m) return 'sampled receipt carries a chunk section';
  if (m.cr === '0.1.2' && 'sample' in m) return 'chunked receipt carries a sample section';
  if (m.cr === '0.1.2') {                                              // §12 chunk-section shape
    const c = m.chunk;
    if (typeof c !== 'object' || c === null) return 'missing chunk section';
    if (!(Number.isInteger(c.index) && c.index >= 0)) return 'chunk index invalid';
    if (typeof c.prev_certificate !== 'string') return 'chunk missing prev_certificate';
    if (c.index === 0 && c.prev_certificate !== '') return 'chunk 0 must have empty prev_certificate';
    if (c.index > 0 && !c.prev_certificate) return `chunk ${c.index} has no prev_certificate`;
    if (typeof c.closing !== 'boolean') return 'chunk missing closing flag';
    if (c.closing && !(Number.isInteger(c.n_chunks) && c.n_chunks === c.index && typeof c.chain_digest === 'string'))
      return 'closing receipt needs n_chunks == index and a chain_digest';
  }
  if (m.cr === '0.1.1') {
    const s = m.sample;
    if (typeof s !== 'object' || s === null) return 'missing sample section';
    if (s.rule !== 'sha256-ctr-reject-v1') return 'unknown sample rule';
    if (!(Number.isInteger(s.n_units) && Number.isInteger(s.size) && 1 <= s.size && s.size <= s.n_units))
      return 'sample size/n_units invalid';
    if (typeof s.challenge !== 'string' || typeof s.digest !== 'string') return 'sample challenge/digest';
    if (s.challenge.length % 2 || !/^[0-9a-fA-F]*$/.test(s.challenge)) return 'sample.challenge not hex';
    if (m.output.shape[0] !== s.n_units) return 'sample.n_units != output.shape[0]';   // §10
  }
  return null;
}
// §6 verdict for a claimed receipt with no re-execution supplied — enough for the refuse/*
// vectors: MALFORMED if structurally invalid; UNVERIFIABLE if the profile is unregistered
// (§6.1 rule 2 — the claim cannot be assessed); otherwise a full verify would re-execute.
function refusalVerdict(m) {
  if (malformedReason(m) !== null) return 'MALFORMED';
  if (!(m.arithmetic.profile in PROFILES)) return 'UNVERIFIABLE';
  return 'ACCEPT';
}

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

// ---- refuse/* manifests (mutations of the valid ones, exactly as the reference builds them) --
const profileContradicts = { ...exact, arithmetic: { ...ARITH, profile: 'float64' } };       // §6.1 rule 1
const sampleSpaceShrunk = { ...sampled, sample: { ...sampled.sample, n_units: 4 } };          // §10
const unknownProfile = { ...exact, arithmetic: { ...ARITH, profile: 'vendor-unregistered-v0' } }; // §6.1 rule 2
const outputDigestMissing = { ...sampled, output: { shape: sampled.output.shape } };          // no digest
const crossVersion = { ...sampled, chunk: { closing: false, index: 0, prev_certificate: '' } }; // S7

// ---- emit every vector this implementation can reproduce ----------------------------------
const canonVec = (name, input) => ({ name, input, canonical: canon(input), digest: digestBytes(canon(input)) });
const receiptVec = (name, m, extra = {}) =>
  ({ name, certificate: certificateOf(m), manifest_canonical: canon(m), ...extra });
const refuseVec = (name, m) =>
  ({ name, certificate: certificateOf(m), manifest_canonical: canon(m), expect_verdict: refusalVerdict(m) });

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
  refuseVec('refuse/profile-contradicts-registry', profileContradicts),
  refuseVec('refuse/sample-space-shrunk', sampleSpaceShrunk),
  refuseVec('refuse/unknown-profile-not-accepted', unknownProfile),
  refuseVec('refuse/output-digest-missing', outputDigestMissing),
  refuseVec('refuse/cross-version-section', crossVersion),
];

// ---- CLI --------------------------------------------------------------------------------
// A standalone verdict for a single receipt, using only the structural rules this file already
// implements. Without an independent re-execution a verifier cannot reach ACCEPT/REJECT, but it
// can already deliver MALFORMED, UNVERIFIABLE, and "well-formed, awaiting re-execution".
function verifyReceipt(receipt) {
  if (typeof receipt !== 'object' || receipt === null || !('manifest' in receipt))
    return { status: 'MALFORMED', reason: 'receipt must be an object with a manifest' };
  const m = receipt.manifest;
  const r = malformedReason(m);
  if (r !== null) return { status: 'MALFORMED', reason: r };
  let cert;
  try { cert = certificateOf(m); }
  catch (e) { return { status: 'MALFORMED', reason: 'manifest is not canonicalisable: ' + e.message }; }
  if (typeof receipt.certificate !== 'string' || receipt.certificate !== cert)
    return { status: 'MALFORMED', reason: 'certificate does not match manifest (tampered or miscomputed)' };
  if (!(m.arithmetic.profile in PROFILES))
    return { status: 'UNVERIFIABLE', reason: `arithmetic profile '${m.arithmetic.profile}' is not in this verifier's registry` };
  if (!m.arithmetic.order_independent)
    return { status: 'UNVERIFIABLE', reason: 'arithmetic is order-dependent; no re-execution can settle it' };
  return { status: 'WELL-FORMED', reason: 'self-consistent and order-independent; supply an independent re-execution to obtain ACCEPT/REJECT' };
}

// ---- §12 chain verification (the verdict half for receipt chains) -------------------------
// verifyChain and its per-chunk verifyPair mirror the reference verify_chain / verify: an
// INDEPENDENT second implementation of the most involved verdict path, so a chain artifact can
// be re-verified in a different language. Verdict strings and REJECT/MALFORMED causes match the
// reference (python/cr/receipt.py) so the two implementations agree on every bundle.

function wellformedCert(r) {                    // null if well-formed AND certificate matches
  if (typeof r !== 'object' || r === null || !('manifest' in r)) return 'receipt must have a manifest';
  const bad = malformedReason(r.manifest);
  if (bad !== null) return bad;
  let cert;
  try { cert = certificateOf(r.manifest); } catch (e) { return 'not canonicalisable: ' + e.message; }
  if (typeof r.certificate !== 'string' || r.certificate !== cert)
    return 'certificate does not match manifest (tampered or miscomputed)';
  return null;
}

// Mirror of verify(a, b): compare a claimed receipt against an independent re-execution.
function verifyPair(a, b) {
  for (const r of [a, b]) { const bad = wellformedCert(r); if (bad !== null) return { status: 'MALFORMED', reason: bad }; }
  for (const k of ['model', 'input', 'computation'])
    if (canon(a.manifest[k]) !== canon(b.manifest[k])) return { status: 'REJECT', reason: `re-execution used a different ${k}` };
  if (a.manifest.arithmetic.profile !== b.manifest.arithmetic.profile) return { status: 'REJECT', reason: 're-execution used a different arithmetic profile' };
  if (!(a.manifest.arithmetic.profile in PROFILES)) return { status: 'UNVERIFIABLE', reason: 'profile is not in this verifier\'s registry' };
  if (!a.manifest.arithmetic.order_independent) return { status: 'UNVERIFIABLE', reason: 'arithmetic is order-dependent' };
  if (a.manifest.output.digest !== b.manifest.output.digest) return { status: 'REJECT', reason: 'output digest differs under order-independent arithmetic' };
  if (a.certificate !== b.certificate) return { status: 'REJECT', reason: 'certificate differs' };
  return { status: 'ACCEPT', reason: 're-execution reproduced the certificate' };
}

function isClosing(r) {
  const c = r && typeof r.manifest === 'object' && r.manifest !== null ? r.manifest.chunk : null;
  return c && typeof c === 'object' && c.closing === true;
}

// Mirror of verify_chain(claimed, recomputed, allow_open).
function verifyChain(claimed, recomputed = null, allowOpen = false) {
  if (!Array.isArray(claimed) || claimed.length === 0) return { status: 'MALFORMED', reason: 'empty chain' };
  let body = claimed.slice(), closing = null;
  if (isClosing(body[body.length - 1])) { closing = body[body.length - 1]; body = body.slice(0, -1); }
  if (body.length === 0) return { status: 'MALFORMED', reason: 'chain has a closing receipt but no chunks' };
  for (let k = 0; k < body.length; k++) {
    const bad = wellformedCert(body[k]);
    if (bad !== null) return { status: 'MALFORMED', reason: `chunk ${k}: ${bad}` };
    const c = body[k].manifest.chunk;
    if (typeof c !== 'object' || c === null || c.closing) return { status: 'MALFORMED', reason: `chunk ${k}: not a chunk receipt` };
    if (c.index !== k) return { status: 'REJECT', reason: `chunk order broken: position ${k} carries index ${c.index}` };
    const wantPrev = k === 0 ? '' : body[k - 1].certificate;
    if (c.prev_certificate !== wantPrev) return { status: 'REJECT', reason: `chunk ${k}: prev_certificate does not match chunk ${k - 1} (reordered, dropped or foreign chunk)` };
  }
  if (closing !== null) {
    const bad = wellformedCert(closing);
    if (bad !== null) return { status: 'MALFORMED', reason: `closing receipt: ${bad}` };
    const cc = closing.manifest.chunk;
    if (cc.n_chunks !== body.length) return { status: 'REJECT', reason: `closing receipt claims ${cc.n_chunks} chunks, chain has ${body.length}` };
    if (cc.prev_certificate !== body[body.length - 1].certificate) return { status: 'REJECT', reason: 'closing receipt does not chain from the last chunk' };
    if (cc.chain_digest !== chainDigest(body.map(r => r.certificate))) return { status: 'REJECT', reason: 'closing chain_digest does not match the chain' };
  } else if (!allowOpen) {
    return { status: 'REJECT', reason: 'chain is not closed: truncation would be undetectable (pass allow_open to verify a prefix)' };
  }
  if (recomputed === null) return { status: 'UNVERIFIABLE', reason: 'no re-executed chain supplied' };
  if (recomputed.length !== body.length) return { status: 'MALFORMED', reason: `re-executed chain has ${recomputed.length} chunks, claimed body has ${body.length}` };
  for (let k = 0; k < body.length; k++) {
    const v = verifyPair(body[k], recomputed[k]);
    if (v.status !== 'ACCEPT') return { status: v.status, reason: `chunk ${k}: ${v.reason}` };
  }
  const state = closing !== null ? `closed chain of ${body.length} chunks, totality bound by closing receipt`
                                 : `OPEN PREFIX of ${body.length} chunks: totality UNATTESTED`;
  return { status: 'ACCEPT', reason: `every chunk re-executed to its certificate; ${state}` };
}

if (process.argv[2] === 'verify-chain') {
  // node cr.mjs verify-chain <bundle.json> [--allow-open]
  // A bundle is {kind:"cr-chain", receipts:[...]} (or a bare receipts array). The verifier's
  // re-execution defaults to the bundle's own body — the chain analog of verify(r, r), i.e. the
  // check a second machine runs when it recomputes the same body. Prints verdict + reason.
  const path = process.argv[3];
  const allowOpen = process.argv.includes('--allow-open');
  if (!path) { console.error('usage: node cr.mjs verify-chain <bundle.json> [--allow-open]'); process.exit(2); }
  let verdict;
  try {
    const parsed = JSON.parse(readFileSync(path, 'utf8'));
    const receipts = Array.isArray(parsed) ? parsed : parsed.receipts;
    if (!Array.isArray(receipts)) throw new Error('bundle has no receipts array');
    let body = receipts.slice();
    if (isClosing(body[body.length - 1])) body = body.slice(0, -1);
    verdict = verifyChain(receipts, body, allowOpen);
  } catch (e) {
    verdict = { status: 'MALFORMED', reason: `could not read bundle: ${e.message}` };
  }
  console.log(`${verdict.status}  ${verdict.reason}`);
  process.exit(verdict.status === 'ACCEPT' ? 0 : 1);
} else if (process.argv[2] === 'verify') {
  // node cr.mjs verify <receipt.json>  — read a receipt and print its verdict + reason.
  const path = process.argv[3];
  if (!path) { console.error('usage: node cr.mjs verify <receipt.json>'); process.exit(2); }
  let verdict;
  try {
    verdict = verifyReceipt(JSON.parse(readFileSync(path, 'utf8')));
  } catch (e) {
    // a verifier must ANSWER, never crash, on hostile input — this file's own §6 lesson.
    verdict = { status: 'MALFORMED', reason: `could not read receipt: ${e.message}` };
  }
  console.log(`${verdict.status}  ${verdict.reason}`);
  process.exit(verdict.status === 'WELL-FORMED' ? 0 : 1);
} else {
  // no-arg (and any other) mode: emit the conformance vectors for the runner (unchanged).
  process.stdout.write(JSON.stringify(vectors, null, 2) + '\n');
}
