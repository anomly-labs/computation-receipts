// Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
//
// cr_canonical.mjs — a SECOND, from-scratch CR implementation in JavaScript.
//
// It is written from spec/CR-v0.1.md §2-§3 alone (Node stdlib only, no CR code, no reading of
// the reference vectors) and emits the canonicalisation-layer conformance vectors in the
// published schema. Pipe it into the conformance runner to prove a non-Python implementation
// reproduces the vectors from the spec text:
//
//   node examples/js/cr_canonical.mjs > /tmp/js.json
//   python3 python/conformance_runner.py /tmp/js.json      # expect: canonicalisation 8/8
//
// This is the real test of §9 ("implement CR without this repository") and a direct check that
// the pinned rules (S1 escaping, S2 single-byte dtype, S3 code-point collection sort) are
// sufficient for an independent implementer.

import { createHash } from 'node:crypto';

// ---- §2 canonical form -------------------------------------------------------------------

// §2 rule 6: escape only " \ and the C0 controls (short forms where defined, else lowercase
// \u00xx); the solidus / is NOT escaped; every other character (incl. non-ASCII) is literal.
function escStr(s) {
  let out = '"';
  for (const ch of s) {                    // for..of iterates by code point (handles surrogates)
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

// §2 rule 2: sort keys by Unicode code point (not UTF-16 code unit).
function cmpCodepoint(a, b) {
  const ap = [...a], bp = [...b];          // spread splits by code point
  for (let i = 0; i < Math.min(ap.length, bp.length); i++) {
    const d = ap[i].codePointAt(0) - bp[i].codePointAt(0);
    if (d !== 0) return d;
  }
  return ap.length - bp.length;
}

function canon(v) {
  if (v === null) return 'null';
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (typeof v === 'bigint') return v.toString();
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) throw new Error('§2 rule 4: NaN/Infinity must not appear');
    if (!Number.isInteger(v)) throw new Error('§2 rule 7: manifests carry integers only');
    return String(v);                       // §2 rule 7: base-10 integer, no decimal point
  }
  if (typeof v === 'string') return escStr(v);
  if (Array.isArray(v)) return '[' + v.map(canon).join(',') + ']';
  if (typeof v === 'object') {
    const keys = Object.keys(v).sort(cmpCodepoint);
    return '{' + keys.map(k => escStr(k) + ':' + canon(v[k])).join(',') + '}';
  }
  throw new Error('unserialisable value');
}

const sha256hex = buf => createHash('sha256').update(buf).digest('hex');
const digestOf = str => 'sha256:' + sha256hex(Buffer.from(str, 'utf8'));

// ---- §3 digests --------------------------------------------------------------------------

// §3.1: H( canonical({"dtype":code,"shape":[dims]}) || raw little-endian C-order bytes ).
function tensorDigest(code, shape, raw) {
  const h = createHash('sha256');
  h.update(Buffer.from(canon({ dtype: code, shape }), 'utf8'));
  h.update(raw);
  return 'sha256:' + h.digest('hex');
}

// §3.2: iterate names sorted by code point; update with canonical({"name":n,"digest":d}).
function namedDigest(named) {
  const h = createHash('sha256');
  for (const name of Object.keys(named).sort(cmpCodepoint)) {
    h.update(Buffer.from(canon({ name, digest: named[name] }), 'utf8'));
  }
  return 'sha256:' + h.digest('hex');
}

// raw little-endian bytes for the reference tensors (reconstructed from the arange convention)
function f64le(vals) { const b = Buffer.allocUnsafe(8 * vals.length); vals.forEach((v, i) => b.writeDoubleLE(v, 8 * i)); return b; }
function i32le(vals) { const b = Buffer.allocUnsafe(4 * vals.length); vals.forEach((v, i) => b.writeInt32LE(v, 4 * i)); return b; }
function i8(vals)    { const b = Buffer.allocUnsafe(vals.length);     vals.forEach((v, i) => b.writeInt8(v, i)); return b; }

// ---- emit the canonicalisation-layer vectors in the published schema ---------------------

const F64 = tensorDigest('f8', [2, 3], f64le([0, 1, 2, 3, 4, 5]));   // np.arange(6,f8).reshape(2,3)
const I32 = tensorDigest('i4', [4], i32le([0, 1, 2, 3]));            // np.arange(4,i4)
const I8  = tensorDigest('i1', [6], i8([-3, -2, -1, 0, 1, 2]));      // np.arange(-3,3,i1)

const canonVec = (name, input) => ({ name, input, canonical: canon(input), digest: digestOf(canon(input)) });

const vectors = [
  canonVec('canonical/empty-object', {}),
  canonVec('canonical/key-order', { b: 1, a: { d: [1, 2, 3], c: 'x' } }),
  canonVec('canonical/utf8', { k: 'café-é中' }),
  canonVec('canonical/escaping', { path: 'model/forward', ctl: 'a\tb\ncd', q: 'he said "hi"\\done' }),
  { name: 'tensor/float64-2x3', digest: F64 },
  { name: 'tensor/int32-4', digest: I32 },
  { name: 'tensor/int8-6', digest: I8 },
  { name: 'tensors/named-order-independent', digest: namedDigest({ w2: I32, w1: F64 }) },
];

process.stdout.write(JSON.stringify(vectors, null, 2) + '\n');
