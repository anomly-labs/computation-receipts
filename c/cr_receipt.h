/*
 * Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe. Licensed per LICENSES.md (Apache-2.0 code / CC-BY-4.0 docs).
 *
 * cr_receipt.h — Computation Receipt (CR) v0.1 emitter, in dependency-free C.
 *
 * WHY THIS EXISTS. The milestone-1 demo shipped a bare SHA-256 of the quire.
 * CR v0.1 §5 is explicit that such a hash cannot carry a claim: the same output
 * from a different model or a different computation is indistinguishable. This
 * module upgrades the on-card result to a real receipt whose certificate binds
 * model, input, computation, arithmetic AND output together, so an on-card run
 * ships something a third party can actually verify.
 *
 * IT IS ALSO THE STANDARD'S OWN TEST. CR claims to be implementable by anyone
 * without our library. This is an independent implementation in another
 * language, on the FPGA host, sharing no code with the Python reference — and
 * cross_check_receipt.py asserts the two produce byte-identical certificates.
 * If that ever fails, CR is not a standard and we should stop calling it one.
 *
 * Scope: emit only. Verification lives in the reference implementation and the
 * `space-time receipt` CLI; a prover does not verify its own receipts.
 *
 * Canonicalisation implemented here (CR §2, §3):
 *   - UTF-8 JSON, keys sorted by code point, separators exactly ',' and ':'
 *   - tensor digest = SHA256( canonical({"dtype":..,"shape":[..]}) || LE bytes )
 *   - named collection = SHA256 over sorted canonical({"name":..,"digest":..})
 *   - certificate = SHA256( canonical(manifest) )
 * Keys are emitted in hand-verified sorted order rather than sorted at runtime;
 * cross_check_receipt.py is what proves that ordering correct.
 */
#ifndef CR_RECEIPT_H
#define CR_RECEIPT_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Self-contained SHA-256 (public-domain FIPS 180-4 implementation) so this header
 * links with no external dependency — `#include "cr_receipt.h"` and compile. Static,
 * matching the rest of this header-only library; every consuming TU gets its own copy.
 * Correctness is not asserted by faith: the differential test against Python's hashlib
 * (docs/audit/.../differential_c_python.py) fails on a single wrong bit. */
typedef struct { uint32_t h[8]; uint64_t len; uint8_t buf[64]; size_t fill; } sha256_t;

static void sha256_init(sha256_t *s) {
    static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                   0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    memcpy(s->h, iv, sizeof iv); s->len = 0; s->fill = 0;
}

static void cr__sha256_block(sha256_t *s, const uint8_t *p) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
        0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
        0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
        0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
        0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
        0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
#define CR__ROR(x,n) (((x)>>(n))|((x)<<(32-(n))))
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)p[i*4]<<24 | (uint32_t)p[i*4+1]<<16 |
               (uint32_t)p[i*4+2]<<8 | (uint32_t)p[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = CR__ROR(w[i-15],7)^CR__ROR(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1 = CR__ROR(w[i-2],17)^CR__ROR(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],
             e=s->h[4],f=s->h[5],g=s->h[6],hh=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = CR__ROR(e,6)^CR__ROR(e,11)^CR__ROR(e,25);
        uint32_t ch = (e&f)^(~e&g);
        uint32_t t1 = hh+S1+ch+k[i]+w[i];
        uint32_t S0 = CR__ROR(a,2)^CR__ROR(a,13)^CR__ROR(a,22);
        uint32_t maj = (a&b)^(a&c)^(b&c);
        uint32_t t2 = S0+maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=hh;
#undef CR__ROR
}

static void sha256_update(sha256_t *s, const void *data, size_t n) {
    const uint8_t *p = (const uint8_t *)data;
    s->len += n;
    while (n) {
        size_t take = 64 - s->fill; if (take > n) take = n;
        memcpy(s->buf + s->fill, p, take);
        s->fill += take; p += take; n -= take;
        if (s->fill == 64) { cr__sha256_block(s, s->buf); s->fill = 0; }
    }
}

static void sha256_hex(sha256_t *s, char out[65]) {
    uint64_t bits = s->len * 8;
    uint8_t pad = 0x80;
    sha256_update(s, &pad, 1);
    uint8_t zero = 0;
    while (s->fill != 56) sha256_update(s, &zero, 1);
    uint8_t lenbe[8];
    for (int i = 0; i < 8; i++) lenbe[i] = (uint8_t)(bits >> (56 - 8*i));
    /* append length WITHOUT re-counting it into len (write the block directly) */
    memcpy(s->buf + 56, lenbe, 8);
    cr__sha256_block(s, s->buf);
    static const char hexd[] = "0123456789abcdef";
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 4; j++) {
            uint8_t byte = (uint8_t)(s->h[i] >> (24 - 8*j));
            out[(i*4+j)*2]   = hexd[byte >> 4];
            out[(i*4+j)*2+1] = hexd[byte & 15];
        }
    out[64] = 0;
}

/* SHA-256 of an empty message — the digest of an empty tensor collection. */
#define CR_EMPTY_COLLECTION_DIGEST \
    "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

/* digest of one tensor: dtype is the numpy code without byte-order prefix
 * ("u2" = uint16, "u4" = uint32); data must already be little-endian. */
/* hash a NUL-terminated string by its true length — avoids fixed accumulation buffers.
 * MEMORY-SAFETY (found 2026-08-08, open-cr pre-release audit, confirmed by ASan): the
 * previous versions built the header/entry with snprintf into fixed buffers and then fed
 * sha256_update the snprintf RETURN value — which is the would-be length, not the
 * truncated one — so a long dtype or name caused an out-of-bounds read/write past the
 * buffer. Hashing pieces directly cannot overflow and is byte-identical for valid input. */
static void cr__sha_str(sha256_t *s, const char *str) {
    sha256_update(s, str, strlen(str));
}

/* bytes ACTUALLY written by snprintf into a buffer of capacity `cap`. snprintf returns
 * the would-be length, not the truncated one, so using its return as a hash/copy length
 * reads past the buffer. MEMORY-SAFETY (found 2026-08-08, ASan-confirmed): this recurred
 * at every snprintf-into-fixed-buffer-then-hash site (all four manifest builders,
 * cr_chain_digest, cr_sample_seed); clamping here eliminates the whole class. A truncated
 * buffer is caller misuse (buffer too small) and yields a certificate that will not verify
 * against a correct re-execution — safe by consequence, never an out-of-bounds access. */
static size_t cr__wrote(int n, size_t cap) {
    if (n < 0) return 0;
    return (size_t)n < cap ? (size_t)n : (cap ? cap - 1 : 0);
}

#define CR_JSON_ESC_MAX 1024

/* JSON-escape a caller string into `out` (NUL-terminated), matching the CR canonical form
 * (spec §2) and Python's json.dumps(ensure_ascii=False): escape " and \, the short controls
 * \b \t \n \f \r, any other control byte (< 0x20) as \u00xx (lowercase hex); every other
 * byte — INCLUDING non-ASCII >= 0x80 — is copied literally (spec §2 rule 5, so a UTF-8
 * "café" hashes identically on both sides).
 * CONFORMANCE (found 2026-08-08, C<->Python emitter differential on special characters):
 * the manifest builders previously interpolated computation id / version and tensor names
 * with a raw %s, so a name containing a quote, backslash or control character produced bytes
 * that were not valid JSON and did NOT match the Python reference — silently breaking the
 * cross-implementation byte-identity the whole format rests on (an honest on-card prover
 * would emit a certificate the verifier could not reproduce). This restores it. If the
 * escaped result would exceed `cap` it stops on an escape boundary (never a partial escape),
 * yielding a certificate that will not verify — caller misuse, never an out-of-bounds write. */
static void cr__json_escape(const char *in, char *out, size_t cap) {
    static const char hexd[] = "0123456789abcdef";
    size_t o = 0;
    if (cap == 0) return;
    for (const unsigned char *p = (const unsigned char *)in; *p; p++) {
        unsigned char c = *p;
        char esc[6];
        size_t elen;
        switch (c) {
            case '"':  esc[0] = '\\'; esc[1] = '"';  elen = 2; break;
            case '\\': esc[0] = '\\'; esc[1] = '\\'; elen = 2; break;
            case '\b': esc[0] = '\\'; esc[1] = 'b';  elen = 2; break;
            case '\t': esc[0] = '\\'; esc[1] = 't';  elen = 2; break;
            case '\n': esc[0] = '\\'; esc[1] = 'n';  elen = 2; break;
            case '\f': esc[0] = '\\'; esc[1] = 'f';  elen = 2; break;
            case '\r': esc[0] = '\\'; esc[1] = 'r';  elen = 2; break;
            default:
                if (c < 0x20) {
                    esc[0] = '\\'; esc[1] = 'u'; esc[2] = '0'; esc[3] = '0';
                    esc[4] = hexd[c >> 4]; esc[5] = hexd[c & 15]; elen = 6;
                } else {
                    esc[0] = (char)c; elen = 1;
                }
        }
        if (o + elen >= cap) break;   /* leave room for the NUL; stop on an escape boundary */
        for (size_t k = 0; k < elen; k++) out[o++] = esc[k];
    }
    out[o] = 0;
}

static void cr_digest_tensor(const char *dtype, const int *shape, int ndim,
                             const void *data, size_t nbytes, char out[72])
{
    sha256_t s; sha256_init(&s);
    cr__sha_str(&s, "{\"dtype\":\"");
    cr__sha_str(&s, dtype);
    cr__sha_str(&s, "\",\"shape\":[");
    for (int i = 0; i < ndim; i++) {
        char d[24];                                  /* ",-2147483648" fits in 13 */
        int m = snprintf(d, sizeof d, "%s%d", i ? "," : "", shape[i]);
        if (m > 0 && (size_t)m < sizeof d) sha256_update(&s, d, (size_t)m);
    }
    cr__sha_str(&s, "]}");
    sha256_update(&s, data, nbytes);
    char hex[65]; sha256_hex(&s, hex);
    snprintf(out, 72, "sha256:%s", hex);
}

/* digest of a named collection; names MUST be passed already sorted. */
static void cr_digest_named(const char *const *names, const char *const *digests,
                            int count, char out[72])
{
    sha256_t s; sha256_init(&s);
    for (int i = 0; i < count; i++) {
        char name_e[CR_JSON_ESC_MAX];
        cr__json_escape(names[i], name_e, sizeof name_e);
        cr__sha_str(&s, "{\"digest\":\"");
        cr__sha_str(&s, digests[i]);         /* a sha256:<hex> digest — no escaping needed */
        cr__sha_str(&s, "\",\"name\":\"");
        cr__sha_str(&s, name_e);
        cr__sha_str(&s, "\"}");
    }
    char hex[65]; sha256_hex(&s, hex);
    snprintf(out, 72, "sha256:%s", hex);
}

/*
 * Build the canonical manifest for one on-card exact-quire MAC and its
 * certificate. Key order below is the CR canonical (sorted) order:
 *   arithmetic < computation < cr < digest_alg < input < model < output
 * and within arithmetic: accumulation < order_independent < params < profile;
 * within params: es < frac_bits < n < quire_bits.
 *
 * ESCAPING BOUNDARY (verified 2026-08-08, C<->Python differential): the FREE-TEXT string
 * parameters — computation_id, computation_version (here) and tensor names / host_note
 * (elsewhere) — are JSON-escaped via cr__json_escape, because a caller legitimately chooses
 * them and they can contain any character. The remaining %s parameters (input_digest,
 * output_digest, sample_digest, prev_certificate, challenge_hex, chain certs) are
 * FORMAT-CONSTRAINED to `sha256:<hex>` or hex by this emitter's contract: they are hash
 * outputs / prior certificates / beacon randomness, never free text. Valid hex contains no
 * escapable character, so escaping them would be a no-op; a special character in one is
 * caller misuse that yields a certificate which will not verify against a correct
 * re-execution — safe by consequence (same principle as cr__wrote), not a reachable defect.
 */
static void cr_build_manifest(char *buf, size_t buflen,
                              const char *computation_id,
                              const char *computation_version,
                              const char *input_digest, int n_inputs,
                              const char *output_digest, int output_len,
                              char certificate[72])
{
    char id_e[CR_JSON_ESC_MAX], ver_e[CR_JSON_ESC_MAX];
    cr__json_escape(computation_id, id_e, sizeof id_e);
    cr__json_escape(computation_version, ver_e, sizeof ver_e);
    int n = snprintf(buf, buflen,
        "{\"arithmetic\":{\"accumulation\":\"exact\",\"order_independent\":true,"
        "\"params\":{\"es\":3,\"frac_bits\":96,\"n\":16,\"quire_bits\":256},"
        "\"profile\":\"bposit16-quire256\"},"
        "\"computation\":{\"id\":\"%s\",\"version\":\"%s\"},"
        "\"cr\":\"0.1\",\"digest_alg\":\"sha256\","
        "\"input\":{\"digest\":\"%s\",\"n_tensors\":%d},"
        "\"model\":{\"digest\":\"" CR_EMPTY_COLLECTION_DIGEST "\",\"n_tensors\":0},"
        "\"output\":{\"digest\":\"%s\",\"shape\":[%d]}}",
        id_e, ver_e, input_digest, n_inputs,
        output_digest, output_len);

    sha256_t s; sha256_init(&s);
    sha256_update(&s, buf, cr__wrote(n, buflen));
    char hex[65]; sha256_hex(&s, hex);
    snprintf(certificate, 72, "sha256:%s", hex);
}

/*
 * ---------------------------------------------------------------------------
 * CR v0.1.1 sampled receipts (spec §10) — the part most likely to diverge
 * between implementations, and therefore the part most worth proving.
 *
 * The verifier re-executes only a sample of output units. The sample indices
 * are NOT chosen by the prover: they are a PRF of the manifest-minus-sample
 * (which already commits the FULL output digest) and an optional challenge.
 * Rule `sha256-ctr-reject-v1`:
 *   seed    = SHA256( canonical(manifest_without_sample) || challenge )
 *   draw    = first 8 bytes of SHA256(seed || ctr_be8), big-endian u64
 *   reject  >= floor(2^64 / n) * n   (modulo-bias rejection)
 *   index   = draw mod n, skipping repeats, until `size` distinct, sorted asc
 * ---------------------------------------------------------------------------
 */

/* raw (non-hex) SHA-256 digest, needed for the PRF seed */
static void cr_sha256_raw(const void *data, size_t n, uint8_t out[32])
{
    sha256_t s; sha256_init(&s); sha256_update(&s, data, n);
    char hex[65]; sha256_hex(&s, hex);
    for (int i = 0; i < 32; i++) {
        unsigned v; sscanf(hex + 2 * i, "%2x", &v); out[i] = (uint8_t)v;
    }
}

/* sample-without-replacement per SAMPLE_RULE; returns count written. */
static int cr_sampled_indices(int n_units, int size, const uint8_t seed[32],
                              int *out)
{
    if (size < 1 || n_units < 1 || size > n_units) return -1;
    /* Rejection bound = floor(2^64 / n) * n (spec). When n is a power of two, 2^64 is
     * an exact multiple: there is no modulo bias and NOTHING is rejected — but that
     * bound equals 2^64, which does not fit in a uint64, so it must be special-cased.
     * For every non-power-of-two n, floor((2^64-1)/n) == floor(2^64/n), so the
     * expression below is exact.
     * BUG FIXED 2026-08-08 (open-cr pre-release audit): the previous
     * (2^64-1)/n*n wrongly rejected the top n draw-values for power-of-two n (16, 32,
     * 256, 1024, ... — common ML batch sizes), diverging from the Python reference and
     * so from the spec. The collision is ~n/2^64 per draw (unreachable by fuzzing) but
     * a real conformance divergence: same seed must give same indices on every impl. */
    const int cr__pow2 = (n_units & (n_units - 1)) == 0;
    /* n_units >= 1 is guaranteed by the guard above, so (uint64_t)n_units is exact;
     * UINT64_MAX keeps both operands uint64_t so the build is -Wconversion-clean. */
    const uint64_t nu = (uint64_t)n_units;
    const uint64_t limit = cr__pow2 ? 0u : (UINT64_MAX / nu) * nu;
    int got = 0;
    uint64_t ctr = 0;
    while (got < size) {
        uint8_t msg[40], dig[32];
        memcpy(msg, seed, 32);
        for (int b = 0; b < 8; b++) msg[32 + b] = (uint8_t)(ctr >> (8 * (7 - b)));
        ctr++;
        cr_sha256_raw(msg, sizeof msg, dig);
        uint64_t v = 0;
        for (int b = 0; b < 8; b++) v = (v << 8) | dig[b];
        if (!cr__pow2 && v >= limit) continue;       /* modulo-bias rejection (none for pow2) */
        int idx = (int)(v % (uint64_t)n_units);
        int dup = 0;
        for (int k = 0; k < got; k++) if (out[k] == idx) { dup = 1; break; }
        if (!dup) out[got++] = idx;
    }
    for (int i = 1; i < got; i++) {                  /* sort ascending */
        int key = out[i], j = i - 1;
        while (j >= 0 && out[j] > key) { out[j + 1] = out[j]; j--; }
        out[j + 1] = key;
    }
    return got;
}

/* hex-encode a challenge for the manifest (empty challenge -> empty string) */
static void cr_hex(const uint8_t *p, size_t n, char *out)
{
    for (size_t i = 0; i < n; i++) sprintf(out + 2 * i, "%02x", p[i]);
    out[2 * n] = 0;
}

/*
 * Build a v0.1.1 sampled manifest. `sample_digest` is the tensor digest of
 * output[indices]; the caller derives the indices with cr_sample_seed +
 * cr_sampled_indices and slices its own output accordingly.
 * Key order: ... output < sample (o < s), and within sample:
 *   challenge < digest < n_units < rule < size
 */
static void cr_build_sampled_manifest(char *buf, size_t buflen,
                                      const char *computation_id,
                                      const char *computation_version,
                                      const char *input_digest, int n_inputs,
                                      const char *output_digest, int output_len,
                                      int n_units, int size,
                                      const char *challenge_hex,
                                      const char *sample_digest,
                                      char certificate[72])
{
    char id_e[CR_JSON_ESC_MAX], ver_e[CR_JSON_ESC_MAX];
    cr__json_escape(computation_id, id_e, sizeof id_e);
    cr__json_escape(computation_version, ver_e, sizeof ver_e);
    int n = snprintf(buf, buflen,
        "{\"arithmetic\":{\"accumulation\":\"exact\",\"order_independent\":true,"
        "\"params\":{\"es\":3,\"frac_bits\":96,\"n\":16,\"quire_bits\":256},"
        "\"profile\":\"bposit16-quire256\"},"
        "\"computation\":{\"id\":\"%s\",\"version\":\"%s\"},"
        "\"cr\":\"0.1.1\",\"digest_alg\":\"sha256\","
        "\"input\":{\"digest\":\"%s\",\"n_tensors\":%d},"
        "\"model\":{\"digest\":\"" CR_EMPTY_COLLECTION_DIGEST "\",\"n_tensors\":0},"
        "\"output\":{\"digest\":\"%s\",\"shape\":[%d]},"
        "\"sample\":{\"challenge\":\"%s\",\"digest\":\"%s\",\"n_units\":%d,"
        "\"rule\":\"sha256-ctr-reject-v1\",\"size\":%d}}",
        id_e, ver_e, input_digest, n_inputs,
        output_digest, output_len, challenge_hex, sample_digest, n_units, size);

    sha256_t s; sha256_init(&s);
    sha256_update(&s, buf, cr__wrote(n, buflen));
    char hex[65]; sha256_hex(&s, hex);
    snprintf(certificate, 72, "sha256:%s", hex);
}

/* The seed commits everything EXCEPT the sample section — build that manifest
 * (cr = 0.1.1, no "sample" key) and hash it with the challenge appended. */
static void cr_sample_seed(const char *computation_id, const char *computation_version,
                           const char *input_digest, int n_inputs,
                           const char *output_digest, int output_len,
                           const uint8_t *challenge, size_t challenge_len,
                           uint8_t seed[32])
{
    char base[1024];
    char id_e[CR_JSON_ESC_MAX], ver_e[CR_JSON_ESC_MAX];
    cr__json_escape(computation_id, id_e, sizeof id_e);
    cr__json_escape(computation_version, ver_e, sizeof ver_e);
    int n = snprintf(base, sizeof base,
        "{\"arithmetic\":{\"accumulation\":\"exact\",\"order_independent\":true,"
        "\"params\":{\"es\":3,\"frac_bits\":96,\"n\":16,\"quire_bits\":256},"
        "\"profile\":\"bposit16-quire256\"},"
        "\"computation\":{\"id\":\"%s\",\"version\":\"%s\"},"
        "\"cr\":\"0.1.1\",\"digest_alg\":\"sha256\","
        "\"input\":{\"digest\":\"%s\",\"n_tensors\":%d},"
        "\"model\":{\"digest\":\"" CR_EMPTY_COLLECTION_DIGEST "\",\"n_tensors\":0},"
        "\"output\":{\"digest\":\"%s\",\"shape\":[%d]}}",
        id_e, ver_e, input_digest, n_inputs,
        output_digest, output_len);

    size_t bn = cr__wrote(n, sizeof base);          /* never the would-be length */
    uint8_t *msg = (uint8_t *)malloc(bn + challenge_len);
    memcpy(msg, base, bn);
    if (challenge_len) memcpy(msg + bn, challenge, challenge_len);
    cr_sha256_raw(msg, bn + challenge_len, seed);
    free(msg);
}

/*
 * ---------------------------------------------------------------------------
 * CR v0.1.2 chunked receipts / receipt chains (spec §12).
 * Chunk k is a full receipt over that chunk plus:
 *   "chunk":{"closing":false,"index":k,"prev_certificate":"<cert of k-1>"}
 * Key order within chunk: closing < index < prev_certificate; and the chunk
 * key sorts after "cr" but before "digest_alg" (c-h-u < c-r is false: "chunk"
 * < "computation" < "cr"), so it is emitted FIRST of the c-keys.
 * ---------------------------------------------------------------------------
 */
static void cr_build_chunk_manifest(char *buf, size_t buflen,
                                    const char *computation_id,
                                    const char *computation_version,
                                    const char *input_digest, int n_inputs,
                                    const char *output_digest, int output_len,
                                    int chunk_index, const char *prev_certificate,
                                    char certificate[72])
{
    char id_e[CR_JSON_ESC_MAX], ver_e[CR_JSON_ESC_MAX];
    cr__json_escape(computation_id, id_e, sizeof id_e);
    cr__json_escape(computation_version, ver_e, sizeof ver_e);
    int n = snprintf(buf, buflen,
        "{\"arithmetic\":{\"accumulation\":\"exact\",\"order_independent\":true,"
        "\"params\":{\"es\":3,\"frac_bits\":96,\"n\":16,\"quire_bits\":256},"
        "\"profile\":\"bposit16-quire256\"},"
        "\"chunk\":{\"closing\":false,\"index\":%d,\"prev_certificate\":\"%s\"},"
        "\"computation\":{\"id\":\"%s\",\"version\":\"%s\"},"
        "\"cr\":\"0.1.2\",\"digest_alg\":\"sha256\","
        "\"input\":{\"digest\":\"%s\",\"n_tensors\":%d},"
        "\"model\":{\"digest\":\"" CR_EMPTY_COLLECTION_DIGEST "\",\"n_tensors\":0},"
        "\"output\":{\"digest\":\"%s\",\"shape\":[%d]}}",
        chunk_index, prev_certificate, id_e, ver_e,
        input_digest, n_inputs, output_digest, output_len);

    sha256_t s; sha256_init(&s);
    sha256_update(&s, buf, cr__wrote(n, buflen));
    char hex[65]; sha256_hex(&s, hex);
    snprintf(certificate, 72, "sha256:%s", hex);
}

/* Build the closing receipt that terminates a chain (spec §12): a full receipt over the
 * TOTAL output plus a closing chunk section binding the chunk count and the chain digest.
 * Mirrors the reference build_closing_receipt for a chain whose chunks carry model=None
 * (model/input here are the empty collection, as build_closing_receipt emits). Chunk-section
 * key order (sorted): chain_digest < closing < index < n_chunks < prev_certificate. */
static void cr_build_closing_manifest(char *buf, size_t buflen,
                                      const char *computation_id,
                                      const char *computation_version,
                                      const char *output_digest, int output_len,
                                      int n_chunks, const char *prev_certificate,
                                      const char *chain_digest,
                                      char certificate[72])
{
    char id_e[CR_JSON_ESC_MAX], ver_e[CR_JSON_ESC_MAX];
    cr__json_escape(computation_id, id_e, sizeof id_e);
    cr__json_escape(computation_version, ver_e, sizeof ver_e);
    int n = snprintf(buf, buflen,
        "{\"arithmetic\":{\"accumulation\":\"exact\",\"order_independent\":true,"
        "\"params\":{\"es\":3,\"frac_bits\":96,\"n\":16,\"quire_bits\":256},"
        "\"profile\":\"bposit16-quire256\"},"
        "\"chunk\":{\"chain_digest\":\"%s\",\"closing\":true,\"index\":%d,"
        "\"n_chunks\":%d,\"prev_certificate\":\"%s\"},"
        "\"computation\":{\"id\":\"%s\",\"version\":\"%s\"},"
        "\"cr\":\"0.1.2\",\"digest_alg\":\"sha256\","
        "\"input\":{\"digest\":\"" CR_EMPTY_COLLECTION_DIGEST "\",\"n_tensors\":0},"
        "\"model\":{\"digest\":\"" CR_EMPTY_COLLECTION_DIGEST "\",\"n_tensors\":0},"
        "\"output\":{\"digest\":\"%s\",\"shape\":[%d]}}",
        chain_digest, n_chunks, n_chunks, prev_certificate, id_e, ver_e,
        output_digest, output_len);

    sha256_t s; sha256_init(&s);
    sha256_update(&s, buf, cr__wrote(n, buflen));
    char hex[65]; sha256_hex(&s, hex);
    snprintf(certificate, 72, "sha256:%s", hex);
}

/* chain digest = SHA256 over canonical({"certificate":..}) for each link in order */
static void cr_chain_digest(const char (*certs)[72], int count, char out[72])
{
    sha256_t s; sha256_init(&s);
    for (int i = 0; i < count; i++) {
        char ent[128];
        int n = snprintf(ent, sizeof ent, "{\"certificate\":\"%s\"}", certs[i]);
        sha256_update(&s, ent, cr__wrote(n, sizeof ent));
    }
    char hex[65]; sha256_hex(&s, hex);
    snprintf(out, 72, "sha256:%s", hex);
}

/* Emit the receipt envelope. `meta` is untrusted provenance and is deliberately
 * NOT covered by the certificate (CR §5) — otherwise a receipt produced on this
 * card could never verify on another machine, which is the whole point. */
static void cr_print_receipt(FILE *f, const char *manifest, const char *certificate,
                             const char *host_note)
{
    /* host_note is a free-text string field, so it must be JSON-escaped like any other
     * caller string (found 2026-08-08, same class as the id/version escaping fix): a note
     * containing a quote/backslash/control char otherwise emits INVALID JSON that a
     * verifier's parser rejects. `certificate` is a sha256:<hex> literal and `manifest` is
     * an already-built JSON *value* (not a string), so neither needs escaping. */
    char note_e[CR_JSON_ESC_MAX];
    cr__json_escape(host_note, note_e, sizeof note_e);
    fprintf(f, "{\"certificate\":\"%s\",\"manifest\":%s,"
               "\"meta\":{\"emitter\":\"f2-cl_bposit-host\",\"note\":\"%s\"}}\n",
            certificate, manifest, note_e);
}

#endif /* CR_RECEIPT_H */
