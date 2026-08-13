/* Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
 *
 * cr_chain_emit_demo.c — emit a receipt CHAIN from C, the way an on-card prover would.
 *
 * The C header is an EMITTER: a chip/FPGA prover builds receipts, a host verifies them. This
 * demonstrates the emitter can build a whole §12 chain (chunk receipts + a closing receipt) whose
 * certificates are byte-identical to the Python reference, so a silicon prover can attest a
 * multi-step on-card computation as one tamper-evident chain that a different machine (Python or the
 * JS verifier) then re-verifies. It prints a cr-chain BUNDLE to stdout:
 *   {"kind":"cr-chain","cr_chain_version":"0.1.2","receipts":[<chunk0>,<chunk1>,<closing>]}
 * Each step's inputs/outputs are small int32 tensors; the Python cross-check rebuilds the identical
 * chain and asserts every certificate matches, then verifies the bundle with verify_chain.
 *
 * Build:  cc -std=c11 -O2 -o cr_chain_emit_demo cr_chain_emit_demo.c
 * Run:    ./cr_chain_emit_demo
 */
#include <stdio.h>
#include <string.h>
#include "cr_receipt.h"

#define NSTEPS 2
#define COMP_ID "c.chain.demo"
#define COMP_VER "1"

/* little-endian raw bytes of an int32 array (the CR tensor byte order) */
static void i32le(const int *v, int n, unsigned char *out) {
    for (int i = 0; i < n; i++) {
        unsigned int u = (unsigned int)v[i];
        out[4*i+0] = u & 0xff; out[4*i+1] = (u>>8) & 0xff;
        out[4*i+2] = (u>>16) & 0xff; out[4*i+3] = (u>>24) & 0xff;
    }
}

int main(void) {
    char certs[NSTEPS][72];
    char manifests[NSTEPS][2048];
    int  out_lens[NSTEPS];
    char total_out_digest[72];
    int  total_len = 0;

    char prev[72]; prev[0] = 0;                 /* chunk 0 prev_certificate = "" */

    printf("{\"cr_chain_version\":\"0.1.2\",\"kind\":\"cr-chain\",\"receipts\":[");

    for (int t = 0; t < NSTEPS; t++) {
        int x[3] = { t, t+1, t+2 };             /* input tensor {"x": x}  (i4, len 3) */
        int y[2] = { 10*t, 10*t + 1 };          /* output tensor          (i4, len 2) */
        out_lens[t] = 2;

        unsigned char xb[12], yb[8];
        i32le(x, 3, xb); i32le(y, 2, yb);

        int shp3[1] = {3}, shp2[1] = {2};
        char x_dig[72], y_dig[72], in_dig[72];
        cr_digest_tensor("i4", shp3, 1, xb, sizeof xb, x_dig);
        cr_digest_tensor("i4", shp2, 1, yb, sizeof yb, y_dig);
        const char *names[1] = { "x" }; const char *digs[1] = { x_dig };
        cr_digest_named(names, digs, 1, in_dig);       /* input {"x": x} */

        cr_build_chunk_manifest(manifests[t], sizeof manifests[t], COMP_ID, COMP_VER,
                                in_dig, 1, y_dig, out_lens[t], t, prev, certs[t]);
        printf("%s{\"certificate\":\"%s\",\"manifest\":%s,\"meta\":{}}",
               t ? "," : "", certs[t], manifests[t]);
        strcpy(prev, certs[t]);
        /* the total output the closing receipt binds = the last step's output */
        strcpy(total_out_digest, y_dig); total_len = out_lens[t];
    }

    char chain_dig[72];
    cr_chain_digest(certs, NSTEPS, chain_dig);
    char closing[2048], closing_cert[72];
    cr_build_closing_manifest(closing, sizeof closing, COMP_ID, COMP_VER,
                              total_out_digest, total_len, NSTEPS, prev, chain_dig, closing_cert);
    printf(",{\"certificate\":\"%s\",\"manifest\":%s,\"meta\":{}}", closing_cert, closing);
    printf("]}\n");
    return 0;
}
