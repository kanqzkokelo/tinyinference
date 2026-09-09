// test_quant_ref.c: roundtrip gate for the Q4_0 reference quantizer.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "dequant_ref.h"
#include "quant_ref.h"

static unsigned int rng = 0x12345678u;
static float frand01(void) {
    rng = rng * 1103515245u + 12345u;
    return (float)(rng >> 8) * (1.0f / 16777216.0f);
}

static int fails = 0;
static long nblocks = 0;
static double worst = 0.0;

static void check_block(const float *x, const char *tag) {
    unsigned char q[18];
    float y[32];
    float amax = 0.0f;
    int j;
    long bw = quantize_row_q4_0(x, q, 32);
    long n;
    if (bw != 18) {
        printf("FAIL %s: bytes=%ld\n", tag, bw);
        fails++;
        return;
    }
    n = ttq_dequant(q, TTQ_Q4_0, 32, y);
    if (n != 32) {
        printf("FAIL %s: dequant=%ld\n", tag, n);
        fails++;
        return;
    }
    for (j = 0; j < 32; j++) {
        float a = x[j] < 0.0f ? -x[j] : x[j];
        if (a > amax) amax = a;
    }
    for (j = 0; j < 32; j++) {
        double e = fabs((double)y[j] - (double)x[j]);
        double tol = (double)amax * (1.0 / 8.0 + 0.001) + 1e-7;
        double r = e / (tol > 0.0 ? tol : 1.0);
        if (r > worst) worst = r;
        if (e > tol) {
            printf("FAIL %s j=%d x=%f y=%f tol=%f\n", tag, j, x[j], y[j], tol);
            fails++;
            return;
        }
    }
    nblocks++;
}

static void uniform_block(float *x, float lo, float hi) {
    int j;
    for (j = 0; j < 32; j++) x[j] = lo + (hi - lo) * frand01();
}

int main(void) {
    float x[32];
    float cs[5];
    float sc[3];
    unsigned u;
    int t;
    for (t = 0; t < 32; t++) x[t] = 0.0f;
    check_block(x, "zeros");
    cs[0] = 0.5f; cs[1] = -3.0f; cs[2] = 100.0f; cs[3] = -0.001f; cs[4] = 1e-6f;
    for (u = 0; u < 5; u++) {
        for (t = 0; t < 32; t++) x[t] = cs[u];
        check_block(x, "const");
    }
    sc[0] = 1e-3f; sc[1] = 1.0f; sc[2] = 40.0f;
    for (u = 0; u < 3; u++) {
        for (t = 0; t < 200; t++) { uniform_block(x, -sc[u], sc[u]); check_block(x, "sym"); }
        for (t = 0; t < 200; t++) { uniform_block(x, 0.0f, sc[u]); check_block(x, "asym-pos"); }
        for (t = 0; t < 200; t++) { uniform_block(x, -sc[u], 0.0f); check_block(x, "asym-neg"); }
    }
    for (t = 0; t < 100; t++) {
        int j;
        float s = (t & 1) ? 50.0f : -50.0f;
        for (j = 0; j < 32; j++) x[j] = (frand01() - 0.5f) * 2e-3f;
        x[t % 32] = s;
        check_block(x, "outlier");
    }
    for (t = 0; t < 50; t++) {
        int j;
        for (j = 0; j < 32; j++) x[j] = (j & 1) ? 7.0f : -7.0f;
        check_block(x, "alternating");
    }
    if (quantize_row_q4_0(x, NULL, 32) != -1) { printf("FAIL null-y\n"); fails++; }
    {
        unsigned char q[64];
        if (quantize_row_q4_0(x, q, 33) != -1) { printf("FAIL bad-k\n"); fails++; }
    }
    printf("blocks=%ld worst_tol_ratio=%.3f fails=%d\n", nblocks, worst, fails);
    return fails != 0;
}
