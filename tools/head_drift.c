// head_drift.c: Q6_K lm_head -> Q4_0 requant fidelity gate on REAL weights.
// Row-chunked: dequant orig chunk, requant rows to Q4_0, dequant back, then
// CPU dot products vs N_SAMPLES rmsnorm-like vectors (last 4 carry sparse
// outliers to stress clamp asymmetry). Reports top-1 agreement, drift
// stats, and drift-vs-margin ratio per sample.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dequant_ref.h"
#include "loader_gguf.h"
#include "quant_ref.h"

#define K_DIM 2048
#define CHUNK_ROWS 2048
#define N_SAMPLES 16

static unsigned int rng = 0x9e3779b9u;
static float frand01(void) {
    rng = rng * 1103515245u + 12345u;
    return (float)(rng >> 8) * (1.0f / 16777216.0f);
}

static float dot_f32(const float *a, const float *b, long n) {
    double s = 0.0;
    long i;
    for (i = 0; i < n; i++) s += (double)a[i] * (double)b[i];
    return (float)s;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "/home/mitesh/Storage/repos/nnfromscratch/data/testmodels/llama-3.2-1b-q4_0.gguf";
    GGUFModel *m;
    GGUFTensor *tw;
    long M, K = K_DIM, r, i;
    int s;
    size_t q6rb, q4rb;
    float *xs, *lo, *lq, *crow, *qrow;
    unsigned char *q4c;
    int agree = 0;
    double maxdrift = 0.0, sumdrift = 0.0, minratio = 1e30;
    long nlog;
    m = gguf_load(path);
    if (!m) { printf("FAIL: gguf_load\n"); return 2; }
    tw = gguf_get_tensor(m, "token_embd.weight");
    if (!tw || !tw->data) { printf("FAIL: no token_embd.weight\n"); return 2; }
    if (tw->type != 14) { printf("FAIL: head type=%d want 14\n", tw->type); return 2; }
    M = (long)(tw->size_bytes / ((K / 256) * 210));
    if (M * (K / 256) * 210 != (long)tw->size_bytes) { printf("FAIL: size math\n"); return 2; }
    printf("head rows=%ld K=%ld MB=%.1f\n", M, K, (double)tw->size_bytes / 1e6);
    q6rb = (size_t)(K / 256) * 210;
    q4rb = (size_t)(K / 32) * 18;
    xs = malloc((size_t)N_SAMPLES * K * sizeof(float));
    lo = malloc((size_t)N_SAMPLES * M * sizeof(float));
    lq = malloc((size_t)N_SAMPLES * M * sizeof(float));
    crow = malloc((size_t)CHUNK_ROWS * K * sizeof(float));
    qrow = malloc((size_t)CHUNK_ROWS * K * sizeof(float));
    q4c = malloc((size_t)CHUNK_ROWS * q4rb);
    if (!xs || !lo || !lq || !crow || !qrow || !q4c) { printf("FAIL: oom\n"); return 2; }
    for (s = 0; s < N_SAMPLES; s++) {
        float *x = xs + (long)s * K;
        long k;
        for (k = 0; k < K; k++) x[k] = (frand01() - 0.5f) * (s < 12 ? 6.0f : 2.0f);
        if (s >= 12) for (k = 0; k < 8; k++) x[(long)(frand01() * K)] = (k & 1) ? 25.0f : -25.0f;
    }
    if (argc > 2) {
        FILE *fx = fopen(argv[2], "rb");
        if (fx) {
            size_t got = fread(xs, sizeof(float), (size_t)N_SAMPLES * K, fx);
            int nreal = (int)(got / (size_t)K);
            fclose(fx);
            printf("real-xn vecs=%d\n", nreal);
            if (nreal > 0) {
                for (s = nreal; s < N_SAMPLES; s++)
                    memcpy(xs + (long)s * K, xs + (long)(s % nreal) * K,
                           (size_t)K * sizeof(float));
            }
        }
    }
    for (r = 0; r < M; r += CHUNK_ROWS) {
        long nr = M - r < CHUNK_ROWS ? M - r : CHUNK_ROWS;
        const unsigned char *raw = (const unsigned char *)tw->data + (size_t)r * q6rb;
        if (ttq_dequant(raw, 14, nr * K, crow) != nr * K) { printf("FAIL: dq orig\n"); return 2; }
        for (i = 0; i < nr; i++)
            if (quantize_row_q4_0(crow + i * K, q4c + (size_t)i * q4rb, K) != (long)q4rb) {
                printf("FAIL: quant row %ld\n", r + i);
                return 2;
            }
        if (ttq_dequant(q4c, TTQ_Q4_0, nr * K, qrow) != nr * K) { printf("FAIL: dq q4\n"); return 2; }
        for (s = 0; s < N_SAMPLES; s++) {
            const float *x = xs + (long)s * K;
            float *po = lo + (long)s * M + r;
            float *pq = lq + (long)s * M + r;
            for (i = 0; i < nr; i++) {
                po[i] = dot_f32(crow + i * K, x, K);
                pq[i] = dot_f32(qrow + i * K, x, K);
            }
        }
        if ((r / CHUNK_ROWS) % 16 == 0) { printf("chunk %ld/%ld\n", r, M); fflush(stdout); }
    }
    nlog = (long)N_SAMPLES * M;
    for (s = 0; s < N_SAMPLES; s++) {
        const float *po = lo + (long)s * M;
        const float *pq = lq + (long)s * M;
        long bo = 0, bq = 0;
        float m1 = -1e30f, m2 = -1e30f;
        for (i = 0; i < M; i++) {
            double e = fabs((double)po[i] - (double)pq[i]);
            if (e > maxdrift) maxdrift = e;
            sumdrift += e;
            if (po[i] > m1) { m2 = m1; m1 = po[i]; bo = i; }
            else if (po[i] > m2) { m2 = po[i]; }
            if (pq[i] > (bq == 0 && i == 0 ? -1e30f : pq[bq])) bq = i;
        }
        if (bo == bq) agree++;
        {
            double margin = (double)m1 - (double)m2;
            double ratio = margin > 0 ? maxdrift / margin : 1e30;
            if (ratio < minratio) minratio = ratio;
            printf("s=%d argmax_o=%ld argmax_q=%ld %s margin=%.3f\n",
                s, bo, bq, bo == bq ? "OK" : "FLIP", margin);
        }
    }
    printf("agree=%d/%d maxdrift=%.4f meandrift=%.5f\n", agree, N_SAMPLES, maxdrift, sumdrift / nlog);
    gguf_free(m);
    return agree == N_SAMPLES ? 0 : 1;
}
