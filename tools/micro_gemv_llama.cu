// Task 2 microbench: time existing GEMV entry points per decode shape class.
// Q4_0 weights, K=896 (qwen2.5-0.5b dims). Links kernels/* like tests do;
// no kernel code here, only timing of public launchers.
//
// Effective bandwidth formula:
//   Q4_0 weight bytes per launch = M * (K/32) * 18  (16B qs + 2B scale / 32 elems)
//   Q8_0 weight bytes per launch = M * (K/32) * 34  (32B qs + 2B scale / 32 elems)
//   GB/s = weight_bytes / median_sec / 1e9  (x vector traffic ignored, tiny vs weights)
// Batch4 launchers stream weights once for 4 tokens; GB/s uses the same
// single-pass weight bytes, per-token us = launch_us / 4.
//
// Method: 20 warmup + 200 timed iters, per-iter CUDA events, median.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

extern "C" {
extern int tt_gemv_q4_0(const void *dW, const float *dx, float *dy,
                        int M, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy,
                           int M, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_dispatch(const void *dW, const float *dx, float *dy,
                                 int M, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_batch4(const void *dW, const float *dX_4xK, float *dY_4xM,
                               int M, int K, cudaStream_t stream);
extern int tt_logits_q4_0(const void *dW, const float *dx, float *dlogits,
                          int vocab, int K, cudaStream_t stream);
extern int tt_logits_q4_0_v2(const void *dW, const float *dx, float *dlogits,
                             int vocab, int K, cudaStream_t stream);
extern int tt_logits_q4_0_v4(const void *dW, const float *dx, float *dlogits,
                             int vocab, int K, cudaStream_t stream);
extern int tt_logits_q4_0_batch4(const void *dW, const float *dX_4xK,
                                  float *dL_4xVocab, int vocab, int K,
                                  cudaStream_t stream);
extern int tt_logits_dispatch(const void *dW, int dtype, const float *dx,
                              float *dlogits, int vocab, int K,
                              cudaStream_t stream);
extern int tt_logits_q8_0(const void *dW, const float *dx, float *dlogits,
                           int vocab, int K, cudaStream_t stream);
extern int tt_logits_q8_0_v4(const void *dW, const float *dx, float *dlogits,
                             int vocab, int K, cudaStream_t stream);
}

static const int WARMUP = 20;
static const int ITERS = 200;

// Fill device-ready random quantized rows on host. d bits = 1.0 fp16.
static void fill_q4(uint8_t *hW, size_t wbytes, unsigned seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < wbytes; i++) {
        s = s * 1103515245u + 12345u;
        hW[i] = (uint8_t)(s >> 16);
    }
    int nb_total = (int)(wbytes / 18);
    for (int b = 0; b < nb_total; b++) {
        hW[b * 18 + 0] = 0x00;
        hW[b * 18 + 1] = 0x3C; // 1.0f
    }
}

static void fill_q8(uint8_t *hW, size_t wbytes, unsigned seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < wbytes; i++) {
        s = s * 1103515245u + 12345u;
        hW[i] = (uint8_t)(s >> 16);
    }
    int nb_total = (int)(wbytes / 34);
    for (int b = 0; b < nb_total; b++) {
        hW[b * 34 + 0] = 0x00;
        hW[b * 34 + 1] = 0x3C;
    }
}

typedef int (*single_fn)(const void *, const float *, float *, int, int, cudaStream_t);

static double bench_single(const char *cls, int M, int K, const char *entry,
                           single_fn fn, const void *dW, size_t wbytes_q,
                           int blksz, const float *dx, float *dy) {
    (void)wbytes_q;
    for (int i = 0; i < WARMUP; i++) {
        int rc = fn(dW, dx, dy, M, K, 0);
        if (rc != 0) {
            printf("[%-7s] M=%6d K=%4d entry=%-22s LAUNCH-FAIL rc=%d\n", cls, M, K, entry, rc);
            return -1.0;
        }
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> ms;
    ms.reserve(ITERS);
    for (int i = 0; i < ITERS; i++) {
        CK(cudaEventRecord(a, 0));
        int rc = fn(dW, dx, dy, M, K, 0);
        CK(cudaEventRecord(b, 0));
        CK(cudaEventSynchronize(b));
        if (rc != 0) {
            printf("[%-7s] M=%6d K=%4d entry=%-22s LAUNCH-FAIL rc=%d\n", cls, M, K, entry, rc);
            return -1.0;
        }
        float m = 0;
        CK(cudaEventElapsedTime(&m, a, b));
        ms.push_back(m);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    std::sort(ms.begin(), ms.end());
    double med_ms = ms[ITERS / 2];
    double med_us = med_ms * 1000.0;
    double wbytes = (double)M * (K / 32) * blksz;
    double gbs = wbytes / (med_ms / 1000.0) / 1e9;
    printf("[%-7s] M=%6d K=%4d entry=%-22s median=%9.2f us  GB/s=%7.2f\n",
           cls, M, K, entry, med_us, gbs);
    return med_us;
}

typedef int (*batch_fn)(const void *, const float *, float *, int, int, cudaStream_t);

static double bench_batch4(const char *cls, int M, int K, const char *entry,
                           batch_fn fn, const void *dW,
                           int blksz, const float *dX, float *dY) {
    for (int i = 0; i < WARMUP; i++) {
        int rc = fn(dW, dX, dY, M, K, 0);
        if (rc != 0) {
            printf("[%-7s] M=%6d K=%4d entry=%-22s LAUNCH-FAIL rc=%d\n", cls, M, K, entry, rc);
            return -1.0;
        }
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::vector<float> ms;
    ms.reserve(ITERS);
    for (int i = 0; i < ITERS; i++) {
        CK(cudaEventRecord(a, 0));
        int rc = fn(dW, dX, dY, M, K, 0);
        CK(cudaEventRecord(b, 0));
        CK(cudaEventSynchronize(b));
        if (rc != 0) {
            printf("[%-7s] M=%6d K=%4d entry=%-22s LAUNCH-FAIL rc=%d\n", cls, M, K, entry, rc);
            return -1.0;
        }
        float m = 0;
        CK(cudaEventElapsedTime(&m, a, b));
        ms.push_back(m);
    }
    CK(cudaEventDestroy(a));
    CK(cudaEventDestroy(b));
    std::sort(ms.begin(), ms.end());
    double med_ms = ms[ITERS / 2];
    double med_us = med_ms * 1000.0;
    double wbytes = (double)M * (K / 32) * blksz;
    double gbs = wbytes / (med_ms / 1000.0) / 1e9;
    printf("[%-7s] M=%6d K=%4d entry=%-22s median=%9.2f us/launch (%8.2f us/tok)  GB/s=%7.2f\n",
           cls, M, K, entry, med_us, med_us / 4.0, gbs);
    return med_us;
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s\n", p.name);
    printf("method: %d warmup + %d timed iters, per-iter CUDA events, median us\n", WARMUP, ITERS);
    printf("bw formula: Q4_0 bytes=M*(K/32)*18, Q8_0 bytes=M*(K/32)*34; GB/s=bytes/med_s/1e9\n");

    const int K = 896;
    const int qms[] = {896, 1152, 1536};
    const int ffnM = 4864;
    const int lmM = 151936;

    // Q4_0 weights: biggest shape first, slice per M via row offsets.
    size_t w_lm = (size_t)lmM * (K / 32) * 18;
    uint8_t *hW4 = (uint8_t *)malloc(w_lm);
    if (!hW4) { fprintf(stderr, "host OOM\n"); return 2; }
    fill_q4(hW4, w_lm, 0x1234u);
    uint8_t *dW4 = NULL;
    CK(cudaMalloc(&dW4, w_lm));
    CK(cudaMemcpy(dW4, hW4, w_lm, cudaMemcpyHostToDevice));

    size_t w8_lm = (size_t)lmM * (K / 32) * 34;
    uint8_t *hW8 = (uint8_t *)malloc(w8_lm);
    if (!hW8) { fprintf(stderr, "host OOM\n"); return 2; }
    fill_q8(hW8, w8_lm, 0x5678u);
    uint8_t *dW8 = NULL;
    CK(cudaMalloc(&dW8, w8_lm));
    CK(cudaMemcpy(dW8, hW8, w8_lm, cudaMemcpyHostToDevice));

    const int KMAX = 8192;
    float *hX = (float *)malloc((size_t)KMAX * sizeof(float));
    for (int k = 0; k < KMAX; k++) hX[k] = sinf(0.7f * k + 0.3f);
    float *dx = NULL, *dy_big = NULL;
    CK(cudaMalloc(&dx, (size_t)KMAX * sizeof(float)));
    CK(cudaMemcpy(dx, hX, (size_t)KMAX * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy_big, (size_t)lmM * sizeof(float)));

    float *dX4 = NULL, *dY4_big = NULL;
    CK(cudaMalloc(&dX4, (size_t)4 * KMAX * sizeof(float)));
    CK(cudaMalloc(&dY4_big, (size_t)4 * lmM * sizeof(float)));
    CK(cudaMemcpy(dX4, hX, (size_t)KMAX * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX4 + KMAX, hX, (size_t)KMAX * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX4 + 2 * KMAX, hX, (size_t)KMAX * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX4 + 3 * KMAX, hX, (size_t)KMAX * sizeof(float), cudaMemcpyHostToDevice));

    // Llama-3.2-1B decode GEMV shapes (dim=2048 hidden=8192).
    // dW4 base holds 76MB of random Q4 rows; every shape below fits inside.
    // dx holds KMAX floats; each bench reads the first K it needs.
    (void)qms; (void)ffnM; (void)lmM; (void)dW8; (void)dX4; (void)dY4_big;
    printf("--- llama gate/up M=8192 K=2048 ---\n");
    bench_single("gate", 8192, 2048, "tt_gemv_q4_0", tt_gemv_q4_0, dW4, 0, 18, dx, dy_big);
    bench_single("gate", 8192, 2048, "tt_gemv_q4_0_v4", tt_gemv_q4_0_v4, dW4, 0, 18, dx, dy_big);
    bench_single("gate", 8192, 2048, "tt_gemv_q4_0_dispatch", tt_gemv_q4_0_dispatch, dW4, 0, 18, dx, dy_big);
    printf("--- llama down M=2048 K=8192 ---\n");
    bench_single("down", 2048, 8192, "tt_gemv_q4_0", tt_gemv_q4_0, dW4, 0, 18, dx, dy_big);
    bench_single("down", 2048, 8192, "tt_gemv_q4_0_v4", tt_gemv_q4_0_v4, dW4, 0, 18, dx, dy_big);
    bench_single("down", 2048, 8192, "tt_gemv_q4_0_dispatch", tt_gemv_q4_0_dispatch, dW4, 0, 18, dx, dy_big);
    printf("--- llama o-proj M=2048 K=2048 ---\n");
    bench_single("oproj", 2048, 2048, "tt_gemv_q4_0", tt_gemv_q4_0, dW4, 0, 18, dx, dy_big);
    bench_single("oproj", 2048, 2048, "tt_gemv_q4_0_v4", tt_gemv_q4_0_v4, dW4, 0, 18, dx, dy_big);
    bench_single("oproj", 2048, 2048, "tt_gemv_q4_0_dispatch", tt_gemv_q4_0_dispatch, dW4, 0, 18, dx, dy_big);

    // Llama-3.2-1B LM head: V=128256 K=2048 (nb=64 even, V pct 4 == 0).
    // Own weight buffer: 128256*64*18 = ~148 MB.
    printf("--- llama lm-head V=128256 K=2048 ---\n");
    {
        const int V = 128256, KH = 2048;
        size_t wh = (size_t)V * (KH / 32) * 18;
        uint8_t *hWh = (uint8_t *)malloc(wh);
        if (!hWh) { fprintf(stderr, "host OOM head\n"); return 2; }
        fill_q4(hWh, wh, 0x9abcu);
        uint8_t *dWh = NULL;
        CK(cudaMalloc(&dWh, wh));
        CK(cudaMemcpy(dWh, hWh, wh, cudaMemcpyHostToDevice));
        float *dL = NULL;
        CK(cudaMalloc(&dL, (size_t)V * sizeof(float)));
        bench_single("lmhead", V, KH, "tt_logits_q4_0", tt_logits_q4_0, dWh, 0, 18, dx, dL);
        bench_single("lmhead", V, KH, "tt_logits_q4_0_v2", tt_logits_q4_0_v2, dWh, 0, 18, dx, dL);
        bench_single("lmhead", V, KH, "tt_logits_q4_0_v4", tt_logits_q4_0_v4, dWh, 0, 18, dx, dL);
        {
            for (int i = 0; i < WARMUP; i++) tt_logits_dispatch(dWh, 2, dx, dL, V, KH, 0);
            CK(cudaDeviceSynchronize());
            cudaEvent_t a, b;
            CK(cudaEventCreate(&a));
            CK(cudaEventCreate(&b));
            std::vector<float> ms;
            ms.reserve(ITERS);
            for (int i = 0; i < ITERS; i++) {
                CK(cudaEventRecord(a, 0));
                tt_logits_dispatch(dWh, 2, dx, dL, V, KH, 0);
                CK(cudaEventRecord(b, 0));
                CK(cudaEventSynchronize(b));
                float m = 0;
                CK(cudaEventElapsedTime(&m, a, b));
                ms.push_back(m);
            }
            CK(cudaEventDestroy(a));
            CK(cudaEventDestroy(b));
            std::sort(ms.begin(), ms.end());
            double med_ms = ms[ITERS / 2];
            double wbytes = (double)V * (KH / 32) * 18;
            double gbs = wbytes / (med_ms / 1000.0) / 1e9;
            printf("[lmhead ] V=%6d K=%4d entry=%-22s median=%9.2f us  GB/s=%7.2f\n",
                   V, KH, "tt_logits_dispatch", med_ms * 1000.0, gbs);
        }
        {
            float *hA = (float *)malloc((size_t)V * sizeof(float));
            float *hB = (float *)malloc((size_t)V * sizeof(float));
            float *hC = (float *)malloc((size_t)V * sizeof(float));
            tt_logits_q4_0(dWh, dx, dL, V, KH, 0); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hA, dL, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
            tt_logits_q4_0_v2(dWh, dx, dL, V, KH, 0); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hB, dL, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
            tt_logits_q4_0_v4(dWh, dx, dL, V, KH, 0); CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hC, dL, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
            double m12 = 0, m14 = 0;
            int bad = 0;
            for (int i = 0; i < V; i++) {
                double d12 = fabs((double)hA[i] - hB[i]);
                double d14 = fabs((double)hA[i] - hC[i]);
                if (d12 != d12 || d14 != d14) { bad = 1; break; }
                if (d12 > m12) m12 = d12;
                if (d14 > m14) m14 = d14;
            }
            printf("[lmhead ] exactness: bad=%d max|s-V2|=%.6g max|s-V4|=%.6g\n", bad, m12, m14);
            free(hA); free(hB); free(hC);
        }
        cudaFree(dWh);
        cudaFree(dL);
        free(hWh);
    }

    printf("done\n");

    cudaFree(dW4);
    cudaFree(dW8);
    cudaFree(dx);
    cudaFree(dy_big);
    cudaFree(dX4);
    cudaFree(dY4_big);
    free(hW4);
    free(hW8);
    free(hX);
    return 0;
}
