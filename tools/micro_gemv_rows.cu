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

    float *hX = (float *)malloc((size_t)K * sizeof(float));
    for (int k = 0; k < K; k++) hX[k] = sinf(0.7f * k + 0.3f);
    float *dx = NULL, *dy_big = NULL;
    CK(cudaMalloc(&dx, (size_t)K * sizeof(float)));
    CK(cudaMemcpy(dx, hX, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dy_big, (size_t)lmM * sizeof(float)));

    float *dX4 = NULL, *dY4_big = NULL;
    CK(cudaMalloc(&dX4, (size_t)4 * K * sizeof(float)));
    CK(cudaMalloc(&dY4_big, (size_t)4 * lmM * sizeof(float)));
    CK(cudaMemcpy(dX4, hX, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX4 + K, hX, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX4 + 2 * K, hX, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dX4 + 3 * K, hX, (size_t)K * sizeof(float), cudaMemcpyHostToDevice));

    size_t rowbytes4 = (size_t)(K / 32) * 18;
    for (int i = 0; i < 3; i++) {
        int M = qms[i];
        const void *dW = dW4; // M rows from base cover M<=1536
        printf("--- class=Q-proj M=%d K=%d ---\n", M, K);
        bench_single("Q-proj", M, K, "tt_gemv_q4_0", tt_gemv_q4_0, dW, 0, 18, dx, dy_big);
        bench_single("Q-proj", M, K, "tt_gemv_q4_0_v4", tt_gemv_q4_0_v4, dW, 0, 18, dx, dy_big);
        bench_single("Q-proj", M, K, "tt_gemv_q4_0_dispatch", tt_gemv_q4_0_dispatch,
                     dW, 0, 18, dx, dy_big);
        bench_batch4("Q-proj", M, K, "tt_gemv_q4_0_batch4", tt_gemv_q4_0_batch4,
                     dW, 18, dX4, dY4_big);
        (void)rowbytes4;
    }

    printf("--- class=FFN M=%d K=%d ---\n", ffnM, K);
    bench_single("FFN", ffnM, K, "tt_gemv_q4_0", tt_gemv_q4_0, dW4, 0, 18, dx, dy_big);
    bench_single("FFN", ffnM, K, "tt_gemv_q4_0_v4", tt_gemv_q4_0_v4, dW4, 0, 18, dx, dy_big);
    bench_single("FFN", ffnM, K, "tt_gemv_q4_0_dispatch", tt_gemv_q4_0_dispatch,
                 dW4, 0, 18, dx, dy_big);
    bench_batch4("FFN", ffnM, K, "tt_gemv_q4_0_batch4", tt_gemv_q4_0_batch4,
                 dW4, 18, dX4, dY4_big);

    printf("--- class=LM-head M=%d K=%d ---\n", lmM, K);
    bench_single("LM-head", lmM, K, "tt_gemv_q4_0", tt_gemv_q4_0, dW4, 0, 18, dx, dy_big);
    bench_single("LM-head", lmM, K, "tt_gemv_q4_0_v4", tt_gemv_q4_0_v4, dW4, 0, 18, dx, dy_big);
    bench_single("LM-head", lmM, K, "tt_logits_q4_0", tt_logits_q4_0, dW4, 0, 18, dx, dy_big);
    bench_single("LM-head", lmM, K, "tt_logits_q4_0_v2", tt_logits_q4_0_v2, dW4, 0, 18, dx, dy_big);
    bench_single("LM-head", lmM, K, "tt_logits_q4_0_v4", tt_logits_q4_0_v4, dW4, 0, 18, dx, dy_big);
    bench_batch4("LM-head", lmM, K, "tt_logits_q4_0_batch4", tt_logits_q4_0_batch4,
                 dW4, 18, dX4, dY4_big);
    bench_single("LM-head", lmM, K, "tt_logits_q8_0", tt_logits_q8_0, dW8, 0, 34, dx, dy_big);
    bench_single("LM-head", lmM, K, "tt_logits_q8_0_v4", tt_logits_q8_0_v4, dW8, 0, 34, dx, dy_big);

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
