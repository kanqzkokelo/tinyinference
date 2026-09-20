// micro_prefill_smalln: time prefill GEMM kernels at verify sizes.
// tt_gemm_q4_0_prefill (cuda-core) vs tt_gemm_wmma_q4_0_prefill (wmma)
// at M=8192 K=2048 (llama gate) and M=2048 K=8192 (llama down), N=1..8.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
extern "C" int tt_gemm_q4_0_prefill(const void *dW, const float *dX, float *dY,
    int M, int K, int N, cudaStream_t s);
extern "C" int tt_gemm_wmma_q4_0_prefill(const void *dW, const float *dX, float *dY,
    int M, int K, int N, cudaStream_t s);
static double tcall(int wmma, const void *dW, const float *dX, float *dY,
    int M, int K, int N) {
    for (int i = 0; i < 10; i++) {
        if (wmma) tt_gemm_wmma_q4_0_prefill(dW, dX, dY, M, K, N, 0);
        else tt_gemm_q4_0_prefill(dW, dX, dY, M, K, N, 0);
    }
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    double best = 1e18;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a, 0);
        if (wmma) tt_gemm_wmma_q4_0_prefill(dW, dX, dY, M, K, N, 0);
        else tt_gemm_q4_0_prefill(dW, dX, dY, M, K, N, 0);
        cudaEventRecord(b, 0); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best * 1000.0;
}
extern "C" int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy,
    int M, int K, cudaStream_t s);
extern "C" int tt_gemv_q4_0_batch4(const void *dW, const float *dX4, float *dY4,
    int M, int K, cudaStream_t s);
static double tseq4(const void *dW, const float *dX4, float *dY4, int M, int K) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; i++)
        for (int c = 0; c < 4; c++)
            tt_gemv_q4_0_v4(dW, dX4 + (size_t)c * K, dY4 + (size_t)c * M, M, K, 0);
    cudaDeviceSynchronize();
    double best = 1e18;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a, 0);
        for (int c = 0; c < 4; c++)
            tt_gemv_q4_0_v4(dW, dX4 + (size_t)c * K, dY4 + (size_t)c * M, M, K, 0);
        cudaEventRecord(b, 0); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best * 1000.0;
}
static double tb4(const void *dW, const float *dX4, float *dY4, int M, int K, int *rc) {
    *rc = tt_gemv_q4_0_batch4(dW, dX4, dY4, M, K, 0);
    if (*rc) return -1;
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) { *rc = -99; return -1; }
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    double best = 1e18;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a, 0);
        tt_gemv_q4_0_batch4(dW, dX4, dY4, M, K, 0);
        cudaEventRecord(b, 0); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best * 1000.0;
}
int main(void) {
    static const int ns[] = {1, 2, 4, 8, 16, 32};
    static const int shapes[][2] = {{8192, 2048}, {2048, 8192}};
    for (int s = 0; s < 2; s++) {
        int M = shapes[s][0], K = shapes[s][1];
        size_t wb = (size_t)M * (K / 32) * 18;
        void *dW = NULL;
        float *dX = NULL, *dY = NULL;
        cudaMalloc(&dW, wb);
        cudaMalloc(&dX, (size_t)32 * K * sizeof(float));
        cudaMalloc(&dY, (size_t)32 * M * sizeof(float));
        cudaMemset(dW, 0x11, wb);
        for (int ni = 0; ni < 6; ni++) {
            int N = ns[ni];
            double t0 = tcall(0, dW, dX, dY, M, K, N);
            double t1 = tcall(1, dW, dX, dY, M, K, N);
            printf("M=%d K=%d N=%d cudacore_us=%.1f wmma_us=%.1f\n",
                M, K, N, t0, t1);
        }
        cudaFree(dW); cudaFree(dX); cudaFree(dY);
    }
    static const int bs[][2] = {{8192, 2048}, {2048, 2048}, {2048, 8192}};
    for (int s = 0; s < 3; s++) {
        int M = bs[s][0], K = bs[s][1];
        size_t wb = (size_t)M * (K / 32) * 18;
        void *dW = NULL;
        float *dX4 = NULL, *dY4 = NULL;
        cudaMalloc(&dW, wb);
        cudaMalloc(&dX4, (size_t)4 * K * sizeof(float));
        cudaMalloc(&dY4, (size_t)4 * M * sizeof(float));
        cudaMemset(dW, 0x11, wb);
        double ts = tseq4(dW, dX4, dY4, M, K);
        int rc = 0;
        double tb = tb4(dW, dX4, dY4, M, K, &rc);
        double gbps = wb / (tb / 1e6) / 1e9;
        printf("B4 M=%d K=%d seq4_us=%.1f b4_us=%.1f rc=%d gbps=%.1f speedup=%.2f\n",
            M, K, ts, tb, rc, rc ? -1 : gbps, rc ? -1 : ts / tb);
        cudaFree(dW); cudaFree(dX4); cudaFree(dY4);
    }
    return 0;
}
