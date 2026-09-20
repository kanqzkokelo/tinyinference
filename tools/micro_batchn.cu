// micro_batchn: validate + time tt_gemv_q4_0_batchn (P1, weights streamed once
// for N tokens, X in registers via __ldg, no smem) against:
//   - reference: N sequential tt_gemv_q4_0_v4 launches (bit-exactness target)
//   - current small-N engine path: tt_gemm_q4_0_prefill (N-independent cost)
//
// Build:
//   $HOME/mmcuda/bin/nvcc -O3 -gencode arch=compute_86,code=sm_86 \
//     -I$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/include \
//     -Iinclude -Xcompiler -fPIC -o build/micro_batchn tools/micro_batchn.cu \
//     kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
//     -L$HOME/mmcuda/lib -lcudart -lpthread -lm
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

extern "C" int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy,
                               int M, int K, cudaStream_t s);
extern "C" int tt_gemv_q4_0_batchn(const void *dW, const float *dX, float *dY,
                                   int M, int K, int N, cudaStream_t s);
extern "C" int tt_gemm_q4_0_prefill(const void *dW, const float *dX, float *dY,
                                    int M, int K, int N, cudaStream_t s);

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at line %d: %s\n", #x, __LINE__, \
            cudaGetErrorString(e_)); exit(2); } } while (0)

static uint32_t g_seed = 12345u;
static uint32_t lcg(void) { g_seed = g_seed * 1103515245u + 12345u; return g_seed >> 8; }

// Q4_0 row fabric: every 18-byte block gets a valid fp16 scale at bytes [0,1]
// (0x2C00 = 0.0625) and random nibble payload in bytes [2,18).
static void fill_q4(uint8_t *hW, size_t wbytes) {
    memset(hW, 0, wbytes);
    const size_t nb = wbytes / 18;
    for (size_t b = 0; b < nb; b++) {
        uint16_t scale = 0x2C00;
        memcpy(hW + b * 18, &scale, 2);
        for (int j = 2; j < 18; j++) hW[b * 18 + j] = (uint8_t)(lcg() & 0xFF);
    }
}

static double time_v4_seq(const void *dW, const float *dX, float *dY,
                          int M, int K, int N) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++)
        for (int c = 0; c < N; c++)
            tt_gemv_q4_0_v4(dW, dX + (size_t)c * K, dY + (size_t)c * M, M, K, 0);
    cudaDeviceSynchronize();
    double best = 1e18;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(a, 0);
        for (int c = 0; c < N; c++)
            tt_gemv_q4_0_v4(dW, dX + (size_t)c * K, dY + (size_t)c * M, M, K, 0);
        cudaEventRecord(b, 0); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best * 1000.0;
}

static double time_bn(const void *dW, const float *dX, float *dY,
                      int M, int K, int N) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) tt_gemv_q4_0_batchn(dW, dX, dY, M, K, N, 0);
    cudaDeviceSynchronize();
    double best = 1e18;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(a, 0);
        tt_gemv_q4_0_batchn(dW, dX, dY, M, K, N, 0);
        cudaEventRecord(b, 0); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best * 1000.0;
}

static double time_gemm(const void *dW, const float *dX, float *dY,
                        int M, int K, int N) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 3; i++) tt_gemm_q4_0_prefill(dW, dX, dY, M, K, N, 0);
    cudaDeviceSynchronize();
    double best = 1e18;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(a, 0);
        tt_gemm_q4_0_prefill(dW, dX, dY, M, K, N, 0);
        cudaEventRecord(b, 0); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best * 1000.0;
}

int main(void) {
    /* Exhaustive N=1..32: the 4/2/1 chunker had a latent rem==3 gap, and the old
     * sparse grid {1,2,4,8,16,32} never exercised N%4==3. */
    static const int ns[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                             28, 29, 30, 31, 32};
    static const int nns = (int)(sizeof(ns) / sizeof(ns[0]));
    static const int shapes[][2] = {
        {8192, 2048},   // llama/qwen FFN gate+up
        {2048, 8192},   // FFN down (K large; batch4 cannot do this)
        {2048, 2048},   // o-proj / llama q
        {4864, 896},    // qwen2.5-0.5B FFN
        {896, 896},     // qwen2.5-0.5B q-proj
    };
    const int nshapes = 5;
    printf("# micro_batchn: tt_gemv_q4_0_batchn vs N x tt_gemv_q4_0_v4 vs prefill GEMM\n");
    printf("# method: 3 warmup + 20 timed iters, best-of, CUDA events. Q4_0 random nibbles,\n");
    printf("#         fp16 scale 0x2C00. maxabs MUST be 0.0 (bit-exact), nonfinite MUST be 0.\n");
    for (int s = 0; s < nshapes; s++) {
        const int M = shapes[s][0], K = shapes[s][1];
        const size_t wb = (size_t)M * (K / 32) * 18;
        uint8_t *hW = (uint8_t *)malloc(wb);
        fill_q4(hW, wb);
        void *dW = NULL; float *dX = NULL, *dY = NULL, *dYref = NULL;
        CK(cudaMalloc(&dW, wb));
        CK(cudaMalloc(&dX, (size_t)32 * K * sizeof(float)));
        CK(cudaMalloc(&dY, (size_t)32 * M * sizeof(float)));
        CK(cudaMalloc(&dYref, (size_t)32 * M * sizeof(float)));
        CK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));
        float *hX = (float *)malloc((size_t)32 * K * sizeof(float));
        for (size_t i = 0; i < (size_t)32 * K; i++)
            hX[i] = (float)((int)(lcg() % 2001) - 1000) / 1000.0f;
        CK(cudaMemcpy(dX, hX, (size_t)32 * K * sizeof(float), cudaMemcpyHostToDevice));
        float *hY = (float *)malloc((size_t)32 * M * sizeof(float));
        float *hYr = (float *)malloc((size_t)32 * M * sizeof(float));

        printf("\n--- M=%d K=%d  weight_bytes=%.2f MB ---\n", M, K, wb / 1e6);
        for (int ni = 0; ni < nns; ni++) {
            const int N = ns[ni];
            /* Poison the output buffer first: a chunking bug that skips rows
             * would otherwise compare stale-correct values and pass. */
            CK(cudaMemset(dY, 0xCD, (size_t)32 * M * sizeof(float)));
            for (int c = 0; c < N; c++)
                tt_gemv_q4_0_v4(dW, dX + (size_t)c * K, dYref + (size_t)c * M, M, K, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy(hYr, dYref, (size_t)N * M * sizeof(float), cudaMemcpyDeviceToHost));
            int rc = tt_gemv_q4_0_batchn(dW, dX, dY, M, K, N, 0);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            CK(cudaMemcpy(hY, dY, (size_t)N * M * sizeof(float), cudaMemcpyDeviceToHost));
            double maxabs = 0.0; int nonfinite = 0;
            for (size_t i = 0; i < (size_t)N * M; i++) {
                if (!isfinite(hY[i]) || !isfinite(hYr[i])) nonfinite++;
                double d = fabs((double)hY[i] - (double)hYr[i]);
                if (d > maxabs) maxabs = d;
            }
            const double t_seq = time_v4_seq(dW, dX, dY, M, K, N);
            const double t_bn = time_bn(dW, dX, dY, M, K, N);
            const double t_gemm = time_gemm(dW, dX, dY, M, K, N);
            /* 4/2/1 chunker: 4s while rem>=4, one extra chunk for rem 3 or 2. */
            const int chunks = (N / 4) + ((N % 4) ? 1 : 0);
            const double bytes = (double)chunks * (double)wb;
            printf("N=%-3d rc=%-2d maxabs=%.6g nonfinite=%d | seq_v4=%8.1fus batchn=%8.1fus "
                   "gemm=%8.1fus | bn_per_tok=%6.2fus gbps_eff=%6.1f bn_vs_seq=%.2fx bn_vs_gemm=%.2fx\n",
                   N, rc, maxabs, nonfinite, t_seq, t_bn, t_gemm, t_bn / N,
                   bytes / (t_bn / 1e6) / 1e9, t_seq / t_bn, t_gemm / t_bn);
        }
        free(hX); free(hY); free(hYr); free(hW);
        cudaFree(dW); cudaFree(dX); cudaFree(dY); cudaFree(dYref);
    }
    return 0;
}

