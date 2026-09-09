// Microbench: R8 (8-rows/warp) vs V4 (4-rows/warp) q4_0 layer GEMV.
// Llama-3.2-1B FFN shapes (gate/up M=8192,K=2048; down M=2048,K=8192).
// Method: 20 warmup + 200 timed iters, per-iter CUDA events, median.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

extern "C" {
extern int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_r8(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
}

static void bench(const char *tag, int M, int K,
                  int (*fn)(const void *, const float *, float *, int, int, cudaStream_t),
                  const void *dW, const float *dx, float *dy) {
    for (int i = 0; i < 20; i++) fn(dW, dx, dy, M, K, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i = 0; i < 200; i++) {
        CK(cudaEventRecord(a));
        fn(dW, dx, dy, M, K, 0);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float m = 0; CK(cudaEventElapsedTime(&m, a, b));
        ms.push_back(m * 1000.0f);
    }
    std::sort(ms.begin(), ms.end());
    float med = ms[ms.size()/2];
    double wbytes = (double)M * (K/32) * 18.0;
    double gbs = wbytes / (med/1e6) / 1e9;
    printf("%-16s M=%5d K=%5d median=%8.2f us GB/s=%7.1f\n", tag, M, K, med, gbs);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}

int main() {
    struct Shape { const char *name; int M, K; };
    Shape shapes[] = { {"llama-gate",8192,2048}, {"llama-down",2048,8192}, {"qwen3-gate",3072,1024}, {"qwen3-down",1024,3072} };
    for (int si = 0; si < 4; si++) {
        int M = shapes[si].M, K = shapes[si].K;
        size_t wb = (size_t)M * (K/32) * 18;
        uint8_t *hW = (uint8_t*)malloc(wb);
        uint32_t st = 12345;
        for (size_t i = 0; i < wb; i++) { st = st*1103515245u + 12345u; hW[i] = (uint8_t)(st>>16); }
        float *hX = (float*)malloc((size_t)K*4);
        for (int i = 0; i < K; i++) hX[i] = sinf(0.7f*i);
        void *dW = NULL; float *dx = NULL, *dyV = NULL, *dyR = NULL;
        CK(cudaMalloc(&dW, wb));
        CK(cudaMalloc(&dx, (size_t)K*4));
        CK(cudaMalloc(&dyV, (size_t)M*4));
        CK(cudaMalloc(&dyR, (size_t)M*4));
        CK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hX, (size_t)K*4, cudaMemcpyHostToDevice));
        bench(shapes[si].name, M, K, tt_gemv_q4_0_v4, dW, dx, dyV);
        bench(shapes[si].name, M, K, tt_gemv_q4_0_r8, dW, dx, dyR);
        std::vector<float> yV(M), yR(M);
        CK(cudaMemcpy(yV.data(), dyV, (size_t)M*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(yR.data(), dyR, (size_t)M*4, cudaMemcpyDeviceToHost));
        float md = 0; for (int i = 0; i < M; i++) md = std::max(md, fabsf(yV[i]-yR[i]));
        printf("%-16s maxdiff V4-vs-R8 = %g\n", shapes[si].name, md);
        cudaFree(dW); cudaFree(dx); cudaFree(dyV); cudaFree(dyR);
        free(hW); free(hX);
    }
    printf("done\n");
    return 0;
}

