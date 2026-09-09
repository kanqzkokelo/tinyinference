// Fused SwiGLU (V2 math) vs 2x V4 GEMV + epilogue allowance, llama gate/up shape.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

extern "C" {
extern int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
extern int tt_ffn_q4_0(const void *dGate, const void *dUp, const float *dx, float *dh, int M, int K, int act, cudaStream_t stream);
}

static float med2(const char *tag, int M, int K, bool fused, const void *dG, const void *dU, const float *dx, float *d1, float *d2) {
    for (int i = 0; i < 20; i++) { if (fused) tt_ffn_q4_0(dG, dU, dx, d1, M, K, 0, 0); else { tt_gemv_q4_0_v4(dG, dx, d1, M, K, 0); tt_gemv_q4_0_v4(dU, dx, d2, M, K, 0); } }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> ms;
    for (int i = 0; i < 200; i++) {
        CK(cudaEventRecord(a));
        if (fused) tt_ffn_q4_0(dG, dU, dx, d1, M, K, 0, 0); else { tt_gemv_q4_0_v4(dG, dx, d1, M, K, 0); tt_gemv_q4_0_v4(dU, dx, d2, M, K, 0); }
        CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        float m = 0; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m*1000.0f);
    }
    std::sort(ms.begin(), ms.end());
    float med = ms[ms.size()/2];
    double wB = 2.0*(double)M*(K/32)*18.0;
    printf("%-14s M=%5d K=%5d median=%8.2f us eff-GB/s=%6.1f\n", tag, M, K, med, wB/(med/1e6)/1e9);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return med;
}

int main() {
    struct S { const char *n; int M, K; }; S sh[2] = {{"llama-gu",8192,2048},{"qwen3-gu",3072,1024}};
    for (int s = 0; s < 2; s++) {
        int M = sh[s].M, K = sh[s].K;
        size_t wb = (size_t)M*(K/32)*18;
        uint8_t *h = (uint8_t*)malloc(wb);
        uint32_t st = 4242; for (size_t i = 0; i < wb; i++) { st = st*1103515245u + 12345u; h[i] = (uint8_t)(st>>16); }
        void *dG = NULL, *dU = NULL; float *dx = NULL, *d1 = NULL, *d2 = NULL;
        CK(cudaMalloc(&dG, wb)); CK(cudaMalloc(&dU, wb));
        CK(cudaMalloc(&dx, (size_t)K*4)); CK(cudaMalloc(&d1, (size_t)M*4)); CK(cudaMalloc(&d2, (size_t)M*4));
        CK(cudaMemcpy(dG, h, wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dU, h, wb, cudaMemcpyHostToDevice));
        float *hx = (float*)malloc((size_t)K*4); for (int i = 0; i < K; i++) hx[i] = sinf(0.5f*i);
        CK(cudaMemcpy(dx, hx, (size_t)K*4, cudaMemcpyHostToDevice));
        float f = med2("fused-V2", M, K, true, dG, dU, dx, d1, d2);
        float v = med2("2xV4", M, K, false, dG, dU, dx, d1, d2);
        printf("%s fused/2xV4 = %.3f (fused saves one x-pass + SiLU kernel)\n", sh[s].n, f/v);
        cudaFree(dG); cudaFree(dU); cudaFree(dx); cudaFree(d1); cudaFree(d2); free(h); free(hx);
    }
    printf("done\n"); return 0;
}
