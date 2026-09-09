// Full-stack microbench: 16x Llama layer GEMV sequences (q,k,v,o,gate,up,down)
// with distinct weights, back-to-back. Predicts roofline step time ex-logits.
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
extern int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
}

static void *allocW(size_t b) { void *p = NULL; CK(cudaMalloc(&p, b)); return p; }
static void fillW(void *dW, size_t b, uint32_t seed) {
    uint8_t *h = (uint8_t*)malloc(b);
    uint32_t s = seed;
    for (size_t i = 0; i < b; i++) { s = s*1103515245u + 12345u; h[i] = (uint8_t)(s>>16); }
    CK(cudaMemcpy(dW, h, b, cudaMemcpyHostToDevice)); free(h); }

int main() {
    const int NL = 16;
    // Q:2048x2048 K:1024x2048 V:1024x2048 O:2048x2048 gate/up:8192x2048 down:2048x8192
    struct S { int M, K; }; S shapes[7] = {{2048,2048},{1024,2048},{1024,2048},{2048,2048},{8192,2048},{8192,2048},{2048,8192}};
    void *dW[7]; size_t wb[7];
    for (int i = 0; i < 7; i++) { wb[i] = (size_t)shapes[i].M*(shapes[i].K/32)*18; dW[i] = allocW(wb[i]); fillW(dW[i], wb[i], 1000+i); }
    double totB = 0; for (int i = 0; i < 7; i++) totB += (double)wb[i]*NL;
    printf("weight MB per step: %.1f\n", totB/1e6);
    float *dx = NULL, *dy = NULL;
    CK(cudaMalloc(&dx, (size_t)8192*4)); CK(cudaMalloc(&dy, (size_t)8192*4));
    float *hx = (float*)malloc(8192*4);
    for (int i = 0; i < 8192; i++) hx[i] = sinf(0.17f*i);
    CK(cudaMemcpy(dx, hx, 8192*4, cudaMemcpyHostToDevice));
    for (int w = 0; w < 10; w++)
        for (int l = 0; l < NL; l++)
            for (int i = 0; i < 7; i++)
                tt_gemv_q4_0_v4(dW[i], dx, dy, shapes[i].M, shapes[i].K, 0);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> ms;
    for (int it = 0; it < 100; it++) {
        CK(cudaEventRecord(a));
        for (int l = 0; l < NL; l++)
            for (int i = 0; i < 7; i++)
                tt_gemv_q4_0_v4(dW[i], dx, dy, shapes[i].M, shapes[i].K, 0);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float m = 0; CK(cudaEventElapsedTime(&m, a, b));
        ms.push_back(m*1000.0f);
    }
    std::sort(ms.begin(), ms.end());
    float med = ms[ms.size()/2];
    printf("16-layer stack median=%8.2f us  eff-GB/s=%.1f\n", med, totB/(med/1e6)/1e9);
    printf("done\n");
    return 0;
}
