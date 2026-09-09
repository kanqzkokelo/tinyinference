// Chain microbench: gate+up+down back-to-back with distinct weights,
// mimics engine order. Compares per-kernel medians vs isolated launches.
// If inflated -> L2/TLB churn or clock effect in-engine.
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
}

int main() {
    const int Mg = 8192, Kg = 2048, Md = 2048, Kd = 8192;
    size_t wbg = (size_t)Mg*(Kg/32)*18, wbd = (size_t)Md*(Kd/32)*18;
    uint8_t *hW = (uint8_t*)malloc(wbg);
    uint32_t st = 999;
    for (size_t i = 0; i < wbg; i++) { st = st*1103515245u + 12345u; hW[i] = (uint8_t)(st>>16); }
    void *dWg = NULL, *dWu = NULL, *dWd = NULL;
    CK(cudaMalloc(&dWg, wbg)); CK(cudaMalloc(&dWu, wbg)); CK(cudaMalloc(&dWd, wbd));
    CK(cudaMemcpy(dWg, hW, wbg, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWu, hW, wbg, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dWd, hW, wbd > wbg ? wbg : wbd, cudaMemcpyHostToDevice));
    float *dx = NULL, *dg = NULL, *du = NULL, *dd = NULL, *hx = NULL;
    CK(cudaMalloc(&dx, (size_t)8192*4)); CK(cudaMalloc(&dg, (size_t)8192*4));
    CK(cudaMalloc(&du, (size_t)8192*4)); CK(cudaMalloc(&dd, (size_t)2048*4));
    hx = (float*)malloc(8192*4);
    for (int i = 0; i < 8192; i++) hx[i] = sinf(0.31f*i);
    CK(cudaMemcpy(dx, hx, 8192*4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 20; i++) {
        tt_gemv_q4_0_v4(dWg, dx, dg, Mg, Kg, 0);
        tt_gemv_q4_0_v4(dWu, dx, du, Mg, Kg, 0);
        tt_gemv_q4_0_v4(dWd, dx, dd, Md, Kd, 0);
    }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    const char *nm[3] = {"gate","up","down"};
    for (int s = 0; s < 3; s++) {
        std::vector<float> ms;
        for (int i = 0; i < 200; i++) {
            if (s != 0) tt_gemv_q4_0_v4(dWg, dx, dg, Mg, Kg, 0);
            if (s != 1) tt_gemv_q4_0_v4(dWu, dx, du, Mg, Kg, 0);
            CK(cudaEventRecord(a));
            if (s == 0) tt_gemv_q4_0_v4(dWg, dx, dg, Mg, Kg, 0);
            if (s == 1) tt_gemv_q4_0_v4(dWu, dx, du, Mg, Kg, 0);
            if (s == 2) tt_gemv_q4_0_v4(dWd, dx, dd, Md, Kd, 0);
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float m = 0; CK(cudaEventElapsedTime(&m, a, b));
            ms.push_back(m*1000.0f);
        }
        std::sort(ms.begin(), ms.end());
        printf("chain-%s median=%8.2f us\n", nm[s], ms[ms.size()/2]);
    }
    printf("done\n");
    return 0;
}
