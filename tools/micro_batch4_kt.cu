// micro_batch4_kt: kt2 (K-tiled x2) vs seq v4 at M=2048 K=8192 (llama down).
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
extern "C" int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t s);
extern "C" int tt_gemv_q4_0_batch4_kt2(const void *dW, const float *dX4, float *dY4, int M, int K, cudaStream_t s);
static double tevt(void (*fn)(void), int reps=30) { (void)fn; return 0; }
int main(void) {
    int M = 2048, K = 8192;
    size_t wb = (size_t)M * (K / 32) * 18;
    size_t xb = (size_t)4 * K * sizeof(float);
    size_t yb = (size_t)4 * M * sizeof(float);
    float *hX = (float*)malloc(xb), *hYref = (float*)malloc(yb), *hY = (float*)malloc(yb);
    srand(7);
    for (size_t i = 0; i < (size_t)4*K; i++) hX[i] = (float)(rand() % 2000 - 1000) / 1000.f;
    void *dW=NULL; float *dX=NULL, *dY=NULL;
    cudaMalloc(&dW, wb); cudaMalloc(&dX, xb); cudaMalloc(&dY, yb);
    cudaMemset(dW, 0x11, wb);
    cudaMemcpy(dX, hX, xb, cudaMemcpyHostToDevice);
    // ref: 4x v4
    for (int c=0;c<4;c++) tt_gemv_q4_0_v4(dW, dX+(size_t)c*K, dY+(size_t)c*M, M, K, 0);
    cudaDeviceSynchronize();
    cudaMemcpy(hYref, dY, yb, cudaMemcpyDeviceToHost);
    int rc = tt_gemv_q4_0_batch4_kt2(dW, dX, dY, M, K, 0);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    printf("launch rc=%d cuda=%d (%s)\n", rc, (int)e, cudaGetErrorString(e));
    if (rc==0 && e==cudaSuccess) {
        cudaMemcpy(hY, dY, yb, cudaMemcpyDeviceToHost);
        double mx=0; for (size_t i=0;i<(size_t)4*M;i++){ double d=fabs(hY[i]-hYref[i]); if(d>mx)mx=d; }
        printf("maxabs=%.6f\n", mx);
    }
    // perf: seq vs kt2
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    for(int i=0;i<5;i++) for(int c=0;c<4;c++) tt_gemv_q4_0_v4(dW, dX+(size_t)c*K, dY+(size_t)c*M, M, K, 0);
    cudaDeviceSynchronize();
    double bs=1e18;
    for(int i=0;i<30;i++){ cudaEventRecord(a,0); for(int c=0;c<4;c++) tt_gemv_q4_0_v4(dW, dX+(size_t)c*K, dY+(size_t)c*M, M, K, 0); cudaEventRecord(b,0); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); if(ms*1000<bs)bs=ms*1000; }
    for(int i=0;i<10;i++) tt_gemv_q4_0_batch4_kt2(dW,dX,dY,M,K,0);
    cudaDeviceSynchronize();
    double bk=1e18;
    for(int i=0;i<30;i++){ cudaEventRecord(a,0); tt_gemv_q4_0_batch4_kt2(dW,dX,dY,M,K,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float ms=0; cudaEventElapsedTime(&ms,a,b); if(ms*1000<bk)bk=ms*1000; }
    double gbps = (double)wb / (bk/1e6) / 1e9;
    printf("M=%d K=%d seq4_us=%.1f kt2_us=%.1f gbps=%.1f speedup=%.2f\n", M,K,bs,bk,gbps,bs/bk);
    return 0;
}
