// micro_gemv_q6head.cu: production Q6_K head at llama-3.2-1B shape.
// Times tt_gemv_typed dtype 14 at M=128256 K=2048, v2 vs scalar.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) exit(2); } while (0)
extern "C" int tt_gemv_typed(const void* W, int dtype, const float* x, float* y, int M, int K, cudaStream_t stream);
static double bench(const char* tag, const void* dW, const float* dx, float* dy, int M, int K, size_t wb) {
return 0.0;
}
__device__ __forceinline__ float q6v4_red(float v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float q6v4_half(const uint8_t *p) {
    return __half2float(*(const __half *)p);
}
/* Q6_K 4-rows/warp: v2 math x4 rows for 4x ILP. M%4==0 required. */
__global__ void k_gemv_q6_K_v4(const uint8_t *__restrict__ W,
                               const float *__restrict__ x,
                               float *__restrict__ y,
                               int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= M) return;
    const int lane = threadIdx.x;
    const int nsb = K / 256;
    const uint8_t *rw0 = W + (long)row0 * nsb * 210;
    const uint8_t *rw1 = W + (long)(row0 + 1) * nsb * 210;
    const uint8_t *rw2 = W + (long)(row0 + 2) * nsb * 210;
    const uint8_t *rw3 = W + (long)(row0 + 3) * nsb * 210;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    const int is = lane >> 4;
    for (int sb = 0; sb < nsb; sb++) {
        const uint8_t *b0 = rw0 + sb * 210;
        const uint8_t *b1 = rw1 + sb * 210;
        const uint8_t *b2 = rw2 + sb * 210;
        const uint8_t *b3 = rw3 + sb * 210;
        const float *xb = x + (long)sb * 256;
        const float d0 = q6v4_half(b0 + 208);
        const float d1 = q6v4_half(b1 + 208);
        const float d2 = q6v4_half(b2 + 208);
        const float d3 = q6v4_half(b3 + 208);
#pragma unroll
        for (int chunk = 0; chunk < 2; chunk++) {
            const uint8_t *ql0 = b0 + chunk * 64;
            const uint8_t *qh0 = b0 + 128 + chunk * 32;
            const int8_t  *sc0 = (const int8_t *)(b0 + 192 + chunk * 8);
            const uint8_t *ql1 = b1 + chunk * 64;
            const uint8_t *qh1 = b1 + 128 + chunk * 32;
            const int8_t  *sc1 = (const int8_t *)(b1 + 192 + chunk * 8);
            const uint8_t *ql2 = b2 + chunk * 64;
            const uint8_t *qh2 = b2 + 128 + chunk * 32;
            const int8_t  *sc2 = (const int8_t *)(b2 + 192 + chunk * 8);
            const uint8_t *ql3 = b3 + chunk * 64;
            const uint8_t *qh3 = b3 + 128 + chunk * 32;
            const int8_t  *sc3 = (const int8_t *)(b3 + 192 + chunk * 8);
            const float *xc = xb + chunk * 128;
            const int a0 = ql0[lane], a1 = ql0[lane + 32], ha = qh0[lane];
            const int b_0 = ql1[lane], b_1 = ql1[lane + 32], hb = qh1[lane];
            const int c0 = ql2[lane], c1 = ql2[lane + 32], hc = qh2[lane];
            const int e0 = ql3[lane], e1 = ql3[lane + 32], he = qh3[lane];
            const float x0 = xc[lane], x1 = xc[lane + 32];
            const float x2 = xc[lane + 64], x3 = xc[lane + 96];
            s0 += d0 * (sc0[is] * (float)((int8_t)((a0 & 0xF) | (((ha >> 0) & 3) << 4)) - 32) * x0 +
                        sc0[is + 2] * (float)((int8_t)((a1 & 0xF) | (((ha >> 2) & 3) << 4)) - 32) * x1 +
                        sc0[is + 4] * (float)((int8_t)(((a0 >> 4) & 0xF) | (((ha >> 4) & 3) << 4)) - 32) * x2 +
                        sc0[is + 6] * (float)((int8_t)(((a1 >> 4) & 0xF) | (((ha >> 6) & 3) << 4)) - 32) * x3);
            s1 += d1 * (sc1[is] * (float)((int8_t)((b_0 & 0xF) | (((hb >> 0) & 3) << 4)) - 32) * x0 +
                        sc1[is + 2] * (float)((int8_t)((b_1 & 0xF) | (((hb >> 2) & 3) << 4)) - 32) * x1 +
                        sc1[is + 4] * (float)((int8_t)(((b_0 >> 4) & 0xF) | (((hb >> 4) & 3) << 4)) - 32) * x2 +
                        sc1[is + 6] * (float)((int8_t)(((b_1 >> 4) & 0xF) | (((hb >> 6) & 3) << 4)) - 32) * x3);
            s2 += d2 * (sc2[is] * (float)((int8_t)((c0 & 0xF) | (((hc >> 0) & 3) << 4)) - 32) * x0 +
                        sc2[is + 2] * (float)((int8_t)((c1 & 0xF) | (((hc >> 2) & 3) << 4)) - 32) * x1 +
                        sc2[is + 4] * (float)((int8_t)(((c0 >> 4) & 0xF) | (((hc >> 4) & 3) << 4)) - 32) * x2 +
                        sc2[is + 6] * (float)((int8_t)(((c1 >> 4) & 0xF) | (((hc >> 6) & 3) << 4)) - 32) * x3);
            s3 += d3 * (sc3[is] * (float)((int8_t)((e0 & 0xF) | (((he >> 0) & 3) << 4)) - 32) * x0 +
                        sc3[is + 2] * (float)((int8_t)((e1 & 0xF) | (((he >> 2) & 3) << 4)) - 32) * x1 +
                        sc3[is + 4] * (float)((int8_t)(((e0 >> 4) & 0xF) | (((he >> 4) & 3) << 4)) - 32) * x2 +
                        sc3[is + 6] * (float)((int8_t)(((e1 >> 4) & 0xF) | (((he >> 6) & 3) << 4)) - 32) * x3);
        }
    }
    s0 = q6v4_red(s0); s1 = q6v4_red(s1); s2 = q6v4_red(s2); s3 = q6v4_red(s3);
    if (lane == 0) {
        y[row0] = s0; y[row0 + 1] = s1; y[row0 + 2] = s2; y[row0 + 3] = s3;
    }
}
static int tt_q6v4(const void *dW, const float *dx, float *dy, int M, int K) {
    dim3 b; b.x = 32; b.y = 8; b.z = 1;
    dim3 g; g.x = (M / 4 + 7) / 8; g.y = 1; g.z = 1;
    k_gemv_q6_K_v4<<<g, b>>>((const uint8_t *)dW, dx, dy, M, K);
    return (int)cudaGetLastError();
}
static double bench_v4(const void* dW, const float* dx, float* dy, int M, int K, size_t wb) {
for (int i = 0; i < 10; i++) tt_q6v4(dW, dx, dy, M, K);
CK(cudaDeviceSynchronize());
cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
std::vector<float> ms; ms.reserve(50);
for (int i = 0; i < 50; i++) { CK(cudaEventRecord(a, 0)); tt_q6v4(dW, dx, dy, M, K); CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b)); float m = 0; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
std::sort(ms.begin(), ms.end()); double med = ms[25];
printf("v4    : median=%.3f ms GBps=%.1f\n", med, wb / (med / 1000.0) / 1e9);
return med;
}
static double benchX(const char* tag, const void* dW, const float* dx, float* dy, int M, int K, size_t wb) {
for (int i = 0; i < 10; i++) tt_gemv_typed(dW, 14, dx, dy, M, K, 0);
CK(cudaDeviceSynchronize());
cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
std::vector<float> ms; ms.reserve(50);
for (int i = 0; i < 50; i++) { CK(cudaEventRecord(a, 0)); tt_gemv_typed(dW, 14, dx, dy, M, K, 0); CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b)); float m = 0; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
std::sort(ms.begin(), ms.end()); double med = ms[25];
printf("%s median=%.3f ms GBps=%.1f\n", tag, med, wb / (med / 1000.0) / 1e9);
return med;
}
int main() {
const int M = 128256; const int K = 2048; const int nsb = 8;
size_t wb = (size_t)M * (size_t)nsb * 210;
printf("q6head MB=%.1f\n", wb / 1e6);
uint8_t* hW = (uint8_t*)malloc(wb);
uint32_t s = 4660u;
for (size_t i = 0; i < wb; i++) { s = s * 1103515245u + 12345u; hW[i] = (uint8_t)(s >> 16); }
for (int r = 0; r < M; r++) for (int sb = 0; sb < nsb; sb++) { uint8_t* blk = hW + ((size_t)r * nsb + sb) * 210; blk[208] = 0; blk[209] = 60; }
float* hX = (float*)malloc((size_t)K * 4);
for (int k = 0; k < K; k++) hX[k] = 0.01f * (float)((k % 7) - 3);
uint8_t* dW = 0; float* dx = 0; float* dy = 0;
CK(cudaMalloc((void**)&dW, wb)); CK(cudaMalloc((void**)&dx, (size_t)K * 4)); CK(cudaMalloc((void**)&dy, (size_t)M * 4));
CK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, hX, (size_t)K * 4, cudaMemcpyHostToDevice));
unsetenv("TT_DISABLE_Q6_V2");
double m_v2 = benchX("v2    ", dW, dx, dy, M, K, wb);
std::vector<float> yv2(M); CK(cudaMemcpy(yv2.data(), dy, (size_t)M * 4, cudaMemcpyDeviceToHost));
setenv("TT_DISABLE_Q6_V2", "1", 1);
double m_sc = benchX("scalar", dW, dx, dy, M, K, wb);
std::vector<float> ysc(M); CK(cudaMemcpy(ysc.data(), dy, (size_t)M * 4, cudaMemcpyDeviceToHost));
double mx = 0; double mxr = 0;
for (int i = 0; i < M; i++) { double d = (double)yv2[i] - (double)ysc[i]; double a = d < 0 ? -d : d; if (a > mx) mx = a; double r = ysc[i] < 0 ? -ysc[i] : ysc[i]; if (r > mxr) mxr = r; }
printf("maxabs=%.3e maxref=%.3e speedup=%.2f\n", mx, mxr, m_sc / m_v2);
double m_v4 = bench_v4(dW, dx, dy, M, K, wb);
std::vector<float> yv4(M); CK(cudaMemcpy(yv4.data(), dy, (size_t)M * 4, cudaMemcpyDeviceToHost));
mx = 0; for (int i = 0; i < M; i++) { double d = (double)yv4[i] - (double)yv2[i]; double a = d < 0 ? -d : d; if (a > mx) mx = a; }
printf("v4-vs-v2 maxabs=%.3e speedup=%.2f\n", mx, m_v2 / m_v4);
return 0;
}
