// Q6_K v3: 4 rows per warp, x regs shared across 4 rows. Same per-row op order as V2.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA err %d\n", (int)e_); exit(2); } } while (0)
extern "C" int tt_gemv_typed(const void* W, int dtype, const float* x, float* y, int M, int K, cudaStream_t stream);
static __device__ __forceinline__ float wrsum(float v) {
#pragma unroll
for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
return v; }
static __device__ __forceinline__ float h6_at(const uint8_t *p) {
__half h; memcpy(&h, p, 2); return __half2float(h); }
#define Q6ROW(bR, sR, dR) { const uint8_t *qq = bR + chunk * 64; const uint8_t *hh = bR + 128 + chunk * 32; const int8_t *cc = (const int8_t *)(bR + 192 + chunk * 8); int a0 = qq[lane + 0]; int a1 = qq[lane + 32]; int hv = hh[lane]; float v0 = (float)((int8_t)((a0 & 0xF) | (((hv >> 0) & 3) << 4)) - 32); float v1 = (float)((int8_t)((a1 & 0xF) | (((hv >> 2) & 3) << 4)) - 32); float v2 = (float)((int8_t)(((a0 >> 4) & 0xF) | (((hv >> 4) & 3) << 4)) - 32); float v3 = (float)((int8_t)(((a1 >> 4) & 0xF) | (((hv >> 6) & 3) << 4)) - 32); sR += dR * (cc[is + 0] * v0 * x0 + cc[is + 2] * v1 * x1 + cc[is + 4] * v2 * x2 + cc[is + 6] * v3 * x3); }
__global__ void k_gemv_q6_K_v3(const uint8_t *__restrict__ W, const float *__restrict__ x, float *__restrict__ y, int M, int K) {
const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
if (row0 + 3 >= M) return;
const int lane = threadIdx.x;
const int nsb = K / 256;
const uint8_t *rw0 = W + (long)row0 * nsb * 210;
const uint8_t *rw1 = rw0 + (long)nsb * 210;
const uint8_t *rw2 = rw1 + (long)nsb * 210;
const uint8_t *rw3 = rw2 + (long)nsb * 210;
float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
const int is = lane >> 4;
for (int sb = 0; sb < nsb; sb++) {
const uint8_t *b0 = rw0 + sb * 210;
const uint8_t *b1 = rw1 + sb * 210;
const uint8_t *b2 = rw2 + sb * 210;
const uint8_t *b3 = rw3 + sb * 210;
const float *xb = x + (long)sb * 256;
const float d0 = h6_at(b0 + 208);
const float d1 = h6_at(b1 + 208);
const float d2 = h6_at(b2 + 208);
const float d3 = h6_at(b3 + 208);
#pragma unroll
for (int chunk = 0; chunk < 2; chunk++) {
const float *xc = xb + chunk * 128;
const float x0 = xc[lane + 0];
const float x1 = xc[lane + 32];
const float x2 = xc[lane + 64];
const float x3 = xc[lane + 96];
Q6ROW(b0, s0, d0); Q6ROW(b1, s1, d1); Q6ROW(b2, s2, d2); Q6ROW(b3, s3, d3);
}
}
s0 = wrsum(s0); s1 = wrsum(s1); s2 = wrsum(s2); s3 = wrsum(s3);
if (lane == 0) { y[row0] = s0; y[row0+1] = s1; y[row0+2] = s2; y[row0+3] = s3; }
}
static double bench_v2(const void *dW, const float *dx, float *dy, int M, int K, size_t wb, const char *tag) {
for (int i = 0; i < 10; i++) tt_gemv_typed(dW, 14, dx, dy, M, K, 0);
CK(cudaDeviceSynchronize());
cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
std::vector<float> ms; ms.reserve(50);
for (int i = 0; i < 50; i++) { CK(cudaEventRecord(a, 0)); tt_gemv_typed(dW, 14, dx, dy, M, K, 0); CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b)); float m = 0; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
std::sort(ms.begin(), ms.end());
printf("%s median=%.3f ms GBps=%.1f\n", tag, ms[25], wb / (ms[25] / 1000.0) / 1e9);
return ms[25]; }
static double bench_v3(const void *dW, const float *dx, float *dy, int M, int K, size_t wb, const char *tag) {
dim3 blk(32, 4); dim3 grd(M / 16);
for (int i = 0; i < 10; i++) k_gemv_q6_K_v3<<<grd, blk>>>((const uint8_t *)dW, dx, dy, M, K);
CK(cudaDeviceSynchronize());
cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
std::vector<float> ms; ms.reserve(50);
for (int i = 0; i < 50; i++) { CK(cudaEventRecord(a, 0)); k_gemv_q6_K_v3<<<grd, blk>>>((const uint8_t *)dW, dx, dy, M, K); CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b)); float m = 0; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
std::sort(ms.begin(), ms.end());
printf("%s median=%.3f ms GBps=%.1f\n", tag, ms[25], wb / (ms[25] / 1000.0) / 1e9);
return ms[25]; }
int main() {
CK(cudaSetDevice(0));
const int M = 128256; const int K = 2048; const int nsb = 8;
size_t wb = (size_t)M * (size_t)nsb * 210;
uint8_t *hW = (uint8_t *)malloc(wb);
uint32_t s = 4660u;
for (size_t i = 0; i < wb; i++) { s = s * 1103515245u + 12345u; hW[i] = (uint8_t)(s >> 16); }
for (int r = 0; r < M; r++) for (int sb = 0; sb < nsb; sb++) { uint8_t *blk = hW + ((size_t)r * nsb + sb) * 210; blk[208] = 0; blk[209] = 60; }
float *hX = (float *)malloc((size_t)K * 4);
for (int k = 0; k < K; k++) hX[k] = 0.01f * (float)((k % 7) - 3);
uint8_t *dW = 0; float *dx = 0; float *dy = 0;
CK(cudaMalloc((void **)&dW, wb)); CK(cudaMalloc((void **)&dx, (size_t)K * 4)); CK(cudaMalloc((void **)&dy, (size_t)M * 4));
CK(cudaMemcpy(dW, hW, wb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx, hX, (size_t)K * 4, cudaMemcpyHostToDevice));
double m2 = bench_v2(dW, dx, dy, M, K, wb, "v2");
std::vector<float> y2(M); CK(cudaMemcpy(y2.data(), dy, (size_t)M * 4, cudaMemcpyDeviceToHost));
double m3 = bench_v3(dW, dx, dy, M, K, wb, "v3");
std::vector<float> y3(M); CK(cudaMemcpy(y3.data(), dy, (size_t)M * 4, cudaMemcpyDeviceToHost));
double mx = 0;
for (int i = 0; i < M; i++) { double a = fabs((double)y2[i] - (double)y3[i]); if (a > mx) mx = a; }
printf("maxabs=%.3e speedup=%.2f\n", mx, m2 / m3);
return 0; }
