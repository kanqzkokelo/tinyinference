// micro_head_q4proxy.cu: Q6_K head vs Q4_0 traffic proxy, llama-3.2-1B shape.
// M=128256 K=2048. Q6 215.5MB vs Q4_0 ~147.7MB. Timing only, no fidelity check.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) exit(2); } while (0)
extern "C" int tt_gemv_typed(const void* W, int dtype, const float* x, float* y, int M, int K, cudaStream_t stream);
static double bench(const char* tag, const void* dW, int dtype, const float* dx, float* dy, int M, int K, size_t wb) {
for (int i = 0; i < 10; i++) tt_gemv_typed(dW, dtype, dx, dy, M, K, 0);
CK(cudaDeviceSynchronize());
cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
std::vector<float> ms; ms.reserve(50);
for (int i = 0; i < 50; i++) { CK(cudaEventRecord(a, 0)); tt_gemv_typed(dW, dtype, dx, dy, M, K, 0); CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b)); float m = 0; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m); }
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
double m_v2 = bench("v2      ", dW, 14, dx, dy, M, K, wb);
size_t wb4 = (size_t)M * (size_t)K * 18 / 32;
printf("q4head MB=%.1f\n", wb4 / 1e6);
uint8_t* hW4 = (uint8_t*)malloc(wb4);
for (size_t i = 0; i < wb4; i++) { s = s * 1103515245u + 12345u; hW4[i] = (uint8_t)(s >> 16); }
uint8_t* dW4 = 0;
CK(cudaMalloc((void**)&dW4, wb4));
CK(cudaMemcpy(dW4, hW4, wb4, cudaMemcpyHostToDevice));
double m_q4 = bench("q4proxy ", dW4, 2, dx, dy, M, K, wb4);
printf("q4-vs-q6 speedup=%.2f saveMB=%.1f\n", m_v2 / m_q4, (wb - wb4) / 1e6);
return 0;
}
