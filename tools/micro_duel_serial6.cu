// Duel serial vs split S8 Llama H32 KV8 D64 pos511
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA err %d\n", (int)e_); exit(2); } } while (0)
extern "C" int tt_flash_gqa_q8_0(const float *q, const void *Kc, const void *Vc, float *out, const int *d_pos, int nh, int nkv, int hd, int mctx, float sc, int win, cudaStream_t st);
extern "C" int tt_flash_gqa_q8_0_splitk(const float *q, const void *Kc, const void *Vc, float *pa, float *pm, float *pl, float *out, const int *d_pos, int nh, int nkv, int hd, float sc, int win, int S, cudaStream_t st);
struct BlockQ8_0 { __half d; int8_t qs[32]; };
int main() {
CK(cudaSetDevice(0));
const int H = 32;
const int KV = 8;
const int D = 64;
const int T = 512;
const int S = 8;
const int POS = 511;
const float scale = 0.125f;
const int bps = KV * D / 32;
size_t nblk = (size_t)T * bps;
std::vector<float> h_q((size_t)H * D);
std::vector<BlockQ8_0> h_K(nblk);
std::vector<BlockQ8_0> h_V(nblk);
srand(1234);
for (size_t i = 0; i < h_q.size(); i++) h_q[i] = (float)(rand() % 2000 - 1000) / 1000.0f;
for (size_t i = 0; i < nblk; i++) {
h_K[i].d = __float2half(0.02f);
h_V[i].d = __float2half(0.02f);
for (int j = 0; j < 32; j++) { h_K[i].qs[j] = (int8_t)(rand() % 5 - 2); h_V[i].qs[j] = (int8_t)(rand() % 5 - 2); }
}
float *d_q = 0;
float *d_os = 0;
float *d_ok = 0;
float *d_acc = 0;
float *d_m = 0;
float *d_l = 0;
BlockQ8_0 *d_K = 0;
BlockQ8_0 *d_V = 0;
int *d_pos = 0;
CK(cudaMalloc(&d_q, sizeof(float) * H * D));
CK(cudaMalloc(&d_K, sizeof(BlockQ8_0) * nblk));
CK(cudaMalloc(&d_V, sizeof(BlockQ8_0) * nblk));
CK(cudaMalloc(&d_os, sizeof(float) * H * D));
CK(cudaMalloc(&d_ok, sizeof(float) * H * D));
CK(cudaMalloc(&d_acc, sizeof(float) * (size_t)S * H * D));
CK(cudaMalloc(&d_m, sizeof(float) * (size_t)S * H));
CK(cudaMalloc(&d_l, sizeof(float) * (size_t)S * H));
CK(cudaMalloc(&d_pos, sizeof(int)));
CK(cudaMemcpy(d_q, h_q.data(), sizeof(float) * H * D, cudaMemcpyHostToDevice));
CK(cudaMemcpy(d_K, h_K.data(), sizeof(BlockQ8_0) * nblk, cudaMemcpyHostToDevice));
CK(cudaMemcpy(d_V, h_V.data(), sizeof(BlockQ8_0) * nblk, cudaMemcpyHostToDevice));
CK(cudaMemcpy(d_pos, &POS, sizeof(int), cudaMemcpyHostToDevice));
tt_flash_gqa_q8_0(d_q, d_K, d_V, d_os, d_pos, H, KV, D, T, scale, 0, 0);
tt_flash_gqa_q8_0_splitk(d_q, d_K, d_V, d_acc, d_m, d_l, d_ok, d_pos, H, KV, D, scale, 0, S, 0);
CK(cudaDeviceSynchronize());
std::vector<float> h_os(H * D);
std::vector<float> h_ok(H * D);
CK(cudaMemcpy(h_os.data(), d_os, sizeof(float) * H * D, cudaMemcpyDeviceToHost));
CK(cudaMemcpy(h_ok.data(), d_ok, sizeof(float) * H * D, cudaMemcpyDeviceToHost));
double mx = 0;
double ss = 0;
double sr = 0;
for (size_t i = 0; i < h_os.size(); i++) { double a = fabs((double)h_os[i] - (double)h_ok[i]); if (a > mx) mx = a; ss += (double)h_os[i] * (double)h_os[i]; sr += (double)h_ok[i] * (double)h_ok[i]; }
printf("validate maxabs=%.3e rmsA=%.3f rmsB=%.3f\n", mx, sqrt(ss / h_os.size()), sqrt(sr / h_os.size()));
for (int i = 0; i < 50; i++) tt_flash_gqa_q8_0(d_q, d_K, d_V, d_os, d_pos, H, KV, D, T, scale, 0, 0);
for (int i = 0; i < 50; i++) tt_flash_gqa_q8_0_splitk(d_q, d_K, d_V, d_acc, d_m, d_l, d_ok, d_pos, H, KV, D, scale, 0, S, 0);
CK(cudaDeviceSynchronize());
cudaEvent_t ea;
cudaEvent_t eb;
CK(cudaEventCreate(&ea));
CK(cudaEventCreate(&eb));
for (int w = 0; w < 4; w++) {
int serial = (w % 2 == 0);
const char *tag = serial ? "serial" : "split8";
if (w == 2) tag = "serial2";
if (w == 3) tag = "split8b";
std::vector<float> ms;
ms.reserve(300);
for (int i = 0; i < 300; i++) { CK(cudaEventRecord(ea, 0)); if (serial) tt_flash_gqa_q8_0(d_q, d_K, d_V, d_os, d_pos, H, KV, D, T, scale, 0, 0); else tt_flash_gqa_q8_0_splitk(d_q, d_K, d_V, d_acc, d_m, d_l, d_ok, d_pos, H, KV, D, scale, 0, S, 0); CK(cudaEventRecord(eb, 0)); CK(cudaEventSynchronize(eb)); float m = 0; CK(cudaEventElapsedTime(&m, ea, eb)); ms.push_back(m * 1000.0f); }
std::sort(ms.begin(), ms.end());
printf("%s med=%.2f us p10=%.2f p90=%.2f\n", tag, ms[150], ms[30], ms[270]);
}
CK(cudaEventDestroy(ea));
CK(cudaEventDestroy(eb));
return 0;
}
