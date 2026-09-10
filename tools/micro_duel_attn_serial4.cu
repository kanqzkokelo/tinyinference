// Duel serial vs split-K Q8 attention at decode shapes.
// Times tt_flash_gqa_q8_0 vs tt_flash_gqa_q8_0_splitk, checks agreement.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

extern "C" {
extern int tt_flash_gqa_q8_0(const float *q, const void *Kc_q8, const void *Vc_q8, float *out,
    const int *d_pos, int n_heads, int n_kv_heads, int head_dim,
    int max_ctx, float scale, int window, cudaStream_t stream);
// link stub: dequant helper unused by attention kernels
long ttq_dequant(const void *a, int b, long c, float *d) {
    (void)a; (void)b; (void)c; (void)d; return 0;
}
extern int tt_flash_gqa_q8_0_splitk(const float *q, const void *Kc_q8, const void *Vc_q8,
    float *p_acc, float *p_m, float *p_l, float *out,
    const int *d_pos, int n_heads, int n_kv_heads, int head_dim,
    float scale, int window, int S, cudaStream_t stream);
}

struct BlockQ8 { half d; int8_t qs[32]; };

static double med_ms(std::vector<float> &v) {
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

static void run_shape(const char *name, int H, int KV, int HD, int pos) {
    int max_ctx = pos + 64;
    int bph = HD / 32;
    size_t nblk = (size_t)max_ctx * KV * bph;
    float scale = 1.0f / sqrtf((float)HD);
    int S = (pos + 63) / 64;
    if (S < 2) S = 2;
    if (S > 64) S = 64;

    std::vector<float> hq((size_t)H * HD);
    std::vector<BlockQ8> hkv(nblk);
    srand(12345);
    for (auto &x : hq) x = ((rand() % 2000) - 1000) / 1000.0f;
    for (auto &b : hkv) {
        b.d = __float2half(0.02f);
        for (int i = 0; i < 32; i++) b.qs[i] = (int8_t)((rand() % 256) - 128);
    }
    float *dq; BlockQ8 *dK, *dV; float *dout, *dacc, *dm, *dl; int *dpos;
    CK(cudaMalloc(&dq, sizeof(float) * H * HD));
    CK(cudaMalloc(&dK, sizeof(BlockQ8) * nblk));
    CK(cudaMalloc(&dV, sizeof(BlockQ8) * nblk));
    CK(cudaMalloc(&dout, sizeof(float) * H * HD));
    CK(cudaMalloc(&dacc, sizeof(float) * (size_t)S * H * HD));
    CK(cudaMalloc(&dm, sizeof(float) * (size_t)S * H));
    CK(cudaMalloc(&dl, sizeof(float) * (size_t)S * H));
    CK(cudaMalloc(&dpos, sizeof(int)));
    // 64MB L2-flush buffer (RTX3050 L2 ~2MB); memset between iters
    // simulates engine streaming (28 layers x ~1MB KV, all L2 misses).
    char *dflush; CK(cudaMalloc(&dflush, 64<<20));
    CK(cudaMemcpy(dq, hq.data(), sizeof(float) * H * HD, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hkv.data(), sizeof(BlockQ8) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hkv.data(), sizeof(BlockQ8) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));

    // correctness: serial vs split
    std::vector<float> hser(H * HD), hspl(H * HD);
    tt_flash_gqa_q8_0(dq, dK, dV, dout, dpos, H, KV, HD, max_ctx, scale, 0, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hser.data(), dout, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    tt_flash_gqa_q8_0_splitk(dq, dK, dV, dacc, dm, dl, dout, dpos, H, KV, HD, scale, 0, S, 0);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hspl.data(), dout, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    double maxd = 0;
    for (size_t i = 0; i < hser.size(); i++) maxd = std::max<double>(maxd, fabs(hser[i]-hspl[i]));
    printf("[%s] H=%d KV=%d HD=%d pos=%d S=%d serial-vs-split maxdiff=%.5f\n", name, H, KV, HD, pos, S, maxd);

    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int w = 0; w < 20; w++) tt_flash_gqa_q8_0(dq, dK, dV, dout, dpos, H, KV, HD, max_ctx, scale, 0, 0);
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 200; i++) {
        CK(cudaMemset(dflush, i & 255, 64<<20));
        CK(cudaEventRecord(a, 0));
        tt_flash_gqa_q8_0(dq, dK, dV, dout, dpos, H, KV, HD, max_ctx, scale, 0, 0);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m);
    }
    printf("[%s] serial median=%7.2f us\n", name, med_ms(ms)*1000.0);
    for (int w = 0; w < 20; w++) tt_flash_gqa_q8_0_splitk(dq, dK, dV, dacc, dm, dl, dout, dpos, H, KV, HD, scale, 0, S, 0);
    CK(cudaDeviceSynchronize());
    ms.clear();
    for (int i = 0; i < 200; i++) {
        CK(cudaMemset(dflush, i & 255, 64<<20));
        CK(cudaEventRecord(a, 0));
        tt_flash_gqa_q8_0_splitk(dq, dK, dV, dacc, dm, dl, dout, dpos, H, KV, HD, scale, 0, S, 0);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m);
    }
    // traffic: useful KV bytes read per call (K+V, int8+scale)
    double kvbytes = (double)(pos+1) * KV * HD * 2 + (double)(pos+1) * KV * bph * 2 * 2;
    double med = med_ms(ms)*1000.0;
    printf("[%s] splitS%d median=%7.2f us  usefulGB/s=%6.1f\n", name, S, med, kvbytes/(med/1e6)/1e9);
}

static void sweep_S_llama2k() {
    const int H = 32, KV = 8, HD = 64, pos = 2048;
    int max_ctx = pos + 64;
    int bph = HD / 32;
    size_t nblk = (size_t)max_ctx * KV * bph;
    float scale = 1.0f / sqrtf((float)HD);
    std::vector<float> hq((size_t)H * HD);
    std::vector<BlockQ8> hkv(nblk);
    srand(777);
    for (auto &x : hq) x = ((rand() % 2000) - 1000) / 1000.0f;
    for (auto &b : hkv) { b.d = __float2half(0.02f); for (int i = 0; i < 32; i++) b.qs[i] = (int8_t)((rand() % 256) - 128); }
    float *dq; BlockQ8 *dK, *dV; float *dout, *dacc, *dm, *dl; int *dpos;
    CK(cudaMalloc(&dq, sizeof(float) * H * HD));
    CK(cudaMalloc(&dK, sizeof(BlockQ8) * nblk));
    CK(cudaMalloc(&dV, sizeof(BlockQ8) * nblk));
    CK(cudaMalloc(&dout, sizeof(float) * H * HD));
    CK(cudaMalloc(&dacc, sizeof(float) * (size_t)32 * H * HD));
    CK(cudaMalloc(&dm, sizeof(float) * (size_t)32 * H));
    CK(cudaMalloc(&dl, sizeof(float) * (size_t)32 * H));
    CK(cudaMalloc(&dpos, sizeof(int)));
    char *dflush; CK(cudaMalloc(&dflush, 64<<20));
    CK(cudaMemcpy(dq, hq.data(), sizeof(float) * H * HD, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hkv.data(), sizeof(BlockQ8) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hkv.data(), sizeof(BlockQ8) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
    const int Ss[] = {4, 8, 16, 32};
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    double kvbytes = (double)(pos+1) * KV * HD * 2 + (double)(pos+1) * KV * bph * 2 * 2;
    for (int si = 0; si < 4; si++) {
        int S = Ss[si];
        for (int w = 0; w < 20; w++) tt_flash_gqa_q8_0_splitk(dq, dK, dV, dacc, dm, dl, dout, dpos, H, KV, HD, scale, 0, S, 0);
        CK(cudaDeviceSynchronize());
        std::vector<float> ms;
        for (int i = 0; i < 200; i++) {
            CK(cudaMemset(dflush, i & 255, 64<<20));
            CK(cudaEventRecord(a, 0));
            tt_flash_gqa_q8_0_splitk(dq, dK, dV, dacc, dm, dl, dout, dpos, H, KV, HD, scale, 0, S, 0);
            CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
            float m; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m);
        }
        double med = med_ms(ms)*1000.0;
        printf("[sweep] S=%d median=%7.2f us usefulGB/s=%6.1f", S, med, kvbytes/(med/1e6)/1e9);
    }
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}
int main() {
    run_shape("qwen3", 16, 8, 128, 525);
    run_shape("llama", 32, 8, 64, 510);
    run_shape("smollm2", 9, 3, 64, 510);
    run_shape("qwen2.5", 14, 2, 64, 525);
    run_shape("qwen3-2k", 16, 8, 128, 2048);
    run_shape("smollm2-2k", 9, 3, 64, 2048);
    run_shape("llama-2k", 32, 8, 64, 2048);
    sweep_S_llama2k();
    return 0;
}
