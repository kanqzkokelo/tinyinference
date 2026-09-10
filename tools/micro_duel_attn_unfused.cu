// Duel: unfused decode attention (QK-gemv + softmax + PV-gemv, pure
// streaming, no warp-sync chain) vs fused k_fa2_q8_split. Same-process
// A-B on random KV. argv[1] present = hot L2 (skip flush).
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

static int g_noflush = 0;

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

extern "C" {
long ttq_dequant(const void *a, int b, long c, float *d) {
    (void)a; (void)b; (void)c; (void)d; return 0;
}
int tt_flash_gqa_q8_0_splitk(const float *q, const void *Kc_q8, const void *Vc_q8,
    float *acc, float *st_m, float *st_l, float *out, const int *d_pos,
    int n_heads, int n_kv_heads, int head_dim, float scale, int window, int S,
    int cap, cudaStream_t stream);
}

struct BlockQ8KV { signed char qs[32]; half d; short pad; };
struct BlockQ8E { half d; short pad; signed char qs[32]; };

// ---- v2: (kv,t,b)-native layout, smem staging, G-way reuse ----
// new layout: Kn[((kv * cap) + t) * bph + b], cap = n (duel exact)
#define TT2 256
__global__ void k2_qk(const float *__restrict__ q, const BlockQ8KV *__restrict__ Kn,
                      float *__restrict__ scores, int n, int G, int HD, float scale,
                      int PART) {
    int kv = blockIdx.x / PART;
    int pp = blockIdx.x % PART;
    int tid = threadIdx.x, NT = blockDim.x;
    int bph = HD / 32;
    int chunk = (n + PART - 1) / PART;
    int r0 = min(n, pp * chunk), r1 = min(n, r0 + chunk);
    __shared__ float sq[8 * 128];
    for (int i = tid; i < G * HD; i += NT)
        sq[i] = q[(long)(kv * G + i / HD) * HD + i % HD];
    __shared__ half sKd[TT2 * 4];
    __shared__ unsigned sKq[TT2 * 32];
    for (int t0 = r0; t0 < r1; t0 += TT2) {
        int tact = min(TT2, r1 - t0);
        for (int i = tid; i < tact * bph; i += NT) {
            int tt = i / bph, b = i % bph;
            long gi = ((long)kv * n + t0 + tt) * bph + b;
            sKd[tt * bph + b] = Kn[gi].d;
            const unsigned *src = (const unsigned *)&Kn[gi].qs[0];
            unsigned *dst = sKq + ((long)tt * bph + b) * 8;
            dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3];
            dst[4]=src[4]; dst[5]=src[5]; dst[6]=src[6]; dst[7]=src[7];
        }
        __syncthreads();
        for (int gt = tid; gt < G * tact; gt += NT) {
            int g = gt / tact, tt = gt % tact;
            const float *qq = sq + (long)g * HD;
            float dot = 0.f;
            for (int b = 0; b < bph; b++) {
                float dk = __half2float(sKd[tt * bph + b]);
                const unsigned *kw = sKq + ((long)tt * bph + b) * 8;
                const float *qb = qq + b * 32;
                for (int w = 0; w < 8; w++) {
                    unsigned u = kw[w];
                    const float *q4 = qb + w * 4;
                    dot += q4[0] * ((float)((signed char)(u      )) * dk)
                         + q4[1] * ((float)((signed char)(u >>  8)) * dk)
                         + q4[2] * ((float)((signed char)(u >> 16)) * dk)
                         + q4[3] * ((float)((signed char)(u >> 24)) * dk);
                }
            }
            scores[((long)(kv * G + g)) * n + t0 + tt] = dot * scale;
        }
        __syncthreads();
    }
}

// pass3 v2: one block per (kv, d-half); tile loaded ONCE, shared by G heads
__global__ void k2_pv(const float *__restrict__ scores, const BlockQ8KV *__restrict__ Vn,
                      float *__restrict__ out, int n, int G, int HD) {
    int kv = blockIdx.x / 2, cc = blockIdx.x % 2;
    int tid = threadIdx.x, NT = blockDim.x;
    int bph = HD / 32;
    int HD2 = HD / 2, d0 = cc * HD2;
    int GD2 = G * HD2; // <= 256 for our shapes (max 224)
    __shared__ half sVd[TT2 * 4];
    __shared__ unsigned sVq[TT2 * 32];
    __shared__ float sP[8 * 256];
    float acc = 0.f;
    int gd = tid; // one (g,d) per thread
    int g = (gd < GD2) ? gd / HD2 : 0;
    int d = (gd < GD2) ? d0 + gd % HD2 : 0;
    int b_of_d = d / 32;
    const float *prow = (gd < GD2) ? scores + ((long)(kv * G + g)) * n : 0;
    for (int t0 = 0; t0 < n; t0 += TT2) {
        int tact = min(TT2, n - t0);
        for (int i = tid; i < tact * bph; i += NT) {
            int tt = i / bph, b = i % bph;
            long gi = ((long)kv * n + t0 + tt) * bph + b;
            sVd[tt * bph + b] = Vn[gi].d;
            const unsigned *src = (const unsigned *)&Vn[gi].qs[0];
            unsigned *dst = sVq + ((long)tt * bph + b) * 8;
            dst[0]=src[0]; dst[1]=src[1]; dst[2]=src[2]; dst[3]=src[3];
            dst[4]=src[4]; dst[5]=src[5]; dst[6]=src[6]; dst[7]=src[7];
        }
        __syncthreads();
        for (int i = tid; i < G * tact; i += NT)
            sP[(long)(i / tact) * TT2 + i % tact] =
                scores[((long)(kv * G + i / tact)) * n + t0 + i % tact];
        __syncthreads();
        if (gd < GD2) {
            int w_of_d = (d % 32) / 4, sh = (d % 4) * 8;
            for (int tt = 0; tt < tact; tt++) {
                float dv = __half2float(sVd[tt * bph + b_of_d]);
                unsigned u = sVq[((long)tt * bph + b_of_d) * 8 + w_of_d];
                acc += sP[(long)g * TT2 + tt] * ((float)((signed char)(u >> sh)) * dv);
            }
        }
        __syncthreads();
    }
    if (gd < GD2) out[((long)(kv * G + g)) * HD + d] = acc;
}

// pass2: online softmax per head-row (one block per head)
__global__ void k_softmax_rows(float *__restrict__ scores, int n, int H) {
    int h = blockIdx.x;
    int tid = threadIdx.x, NT = blockDim.x;
    float *row = scores + (long)h * n;
    __shared__ float sb[256];
    float m = -1e30f;
    for (int t = tid; t < n; t += NT) m = fmaxf(m, row[t]);
    sb[tid] = m; __syncthreads();
    for (int s = NT >> 1; s > 0; s >>= 1) {
        if (tid < s) sb[tid] = fmaxf(sb[tid], sb[tid + s]);
        __syncthreads();
    }
    m = sb[0]; __syncthreads();
    float l = 0.f;
    for (int t = tid; t < n; t += NT) { float e = expf(row[t] - m); row[t] = e; l += e; }
    sb[tid] = l; __syncthreads();
    for (int s = NT >> 1; s > 0; s >>= 1) {
        if (tid < s) sb[tid] += sb[tid + s];
        __syncthreads();
    }
    l = sb[0];
    for (int t = tid; t < n; t += NT) row[t] /= l;
}

static double med_ms(std::vector<float> &v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void run_shape(const char *name, int H, int KV, int HD, int pos, char *dflush) {
    int G = H / KV, n = pos + 1, S = 32;
    int bph = HD / 32;
    long nblk = (long)n * KV * bph;
    float scale = 1.0f / sqrtf((float)HD);
    std::vector<float> hq((long)H * HD);
    std::vector<BlockQ8E> hkv(nblk);
    srand(12345);
    for (auto &x : hq) x = ((rand() % 2000) - 1000) / 1000.0f;
    for (auto &b : hkv) {
        b.d = __float2half(0.02f); b.pad = 0;
        for (int i = 0; i < 32; i++) b.qs[i] = (signed char)((rand() % 256) - 128);
    }
    std::vector<BlockQ8KV> hkv2(nblk);
    for (long i = 0; i < nblk; i++) {
        for (int j = 0; j < 32; j++) hkv2[i].qs[j] = hkv[i].qs[j];
        hkv2[i].d = hkv[i].d; hkv2[i].pad = 0;
    }
    float *dq, *dout_split, *dout_unf, *dacc, *dm, *dl, *dscores;
    BlockQ8E *dK, *dV;
    int *dpos;
    CK(cudaMalloc(&dq, sizeof(float) * H * HD));
    CK(cudaMalloc(&dK, sizeof(BlockQ8E) * nblk));
    CK(cudaMalloc(&dV, sizeof(BlockQ8E) * nblk));
    CK(cudaMalloc(&dout_split, sizeof(float) * H * HD));
    CK(cudaMalloc(&dout_unf, sizeof(float) * H * HD));
    CK(cudaMalloc(&dscores, sizeof(float) * (long)H * n));
    std::vector<BlockQ8KV> hkv2n(nblk);
    for (int t = 0; t < n; t++) for (int k = 0; k < KV; k++) for (int bb = 0; bb < bph; bb++)
        hkv2n[((long)k * n + t) * bph + bb] = hkv2[((long)t * KV + k) * bph + bb];
    /* split side wants d-first blocks in the same kv-major order */
    std::vector<BlockQ8E> hkvE(nblk);
    for (long i = 0; i < nblk; i++) {
        hkvE[i].d = hkv2n[i].d; hkvE[i].pad = 0;
        for (int j = 0; j < 32; j++) hkvE[i].qs[j] = hkv2n[i].qs[j];
    }
    BlockQ8KV *dK2, *dV2;
    CK(cudaMalloc(&dK2, sizeof(BlockQ8KV) * nblk));
    CK(cudaMalloc(&dV2, sizeof(BlockQ8KV) * nblk));
    CK(cudaMemcpy(dK2, hkv2n.data(), sizeof(BlockQ8KV) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV2, hkv2n.data(), sizeof(BlockQ8KV) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&dacc, sizeof(float) * (long)S * H * HD));
    CK(cudaMalloc(&dm, sizeof(float) * (long)S * H));
    CK(cudaMalloc(&dl, sizeof(float) * (long)S * H));
    CK(cudaMalloc(&dpos, sizeof(int)));
    CK(cudaMemcpy(dq, hq.data(), sizeof(float) * H * HD, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hkvE.data(), sizeof(BlockQ8E) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hkvE.data(), sizeof(BlockQ8E) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto flush = [&]() { if (!g_noflush) CK(cudaMemset(dflush, 0xA5, 64 << 20)); };
    auto time_split = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        tt_flash_gqa_q8_0_splitk(dq, dK, dV, dacc, dm, dl, dout_split, dpos,
                                 H, KV, HD, scale, 0, S, n, 0);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); return m;
    };
    auto time_unf = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        k2_qk<<<KV * 2, 256, 0, 0>>>(dq, dK2, dscores, n, G, HD, scale, 2);
        k_softmax_rows<<<H, 256, 0, 0>>>(dscores, n, H);
        k2_pv<<<KV * 2, 256, 0, 0>>>(dscores, dV2, dout_unf, n, G, HD);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); return m;
    };
    {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        std::vector<float> m1, m2, m3;
        float t;
        for (int i = 0; i < 50; i++) {
            flush(); CK(cudaEventRecord(e0, 0));
            k2_qk<<<KV * 2, 256, 0, 0>>>(dq, dK2, dscores, n, G, HD, scale, 2);
            CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&t, e0, e1)); m1.push_back(t);
        }
        for (int i = 0; i < 50; i++) {
            flush(); CK(cudaEventRecord(e0, 0));
            k_softmax_rows<<<H, 256, 0, 0>>>(dscores, n, H);
            CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&t, e0, e1)); m2.push_back(t);
        }
        for (int i = 0; i < 50; i++) {
            flush(); CK(cudaEventRecord(e0, 0));
            k2_pv<<<KV * 2, 256, 0, 0>>>(dscores, dV2, dout_unf, n, G, HD);
            CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&t, e0, e1)); m3.push_back(t);
        }
        printf("[%s] med us: qk=%.1f sm=%.1f pv=%.1f sum=%.1f ", name,
            med_ms(m1)*1000, med_ms(m2)*1000, med_ms(m3)*1000,
            (med_ms(m1)+med_ms(m2)+med_ms(m3))*1000); fflush(stdout);
    }
    time_split(); time_unf();
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 200; i++) ms.push_back(time_split());
    double split_us = med_ms(ms) * 1000.0;
    ms.clear();
    for (int i = 0; i < 200; i++) ms.push_back(time_unf());
    double unf_us = med_ms(ms) * 1000.0;
    std::vector<float> hs((long)H * HD), hu((long)H * HD);
    CK(cudaMemcpy(hs.data(), dout_split, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hu.data(), dout_unf, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    double md = 0;
    for (size_t i = 0; i < hs.size(); i++) md = std::max(md, (double)fabsf(hs[i] - hu[i]));
    printf("[%s] H=%d KV=%d HD=%d pos=%d split-vs-unfused maxdiff=%.5f\n",
           name, H, KV, HD, pos, md);
    printf("[%s] split median=%8.2f us | unfused median=%8.2f us (%.2fx)\n",
           name, split_us, unf_us, split_us / unf_us);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}

int main(int argc, char **argv) {
    g_noflush = argc > 1;
    char *dflush;
    CK(cudaMalloc(&dflush, 64 << 20));
    run_shape("llama-510", 32, 8, 64, 510, dflush);
    run_shape("llama-2k", 32, 8, 64, 2048, dflush);
    run_shape("qwen3-2k", 16, 8, 128, 2048, dflush);
    return 0;
}
