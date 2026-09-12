// Duel: current unfused (qk + softmax + pv, scores via global) vs fused
// online-softmax (qk + softmax + pv in one kernel, PARTials + combine).
// Llama shapes: H=32 KV=8 HD=64 G=4. argv[1] present = hot L2.
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
static int g_noflush = 0;
#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", \
            cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)
struct BlockQ8KV { signed char qs[32]; half d; short pad; };
#define TT2 256
#define FTT 128
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
__global__ void k2_pv(const float *__restrict__ scores, const BlockQ8KV *__restrict__ Vn,
                      float *__restrict__ out, int n, int G, int HD) {
    int kv = blockIdx.x / 2, cc = blockIdx.x % 2;
    int tid = threadIdx.x, NT = blockDim.x;
    int bph = HD / 32;
    int HD2 = HD / 2, d0 = cc * HD2;
    int GD2 = G * HD2;
    __shared__ half sVd[TT2 * 4];
    __shared__ unsigned sVq[TT2 * 32];
    __shared__ float sP[8 * 256];
    float acc = 0.f;
    int gd = tid;
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
// Fused: one block per (kv, part). Dots land in smem, online softmax
// per head in registers, V consumed from the same tile. GD = G*HD
// must be <= 256 (llama G=4 HD=64 gives 256, one (g,d) per thread).
__global__ void k_unf_fused(const float *__restrict__ q,
                      const BlockQ8KV *__restrict__ Kn,
                      const BlockQ8KV *__restrict__ Vn,
                      float *__restrict__ p_m, float *__restrict__ p_l,
                      float *__restrict__ p_acc,
                      int n, int G, int HD, float scale, int PART) {
    int kv = blockIdx.x / PART;
    int pp = blockIdx.x % PART;
    int tid = threadIdx.x, NT = blockDim.x;
    int bph = HD / 32;
    int GD = G * HD;
    int chunk = (n + PART - 1) / PART;
    int r0 = min(n, pp * chunk), r1 = min(n, r0 + chunk);
    __shared__ float sq[8 * 128];
    for (int i = tid; i < G * HD; i += NT)
        sq[i] = q[(long)(kv * G + i / HD) * HD + i % HD];
    __shared__ half sKd[FTT * 4];
    __shared__ unsigned sKq[FTT * 32];
    __shared__ half sVd[FTT * 4];
    __shared__ unsigned sVq[FTT * 32];
    __shared__ float sP[8 * FTT];
    int gd = tid;
    int g = (gd < GD) ? gd / HD : 0;
    int d = (gd < GD) ? gd % HD : 0;
    int b_of_d = d / 32, w_of_d = (d % 32) / 4, sh = (d % 4) * 8;
    float m = -1e30f, l = 0.f, acc = 0.f;
    for (int t0 = r0; t0 < r1; t0 += FTT) {
        int tact = min(FTT, r1 - t0);
        for (int i = tid; i < tact * bph; i += NT) {
            int tt = i / bph, b = i % bph;
            long gi = ((long)kv * n + t0 + tt) * bph + b;
            sKd[tt * bph + b] = Kn[gi].d;
            const unsigned *sk = (const unsigned *)&Kn[gi].qs[0];
            unsigned *dk2 = sKq + ((long)tt * bph + b) * 8;
            dk2[0]=sk[0]; dk2[1]=sk[1]; dk2[2]=sk[2]; dk2[3]=sk[3];
            dk2[4]=sk[4]; dk2[5]=sk[5]; dk2[6]=sk[6]; dk2[7]=sk[7];
            sVd[tt * bph + b] = Vn[gi].d;
            const unsigned *sv = (const unsigned *)&Vn[gi].qs[0];
            unsigned *dv2 = sVq + ((long)tt * bph + b) * 8;
            dv2[0]=sv[0]; dv2[1]=sv[1]; dv2[2]=sv[2]; dv2[3]=sv[3];
            dv2[4]=sv[4]; dv2[5]=sv[5]; dv2[6]=sv[6]; dv2[7]=sv[7];
        }
        __syncthreads();
        for (int gt = tid; gt < G * tact; gt += NT) {
            int gg = gt / tact, tt = gt % tact;
            const float *qq = sq + (long)gg * HD;
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
            sP[(long)gg * FTT + tt] = dot * scale;
        }
        __syncthreads();
        if (gd < GD) {
            float tm = -1e30f;
            for (int tt = 0; tt < tact; tt++) tm = fmaxf(tm, sP[(long)g * FTT + tt]);
            float nm = fmaxf(m, tm);
            float rs = expf(m - nm);
            float a = acc * rs, lt = 0.f;
            for (int tt = 0; tt < tact; tt++) {
                float e = expf(sP[(long)g * FTT + tt] - nm);
                float dv = __half2float(sVd[tt * bph + b_of_d]);
                unsigned u = sVq[((long)tt * bph + b_of_d) * 8 + w_of_d];
                a += e * ((float)((signed char)(u >> sh)) * dv);
                lt += e;
            }
            acc = a; l = l * rs + lt; m = nm;
        }
        __syncthreads();
    }
    if (gd < GD) {
        p_acc[((long)(kv * PART + pp) * G + g) * HD + d] = acc;
        if (d == 0) {
            p_m[(kv * PART + pp) * G + g] = m;
            p_l[(kv * PART + pp) * G + g] = l;
        }
    }
}
// Combine PARTials: rescale by global max then normalize. One block/kv.
__global__ void k_unf_combine(const float *__restrict__ p_m,
                      const float *__restrict__ p_l,
                      const float *__restrict__ p_acc,
                      float *__restrict__ out,
                      int G, int HD, int KV, int PART) {
    int kv = blockIdx.x;
    int tid = threadIdx.x;
    int GD = G * HD;
    int gd = tid;
    int g = (gd < GD) ? gd / HD : 0;
    int d = (gd < GD) ? gd % HD : 0;
    if (gd >= GD) return;
    float mall = -1e30f;
    for (int pp = 0; pp < PART; pp++)
        mall = fmaxf(mall, p_m[(kv * PART + pp) * G + g]);
    float num = 0.f, den = 0.f;
    for (int pp = 0; pp < PART; pp++) {
        long hb = (kv * PART + pp) * G + g;
        float a = expf(p_m[hb] - mall);
        num += p_acc[hb * HD + d] * a;
        den += p_l[hb] * a;
    }
    out[((long)(kv * G + g)) * HD + d] = num / den;
}

static double med_ms(std::vector<float> &v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static void run_shape(const char *name, int H, int KV, int HD, int pos, char *dflush) {
    int G = H / KV, n = pos + 1, PART = 2, GD = G * HD;
    int bph = HD / 32;
    long nblk = (long)n * KV * bph;
    float scale = 1.0f / sqrtf((float)HD);
    std::vector<float> hq((long)H * HD);
    std::vector<BlockQ8KV> hkv_t((long)n * KV * bph);
    srand(12345);
    for (auto &x : hq) x = ((rand() % 2000) - 1000) / 1000.0f;
    for (auto &b : hkv_t) {
        b.d = __float2half(0.02f); b.pad = 0;
        for (int i = 0; i < 32; i++) b.qs[i] = (signed char)((rand() % 256) - 128);
    }
    // kv-major: Kn[((kv*n)+t)*bph+b]
    std::vector<BlockQ8KV> hkv(nblk);
    for (int t = 0; t < n; t++) for (int k = 0; k < KV; k++) for (int bb = 0; bb < bph; bb++)
        hkv[((long)k * n + t) * bph + bb] = hkv_t[((long)t * KV + k) * bph + bb];
    float *dq, *dscores, *dout_unf, *dout_fus, *p_m, *p_l, *p_acc;
    BlockQ8KV *dK2, *dV2;
    CK(cudaMalloc(&dq, sizeof(float) * H * HD));
    CK(cudaMalloc(&dK2, sizeof(BlockQ8KV) * nblk));
    CK(cudaMalloc(&dV2, sizeof(BlockQ8KV) * nblk));
    CK(cudaMalloc(&dscores, sizeof(float) * (long)H * n));
    CK(cudaMalloc(&dout_unf, sizeof(float) * H * HD));
    CK(cudaMalloc(&dout_fus, sizeof(float) * H * HD));
    CK(cudaMalloc(&p_m, sizeof(float) * (long)KV * PART * G));
    CK(cudaMalloc(&p_l, sizeof(float) * (long)KV * PART * G));
    CK(cudaMalloc(&p_acc, sizeof(float) * (long)KV * PART * G * HD));
    CK(cudaMemcpy(dq, hq.data(), sizeof(float) * H * HD, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK2, hkv.data(), sizeof(BlockQ8KV) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV2, hkv.data(), sizeof(BlockQ8KV) * nblk, cudaMemcpyHostToDevice));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto flush = [&]() { if (!g_noflush) CK(cudaMemset(dflush, 0xA5, 64 << 20)); };
    auto time_unf = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        k2_qk<<<KV * PART, 256, 0, 0>>>(dq, dK2, dscores, n, G, HD, scale, PART);
        k_softmax_rows<<<H, 256, 0, 0>>>(dscores, n, H);
        k2_pv<<<KV * 2, 256, 0, 0>>>(dscores, dV2, dout_unf, n, G, HD);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); return m;
    };
    auto time_fus = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        k_unf_fused<<<KV * PART, 256, 0, 0>>>(dq, dK2, dV2, p_m, p_l, p_acc,
                                             n, G, HD, scale, PART);
        k_unf_combine<<<KV, GD, 0, 0>>>(p_m, p_l, p_acc, dout_fus, G, HD, KV, PART);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); return m;
    };
    time_unf(); time_fus();
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 200; i++) ms.push_back(time_unf());
    double unf_us = med_ms(ms) * 1000.0;
    ms.clear();
    for (int i = 0; i < 200; i++) ms.push_back(time_fus());
    double fus_us = med_ms(ms) * 1000.0;
    std::vector<float> hu((long)H * HD), hf((long)H * HD);
    CK(cudaMemcpy(hu.data(), dout_unf, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hf.data(), dout_fus, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    double md = 0;
    for (size_t i = 0; i < hu.size(); i++) md = std::max(md, (double)fabsf(hu[i] - hf[i]));
    printf("[%s] H=%d KV=%d HD=%d pos=%d unf-vs-fused maxdiff=%.5f\n",
           name, H, KV, HD, pos, md);
    printf("[%s] unfused median=%8.2f us | fused median=%8.2f us (%.2fx)\n",
           name, unf_us, fus_us, unf_us / fus_us);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    CK(cudaFree(dq)); CK(cudaFree(dK2)); CK(cudaFree(dV2));
    CK(cudaFree(dscores)); CK(cudaFree(dout_unf)); CK(cudaFree(dout_fus));
    CK(cudaFree(p_m)); CK(cudaFree(p_l)); CK(cudaFree(p_acc));
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
