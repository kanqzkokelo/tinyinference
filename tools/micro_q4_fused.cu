// Duel: current Q4 split-K (k_fa2_q4_split + combine, per-warp serial)
// vs fused Q4 online-softmax (qk + softmax + pv, PARTials + combine).
// Llama shapes: H=32 KV=8 HD=64 G=4. argv[1] present = hot L2.
#include <cstdio>
#include <cstdint>
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
typedef struct { uint16_t d; uint8_t qs[16]; } BlockQ4_0;
#define BC 64
#define FTT 128
__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off /= 2) v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}
// Baseline: faithful port of k_fa2_q4_split (host n, compact t-major KV).
__global__ void k_q4_split(const float *__restrict__ q,
                      const BlockQ4_0 *__restrict__ Kc,
                      const BlockQ4_0 *__restrict__ Vc,
                      float *__restrict__ p_acc, float *__restrict__ p_m,
                      float *__restrict__ p_l,
                      int n, int H, int KV, int HD, float scale, int S) {
    int s = blockIdx.x, kv = blockIdx.y;
    if (s >= S || kv >= KV) return;
    int G = H / KV;
    int tid = threadIdx.x;
    int lane = tid & 31, warp = tid >> 5;
    if (warp >= G) return;
    int head = kv * G + warp;
    int elems = HD / 32;
    const float *qh = q + (long)head * HD + lane * elems;
    float qreg[4] = {0, 0, 0, 0};
    if (elems == 4) { const float4 v = *(const float4 *)qh;
        qreg[0]=v.x; qreg[1]=v.y; qreg[2]=v.z; qreg[3]=v.w; }
    else if (elems == 2) { const float2 v = *(const float2 *)qh;
        qreg[0]=v.x; qreg[1]=v.y; }
    else { for (int i = 0; i < 4; i++) if (i < elems) qreg[i] = qh[i]; }
    int chunk = (n + S - 1) / S;
    int begin = min(n, s * chunk), end = min(n, begin + chunk);
    if (begin >= end) {
        if (lane == 0) { p_m[(size_t)s * H + head] = -1e30f; p_l[(size_t)s * H + head] = 0; }
        float *ma = p_acc + ((size_t)s * H + head) * HD + lane * elems;
        for (int i = 0; i < 4; i++) if (i < elems) ma[i] = 0;
        return;
    }
    int bph = HD / 32;
    extern __shared__ char raw[];
    half *sK_d = (half *)raw;
    half *sV_d = sK_d + BC * bph;
    uint8_t *sK_q = (uint8_t *)(sV_d + BC * bph);
    uint8_t *sV_q = sK_q + BC * (HD / 2);
    float m_prev = -1e30f, l_prev = 0;
    float acc[4] = {0, 0, 0, 0};
    int block_in_head = (lane * elems) / 32;
    int sub = lane & 7;
    bool is_high = (sub >= 4);
    int byte_offset = (sub & 3) * 4;
    uint32_t shift = is_high ? 4 : 0;
    for (int t0 = begin; t0 < end; t0 += BC) {
        int tact = min(BC, end - t0);
        int total = tact * bph;
        for (int i = tid; i < total; i += blockDim.x) {
            int tok = i / bph, b = i % bph;
            long gi = ((long)(t0 + tok) * KV + kv) * bph + b;
            const BlockQ4_0 bk = Kc[gi], bv = Vc[gi];
            sK_d[tok * bph + b] = __ushort_as_half(bk.d);
            sV_d[tok * bph + b] = __ushort_as_half(bv.d);
            int ro = tok * (HD / 2) + b * 16;
            for (int j = 0; j < 8; j++) {
                ((uint16_t *)&sK_q[ro])[j] = ((const uint16_t *)&bk.qs[0])[j];
                ((uint16_t *)&sV_q[ro])[j] = ((const uint16_t *)&bv.qs[0])[j];
            }
        }
        __syncthreads();
        for (int ti = 0; ti < tact; ti++) {
            float dk = __half2float(sK_d[ti * bph + block_in_head]);
            float dv = __half2float(sV_d[ti * bph + block_in_head]);
            float kval[4] = {0, 0, 0, 0}, vval[4] = {0, 0, 0, 0};
            if (elems == 2) {
                size_t kr = (size_t)ti * (HD / 2) + block_in_head * 16;
                int vb = (lane & 15) * 2;
                uint8_t k0 = sK_q[kr + ((vb    ) & 15)], k1 = sK_q[kr + ((vb + 1) & 15)];
                uint8_t v0 = sV_q[kr + ((vb    ) & 15)], v1 = sV_q[kr + ((vb + 1) & 15)];
                kval[0] = (float)(((int)((k0 >> ((vb     >= 16) ? 4 : 0)) & 15)) - 8);
                kval[1] = (float)(((int)((k1 >> ((vb + 1 >= 16) ? 4 : 0)) & 15)) - 8);
                vval[0] = (float)(((int)((v0 >> ((vb     >= 16) ? 4 : 0)) & 15)) - 8);
                vval[1] = (float)(((int)((v1 >> ((vb + 1 >= 16) ? 4 : 0)) & 15)) - 8);
            } else {
                const uint32_t ku = *(const uint32_t *)&sK_q[ti * (HD / 2) + block_in_head * 16 + byte_offset];
                const uint32_t vu = *(const uint32_t *)&sV_q[ti * (HD / 2) + block_in_head * 16 + byte_offset];
                uint32_t ks = ku >> shift, vs = vu >> shift;
                kval[0]=(float)((int)(ks        & 15)-8); kval[1]=(float)((int)((ks>>8)&15)-8);
                kval[2]=(float)((int)((ks>>16)&15)-8);   kval[3]=(float)((int)((ks>>24)&15)-8);
                vval[0]=(float)((int)(vs        & 15)-8); vval[1]=(float)((int)((vs>>8)&15)-8);
                vval[2]=(float)((int)((vs>>16)&15)-8);   vval[3]=(float)((int)((vs>>24)&15)-8);
            }
            float dp = (qreg[0]*kval[0]+qreg[1]*kval[1]+qreg[2]*kval[2]+qreg[3]*kval[3])*dk;
            float score = __shfl_sync(0xffffffff, warp_sum(dp), 0) * scale;
            float mc = fmaxf(m_prev, score);
            float p = expf(score - mc), alpha = expf(m_prev - mc);
            l_prev = l_prev * alpha + p;
            float pdv = p * dv;
            acc[0]=acc[0]*alpha+pdv*vval[0]; acc[1]=acc[1]*alpha+pdv*vval[1];
            acc[2]=acc[2]*alpha+pdv*vval[2]; acc[3]=acc[3]*alpha+pdv*vval[3];
            m_prev = mc;
        }
        __syncthreads();
    }
    if (lane == 0) { p_m[(size_t)s * H + head] = m_prev; p_l[(size_t)s * H + head] = l_prev; }
    float *ma = p_acc + ((size_t)s * H + head) * HD + lane * elems;
    for (int i = 0; i < 4; i++) if (i < elems) ma[i] = acc[i];
}
__global__ void k_q4_combine(const float *__restrict__ p_acc,
                      const float *__restrict__ p_m, const float *__restrict__ p_l,
                      float *__restrict__ out, int H, int HD, int S) {
    int h = blockIdx.x;
    if (h >= H) return;
    int lane = threadIdx.x, elems = HD / 32;
    float mg = -1e30f, lg = 0, acc[4] = {0, 0, 0, 0};
    for (int s = 0; s < S; s++) {
        float ms = p_m[s * H + h], ls = p_l[s * H + h];
        if (!isfinite(ms) || !isfinite(ls) || ls <= 0 || ms <= -1e20f) continue;
        float mn = fmaxf(mg, ms);
        float ap = expf(mg - mn), as = expf(ms - mn);
        lg = lg * ap + ls * as;
        const float *pp = p_acc + ((size_t)s * H + h) * HD + lane * elems;
        if (elems == 4) { const float4 v = *(const float4 *)pp;
            acc[0]=acc[0]*ap+v.x*as; acc[1]=acc[1]*ap+v.y*as;
            acc[2]=acc[2]*ap+v.z*as; acc[3]=acc[3]*ap+v.w*as; }
        else if (elems == 2) { const float2 v = *(const float2 *)pp;
            acc[0]=acc[0]*ap+v.x*as; acc[1]=acc[1]*ap+v.y*as; }
        else { for (int i = 0; i < 4; i++) if (i < elems) acc[i]=acc[i]*ap+pp[i]*as; }
        mg = mn;
    }
    float *o = out + (long)h * HD + lane * elems;
    if (elems == 4) { float4 v = make_float4(acc[0]/lg, acc[1]/lg, acc[2]/lg, acc[3]/lg);
        *(float4 *)o = v; }
    else if (elems == 2) { float2 v = make_float2(acc[0]/lg, acc[1]/lg); *(float2 *)o = v; }
    else { for (int i = 0; i < 4; i++) if (i < elems) o[i] = acc[i] / lg; }
}
// Fused Q4: one block per (kv, part). Dots in smem, online softmax per
// head in registers, V consumed from the same tile. GD <= 256.
__global__ void k_q4_fused(const float *__restrict__ q,
                      const BlockQ4_0 *__restrict__ Kn,
                      const BlockQ4_0 *__restrict__ Vn,
                      float *__restrict__ p_m, float *__restrict__ p_l,
                      float *__restrict__ p_acc,
                      int n, int H, int KV, int HD, float scale, int PART) {
    int kv = blockIdx.x / PART;
    int pp = blockIdx.x % PART;
    int tid = threadIdx.x, NT = blockDim.x;
    int G = H / KV, bph = HD / 32, GD = G * HD;
    int chunk = (n + PART - 1) / PART;
    int r0 = min(n, pp * chunk), r1 = min(n, r0 + chunk);
    __shared__ float sq[8 * 128];
    for (int i = tid; i < G * HD; i += NT)
        sq[i] = q[(long)(kv * G + i / HD) * HD + i % HD];
    __shared__ half sKd[FTT * 4];
    __shared__ uint8_t sKq[FTT * 64];
    __shared__ half sVd[FTT * 4];
    __shared__ uint8_t sVq[FTT * 64];
    __shared__ float sP[8 * FTT];
    int gd = tid;
    int g = (gd < GD) ? gd / HD : 0;
    int d = (gd < GD) ? gd % HD : 0;
    int b_of_d = d / 32, bd_q4 = d % 32, vbyte_q4 = bd_q4 & 15, vhi_q4 = bd_q4 >> 4;
    float m = -1e30f, l = 0.f, acc = 0.f;
    for (int t0 = r0; t0 < r1; t0 += FTT) {
        int tact = min(FTT, r1 - t0);
        for (int i = tid; i < tact * bph; i += NT) {
            int tt = i / bph, b = i % bph;
            long gi = ((long)(t0 + tt) * KV + kv) * bph + b;
            const BlockQ4_0 bk = Kn[gi], bv = Vn[gi];
            sKd[tt * bph + b] = __ushort_as_half(bk.d);
            sVd[tt * bph + b] = __ushort_as_half(bv.d);
            uint8_t *dk2 = sKq + ((long)tt * bph + b) * 16;
            uint8_t *dv2 = sVq + ((long)tt * bph + b) * 16;
            for (int j = 0; j < 16; j++) { dk2[j] = bk.qs[j]; dv2[j] = bv.qs[j]; }
        }
        __syncthreads();
        for (int gt = tid; gt < G * tact; gt += NT) {
            int gg = gt / tact, tt = gt % tact;
            const float *qq = sq + (long)gg * HD;
            float dot = 0.f;
            for (int b = 0; b < bph; b++) {
                float dk = __half2float(sKd[tt * bph + b]);
                const uint8_t *kb = sKq + ((long)tt * bph + b) * 16;
                const float *qb = qq + b * 32;
                for (int j = 0; j < 16; j++) {
                    uint8_t u = kb[j];
                    dot += qb[j] * ((float)(u & 15) - 8.f) * dk
                         + qb[j+16] * ((float)(u >> 4) - 8.f) * dk;
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
                uint8_t u = sVq[((long)tt * bph + b_of_d) * 16 + vbyte_q4];
                float vv = ((float)((u >> (vhi_q4 * 4)) & 15) - 8.f) * dv;
                a += e * vv;
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
__global__ void k_q4_fcombine(const float *__restrict__ p_m,
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
    int G = H / KV, n = pos + 1, GD = G * HD;
    int bph = HD / 32;
    long nblk = (long)n * KV * bph;
    float scale = 1.0f / sqrtf((float)HD);
    int S = (pos + 63) / 64;
    if (S < 2) S = 2;
    if (S > 64) S = 64;
    int PART = 2;
    std::vector<float> hq((long)H * HD);
    std::vector<BlockQ4_0> hkv(nblk);
    srand(12345);
    for (auto &x : hq) x = ((rand() % 2000) - 1000) / 1000.0f;
    uint16_t dbits; { __half h = __float2half(0.02f); dbits = *reinterpret_cast<uint16_t *>(&h); }
    for (auto &b : hkv) {
        b.d = dbits;
        for (int i = 0; i < 16; i++) b.qs[i] = (uint8_t)(rand() % 256);
    }
    float *dq, *dout_sp, *dout_fus, *p_m, *p_l, *p_acc, *f_m, *f_l, *f_acc;
    BlockQ4_0 *dK, *dV;
    CK(cudaMalloc(&dq, sizeof(float) * H * HD));
    CK(cudaMalloc(&dK, sizeof(BlockQ4_0) * nblk));
    CK(cudaMalloc(&dV, sizeof(BlockQ4_0) * nblk));
    CK(cudaMalloc(&dout_sp, sizeof(float) * H * HD));
    CK(cudaMalloc(&dout_fus, sizeof(float) * H * HD));
    CK(cudaMalloc(&p_m, sizeof(float) * (long)S * H));
    CK(cudaMalloc(&p_l, sizeof(float) * (long)S * H));
    CK(cudaMalloc(&p_acc, sizeof(float) * (long)S * H * HD));
    CK(cudaMalloc(&f_m, sizeof(float) * (long)KV * PART * G));
    CK(cudaMalloc(&f_l, sizeof(float) * (long)KV * PART * G));
    CK(cudaMalloc(&f_acc, sizeof(float) * (long)KV * PART * G * HD));
    CK(cudaMemcpy(dq, hq.data(), sizeof(float) * H * HD, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hkv.data(), sizeof(BlockQ4_0) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hkv.data(), sizeof(BlockQ4_0) * nblk, cudaMemcpyHostToDevice));
    size_t smem_sp = 2 * (size_t)BC * bph * sizeof(half) + 2 * (size_t)BC * (HD / 2);
    dim3 grid_sp(S, KV);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto flush = [&]() { if (!g_noflush) CK(cudaMemset(dflush, 0xA5, 64 << 20)); };
    auto time_sp = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        k_q4_split<<<grid_sp, G * 32, smem_sp, 0>>>(dq, dK, dV, p_acc, p_m, p_l,
                                                   n, H, KV, HD, scale, S);
        k_q4_combine<<<H, 32, 0, 0>>>(p_acc, p_m, p_l, dout_sp, H, HD, S);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); return m;
    };
    auto time_fus = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        k_q4_fused<<<KV * PART, 256, 0, 0>>>(dq, dK, dV, f_m, f_l, f_acc,
                                            n, H, KV, HD, scale, PART);
        k_q4_fcombine<<<KV, GD, 0, 0>>>(f_m, f_l, f_acc, dout_fus, G, HD, KV, PART);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); return m;
    };
    time_sp(); time_fus();
    CK(cudaDeviceSynchronize());
    std::vector<float> ms;
    for (int i = 0; i < 200; i++) ms.push_back(time_sp());
    double sp_us = med_ms(ms) * 1000.0;
    ms.clear();
    for (int i = 0; i < 200; i++) ms.push_back(time_fus());
    double fus_us = med_ms(ms) * 1000.0;
    std::vector<float> hs((long)H * HD), hf((long)H * HD);
    CK(cudaMemcpy(hs.data(), dout_sp, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hf.data(), dout_fus, sizeof(float) * H * HD, cudaMemcpyDeviceToHost));
    double md = 0, ref = 0;
    for (size_t i = 0; i < hs.size(); i++) {
        md = std::max(md, (double)fabsf(hs[i] - hf[i]));
        ref = std::max(ref, (double)fabsf(hs[i]));
    }
    printf("[%s] H=%d KV=%d HD=%d pos=%d S=%d maxdiff=%.6f (refmax=%.3f)\n",
           name, H, KV, HD, pos, S, md, ref);
    printf("[%s] split median=%8.2f us | fused median=%8.2f us (%.2fx)\n",
           name, sp_us, fus_us, sp_us / fus_us);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}
int main(int argc, char **argv) {
    g_noflush = argc > 1;
    char *dflush;
    CK(cudaMalloc(&dflush, 64 << 20));
    run_shape("llama-510", 32, 8, 64, 510, dflush);
    run_shape("llama-2k", 32, 8, 64, 2048, dflush);
    return 0;
}
