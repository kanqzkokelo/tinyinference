// Duel: unfused decode attention over Q4_0 KV (QK-gemv + softmax + PV-gemv)
// vs fused k_fa2_q4_split (tt_flash_gqa_q4_0_splitk). Same-process A-B.
// NOTE Q4 engine cache is t-major: blk[((t*KV)+kv)*bph+b], d raw fp16 bits.
// argv[1] present = hot L2 (skip flush).
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

static int g_noflush = 0;

static void ck_(cudaError_t e, const char *f, int l) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error %d at %s:%d\n", (int)e, f, l);
        exit(1);
    }
}
#define CK(x) ck_((x), __FILE__, __LINE__)

extern "C" {
int tt_flash_gqa_q4_0_splitk(const float *q, const void *Kc_q4, const void *Vc_q4,
    float *acc, float *st_m, float *st_l, float *out, const int *d_pos,
    int n_heads, int n_kv_heads, int head_dim, float scale, int window, int S,
    cudaStream_t stream);
}

struct BlockQ4KV { uint16_t d; uint8_t qs[16]; };

#define TT2 256
__global__ void k4_qk(const float *__restrict__ q, const BlockQ4KV *__restrict__ Kn,
                      float *__restrict__ scores, int n, int KV, int G, int HD,
                      float scale, int PART) {
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
    __shared__ uint8_t sKq[TT2 * 4 * 16];
    for (int t0 = r0; t0 < r1; t0 += TT2) {
        int tact = min(TT2, r1 - t0);
        for (int i = tid; i < tact * bph; i += NT) {
            int tt = i / bph, b = i % bph;
            long gi = ((long)(t0 + tt) * KV + kv) * bph + b;
            BlockQ4KV blk = Kn[gi];
            sKd[tt * bph + b] = __ushort_as_half(blk.d);
            uint8_t *dst = sKq + ((long)tt * bph + b) * 16;
            for (int j = 0; j < 16; j++) dst[j] = blk.qs[j];
        }
        __syncthreads();
        for (int gt = tid; gt < G * tact; gt += NT) {
            int g = gt / tact, tt = gt % tact;
            const float *qq = sq + (long)g * HD;
            float dot = 0.f;
            for (int b = 0; b < bph; b++) {
                float dk = __half2float(sKd[tt * bph + b]);
                const uint8_t *qb = sKq + ((long)tt * bph + b) * 16;
                const float *qf = qq + b * 32;
                for (int v = 0; v < 32; v++) {
                    int by = qb[v & 15];
                    int nib = (v < 16) ? (by & 15) : (by >> 4);
                    dot += qf[v] * ((float)(nib - 8) * dk);
                }
            }
            scores[((long)(kv * G + g)) * n + t0 + tt] = dot * scale;
        }
        __syncthreads();
    }
}

__global__ void k4_pv(const float *__restrict__ scores, const BlockQ4KV *__restrict__ Vn,
                      float *__restrict__ out, int n, int KV, int G, int HD) {
    int kv = blockIdx.x / 2, cc = blockIdx.x % 2;
    int tid = threadIdx.x, NT = blockDim.x;
    int bph = HD / 32;
    int HD2 = HD / 2, d0 = cc * HD2;
    int GD2 = G * HD2;
    __shared__ half sVd[TT2 * 4];
    __shared__ uint8_t sVq[TT2 * 4 * 16];
    __shared__ float sP[8 * 256];
    float acc = 0.f;
    int gd = tid;
    int g = (gd < GD2) ? gd / HD2 : 0;
    int d = (gd < GD2) ? d0 + gd % HD2 : 0;
    int b_of_d = d / 32, v_in_b = d % 32;
    for (int t0 = 0; t0 < n; t0 += TT2) {
        int tact = min(TT2, n - t0);
        for (int i = tid; i < tact * bph; i += NT) {
            int tt = i / bph, b = i % bph;
            long gi = ((long)(t0 + tt) * KV + kv) * bph + b;
            BlockQ4KV blk = Vn[gi];
            sVd[tt * bph + b] = __ushort_as_half(blk.d);
            uint8_t *dst = sVq + ((long)tt * bph + b) * 16;
            for (int j = 0; j < 16; j++) dst[j] = blk.qs[j];
        }
        __syncthreads();
        for (int i = tid; i < G * tact; i += NT)
            sP[(long)(i / tact) * TT2 + i % tact] =
                scores[((long)(kv * G + i / tact)) * n + t0 + i % tact];
        __syncthreads();
        if (gd < GD2) {
            for (int tt = 0; tt < tact; tt++) {
                float dv = __half2float(sVd[tt * bph + b_of_d]);
                int by = sVq[((long)tt * bph + b_of_d) * 16 + (v_in_b & 15)];
                int nib = (v_in_b < 16) ? (by & 15) : (by >> 4);
                acc += sP[(long)g * TT2 + tt] * ((float)(nib - 8) * dv);
            }
        }
        __syncthreads();
    }
    if (gd < GD2) out[((long)(kv * G + g)) * HD + d] = acc;
}

__global__ void k4_softmax_rows(float *__restrict__ scores, int n, int H) {
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
    std::vector<BlockQ4KV> hkv(nblk);
    srand(12345);
    for (size_t i = 0; i < hq.size(); i++) hq[i] = ((rand() % 2000) - 1000) / 1000.0f;
    uint16_t dbits = __half_as_ushort(__float2half(0.02f));
    for (long i = 0; i < nblk; i++) {
        hkv[i].d = dbits;
        for (int j = 0; j < 16; j++) hkv[i].qs[j] = (uint8_t)(rand() % 256);
    }
    float *dq, *dout_split, *dout_unf, *dacc, *dm, *dl, *dscores;
    BlockQ4KV *dK, *dV;
    int *dpos;
    CK(cudaMalloc(&dq, sizeof(float) * H * HD));
    CK(cudaMalloc(&dK, sizeof(BlockQ4KV) * nblk));
    CK(cudaMalloc(&dV, sizeof(BlockQ4KV) * nblk));
    CK(cudaMalloc(&dout_split, sizeof(float) * H * HD));
    CK(cudaMalloc(&dout_unf, sizeof(float) * H * HD));
    CK(cudaMalloc(&dscores, sizeof(float) * (long)H * n));
    CK(cudaMalloc(&dacc, sizeof(float) * (long)S * H * HD));
    CK(cudaMalloc(&dm, sizeof(float) * (long)S * H));
    CK(cudaMalloc(&dl, sizeof(float) * (long)S * H));
    CK(cudaMalloc(&dpos, sizeof(int)));
    CK(cudaMemcpy(dq, hq.data(), sizeof(float) * H * HD, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dK, hkv.data(), sizeof(BlockQ4KV) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dV, hkv.data(), sizeof(BlockQ4KV) * nblk, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, &pos, sizeof(int), cudaMemcpyHostToDevice));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto flush = [&]() { if (!g_noflush) CK(cudaMemset(dflush, 0xA5, 64 << 20)); };
    auto time_split = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        tt_flash_gqa_q4_0_splitk(dq, dK, dV, dacc, dm, dl, dout_split, dpos,
                                 H, KV, HD, scale, 0, S, 0);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m; CK(cudaEventElapsedTime(&m, a, b)); return m;
    };
    auto time_unf = [&]() {
        flush(); CK(cudaEventRecord(a, 0));
        k4_qk<<<KV * 2, 256, 0, 0>>>(dq, dK, dscores, n, KV, G, HD, scale, 2);
        k4_softmax_rows<<<H, 256, 0, 0>>>(dscores, n, H);
        k4_pv<<<KV * 2, 256, 0, 0>>>(dscores, dV, dout_unf, n, KV, G, HD);
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
            k4_qk<<<KV * 2, 256, 0, 0>>>(dq, dK, dscores, n, KV, G, HD, scale, 2);
            CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&t, e0, e1)); m1.push_back(t);
        }
        for (int i = 0; i < 50; i++) {
            flush(); CK(cudaEventRecord(e0, 0));
            k4_softmax_rows<<<H, 256, 0, 0>>>(dscores, n, H);
            CK(cudaEventRecord(e1, 0)); CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&t, e0, e1)); m2.push_back(t);
        }
        for (int i = 0; i < 50; i++) {
            flush(); CK(cudaEventRecord(e0, 0));
            k4_pv<<<KV * 2, 256, 0, 0>>>(dscores, dV, dout_unf, n, KV, G, HD);
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
    run_shape("qwen25-2k", 14, 2, 64, 2048, dflush);
    run_shape("smol-2k", 9, 3, 64, 2048, dflush);
    run_shape("smol-510", 9, 3, 64, 510, dflush);
    return 0;
}
