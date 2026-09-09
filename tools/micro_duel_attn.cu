// Duel attention split-K variants at Llama-3.2-1B shapes (H=32 KV=8 D=64 S=8 pos=511)
// A=baseline B=u32-staging C=B+__expf. Self-contained. Validates m/l/acc match.
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
#define BC_SPLIT 64
struct BlockQ8_0 { __half d; int8_t qs[32]; };
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
template <int VEC_STAGE, int FAST_EXP>
__global__ void k_attn_duel(const float *__restrict__ q, const BlockQ8_0 *__restrict__ Kc,
    const BlockQ8_0 *__restrict__ Vc, float *__restrict__ p_acc, float *__restrict__ p_m,
    float *__restrict__ p_l, int pos, float scale, int S) {
    const int n_heads = 32, n_kv_heads = 8, head_dim = 64;
    const int s = blockIdx.x, kv = blockIdx.y;
    if (s >= S || kv >= n_kv_heads) return;
    const int G = 4;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (warp >= G) return;
    const int head = kv * G + warp, elems = 2, bph = 2;
    const float *qh = q + (long)head * head_dim + lane * elems;
    const float2 qv2 = *(const float2 *)qh;
    float q0 = qv2.x, q1 = qv2.y;
    int nslots = pos + 1;
    int chunk = (nslots + S - 1) / S;
    int begin = s * chunk, end = min(pos + 1, (s + 1) * chunk);
    if (begin >= end) {
        if (lane == 0) { p_m[(size_t)s * n_heads + head] = -1e30f; p_l[(size_t)s * n_heads + head] = 0.0f; }
        float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
        myacc[0] = 0.0f; myacc[1] = 0.0f;
        return;
    }
    extern __shared__ char raw[];
    half *sK_d = (half *)raw;
    half *sV_d = sK_d + BC_SPLIT * bph;
    int8_t *sK_q = (int8_t *)(sV_d + BC_SPLIT * bph);
    int8_t *sV_q = sK_q + BC_SPLIT * head_dim;
    float m_prev = -1e30f, l_prev = 0.0f, a0 = 0.0f, a1 = 0.0f;
    const int bih = (lane * elems) / 32;
    for (int t_tile = begin; t_tile < end; t_tile += BC_SPLIT) {
        int t_end = min(end, t_tile + BC_SPLIT);
        int bc = t_end - t_tile;
        int total = bc * bph;
        for (int i = tid; i < total; i += blockDim.x) {
            int tok = i / bph, b = i % bph;
            long gi = ((long)(t_tile + tok) * n_kv_heads + kv) * bph + b;
            const BlockQ8_0 bk = Kc[gi];
            const BlockQ8_0 bv = Vc[gi];
            sK_d[tok * bph + b] = bk.d;
            sV_d[tok * bph + b] = bv.d;
            int ro = tok * head_dim + b * 32;
            if (VEC_STAGE) {
                uint32_t *dK = (uint32_t *)&sK_q[ro], *dV = (uint32_t *)&sV_q[ro];
                for (int u = 0; u < 8; u++) {
                    uint32_t ku, vu;
                    memcpy(&ku, &bk.qs[u*4], 4); memcpy(&vu, &bv.qs[u*4], 4);
                    dK[u] = ku; dV[u] = vu;
                }
            } else {
#pragma unroll
                for (int j = 0; j < 32; j++) { sK_q[ro + j] = bk.qs[j]; sV_q[ro + j] = bv.qs[j]; }
            }
        }
        __syncthreads();
        for (int t = 0; t < bc; t++) {
            float dk = __half2float(sK_d[t * bph + bih]);
            float dv = __half2float(sV_d[t * bph + bih]);
            int bo = t * head_dim + lane * elems;
            uint16_t ku = *(const uint16_t *)&sK_q[bo];
            uint16_t vu = *(const uint16_t *)&sV_q[bo];
            float k0 = (float)((int8_t)(ku)), k1 = (float)((int8_t)(ku >> 8));
            float v0 = (float)((int8_t)(vu)), v1 = (float)((int8_t)(vu >> 8));
            float dp = (q0 * k0 + q1 * k1) * dk;
            float sc = __shfl_sync(0xffffffff, warp_sum(dp), 0) * scale;
            float mc = fmaxf(m_prev, sc);
            float p = FAST_EXP ? __expf(sc - mc) : expf(sc - mc);
            float al = FAST_EXP ? __expf(m_prev - mc) : expf(m_prev - mc);
            l_prev = l_prev * al + p;
            float pdv = p * dv;
            a0 = a0 * al + pdv * v0;
            a1 = a1 * al + pdv * v1;
            m_prev = mc;
        }
        __syncthreads();
    }
    if (lane == 0) { p_m[(size_t)s * n_heads + head] = m_prev; p_l[(size_t)s * n_heads + head] = l_prev; }
    float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
    myacc[0] = a0; myacc[1] = a1;
}
static void run_one(const char *tag,
    void (*k)(const float *, const BlockQ8_0 *, const BlockQ8_0 *, float *, float *, float *, int, float, int),
    const float *q, const BlockQ8_0 *Kc, const BlockQ8_0 *Vc, float *pa, float *pm, float *pl, size_t smem) {
    dim3 grid(8, 8); dim3 blk(128);
    for (int i = 0; i < 50; i++) k<<<grid, blk, smem>>>(q, Kc, Vc, pa, pm, pl, 511, 0.125f, 8);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> ms; ms.reserve(300);
    for (int i = 0; i < 300; i++) {
        CK(cudaEventRecord(a, 0)); k<<<grid, blk, smem>>>(q, Kc, Vc, pa, pm, pl, 511, 0.125f, 8);
        CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
        float m = 0; CK(cudaEventElapsedTime(&m, a, b)); ms.push_back(m * 1000.0f);
    }
    std::sort(ms.begin(), ms.end());
    printf("%s med=%.2f us p10=%.2f p90=%.2f\n", tag, ms[150], ms[30], ms[270]);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}
int main() {
    CK(cudaSetDevice(0));
    const int H = 32, KV = 8, D = 64, T = 512, S = 8, bph = 2;
    size_t nblk = (size_t)T * KV * bph;
    BlockQ8_0 *hK = (BlockQ8_0 *)malloc(nblk * sizeof(BlockQ8_0));
    BlockQ8_0 *hV = (BlockQ8_0 *)malloc(nblk * sizeof(BlockQ8_0));
    uint32_t st = 999;
    for (size_t i = 0; i < nblk; i++) {
        hK[i].d = __float2half(0.02f);
        hV[i].d = __float2half(0.02f);
        for (int j = 0; j < 32; j++) {
            st = st * 1103515245u + 12345u; hK[i].qs[j] = (int8_t)(st >> 16);
            st = st * 1103515245u + 12345u; hV[i].qs[j] = (int8_t)(st >> 16);
        }
    }
    float *hq = (float *)malloc((size_t)H * D * 4);
    for (int i = 0; i < H * D; i++) hq[i] = sinf(0.37f * i);
    float *q; BlockQ8_0 *Kc, *Vc;
    float *pa, *pm, *pl;
    CK(cudaMalloc(&q, (size_t)H * D * 4));
    CK(cudaMalloc(&Kc, nblk * sizeof(BlockQ8_0)));
    CK(cudaMalloc(&Vc, nblk * sizeof(BlockQ8_0)));
    CK(cudaMalloc(&pa, (size_t)S * H * D * 4));
    CK(cudaMalloc(&pm, (size_t)S * H * 4));
    CK(cudaMalloc(&pl, (size_t)S * H * 4));
    CK(cudaMemcpy(q, hq, (size_t)H * D * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Kc, hK, nblk * sizeof(BlockQ8_0), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Vc, hV, nblk * sizeof(BlockQ8_0), cudaMemcpyHostToDevice));
    size_t smem = (size_t)(BC_SPLIT * bph * 2) * 2 + (size_t)(BC_SPLIT * D * 2);
    printf("smem=%zu\n", smem);
    for (int r = 0; r < 3; r++) {
        run_one("A_base", k_attn_duel<0, 0>, q, Kc, Vc, pa, pm, pl, smem);
        run_one("B_u32 ", k_attn_duel<1, 0>, q, Kc, Vc, pa, pm, pl, smem);
        run_one("C_fast", k_attn_duel<1, 1>, q, Kc, Vc, pa, pm, pl, smem);
    }
    std::vector<float> ref((size_t)S * H * D), refm((size_t)S * H);
    CK(cudaMemset(pa, 0, (size_t)S * H * D * 4));
    k_attn_duel<0, 0><<<dim3(8, 8), dim3(128), smem>>>(q, Kc, Vc, pa, pm, pl, 511, 0.125f, 8);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), pa, (size_t)S * H * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(refm.data(), pm, (size_t)S * H * 4, cudaMemcpyDeviceToHost));
    std::vector<float> got((size_t)S * H * D), gotm((size_t)S * H);
    k_attn_duel<1, 0><<<dim3(8, 8), dim3(128), smem>>>(q, Kc, Vc, pa, pm, pl, 511, 0.125f, 8);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), pa, (size_t)S * H * D * 4, cudaMemcpyDeviceToHost));
    double md = 0; for (size_t i = 0; i < got.size(); i++) md = std::max(md, (double)fabsf(got[i] - ref[i]));
    printf("B-vs-A maxabsdiff=%.3g\n", md);
    k_attn_duel<1, 1><<<dim3(8, 8), dim3(128), smem>>>(q, Kc, Vc, pa, pm, pl, 511, 0.125f, 8);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), pa, (size_t)S * H * D * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gotm.data(), pm, (size_t)S * H * 4, cudaMemcpyDeviceToHost));
    md = 0; for (size_t i = 0; i < got.size(); i++) md = std::max(md, (double)fabsf(got[i] - ref[i]));
    double mm = 0; for (size_t i = 0; i < gotm.size(); i++) mm = std::max(mm, (double)fabsf(gotm[i] - refm[i]));
    printf("C-vs-A maxabsdiff acc=%.3g m=%.3g\n", md, mm);
    printf("done\n"); return 0;
}
