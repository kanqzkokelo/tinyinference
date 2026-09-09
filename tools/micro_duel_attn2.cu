// Attn duel: A=baseline(serial online softmax) vs D=2-way unrolled dual accumulators.
// Llama shapes H=32 KV=8 D=64 S=8 pos=511. Checks acc/m vs baseline.
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
template <int UNROLL2>
__global__ void k_attn(const float *__restrict__ q, const BlockQ8_0 *__restrict__ Kc,
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
    float m0 = -1e30f, l0 = 0.0f, a00 = 0.0f, a01 = 0.0f;
    float m1 = -1e30f, l1 = 0.0f, a10 = 0.0f, a11 = 0.0f;
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
            uint32_t *dK = (uint32_t *)&sK_q[ro], *dV = (uint32_t *)&sV_q[ro];
            for (int u = 0; u < 8; u++) {
                uint32_t ku, vu;
                memcpy(&ku, &bk.qs[u*4], 4); memcpy(&vu, &bv.qs[u*4], 4);
                dK[u] = ku; dV[u] = vu;
            }
        }
        __syncthreads();
        if (UNROLL2) {
            int t = 0;
            for (; t + 1 < bc; t += 2) {
                float dkA = __half2float(sK_d[t * bph + bih]);
                float dvA = __half2float(sV_d[t * bph + bih]);
                float dkB = __half2float(sK_d[(t+1) * bph + bih]);
                float dvB = __half2float(sV_d[(t+1) * bph + bih]);
                int boA = t * head_dim + lane * elems;
                int boB = (t+1) * head_dim + lane * elems;
                uint16_t kuA = *(const uint16_t *)&sK_q[boA];
                uint16_t vuA = *(const uint16_t *)&sV_q[boA];
                uint16_t kuB = *(const uint16_t *)&sK_q[boB];
                uint16_t vuB = *(const uint16_t *)&sV_q[boB];
                float kA0=(float)((int8_t)(kuA)), kA1=(float)((int8_t)(kuA>>8));
                float vA0=(float)((int8_t)(vuA)), vA1=(float)((int8_t)(vuA>>8));
                float kB0=(float)((int8_t)(kuB)), kB1=(float)((int8_t)(kuB>>8));
                float vB0=(float)((int8_t)(vuB)), vB1=(float)((int8_t)(vuB>>8));
                float dpA = (q0*kA0+q1*kA1)*dkA;
                float dpB = (q0*kB0+q1*kB1)*dkB;
                float scA = __shfl_sync(0xffffffff, warp_sum(dpA), 0)*scale;
                float scB = __shfl_sync(0xffffffff, warp_sum(dpB), 0)*scale;
                float mcA = fmaxf(m0, scA);
                float mcB = fmaxf(m1, scB);
                float pA = expf(scA-mcA), alA = expf(m0-mcA);
                float pB = expf(scB-mcB), alB = expf(m1-mcB);
                l0 = l0*alA + pA; l1 = l1*alB + pB;
                a00 = a00*alA + pA*dvA*vA0; a01 = a01*alA + pA*dvA*vA1;
                a10 = a10*alB + pB*dvB*vB0; a11 = a11*alB + pB*dvB*vB1;
                m0 = mcA; m1 = mcB;
            }
            if (t < bc) {
                float dk = __half2float(sK_d[t * bph + bih]);
                float dv = __half2float(sV_d[t * bph + bih]);
                int bo = t * head_dim + lane * elems;
                uint16_t ku = *(const uint16_t *)&sK_q[bo];
                uint16_t vu = *(const uint16_t *)&sV_q[bo];
                float k0=(float)((int8_t)(ku)), k1=(float)((int8_t)(ku>>8));
                float v0=(float)((int8_t)(vu)), v1=(float)((int8_t)(vu>>8));
                float sc = __shfl_sync(0xffffffff, warp_sum((q0*k0+q1*k1)*dk), 0)*scale;
                // merge singleton into stream 0
                float mc = fmaxf(m0, sc);
                float p = expf(sc-mc), al = expf(m0-mc);
                l0 = l0*al + p; a00 = a00*al + p*dv*v0; a01 = a01*al + p*dv*v1; m0 = mc;
            }
            // merge stream1 into stream0
            {
                float mc = fmaxf(m0, m1);
                float al0 = expf(m0-mc), al1 = expf(m1-mc);
                l0 = l0*al0 + l1*al1;
                a00 = a00*al0 + a10*al1; a01 = a01*al0 + a11*al1;
                m0 = mc;
            }
            m1 = -1e30f; l1 = 0.0f; a10 = 0.0f; a11 = 0.0f;
        } else {
            for (int t = 0; t < bc; t++) {
                float dk = __half2float(sK_d[t * bph + bih]);
                float dv = __half2float(sV_d[t * bph + bih]);
                int bo = t * head_dim + lane * elems;
                uint16_t ku = *(const uint16_t *)&sK_q[bo];
                uint16_t vu = *(const uint16_t *)&sV_q[bo];
                float k0=(float)((int8_t)(ku)), k1=(float)((int8_t)(ku>>8));
                float v0=(float)((int8_t)(vu)), v1=(float)((int8_t)(vu>>8));
                float sc = __shfl_sync(0xffffffff, warp_sum((q0*k0+q1*k1)*dk), 0)*scale;
                float mc = fmaxf(m0, sc);
                float p = expf(sc-mc), al = expf(m0-mc);
                l0 = l0*al + p;
                a00 = a00*al + p*dv*v0; a01 = a01*al + p*dv*v1;
                m0 = mc;
            }
        }
        __syncthreads();
    }
    if (lane == 0) { p_m[(size_t)s * n_heads + head] = m0; p_l[(size_t)s * n_heads + head] = l0; }
    float *myacc = p_acc + ((size_t)s * n_heads + head) * head_dim + lane * elems;
    myacc[0] = a00; myacc[1] = a01;
}
static void run_one(const char *tag, void (*k)(const float *, const BlockQ8_0 *, const BlockQ8_0 *, float *, float *, float *, int, float, int),
    const float *q, const BlockQ8_0 *Kc, const BlockQ8_0 *Vc, float *pa, float *pm, float *pl, size_t smem) {
    dim3 grid(8, 8); dim3 blk(128);
    for (int i = 0; i < 50; i++) k<<<grid, blk, smem>>>(q, Kc, Vc, pa, pm, pl, 511, 0.125f, 8);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, e; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&e));
    std::vector<float> ms; ms.reserve(300);
    for (int i = 0; i < 300; i++) {
        CK(cudaEventRecord(a, 0)); k<<<grid, blk, smem>>>(q, Kc, Vc, pa, pm, pl, 511, 0.125f, 8);
        CK(cudaEventRecord(e, 0)); CK(cudaEventSynchronize(e));
        float m = 0; CK(cudaEventElapsedTime(&m, a, e)); ms.push_back(m * 1000.0f);
    }
    std::sort(ms.begin(), ms.end());
    printf("%s med=%.2f us p10=%.2f p90=%.2f\n", tag, ms[150], ms[30], ms[270]);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(e));
}
int main() {
    CK(cudaSetDevice(0));
    const int H = 32, KV = 8, D = 64, T = 512, S = 8, bph = 2;
    size_t nblk = (size_t)T * KV * bph;
    BlockQ8_0 *hK = (BlockQ8_0 *)malloc(nblk * sizeof(BlockQ8_0));
    BlockQ8_0 *hV = (BlockQ8_0 *)malloc(nblk * sizeof(BlockQ8_0));
    uint32_t st = 999;
    for (size_t i = 0; i < nblk; i++) {
        hK[i].d = __float2half(0.02f); hV[i].d = __float2half(0.02f);
        for (int j = 0; j < 32; j++) { st=st*1103515245u+12345u; hK[i].qs[j]=(int8_t)(st>>16); st=st*1103515245u+12345u; hV[i].qs[j]=(int8_t)(st>>16); }
    }
    float *hq = (float *)malloc((size_t)H*D*4);
    for (int i = 0; i < H*D; i++) hq[i] = sinf(0.37f*i);
    float *q; BlockQ8_0 *Kc, *Vc; float *pa, *pm, *pl;
    CK(cudaMalloc(&q,(size_t)H*D*4)); CK(cudaMalloc(&Kc,nblk*sizeof(BlockQ8_0))); CK(cudaMalloc(&Vc,nblk*sizeof(BlockQ8_0)));
    CK(cudaMalloc(&pa,(size_t)S*H*D*4)); CK(cudaMalloc(&pm,(size_t)S*H*4)); CK(cudaMalloc(&pl,(size_t)S*H*4));
    CK(cudaMemcpy(q,hq,(size_t)H*D*4,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Kc,hK,nblk*sizeof(BlockQ8_0),cudaMemcpyHostToDevice));
    CK(cudaMemcpy(Vc,hV,nblk*sizeof(BlockQ8_0),cudaMemcpyHostToDevice));
    size_t smem = (size_t)(BC_SPLIT*bph*2)*2 + (size_t)(BC_SPLIT*D*2);
    for (int r = 0; r < 3; r++) { run_one("A_serial", k_attn<0>, q,Kc,Vc,pa,pm,pl,smem); run_one("D_unroll", k_attn<1>, q,Kc,Vc,pa,pm,pl,smem); }
    std::vector<float> ref((size_t)S*H*D), got((size_t)S*H*D);
    k_attn<0><<<dim3(8,8),dim3(128),smem>>>(q,Kc,Vc,pa,pm,pl,511,0.125f,8); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(),pa,(size_t)S*H*D*4,cudaMemcpyDeviceToHost));
    k_attn<1><<<dim3(8,8),dim3(128),smem>>>(q,Kc,Vc,pa,pm,pl,511,0.125f,8); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(),pa,(size_t)S*H*D*4,cudaMemcpyDeviceToHost));
    double md=0,mr=0; for(size_t i=0;i<got.size();i++){double d=fabs(got[i]-ref[i]); md=std::max(md,d); mr=std::max(mr,d/(fabs(ref[i])+1e-6));}
    printf("D-vs-A maxabs=%.3g maxrel=%.3g\n", md, mr);
    printf("done\n"); return 0;
}
