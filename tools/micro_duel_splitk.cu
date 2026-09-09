// Split-K-within-block duel for Q4_0 layer GEMV. V4 vs SK2 vs SK4.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA err %d\n", (int)e_); exit(2); } } while (0)
typedef struct { __half d; uint8_t qs[16]; } BlockQ4_0;
__device__ __forceinline__ float wrs(float v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
// V4 body factored: rows [row0..row0+3], K-blocks [b0,b1), accumulate to s[4]
__device__ __forceinline__ void v4_range(const BlockQ4_0 *W, const float *x, int row0, int M, int nb, int b0, int b1, int lane, float s[4]) {
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)(row0+1) * nb * 18);
    const uint32_t *rw2 = (const uint32_t *)((const char *)W + (long)(row0+2) * nb * 18);
    const uint32_t *rw3 = (const uint32_t *)((const char *)W + (long)(row0+3) * nb * 18);
    for (int b = b0 + lane; b < b1; b += 32) {
        const int wsc = (18 * b) >> 2;
        const unsigned short d16a = (unsigned short)(((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)(((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const unsigned short d16c = (unsigned short)(((18 * b) & 2) ? (rw2[wsc] >> 16) : (rw2[wsc] & 0xFFFFu));
        const unsigned short d16d = (unsigned short)(((18 * b) & 2) ? (rw3[wsc] >> 16) : (rw3[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const float dc = __half2float(__ushort_as_half(d16c));
        const float dd = __half2float(__ushort_as_half(d16d));
        const int a0 = (18 * b + 2) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t lc = rw2[a0 + k];
            const uint32_t ld = rw3[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const uint32_t vc = sh ? __byte_perm(lc, rw2[a0 + k + 1], 0x5432) : lc;
            const uint32_t vd = sh ? __byte_perm(ld, rw3[a0 + k + 1], 0x5432) : ld;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            s[0] += (float)((int)(va & 0xFu) - 8) * da * xa.x;
            s[0] += (float)((int)((va >> 4) & 0xFu) - 8) * da * xb.x;
            s[0] += (float)((int)((va >> 8) & 0xFu) - 8) * da * xa.y;
            s[0] += (float)((int)((va >> 12) & 0xFu) - 8) * da * xb.y;
            s[0] += (float)((int)((va >> 16) & 0xFu) - 8) * da * xa.z;
            s[0] += (float)((int)((va >> 20) & 0xFu) - 8) * da * xb.z;
            s[0] += (float)((int)((va >> 24) & 0xFu) - 8) * da * xa.w;
            s[0] += (float)((int)(va >> 28) - 8) * da * xb.w;
            s[1] += (float)((int)(vb & 0xFu) - 8) * db * xa.x;
            s[1] += (float)((int)((vb >> 4) & 0xFu) - 8) * db * xb.x;
            s[1] += (float)((int)((vb >> 8) & 0xFu) - 8) * db * xa.y;
            s[1] += (float)((int)((vb >> 12) & 0xFu) - 8) * db * xb.y;
            s[1] += (float)((int)((vb >> 16) & 0xFu) - 8) * db * xa.z;
            s[1] += (float)((int)((vb >> 20) & 0xFu) - 8) * db * xb.z;
            s[1] += (float)((int)((vb >> 24) & 0xFu) - 8) * db * xa.w;
            s[1] += (float)((int)(vb >> 28) - 8) * db * xb.w;
            s[2] += (float)((int)(vc & 0xFu) - 8) * dc * xa.x;
            s[2] += (float)((int)((vc >> 4) & 0xFu) - 8) * dc * xb.x;
            s[2] += (float)((int)((vc >> 8) & 0xFu) - 8) * dc * xa.y;
            s[2] += (float)((int)((vc >> 12) & 0xFu) - 8) * dc * xb.y;
            s[2] += (float)((int)((vc >> 16) & 0xFu) - 8) * dc * xa.z;
            s[2] += (float)((int)((vc >> 20) & 0xFu) - 8) * dc * xb.z;
            s[2] += (float)((int)((vc >> 24) & 0xFu) - 8) * dc * xa.w;
            s[2] += (float)((int)(vc >> 28) - 8) * dc * xb.w;
            s[3] += (float)((int)(vd & 0xFu) - 8) * dd * xa.x;
            s[3] += (float)((int)((vd >> 4) & 0xFu) - 8) * dd * xb.x;
            s[3] += (float)((int)((vd >> 8) & 0xFu) - 8) * dd * xa.y;
            s[3] += (float)((int)((vd >> 12) & 0xFu) - 8) * dd * xb.y;
            s[3] += (float)((int)((vd >> 16) & 0xFu) - 8) * dd * xa.z;
            s[3] += (float)((int)((vd >> 20) & 0xFu) - 8) * dd * xb.z;
            s[3] += (float)((int)((vd >> 24) & 0xFu) - 8) * dd * xa.w;
            s[3] += (float)((int)(vd >> 28) - 8) * dd * xb.w;
        }
    }
}
__global__ void k_v4(const BlockQ4_0 *W, const float *x, float *y, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= M) return;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    float s[4] = {0,0,0,0};
    v4_range(W, x, row0, M, nb, 0, nb, lane, s);
    s[0]=wrs(s[0]); s[1]=wrs(s[1]); s[2]=wrs(s[2]); s[3]=wrs(s[3]);
    if (lane == 0) {
        y[row0]=s[0]; if(row0+1<M) y[row0+1]=s[1]; if(row0+2<M) y[row0+2]=s[2]; if(row0+3<M) y[row0+3]=s[3];
    }
}
template <int SPLIT>
__global__ void k_sk(const BlockQ4_0 *W, const float *x, float *part, int M, int K) {
    const int row0 = (blockIdx.x * blockDim.y + threadIdx.y) * 4;
    if (row0 >= M) return;
    const int s = blockIdx.y;
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const int chunk = (nb + SPLIT - 1) / SPLIT;
    const int b0 = min(s * chunk, nb), b1 = min(b0 + chunk, nb);
    float a[4] = {0,0,0,0};
    v4_range(W, x, row0, M, nb, b0, b1, lane, a);
    a[0]=wrs(a[0]); a[1]=wrs(a[1]); a[2]=wrs(a[2]); a[3]=wrs(a[3]);
    if (lane == 0) {
        float *p = part + ((size_t)s * M + row0) * 1;
        p[0]=a[0]; if(row0+1<M) p[1]=a[1]; if(row0+2<M) p[2]=a[2]; if(row0+3<M) p[3]=a[3];
    }
}
__global__ void k_red(const float *part, float *y, int M, int SPLIT) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= M) return;
    float t = 0;
    for (int s = 0; s < SPLIT; s++) t += part[(size_t)s * M + r];
    y[r] = t;
}
static float timed_ms(std::vector<float> &v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }
template <int SPLIT>
static void duel_sk(const char *tag, int M, int K, BlockQ4_0 *dW, float *dx, float *dy, float *dp) {
    dim3 b(32, 8); dim3 g((M + 31) / 32, SPLIT); dim3 gr((M+255)/256); dim3 br(256);
    for (int i = 0; i < 50; i++) { k_sk<SPLIT><<<g, b>>>(dW, dx, dp, M, K); k_red<<<gr, br>>>(dp, dy, M, SPLIT); }
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, e; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&e));
    std::vector<float> ms; ms.reserve(300);
    for (int i = 0; i < 300; i++) {
        CK(cudaEventRecord(a, 0));
        k_sk<SPLIT><<<g, b>>>(dW, dx, dp, M, K); k_red<<<gr, br>>>(dp, dy, M, SPLIT);
        CK(cudaEventRecord(e, 0)); CK(cudaEventSynchronize(e));
        float m = 0; CK(cudaEventElapsedTime(&m, a, e)); ms.push_back(m * 1000.0f);
    }
    float med = timed_ms(ms);
    std::sort(ms.begin(), ms.end());
    double wB = (double)M * (K/32) * 18.0;
    printf("%s M=%d K=%d med=%.2f us p10=%.2f p90=%.2f GBs=%.1f\n", tag, M, K, med, ms[30], ms[270], wB/(med/1e6)/1e9);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(e));
}
static void duel_v4(const char *tag, int M, int K, BlockQ4_0 *dW, float *dx, float *dy) {
    dim3 b(32, 8); dim3 g((M + 31) / 32);
    for (int i = 0; i < 50; i++) k_v4<<<g, b>>>(dW, dx, dy, M, K);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, e; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&e));
    std::vector<float> ms; ms.reserve(300);
    for (int i = 0; i < 300; i++) {
        CK(cudaEventRecord(a, 0)); k_v4<<<g, b>>>(dW, dx, dy, M, K);
        CK(cudaEventRecord(e, 0)); CK(cudaEventSynchronize(e));
        float m = 0; CK(cudaEventElapsedTime(&m, a, e)); ms.push_back(m * 1000.0f);
    }
    float med = timed_ms(ms);
    std::sort(ms.begin(), ms.end());
    double wB = (double)M * (K/32) * 18.0;
    printf("%s M=%d K=%d med=%.2f us p10=%.2f p90=%.2f GBs=%.1f\n", tag, M, K, med, ms[30], ms[270], wB/(med/1e6)/1e9);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(e));
}
int main() {
    CK(cudaSetDevice(0));
    int Ms[3] = {2048, 2048, 8192}; int Ks[3] = {8192, 2048, 2048};
    const char *Ns[3] = {"down", "o-proj", "gateup"};
    for (int s = 0; s < 3; s++) {
        int M = Ms[s], K = Ks[s];
        size_t nblk = (size_t)M * (K/32);
        BlockQ4_0 *hW = (BlockQ4_0 *)malloc(nblk * sizeof(BlockQ4_0));
        uint32_t st = 1111 + s;
        for (size_t i = 0; i < nblk; i++) {
            hW[i].d = __float2half(0.05f);
            for (int j = 0; j < 16; j++) { st = st*1103515245u+12345u; hW[i].qs[j] = (uint8_t)(st>>16); }
        }
        float *hx = (float *)malloc((size_t)K*4);
        for (int i = 0; i < K; i++) hx[i] = sinf(0.43f*i);
        BlockQ4_0 *dW; float *dx, *dy, *dp;
        CK(cudaMalloc(&dW, nblk*sizeof(BlockQ4_0))); CK(cudaMalloc(&dx,(size_t)K*4));
        CK(cudaMalloc(&dy,(size_t)M*4)); CK(cudaMalloc(&dp,(size_t)4*M*4));
        CK(cudaMemcpy(dW,hW,nblk*sizeof(BlockQ4_0),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx,hx,(size_t)K*4,cudaMemcpyHostToDevice));
        for (int r = 0; r < 2; r++) {
            duel_v4("V4 ", M, K, dW, dx, dy);
            duel_sk<2>("SK2", M, K, dW, dx, dy, dp);
            duel_sk<4>("SK4", M, K, dW, dx, dy, dp);
        }
        printf("-- %s done\n", Ns[s]);
        cudaFree(dW); cudaFree(dx); cudaFree(dy); cudaFree(dp); free(hW); free(hx);
    }
    // correctness at down shape
    {
        int M = 2048, K = 8192;
        size_t nblk = (size_t)M * (K/32);
        BlockQ4_0 *hW = (BlockQ4_0 *)malloc(nblk*sizeof(BlockQ4_0));
        uint32_t st = 777;
        for (size_t i = 0; i < nblk; i++) { hW[i].d = __float2half(0.05f); for (int j=0;j<16;j++){st=st*1103515245u+12345u; hW[i].qs[j]=(uint8_t)(st>>16);} }
        float *hx=(float*)malloc((size_t)K*4); for(int i=0;i<K;i++) hx[i]=sinf(0.43f*i);
        BlockQ4_0 *dW; float *dx,*dy,*dp; CK(cudaMalloc(&dW,nblk*sizeof(BlockQ4_0))); CK(cudaMalloc(&dx,(size_t)K*4)); CK(cudaMalloc(&dy,(size_t)M*4)); CK(cudaMalloc(&dp,(size_t)4*M*4));
        CK(cudaMemcpy(dW,hW,nblk*sizeof(BlockQ4_0),cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx,hx,(size_t)K*4,cudaMemcpyHostToDevice));
        std::vector<float> ref(M), got(M);
        dim3 b(32,8); dim3 g((M+31)/32); dim3 g2((M+31)/32,2); dim3 gr((M+255)/256); dim3 br(256);
        k_v4<<<g,b>>>(dW,dx,dy,M,K); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(ref.data(),dy,(size_t)M*4,cudaMemcpyDeviceToHost));
        k_sk<2><<<g2,b>>>(dW,dx,dp,M,K); k_red<<<gr,br>>>(dp,dy,M,2); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(),dy,(size_t)M*4,cudaMemcpyDeviceToHost));
        double md=0, mr=0; for(int i=0;i<M;i++){double d=fabs(got[i]-ref[i]); md=std::max(md,d); mr=std::max(mr,d/(fabs(ref[i])+1e-6));}
        printf("SK2-vs-V4 maxabs=%.3g maxrel=%.3g\n", md, mr);
    }
    printf("done\n"); return 0;
}
