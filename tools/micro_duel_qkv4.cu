// Duel: 3x V4 sequential vs fused QKV-V4.
// llama-3.2-1b shapes: Mq=2048 Mk=Mv=512 K=2048.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA err\n"); exit(2); } } while (0)
extern "C" {
extern int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_qkv(const void *dWq, const void *dWk, const void *dWv, const float *dx, float *dyq, float *dyk, float *dyv, int Mq, int Mk, int Mv, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_r8(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
}
static float med(std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size()/2]; }
__device__ __forceinline__ float qkv2_reduce(float v) {
#pragma unroll
    for (int o = 16; o > 0; o /= 2) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}
/* 2-rows/warp QKV variant: halve regs vs 4-row fused, 2x warps. Same V4 math. */
__global__ void k_gemv_q4_0_qkv2(const uint32_t *__restrict__ Wq,
                                const uint32_t *__restrict__ Wk,
                                const uint32_t *__restrict__ Wv,
                                const float *__restrict__ x,
                                float *__restrict__ yq,
                                float *__restrict__ yk,
                                float *__restrict__ yv,
                                int Mq, int Mk, int Mv, int K) {
    const int warp = blockIdx.x * blockDim.y + threadIdx.y;
    const int nq2 = Mq / 2, nk2 = Mk / 2;
    const uint32_t *W; float *y; int row0;
    if (warp < nq2) { W = Wq; y = yq; row0 = warp * 2; }
    else if (warp < nq2 + nk2) { W = Wk; y = yk; row0 = (warp - nq2) * 2; }
    else if (warp < nq2 + nk2 + Mv / 2) { W = Wv; y = yv; row0 = (warp - nq2 - nk2) * 2; }
    else return;
    const int M = (W == Wq) ? Mq : ((W == Wk) ? Mk : Mv);
    const int lane = threadIdx.x;
    const int nb = K / 32;
    const uint32_t *rw0 = (const uint32_t *)((const char *)W + (long)row0 * nb * 18);
    const uint32_t *rw1 = (const uint32_t *)((const char *)W + (long)(row0 + 1) * nb * 18);
    float s0 = 0.0f, s1 = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        const int wsc = (18 * b) >> 2;
        const unsigned short d16a = (unsigned short)
            (((18 * b) & 2) ? (rw0[wsc] >> 16) : (rw0[wsc] & 0xFFFFu));
        const unsigned short d16b = (unsigned short)
            (((18 * b) & 2) ? (rw1[wsc] >> 16) : (rw1[wsc] & 0xFFFFu));
        const float da = __half2float(__ushort_as_half(d16a));
        const float db = __half2float(__ushort_as_half(d16b));
        const int a0 = (18 * b + 2) >> 2;
        const int sh  = (18 * b + 2) & 2;
        const float4 *x4 = (const float4 *)(x + b * 32);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const uint32_t la = rw0[a0 + k];
            const uint32_t lb = rw1[a0 + k];
            const uint32_t va = sh ? __byte_perm(la, rw0[a0 + k + 1], 0x5432) : la;
            const uint32_t vb = sh ? __byte_perm(lb, rw1[a0 + k + 1], 0x5432) : lb;
            const float4 xa = x4[k];
            const float4 xb = x4[k + 4];
            s0 += (float)((int)(va         & 0xFu) - 8) * da * xa.x;
            s0 += (float)((int)((va >>  4) & 0xFu) - 8) * da * xb.x;
            s0 += (float)((int)((va >>  8) & 0xFu) - 8) * da * xa.y;
            s0 += (float)((int)((va >> 12) & 0xFu) - 8) * da * xb.y;
            s0 += (float)((int)((va >> 16) & 0xFu) - 8) * da * xa.z;
            s0 += (float)((int)((va >> 20) & 0xFu) - 8) * da * xb.z;
            s0 += (float)((int)((va >> 24) & 0xFu) - 8) * da * xa.w;
            s0 += (float)((int)(va >> 28) - 8) * da * xb.w;
            s1 += (float)((int)(vb         & 0xFu) - 8) * db * xa.x;
            s1 += (float)((int)((vb >>  4) & 0xFu) - 8) * db * xb.x;
            s1 += (float)((int)((vb >>  8) & 0xFu) - 8) * db * xa.y;
            s1 += (float)((int)((vb >> 12) & 0xFu) - 8) * db * xb.y;
            s1 += (float)((int)((vb >> 16) & 0xFu) - 8) * db * xa.z;
            s1 += (float)((int)((vb >> 20) & 0xFu) - 8) * db * xb.z;
            s1 += (float)((int)((vb >> 24) & 0xFu) - 8) * db * xa.w;
            s1 += (float)((int)(vb >> 28) - 8) * db * xb.w;
        }
    }
    s0 = qkv2_reduce(s0);
    s1 = qkv2_reduce(s1);
    if (lane == 0) {
        y[row0] = s0;
        if (row0 + 1 < M) y[row0 + 1] = s1;
    }
}
static int tt_qkv2(const void *dWq, const void *dWk, const void *dWv,
                   const float *dx, float *dyq, float *dyk, float *dyv,
                   int Mq, int Mk, int Mv, int K, cudaStream_t stream) {
    dim3 b; b.x = 32; b.y = 8; b.z = 1;
    const int nwarp = Mq / 2 + Mk / 2 + Mv / 2;
    dim3 g; g.x = (nwarp + 7) / 8; g.y = 1; g.z = 1;
    k_gemv_q4_0_qkv2<<<g, b, 0, stream>>>((const uint32_t *)dWq, (const uint32_t *)dWk,
        (const uint32_t *)dWv, dx, dyq, dyk, dyv, Mq, Mk, Mv, K);
    return (int)cudaGetLastError();
}
int main() {
 const int Mq=2048, Mk=512, Mv=512, K=2048, nb=K/32;
 const size_t wbq=(size_t)Mq*nb*18;
 uint8_t *hW=(uint8_t*)malloc(wbq); uint32_t s=12345;
 for (size_t i=0;i<wbq;i++){ s=s*1103515245u+12345u; hW[i]=(uint8_t)(s>>16); }
 float *hX=(float*)malloc((size_t)K*4);
 for (int i=0;i<K;i++) hX[i]=sinf(0.7f*i)*0.5f;
 void *dW=NULL; float *dx=NULL,*dy=NULL,*dq=NULL,*dk=NULL,*dv=NULL;
 CK(cudaMalloc(&dW,wbq)); CK(cudaMalloc(&dx,(size_t)K*4)); CK(cudaMalloc(&dy,(size_t)Mq*4));
 CK(cudaMalloc(&dq,(size_t)Mq*4)); CK(cudaMalloc(&dk,(size_t)Mk*4)); CK(cudaMalloc(&dv,(size_t)Mv*4));
 CK(cudaMemcpy(dW,hW,wbq,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dx,hX,(size_t)K*4,cudaMemcpyHostToDevice));
 const void *dWq=dW, *dWk=dW, *dWv=dW;
 cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
 for(int i=0;i<50;i++){ tt_gemv_q4_0_v4(dWq,dx,dy,Mq,K,0); tt_gemv_q4_0_qkv(dWq,dWk,dWv,dx,dq,dk,dv,Mq,Mk,Mv,K,0); }
 for(int i=0;i<50;i++){ tt_qkv2(dWq,dWk,dWv,dx,dq,dk,dv,Mq,Mk,Mv,K,0); }
 for(int i=0;i<50;i++){ tt_gemv_q4_0_r8(dWq,dx,dy,Mq,K,0); }
 std::vector<float> t3,tf,t2; float m;
 std::vector<float> t8;
 for(int i=0;i<200;i++){
  CK(cudaEventRecord(a)); tt_gemv_q4_0_v4(dWq,dx,dy,Mq,K,0); tt_gemv_q4_0_v4(dWk,dx,dy,Mk,K,0); tt_gemv_q4_0_v4(dWv,dx,dy,Mv,K,0); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&m,a,b)); t3.push_back(m*1000.0f);
  CK(cudaEventRecord(a)); tt_gemv_q4_0_qkv(dWq,dWk,dWv,dx,dq,dk,dv,Mq,Mk,Mv,K,0); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&m,a,b)); tf.push_back(m*1000.0f);
  CK(cudaEventRecord(a)); tt_qkv2(dWq,dWk,dWv,dx,dq,dk,dv,Mq,Mk,Mv,K,0); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&m,a,b)); t2.push_back(m*1000.0f);
  CK(cudaEventRecord(a)); tt_gemv_q4_0_r8(dWq,dx,dy,Mq,K,0); tt_gemv_q4_0_r8(dWk,dx,dy,Mk,K,0); tt_gemv_q4_0_r8(dWv,dx,dy,Mv,K,0); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); CK(cudaEventElapsedTime(&m,a,b)); t8.push_back(m*1000.0f);
 }
 double wb=((double)Mq+Mk+Mv)*(K/32)*18.0; float m3=med(t3), mf=med(tf), m2=med(t2), m8=med(t8);
 printf("3xV4 : %.2f us GB/s=%.1f\n", m3, wb/(m3/1e6)/1e9);
 printf("fused: %.2f us GB/s=%.1f (%.1f pct)\n", mf, wb/(mf/1e6)/1e9, 100.0*(m3-mf)/m3);
 printf("qkv2 : %.2f us GB/s=%.1f (%.1f pct vs fused)\n", m2, wb/(m2/1e6)/1e9, 100.0*(mf-m2)/mf);
 printf("3xR8 : %.2f us GB/s=%.1f (%.1f pct vs fused)\n", m8, wb/(m8/1e6)/1e9, 100.0*(mf-m8)/mf);
 std::vector<float> yq(Mq), yr(Mq);
 tt_gemv_q4_0_v4(dWq,dx,dy,Mq,K,0); tt_gemv_q4_0_qkv(dWq,dWk,dWv,dx,dq,dk,dv,Mq,Mk,Mv,K,0);
 CK(cudaMemcpy(yr.data(),dy,(size_t)Mq*4,cudaMemcpyDeviceToHost)); CK(cudaMemcpy(yq.data(),dq,(size_t)Mq*4,cudaMemcpyDeviceToHost));
 float md=0; for(int i=0;i<Mq;i++) md=std::max(md,fabsf(yq[i]-yr[i]));
 printf("maxdiff Q = %g\n", md);
 tt_qkv2(dWq,dWk,dWv,dx,dq,dk,dv,Mq,Mk,Mv,K,0);
 CK(cudaMemcpy(yq.data(),dq,(size_t)Mq*4,cudaMemcpyDeviceToHost));
 md=0; for(int i=0;i<Mq;i++) md=std::max(md,fabsf(yq[i]-yr[i]));
 printf("maxdiff Q2 = %g\ndone\n", md);
 return 0;
}
