// Q8 vs Q4 CUDA-core prefill GEMM head-to-head on identical shapes.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "../kernels/gemv_q4_cuda.cu"
static double bench(int (*fn)(const void*,const float*,float*,int,int,int,cudaStream_t), const void* W, const float* X, float* Y, int M, int K, int N, int iters) {
  cudaStream_t s; cudaStreamCreate(&s);
  fn(W,X,Y,M,K,N,s); cudaStreamSynchronize(s);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaEventRecord(a,s);
  for (int i=0;i<iters;i++) fn(W,X,Y,M,K,N,s);
  cudaEventRecord(b,s); cudaEventSynchronize(b);
  float ms=0; cudaEventElapsedTime(&ms,a,b);
  cudaStreamDestroy(s);
  return (double)ms/iters;
}
static void one(int M, int K, int N) {
  size_t nb=(size_t)(K/32);
  void *dW4,*dW8; float *dX,*dY;
  cudaMalloc(&dW4,(size_t)M*nb*18); cudaMalloc(&dW8,(size_t)M*nb*34);
  cudaMalloc(&dX,(size_t)N*K*sizeof(float)); cudaMalloc(&dY,(size_t)N*M*sizeof(float));
  cudaMemset(dW4,0x11,(size_t)M*nb*18); cudaMemset(dW8,0x11,(size_t)M*nb*34);
  cudaMemset(dX,0,(size_t)N*K*sizeof(float));
  double q4=bench(tt_gemm_q4_0_prefill,dW4,dX,dY,M,K,N,10);
  double q8=bench(tt_gemm_q8_0_prefill,dW8,dX,dY,M,K,N,10);
  double flops=2.0*(double)M*K*N;
  printf("M=%d K=%d N=%d: q4 %.3f ms (%.1f GFLOPS) | q8 %.3f ms (%.1f GFLOPS) | q8/q4=%.2f\n", M,K,N, q4, flops/q4/1e6, q8, flops/q8/1e6, q8/q4);
  cudaFree(dW4); cudaFree(dW8); cudaFree(dX); cudaFree(dY);
}
int main() {
  one(3072,1024,512);
  one(1024,1024,512);
  one(2048,1024,512);
  return 0;
}
extern "C" { 
int tt_logits_typed(const void*d,int t,const float*x,float*y,int v,int k,cudaStream_t s){ (void)d;(void)t;(void)x;(void)y;(void)v;(void)k;(void)s; return -99; }
}
