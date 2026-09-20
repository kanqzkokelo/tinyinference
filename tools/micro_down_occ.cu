#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
extern "C" {
extern int tt_gemv_q4_0(const void*, const float*, float*, int, int, cudaStream_t);
extern int tt_gemv_q4_0_v4(const void*, const float*, float*, int, int, cudaStream_t);
extern int tt_gemv_q4_0_v4_res(const void*, const float*, const float*, float*, int, int, cudaStream_t);
extern int tt_ffn_q4_0(const void*, const void*, const float*, float*, int, int, int, cudaStream_t);
}
typedef int (*fn_t)(const void*, const float*, float*, int, int, cudaStream_t);
static double bench(const char* tag, fn_t fn, const void* dW, const float* dx, float* dy, int M, int K, size_t wb) {
  for (int i=0;i<20;i++) fn(dW,dx,dy,M,K,0);
  cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> ms; ms.reserve(200);
  for (int i=0;i<200;i++) { cudaEventRecord(a,0); fn(dW,dx,dy,M,K,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b); ms.push_back(m); }
  cudaEventDestroy(a); cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end()); double med=ms[100];
  double gbs = (double)wb / (med/1000.0) / 1e9;
  printf("%s median=%.2f us GBps=%.1f\n", tag, med*1000.0, gbs);
  return med;
}
typedef int (*res_t)(const void*, const float*, const float*, float*, int, int, cudaStream_t);
static double bench_res(const char* tag, res_t fn, const void* dW, const float* dx, const float* dr, float* dy, int M, int K, size_t wb) {
  for (int i=0;i<20;i++) fn(dW,dx,dr,dy,M,K,0);
  cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> ms; ms.reserve(200);
  for (int i=0;i<200;i++) { cudaEventRecord(a,0); fn(dW,dx,dr,dy,M,K,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b); ms.push_back(m); }
  cudaEventDestroy(a); cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end()); double med=ms[100];
  double gbs = (double)wb / (med/1000.0) / 1e9;
  printf("%s median=%.2f us GBps=%.1f\n", tag, med*1000.0, gbs);
  return med;
}
typedef int (*ffn_t)(const void*, const void*, const float*, float*, int, int, int, cudaStream_t);
static double bench_ffn(const char* tag, ffn_t fn, const void* dG, const void* dU, const float* dx, float* dh, int M, int K, size_t wb2) {
  for (int i=0;i<20;i++) fn(dG,dU,dx,dh,M,K,0,0);
  cudaDeviceSynchronize();
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  std::vector<float> ms; ms.reserve(200);
  for (int i=0;i<200;i++) { cudaEventRecord(a,0); fn(dG,dU,dx,dh,M,K,0,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b); ms.push_back(m); }
  cudaEventDestroy(a); cudaEventDestroy(b);
  std::sort(ms.begin(), ms.end()); double med=ms[100];
  double gbs = (double)wb2 / (med/1000.0) / 1e9;
  printf("%s median=%.2f us GBps=%.1f\n", tag, med*1000.0, gbs);
  return med;
}
int main() {
  cudaSetDevice(0);
  const int Md=2048, Kd=8192, Mg=8192, Kg=2048;
  size_t wbd=(size_t)Md*(Kd/32)*18, wbg=(size_t)Mg*(Kg/32)*18;
  size_t wmax = wbd > wbg ? wbd : wbg;
  uint8_t* hW=(uint8_t*)malloc(wmax);
  uint32_t s=0x1234u; for(size_t i=0;i<wmax;i++){s=s*1103515245u+12345u; hW[i]=(uint8_t)(s>>16);}
  int nb=(int)(wmax/18); for(int b=0;b<nb;b++){hW[b*18]=0; hW[b*18+1]=0x3C;}
  uint8_t* dW=0; cudaMalloc(&dW,wmax); cudaMemcpy(dW,hW,wmax,cudaMemcpyHostToDevice);
  float* hX=(float*)malloc(8192*4); for(int i=0;i<8192;i++) hX[i]=sinf(0.7f*i+0.3f);
  float *dx=0,*dy=0; cudaMalloc(&dx,8192*4); cudaMemcpy(dx,hX,8192*4,cudaMemcpyHostToDevice); cudaMalloc(&dy,8192*4);
  printf("--- down M=2048 K=8192 ---\n");
  bench("down V4   ", tt_gemv_q4_0_v4, dW, dx, dy, Md, Kd, wbd);
  bench("down V2   ", tt_gemv_q4_0, dW, dx, dy, Md, Kd, wbd);
  printf("--- gate M=8192 K=2048 ---\n");
  bench("gate V4   ", tt_gemv_q4_0_v4, dW, dx, dy, Mg, Kg, wbg);
  printf("--- engine paths (in-situ kernels) ---\n");
  float *dRes=0; cudaMalloc(&dRes,8192*4); cudaMemset(dRes,0,8192*4);
  bench_res("down V4res", tt_gemv_q4_0_v4_res, dW, dx, dRes, dy, Md, Kd, wbd);
  { uint8_t *dW2=0; cudaMalloc(&dW2,wmax); cudaMemcpy(dW2,hW,wmax,cudaMemcpyHostToDevice);
    for(int i=0;i<20;i++){ tt_gemv_q4_0_v4_res(i&1?dW2:dW,dx,dRes,dy,Md,Kd,0); }
    cudaDeviceSynchronize(); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> ms; ms.reserve(200);
    for(int i=0;i<200;i++){ const void* W=i&1?dW2:dW; cudaEventRecord(a,0); tt_gemv_q4_0_v4_res(W,dx,dRes,dy,Md,Kd,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b); ms.push_back(m);} 
    std::sort(ms.begin(),ms.end()); printf("down V4res-alternating median=%.2f us\n", ms[100]*1000.0);
    // 16 distinct buffers like 16 layers
    uint8_t* Ws[16]; for(int k=0;k<16;k++){ cudaMalloc(&Ws[k],wbd); cudaMemcpy(Ws[k],hW,wbd,cudaMemcpyHostToDevice); }
    ms.clear();
    for(int i=0;i<200;i++){ const void* W=Ws[i&15]; cudaEventRecord(a,0); tt_gemv_q4_0_v4_res(W,dx,dRes,dy,Md,Kd,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b); ms.push_back(m);} 
    std::sort(ms.begin(),ms.end()); printf("down V4res-16bufs median=%.2f us\n", ms[100]*1000.0); }
  { float *dAlias=0; cudaMalloc(&dAlias,8192*4); cudaMemset(dAlias,0,8192*4);
    for(int i=0;i<20;i++) tt_gemv_q4_0_v4_res(dW,dx,dAlias,dAlias,Md,Kd,0);
    cudaDeviceSynchronize(); cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> ms; ms.reserve(200);
    for(int i=0;i<200;i++){ cudaEventRecord(a,0); tt_gemv_q4_0_v4_res(dW,dx,dAlias,dAlias,Md,Kd,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b); ms.push_back(m);} 
    std::sort(ms.begin(),ms.end()); printf("down V4res-alias median=%.2f us\n", ms[100]*1000.0); }
  bench_ffn("gateup FFN ", tt_ffn_q4_0, dW, dW, dx, dy, Mg, Kg, wbg*2);
  { uint8_t *dFlush=0; size_t FL=128<<20; cudaMalloc(&dFlush,FL);
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> ms; ms.reserve(60);
    for(int i=0;i<60;i++){ cudaMemset(dFlush, i&1, FL); cudaEventRecord(a,0); tt_gemv_q4_0_v4_res(dW,dx,dRes,dy,Md,Kd,0); cudaEventRecord(b,0); cudaEventSynchronize(b); float m=0; cudaEventElapsedTime(&m,a,b); if(i>=10) ms.push_back(m);} 
    std::sort(ms.begin(),ms.end()); printf("down V4res-COLDflush median=%.2f us min=%.2f\n", ms[25]*1000.0, ms[0]*1000.0); }
  printf("done\n"); return 0;
}
