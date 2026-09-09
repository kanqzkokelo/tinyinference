// Duel V2 vs V4 at Llama shapes
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cmath>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA err %d\n", (int)e_); exit(2); } } while (0)
extern "C" int tt_gemv_q4_0(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
extern "C" int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy, int M, int K, cudaStream_t stream);
typedef int (*fn_t)(const void*,const float*,float*,int,int,cudaStream_t);
static void duel(const char *tag, int M, int K, fn_t fn, const char *ename, const void *dW, const float *dx, float *dy) {
    for (int i=0;i<100;i++) fn(dW,dx,dy,M,K,0);
    CK(cudaDeviceSynchronize());
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    std::vector<float> ms; ms.reserve(500);
    for (int i=0;i<500;i++){ CK(cudaEventRecord(a,0)); fn(dW,dx,dy,M,K,0); CK(cudaEventRecord(b,0)); CK(cudaEventSynchronize(b)); float m=0; CK(cudaEventElapsedTime(&m,a,b)); ms.push_back(m*1000.0f); }
    std::sort(ms.begin(),ms.end());
    float med=ms[250], p10=ms[50], p90=ms[450];
    double wB=(double)M*(K/32)*18.0;
    printf("%s %s M=%d K=%d med=%.2f us p10=%.2f p90=%.2f GBs=%.1f\n", tag, ename, M, K, med, p10, p90, wB/(med/1e6)/1e9);
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
}
int main(){
    CK(cudaSetDevice(0));
    float *ta=NULL,*tb=NULL; CK(cudaMalloc(&ta,1<<20)); CK(cudaMalloc(&tb,1<<20));
    for(int i=0;i<50;i++) CK(cudaMemcpy(tb,ta,1<<20,cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize()); cudaFree(ta); cudaFree(tb);
    int Ms[4]={8192,2048,2048,4864}; int Ks[4]={2048,8192,2048,896}; const char *Ns[4]={"llama-gu","llama-dn","llama-qkv","qwen-gu"};
    for(int s=0;s<4;s++){
        int M=Ms[s],K=Ks[s];
        size_t wb=(size_t)M*(K/32)*18;
        uint8_t *h=(uint8_t*)malloc(wb);
        uint32_t st=1234+s; for(size_t i=0;i<wb;i++){st=st*1103515245u+12345u;h[i]=(uint8_t)(st>>16);}
        for(int b=0;b<M*(K/32);b++){h[(size_t)b*18+0]=0x00;h[(size_t)b*18+1]=0x3C;}
        void *dW=NULL; float *dx=NULL,*dy=NULL;
        CK(cudaMalloc(&dW,wb)); CK(cudaMalloc(&dx,(size_t)K*4)); CK(cudaMalloc(&dy,(size_t)M*4));
        CK(cudaMemcpy(dW,h,wb,cudaMemcpyHostToDevice));
        float *hx=(float*)malloc((size_t)K*4); for(int i=0;i<K;i++)hx[i]=sinf(0.5f*i);
        CK(cudaMemcpy(dx,hx,(size_t)K*4,cudaMemcpyHostToDevice));
        for(int r=0;r<3;r++){ duel(Ns[s],M,K,tt_gemv_q4_0,"V2",dW,dx,dy); duel(Ns[s],M,K,tt_gemv_q4_0_v4,"V4",dW,dx,dy); }
        cudaFree(dW);cudaFree(dx);cudaFree(dy);free(h);free(hx);
    }
    printf("done\n"); return 0;
}
