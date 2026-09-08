// R8 (8 rows/warp) Q4_0 decode GEMV test — FFN/Q-proj shapes.
//
// Compares tt_gemv_q4_0_r8 against tt_gemv_q4_0_v4 bit-exact
// (max|D|==0, same math different tiling) and against an fp32
// host reference (<2e-2).
//
// Shapes: M=4864,K=896 (FFN up/gate) and M=896,K=896 (Q-proj).
//
// Build (copies test_batch4_gemv rule pattern, binary to scratch):
//   $HOME/mmcuda/bin/nvcc -O3 -gencode arch=compute_86,code=sm_86 \
//     -I$HOME/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/include \
//     -Iinclude -o $HOME/ti-scratch/test_gemv_r8 tests/test_gemv_r8.c \
//     kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
//     -L$HOME/mmcuda/lib -lcudart -lpthread -lm
// Run: $HOME/ti-scratch/test_gemv_r8  (exit 0 pass, 1 fail)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); \
    if (e_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
            #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
        exit(2); } } while (0)

extern int tt_gemv_q4_0_v4(const void *dW, const float *dx, float *dy,
                           int M, int K, cudaStream_t stream);
extern int tt_gemv_q4_0_r8(const void *dW, const float *dx, float *dy,
                           int M, int K, cudaStream_t stream);

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign << 31; }
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, 4);
    return out;
}

static int run_shape(int M, int K) {
    const int nb = K / 32;
    const size_t wbytes = (size_t)M * nb * 18;
    const size_t xbytes = (size_t)K * sizeof(float);
    const size_t ybytes = (size_t)M * sizeof(float);
    printf("  M=%5d K=%4d  ", M, K);
    fflush(stdout);

    uint8_t *hW = (uint8_t *)malloc(wbytes);
    float *hx = (float *)malloc(xbytes);
    float *href = (float *)malloc(ybytes);
    float *hv4 = (float *)malloc(ybytes);
    float *hr8 = (float *)malloc(ybytes);
    if (!hW || !hx || !href || !hv4 || !hr8) { fprintf(stderr, "OOM\n"); exit(2); }

    unsigned int rng = 0x12345678u ^ ((unsigned)M * 7919u + (unsigned)K);
    for (int m = 0; m < M; m++) {
        for (int b = 0; b < nb; b++) {
            float d = 0.05f + 0.5f * (rng / (float)UINT32_MAX);
            uint32_t bits;
            memcpy(&bits, &d, 4);
            uint16_t d16 = (uint16_t)((bits >> 16) & 0xFFFFu);
            uint8_t *blk = hW + ((size_t)m * nb + b) * 18;
            blk[0] = d16 & 0xFF;
            blk[1] = d16 >> 8;
            for (int j = 0; j < 16; j++) {
                rng = rng * 1103515245u + 12345u;
                blk[2 + j] = (uint8_t)(rng >> 16);
            }
        }
    }
    for (int k = 0; k < K; k++) {
        rng = rng * 1103515245u + 12345u;
        hx[k] = ((rng >> 8) / (float)(1u << 24) - 0.5f) * 2.0f;
    }

    for (int m = 0; m < M; m++) {
        double acc = 0.0;
        for (int b = 0; b < nb; b++) {
            const uint8_t *blk = hW + ((size_t)m * nb + b) * 18;
            uint16_t d16 = (uint16_t)(blk[0] | (blk[1] << 8));
            float d = fp16_to_fp32(d16);
            for (int j = 0; j < 16; j++) {
                acc += ((blk[2 + j] & 0x0F) - 8) * d * hx[b * 32 + j];
                acc += ((blk[2 + j] >> 4) - 8) * d * hx[b * 32 + j + 16];
            }
        }
        href[m] = (float)acc;
    }

    uint8_t *dW = NULL;
    float *dx = NULL, *dy = NULL;
    CK(cudaMalloc(&dW, wbytes));
    CK(cudaMalloc(&dx, xbytes));
    CK(cudaMalloc(&dy, ybytes));
    CK(cudaMemcpy(dW, hW, wbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dx, hx, xbytes, cudaMemcpyHostToDevice));

    int rc = tt_gemv_q4_0_v4(dW, dx, dy, M, K, 0);
    if (rc != 0) { printf("[FAIL: v4 rc=%d]\n", rc); return 1; }
    CK(cudaMemcpy(hv4, dy, ybytes, cudaMemcpyDeviceToHost));

    rc = tt_gemv_q4_0_r8(dW, dx, dy, M, K, 0);
    if (rc != 0) { printf("[FAIL: r8 rc=%d]\n", rc); return 1; }
    CK(cudaMemcpy(hr8, dy, ybytes, cudaMemcpyDeviceToHost));

    double max_v4r8 = 0.0, max_r8ref = 0.0;
    for (int m = 0; m < M; m++) {
        double d1 = fabs((double)hv4[m] - (double)hr8[m]);
        double d2 = fabs((double)hr8[m] - (double)href[m]);
        if (d1 > max_v4r8) max_v4r8 = d1;
        if (d2 > max_r8ref) max_r8ref = d2;
    }
    int bitexact = (max_v4r8 == 0.0);
    int refok = (max_r8ref < 2e-2);
    if (bitexact && refok) {
        printf("[PASS] max|V4-R8|=%.1e max|R8-ref|=%.2e\n", max_v4r8, max_r8ref);
    } else {
        printf("[FAIL] max|V4-R8|=%.2e (need 0) max|R8-ref|=%.2e (need <2e-2)\n",
               max_v4r8, max_r8ref);
    }

    cudaFree(dW); cudaFree(dx); cudaFree(dy);
    free(hW); free(hx); free(href); free(hv4); free(hr8);
    return (bitexact && refok) ? 0 : 1;
}

int main(void) {
    CK(cudaSetDevice(0));
    struct cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("device: %s\n", p.name);
    printf("R8 Q4_0 GEMV: bit-exact vs V4, <2e-2 vs fp32 ref\n");
    int fail = 0;
    fail |= run_shape(4864, 896);
    fail |= run_shape(896, 896);
    if (fail == 0) printf("ALL PASS\n");
    else printf("FAILED\n");
    return fail ? 1 : 0;
}
