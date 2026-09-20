// micro_verify_cost: pin the speculative-decode budget on llama-3.2-1B.
// Phase A: median qwen2_engine_next(). Phase B: median verify(n) for
// n in {1,2,4,8} with TT_SPEC_BATCH=2. Splits fixed vs per-token cost.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#define ITERS 20
#define NMAX 8
static double now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}
static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double med(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), cmpd);
    return v[n / 2];
}
int main(int argc, char **argv) {
    const char *model_path = argc >= 2 ? argv[1]
        : "data/models/llama-3.2-1b-q4_0.gguf";
    static char prompt_buf[8192];
    const char *prompt = prompt_buf;
    if (argc >= 3 && argv[2][0] != (char)64) prompt = argv[2];
    else if (argc >= 3) {
        FILE *pf = fopen(argv[2] + 1, "r");
        size_t nr = pf ? fread(prompt_buf, 1, sizeof(prompt_buf) - 1, pf) : 0;
        if (pf) fclose(pf);
        prompt_buf[nr] = 0;
        if (!nr) strcpy(prompt_buf, "The quick brown fox jumps over the lazy dog. ");
    } else {
        strcpy(prompt_buf, "The quick brown fox jumps over the lazy dog. ");
    }
    if (argc >= 4) setenv("TT_PROFILE", "1", 1);
    setenv("TT_SPEC_BATCH", "2", 1);
    setenv("TT_GREEDY", "1", 1);
    GGUFModel *model = gguf_load(model_path);
    if (!model) { fprintf(stderr, "gguf_load failed\n"); return 1; }
    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok) { fprintf(stderr, "bpe init failed\n"); return 1; }
    TTConfig cfg = tt_config_from_gguf(model, 1024);
    GGUFTensor *tembd = gguf_get_tensor(model, "token_embd.weight");
    if (tembd) cfg.vocab = (int)tembd->shape[tembd->ndim - 1];
    Qwen2Engine *eng = qwen2_engine_create(&cfg, model);
    if (!eng) { fprintf(stderr, "engine create failed\n"); return 1; }
    int ptoks[2048];
    int n_prompt = bpe_encode(tok, prompt, ptoks, 2048);
    if (n_prompt <= 0) { fprintf(stderr, "encode failed\n"); return 1; }
    if (qwen2_engine_prefill(eng, ptoks, n_prompt)) {
        fprintf(stderr, "prefill failed\n"); return 1;
    }
    for (int i = 0; i < 3; i++) qwen2_engine_next(eng);
    double ta[ITERS];
    int saved[ITERS];
    for (int i = 0; i < ITERS; i++) {
        double t0 = now_us();
        int t = qwen2_engine_next(eng);
        cudaDeviceSynchronize();
        ta[i] = now_us() - t0;
        saved[i] = t;
    }
    printf("single_med_us=%.1f\n", med(ta, ITERS));
    float *d_logits = NULL;
    cudaMalloc((void **)&d_logits, sizeof(float) * (size_t)cfg.vocab * NMAX);
    static const int ns[] = {1, 2, 4, 8};
    for (int ni = 0; ni < 4; ni++) {
        int n = ns[ni];
        qwen2_engine_reset(eng);
        if (qwen2_engine_prefill(eng, ptoks, n_prompt)) {
            fprintf(stderr, "re-prefill failed\n"); return 1;
        }
        for (int i = 0; i < 3; i++) qwen2_engine_next(eng);
    double tb[ITERS];
    for (int i = 0; i < ITERS; i++) {
        int cand[NMAX];
        for (int k = 0; k < n; k++) cand[k] = saved[(i + k) % ITERS];
        double t0 = now_us();
        int rc = qwen2_engine_verify_speculative(eng, cand, n, d_logits);
        cudaDeviceSynchronize();
        tb[i] = now_us() - t0;
        if (rc) { fprintf(stderr, "verify rc=%d n=%d iter=%d\n", rc, n, i); return 1; }
    }
        double m = med(tb, ITERS);
        printf("verify_n=%d med_us=%.1f per_tok_us=%.1f\n", n, m, m / n);
    }
    if (argc >= 4) {
        qwen2_debug_profile_reset();
        int c4[4];
        for (int k = 0; k < 4; k++) c4[k] = saved[k];
        qwen2_engine_verify_speculative(eng, c4, 4, d_logits);
        cudaDeviceSynchronize();
        qwen2_debug_profile_report(1);
    }
    return 0;
}
