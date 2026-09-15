// spec_expA_e2e: v6 Experiment A end-to-end driver (optimized carry path).
//
// Carry-based speculative decode: init carry = P(.|prompt) via step_logits
// of last prompt token; each pass drafts j<=4 n-grams, verifies drafts-only,
// full accept takes the bonus from the final verify row (NO step_logits),
// partial/zero accept rolls back and steps the correction. Eager only
// (TT_NO_GRAPH=1: rollback drops graph captures, so graph stays off).
//
// Build: make build/spec_expA_e2e
// Run:   ./build/spec_expA_e2e <model.gguf> <n_predict> <draft_k> <window> <prompt...>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"
#include "ngram_lookup.h"
#define MAX_CTX 1024
#define MAX_HISTORY 4096
#define JMAX 4
#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_)); exit(1); } } while(0)
static int V;
static int argmax_row(const float *m, int row) {
    int best = 0; float bv = m[(long)row * V];
    for (int v = 1; v < V; v++) { float x = m[(long)row * V + v]; if (x > bv) { bv = x; best = v; } }
    return best;
}
static double now_us(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e6 + (double)t.tv_nsec / 1e3;
}
int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s model n_predict draft_k window prompt...\n", argv[0]); return 1; }
    const char *model_path = argv[1];
    int n_predict = atoi(argv[2]);
    int draft_k = atoi(argv[3]);
    int window = atoi(argv[4]);
    if (draft_k < 1) draft_k = 1; if (draft_k > MAX_DRAFT_K) draft_k = MAX_DRAFT_K;
    if (window < 2) window = 2; if (window > 3) window = 3;
    if (n_predict < 1) n_predict = 1;
    char prompt[8192] = "";
    for (int i = 5; i < argc; i++) {
        if (i > 5) strncat(prompt, " ", sizeof(prompt) - strlen(prompt) - 1);
        strncat(prompt, argv[i], sizeof(prompt) - strlen(prompt) - 1);
    }
    setenv("TT_NO_GRAPH", "1", 1);
    GGUFModel *model = gguf_load(model_path);
    if (!model) { fprintf(stderr, "gguf_load failed\n"); return 1; }
    BPETokenizer *tok = bpe_tokenizer_init(model);
    if (!tok) { fprintf(stderr, "bpe init failed\n"); return 1; }
    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    GGUFTensor *tembd = gguf_get_tensor(model, "token_embd.weight");
    if (!tembd) { fprintf(stderr, "embd missing\n"); return 1; }
    cfg.vocab = (int)tembd->shape[tembd->ndim - 1];
    V = cfg.vocab;
    const int eos = tok->eos_id;
    int ptoks[2048];
    int n_prompt = bpe_encode(tok, prompt, ptoks, 2048);
    if (n_prompt < 1) { fprintf(stderr, "encode failed\n"); return 1; }
    Qwen2Engine *e = qwen2_engine_create(&cfg, model);
    if (!e) { fprintf(stderr, "engine create failed\n"); return 1; }
    if (qwen2_engine_prefill(e, ptoks, n_prompt)) { fprintf(stderr, "prefill failed\n"); return 1; }
    int *history = (int *)malloc(sizeof(int) * MAX_HISTORY);
    memcpy(history, ptoks, sizeof(int) * n_prompt);
    int history_n = n_prompt;
    float *carry = (float *)malloc((size_t)V * sizeof(float));
    float *h_rows = (float *)malloc((size_t)JMAX * V * sizeof(float));
    float *d_logits = NULL;
    CUDA_OK(cudaMalloc((void **)&d_logits, (size_t)JMAX * V * sizeof(float)));
    double t0 = now_us();
    if (qwen2_engine_step_logits(e, history[history_n - 1], carry)) { fprintf(stderr, "init step failed\n"); return 1; }
    cudaDeviceSynchronize();
    double init_us = now_us() - t0;
    long emitted = 0, verify_passes = 0, n_full = 0, n_part = 0, n_zero = 0, n_fallback = 0;
    long drafts_att = 0, drafts_acc = 0, fwd_verify = 0, fwd_step = 0;
    double verify_us = 0, correct_us = 0, fallback_us = 0;
    int draft[MAX_DRAFT_K];
    double dec0 = now_us();
    int end = 0;
    while (emitted < n_predict && qwen2_engine_pos(e) < MAX_CTX - 1) {
        int K = ngram_lookup_draft(history, history_n, window, draft_k, draft);
        if (K <= 0) {
            double s0 = now_us();
            int c = argmax_row(carry, 0);
            if (qwen2_engine_step_logits(e, c, carry)) break;
            cudaDeviceSynchronize();
            fallback_us += now_us() - s0;
            n_fallback++; fwd_step++;
            if (history_n < MAX_HISTORY) history[history_n++] = c;
            emitted++;
            if (c == eos) { end = 1; break; }
            continue;
        }
        int j = K < JMAX ? K : JMAX;
        if (argmax_row(carry, 0) != draft[0]) {
            double s0 = now_us();
            int c = argmax_row(carry, 0);
            if (qwen2_engine_step_logits(e, c, carry)) break;
            cudaDeviceSynchronize();
            correct_us += now_us() - s0;
            n_zero++; fwd_step++;
            if (history_n < MAX_HISTORY) history[history_n++] = c;
            emitted++;
            if (c == eos) { end = 1; break; }
            continue;
        }
        double v0 = now_us();
        if (qwen2_engine_verify_speculative(e, draft, j, d_logits)) break;
        CUDA_OK(cudaMemcpy(h_rows, d_logits, (size_t)j * V * sizeof(float), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        verify_us += now_us() - v0;
        verify_passes++; fwd_verify += j; drafts_att += j;
        int m = 1;
        while (m < j && argmax_row(h_rows, m - 1) == draft[m]) m++;
        drafts_acc += m;
        if (m == j) {
            int b = argmax_row(h_rows, j - 1);
            memcpy(carry, h_rows + (long)(j - 1) * V, (size_t)V * sizeof(float));
            for (int i = 0; i < j && history_n < MAX_HISTORY; i++) history[history_n++] = draft[i];
            if (history_n < MAX_HISTORY) history[history_n++] = b;
            emitted += j + 1; n_full++;
            if (b == eos) { end = 1; break; }
        } else {
            double s0 = now_us();
            int c = argmax_row(h_rows, m - 1);
            qwen2_engine_rollback(e, qwen2_engine_pos(e) - (j - m));
            if (qwen2_engine_step_logits(e, c, carry)) break;
            cudaDeviceSynchronize();
            correct_us += now_us() - s0;
            n_part++; fwd_step++;
            for (int i = 0; i < m && history_n < MAX_HISTORY; i++) history[history_n++] = draft[i];
            if (history_n < MAX_HISTORY) history[history_n++] = c;
            emitted += m + 1;
            if (c == eos) { end = 1; break; }
        }
    }
    cudaDeviceSynchronize();
    double dec_us = now_us() - dec0;
    long fwds = fwd_verify + fwd_step;
    printf("E2E emitted=%ld verify=%ld full=%ld part=%ld zero=%ld fallback=%ld dacc=%ld datt=%ld fwd=%ld vrf_us=%.0f cor_us=%.0f fb_us=%.0f init_us=%.0f dec_us=%.0f tps=%.1f tok_per_fwd=%.3f end=%d\n", emitted, verify_passes, n_full, n_part, n_zero, n_fallback, drafts_acc, drafts_att, fwds, verify_us, correct_us, fallback_us, init_us, dec_us, emitted / (dec_us / 1e6), fwds ? (double)emitted / (double)fwds : 0, end);
    return 0;
}
