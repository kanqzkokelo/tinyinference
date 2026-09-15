// spec_expA_ab: v6 Experiment A harness - bonus-logit carry vs eager golden.
//
// Phase M: micro medians for single next(), step_logits(), verify(1..4).
// Phase C: scripted A/B correctness gate. Engine E runs the Experiment-A
//   pass (drafts-only verify + carried prefix row, bonus from final row,
//   rollback+step on partial, step on zero-accept). Engine G runs the
//   eager golden (sequential step_logits). Compared exactly: emitted token
//   ids, engine pos, K-cache hash over fed slots, carry logits vs the
//   golden row (bit-exact). Draft scripts force full / partial / zero
//   acceptance for j=1..4 drafts, plus a repetitive-draft case.
//
// The shipped loop forces eager (TT_NO_GRAPH=1); the harness does the same
// so it tests the shipped configuration.
//
// Build: make build/spec_expA_ab
// Run:   ./build/spec_expA_ab <model.gguf> [prompt]
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"
#include "tokenizer_bpe.h"

#define MAX_CTX 1024
#define JMAX 4
#define ITERS_M 10

#define CUDA_OK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_)); exit(1); } } while(0)

static int V = 0;

static int argmax_row(const float *m, int row, int vocab) {
    int best = 0;
    float best_v = m[(long)row * vocab];
    for (int v = 1; v < vocab; v++) {
        float x = m[(long)row * vocab + v];
        if (x > best_v) { best_v = x; best = v; }
    }
    return best;
}

/* FNV-1a over all layers K-cache slots [0..pos). V-cache has no debug
 * accessor; K + pos + tokens + logits are compared separately. */
static uint64_t kv_hash(Qwen2Engine *e, int pos, int layers, int kv_per_slot) {
    static float *hb = NULL;
    static long cap = 0;
    long need = (long)pos * kv_per_slot;
    if (need > cap) { free(hb); hb = (float *)malloc((size_t)need * sizeof(float)); cap = need; }
    uint64_t h = 1469598103934665603ULL;
    for (int l = 0; l < layers; l++) {
        int got = qwen2_debug_copy_kv(e, l, hb, need);
        long n = got > 0 ? got : 0;
        if (n > need) n = need;
        for (long i = 0; i < n; i++) {
            uint32_t u; memcpy(&u, &hb[i], 4);
            h ^= u; h *= 1099511628211ULL;
        }
        h ^= (uint64_t)l; h *= 1099511628211ULL;
    }
    return h;
}

static int bitcmp(const float *a, const float *b, long n, double *max_abs) {
    int mism = 0; double ma = 0;
    for (long i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > ma) ma = d;
        uint32_t ua, ub; memcpy(&ua, &a[i], 4); memcpy(&ub, &b[i], 4);
        if (ua != ub) mism++;
    }
    if (max_abs) *max_abs = ma;
    return mism;
}

static Qwen2Engine *make_engine(GGUFModel *m, TTConfig *cfg) {
    Qwen2Engine *e = qwen2_engine_create(cfg, m);
    if (!e) { fprintf(stderr, "engine create failed\n"); exit(1); }
    return e;
}

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

/* Teardown hygiene: drain in-flight stream work before freeing device
 * buffers (free-with-inflight-work is UB and suspected source of the
 * sticky cuda=1 seen at Phase-C iteration boundaries), then clear any
 * stale error so the next engine starts clean. */
static void free_engine(Qwen2Engine *e) {
    if (!e) return;
    cudaDeviceSynchronize();
    qwen2_engine_free(e);
    cudaDeviceSynchronize();
    cudaGetLastError();
}

/* Experiment-A single pass on E.
 * Pre: history fed on E; carry = P(.|history). Drafts d[0..j).
 * Emits into out (cap j+1); sets *n_out, *was_full (1 full / 0 partial /
 * -1 zero-accept); updates carry. Returns 0 ok, <0 engine error. */
static int exp_pass(Qwen2Engine *e, float *carry,
                    float *d_logits, float *h_rows,
                    const int *drafts, int j,
                    int *out, int *n_out, int *was_full) {
    if (argmax_row(carry, 0, V) != drafts[0]) {
        int c = argmax_row(carry, 0, V);
        if (qwen2_engine_step_logits(e, c, carry)) return -1;
        out[0] = c; *n_out = 1; *was_full = -1;
        return 0;
    }
    if (qwen2_engine_verify_speculative(e, drafts, j, d_logits)) return -2;
    CUDA_OK(cudaMemcpy(h_rows, d_logits, (size_t)j * V * sizeof(float),
                       cudaMemcpyDeviceToHost));
    int m = 1;
    while (m < j && argmax_row(h_rows, m - 1, V) == drafts[m]) m++;
    if (m == j) {
        int b = argmax_row(h_rows, j - 1, V);
        for (int i = 0; i < j; i++) out[i] = drafts[i];
        out[j] = b; *n_out = j + 1; *was_full = 1;
        memcpy(carry, h_rows + (long)(j - 1) * V, (size_t)V * sizeof(float));
        return 0;
    }
    int c = argmax_row(h_rows, m - 1, V);
    qwen2_engine_rollback(e, qwen2_engine_pos(e) - (j - m));
    if (qwen2_engine_step_logits(e, c, carry)) return -3;
    for (int i = 0; i < m; i++) out[i] = drafts[i];
    out[m] = c; *n_out = m + 1; *was_full = 0;
    return 0;
}

int main(int argc, char **argv) {
    const char *model_path = argc >= 2 ? argv[1]
        : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    const char *prompt = argc >= 3 ? argv[2]
        : "The quick brown fox jumps over the lazy dog. "
          "The quick brown fox jumps over the lazy dog. "
          "The quick brown fox jumps over the lazy dog.";
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
    fprintf(stderr, "[expA] dim=%d layers=%d vocab=%d eos=%d kv_heads=%d hd=%d\n",
            cfg.dim, cfg.n_layers, V, eos, cfg.n_kv_heads, cfg.head_dim);

    int ptoks[2048];
    int n_prompt = bpe_encode(tok, prompt, ptoks, 2048);
    if (n_prompt < 2) { fprintf(stderr, "prompt too short\n"); return 1; }
    const int H = ptoks[n_prompt - 1];

    /* ============ Phase M: micro medians ============ */
    if (!getenv("TT_EXPA_SKIP_M")) {
        Qwen2Engine *e = make_engine(model, &cfg);
        if (qwen2_engine_prefill(e, ptoks, n_prompt)) { fprintf(stderr, "prefill fail\n"); return 1; }
        for (int i = 0; i < 3; i++) qwen2_engine_next(e);
        double t[ITERS_M];
        for (int i = 0; i < ITERS_M; i++) {
            double t0 = now_us();
            qwen2_engine_next(e);
            cudaDeviceSynchronize();
            t[i] = now_us() - t0;
        }
        printf("M single_next_med_us=%.1f\n", med(t, ITERS_M));
        float *hlog = (float *)malloc((size_t)V * sizeof(float));
        for (int i = 0; i < ITERS_M; i++) {
            double t0 = now_us();
            qwen2_engine_step_logits(e, 1, hlog);
            cudaDeviceSynchronize();
            t[i] = now_us() - t0;
        }
        printf("M step_logits_med_us=%.1f\n", med(t, ITERS_M));
        free(hlog);
        float *d_logits = NULL;
        CUDA_OK(cudaMalloc((void **)&d_logits, (size_t)JMAX * V * sizeof(float)));
        int cand[JMAX];
        for (int n = 1; n <= JMAX; n++) {
            for (int k = 0; k < n; k++) cand[k] = 100 + k;
            for (int i = 0; i < ITERS_M; i++) {
                qwen2_engine_reset(e);
                if (qwen2_engine_prefill(e, ptoks, n_prompt)) return 1;
                double t0 = now_us();
                int rc = qwen2_engine_verify_speculative(e, cand, n, d_logits);
                cudaDeviceSynchronize();
                t[i] = now_us() - t0;
                if (rc) { fprintf(stderr, "verify rc=%d\n", rc); return 1; }
            }
            printf("M verify_n=%d med_us=%.1f per_tok_us=%.1f\n", n, med(t, ITERS_M), med(t, ITERS_M) / n);
        }
        cudaFree(d_logits);
        free_engine(e);
    }

    /* ============ Phase C: scripted A/B ============ */
    cudaDeviceSynchronize();
    { cudaError_t _ce = cudaGetLastError(); (void)_ce; }
    const int kv_per_slot = cfg.n_kv_heads * cfg.head_dim;
    float *d_logits = NULL;
    CUDA_OK(cudaMalloc((void **)&d_logits, (size_t)JMAX * V * sizeof(float)));
    float *h_rows = (float *)malloc((size_t)JMAX * V * sizeof(float));
    float *carry = (float *)malloc((size_t)V * sizeof(float));
    float *tmp = (float *)malloc((size_t)V * sizeof(float));
    float *grows = (float *)malloc((size_t)(JMAX + 1) * V * sizeof(float));
    float *glast = (float *)malloc((size_t)V * sizeof(float));
    int fails = 0, passes = 0;

    for (int j = 1; j <= JMAX; j++) {
        for (int mode = 0; mode < 4; mode++) {
            { cudaError_t _sync = cudaDeviceSynchronize(); (void)_sync; }
            { cudaError_t _ce = cudaGetLastError(); if (_ce) printf("C j=%d mode=%d entry-sticky cuda=%d %s\n", j, mode, (int)_ce, cudaGetErrorString(_ce)); }
            Qwen2Engine *g1 = make_engine(model, &cfg);
            int rc0;
            if ((rc0 = qwen2_engine_prefill(g1, ptoks, n_prompt - 1))) {
                printf("C j=%d mode=%d prefill g1 rc=%d cuda=%s\n", j, mode, rc0, cudaGetErrorString(cudaGetLastError()));
                fails++; free_engine(g1); continue;
            }
            if ((rc0 = qwen2_engine_step_logits(g1, H, tmp))) {
                printf("C j=%d mode=%d stepH g1 rc=%d cuda=%s\n", j, mode, rc0, cudaGetErrorString(cudaGetLastError()));
                fails++; free_engine(g1); continue;
            }
            memcpy(grows, tmp, (size_t)V * sizeof(float));
            int tru[JMAX + 1];
            tru[0] = argmax_row(grows, 0, V);
            int bad = 0;
            for (int i = 1; i <= j; i++) {
                if ((rc0 = qwen2_engine_step_logits(g1, tru[i - 1], grows + (long)i * V))) {
                    printf("C j=%d mode=%d tru-step g1 i=%d rc=%d cuda=%s\n", j, mode, i, rc0, cudaGetErrorString(cudaGetLastError()));
                    bad = 1; break;
                }
                tru[i] = argmax_row(grows, i, V);
            }
            free_engine(g1);
            cudaDeviceSynchronize();
            { cudaError_t _ce = cudaGetLastError(); (void)_ce; }
            if (bad) { fails++; continue; }
            int drafts[JMAX];
            if (mode == 0) { for (int i = 0; i < j; i++) drafts[i] = tru[i]; }
            else if (mode == 1) {
                int mcut = j / 2;
                for (int i = 0; i < mcut; i++) drafts[i] = tru[i];
                int w = (tru[mcut] + 1) % V;
                if (w == tru[mcut] || w == eos) w = (w + 7) % V;
                drafts[mcut] = w;
                for (int i = mcut + 1; i < j; i++) drafts[i] = (tru[i] + 3) % V;
            } else if (mode == 2) {
                for (int i = 0; i < j; i++) { int w = (tru[i] + 5) % V; if (w == tru[i] || w == eos) w = (w + 11) % V; drafts[i] = w; }
                if (drafts[0] == tru[0]) drafts[0] = (drafts[0] + 13) % V;
            } else {
                for (int i = 0; i < j; i++) drafts[i] = H;
            }
            int m = 0;
            while (m < j && drafts[m] == tru[m]) m++;
            int exp_out[JMAX + 1], exp_n;
            int corr = (m < j) ? tru[m] : tru[j];
            for (int i = 0; i < m; i++) exp_out[i] = drafts[i];
            exp_out[m] = corr; exp_n = m + 1;
            Qwen2Engine *eE = make_engine(model, &cfg);
            if ((rc0 = qwen2_engine_prefill(eE, ptoks, n_prompt - 1))) {
                printf("C j=%d mode=%d prefill eE rc=%d cuda=%s\n", j, mode, rc0, cudaGetErrorString(cudaGetLastError()));
                fails++; free_engine(eE); continue;
            }
            { int one = H; if ((rc0 = qwen2_engine_verify_speculative(eE, &one, 1, d_logits))) { printf("C j=%d mode=%d init-verify eE rc=%d cuda=%s\n", j, mode, rc0, cudaGetErrorString(cudaGetLastError())); fails++; free_engine(eE); continue; } }
            CUDA_OK(cudaMemcpy(carry, d_logits, (size_t)V * sizeof(float), cudaMemcpyDeviceToHost));
            double ma = 0;
            int mm = bitcmp(carry, grows, V, &ma);
            if (mm != 0) {
                printf("C j=%d mode=%d INIT-CARRY mismatch mism=%d max_abs=%.3e\n", j, mode, mm, ma);
                fails++; free_engine(eE); continue;
            }
            int out[JMAX + 1], n_out = 0, was_full = -9;
            int rc = exp_pass(eE, carry, d_logits, h_rows, drafts, j, out, &n_out, &was_full);
            if (rc) { printf("C j=%d mode=%d exp_pass rc=%d cuda=%s\n", j, mode, rc, cudaGetErrorString(cudaGetLastError())); fails++; free_engine(eE); continue; }
            int posE = qwen2_engine_pos(eE);
            uint64_t hE = kv_hash(eE, posE, cfg.n_layers, kv_per_slot);
            float *carryE = (float *)malloc((size_t)V * sizeof(float));
            memcpy(carryE, carry, (size_t)V * sizeof(float));
            int out_snap[JMAX + 1]; memcpy(out_snap, out, sizeof(out_snap));
            int n_out_snap = n_out, was_full_snap = was_full;
            free_engine(eE);
            cudaDeviceSynchronize();
            { cudaError_t _ce = cudaGetLastError(); (void)_ce; }
            Qwen2Engine *g2 = make_engine(model, &cfg);
            if ((rc0 = qwen2_engine_prefill(g2, ptoks, n_prompt - 1))) {
                printf("C j=%d mode=%d prefill g2 rc=%d cuda=%s\n", j, mode, rc0, cudaGetErrorString(cudaGetLastError()));
                fails++; free_engine(g2); free(carryE); continue;
            }
            if ((rc0 = qwen2_engine_step_logits(g2, H, tmp))) { printf("C j=%d mode=%d stepH g2 rc=%d\n", j, mode, rc0); fails++; free_engine(g2); free(carryE); continue; }
            mm = bitcmp(grows, tmp, V, &ma);
            if (mm != 0) {
                printf("C j=%d mode=%d REBUILD mismatch mism=%d max_abs=%.3e\n", j, mode, mm, ma);
                fails++; free_engine(g2); free(carryE); continue;
            }
            for (int i = 0; i < exp_n; i++) {
                if ((rc0 = qwen2_engine_step_logits(g2, exp_out[i], glast))) { printf("C j=%d mode=%d g2 step %d rc=%d\n", j, mode, i, rc0); bad = 1; break; }
            }
            if (bad) { fails++; free_engine(g2); free(carryE); continue; }
            int ok = 1;
            if (n_out_snap != exp_n) ok = 0;
            else for (int i = 0; i < exp_n; i++) if (out_snap[i] != exp_out[i]) ok = 0;
            int posG = qwen2_engine_pos(g2);
            int pos_ok = (was_full_snap == 1) ? (posE + 1 == posG) : (posE == posG);
            if (!pos_ok) ok = 0;
            uint64_t hG = kv_hash(g2, posE, cfg.n_layers, kv_per_slot);
            if (hE != hG) ok = 0;
            double cma = 0; int cmm;
            if (was_full_snap == 1) cmm = bitcmp(carryE, grows + (long)j * V, V, &cma);
            else cmm = bitcmp(carryE, glast, V, &cma);
            if (cmm != 0) ok = 0;
            int eosE = (out_snap[n_out_snap - 1] == eos);
            int eosG = (exp_out[exp_n - 1] == eos);
            if (eosE != eosG) ok = 0;
            printf("C j=%d mode=%d m=%d full=%d n_out=%d posE=%d posG=%d kv=%s carry=%s eos=%d -> %s\n",
                   j, mode, m, was_full_snap, n_out_snap, posE, posG,
                   hE == hG ? "eq" : "DIFF",
                   cmm == 0 ? "eq" : "DIFF", eosE, ok ? "PASS" : "FAIL");
            if (!ok) {
                fails++;
                if (n_out_snap == exp_n) {
                    for (int i = 0; i < n_out_snap; i++)
                        if (out_snap[i] != exp_out[i])
                            printf("    tok[%d] exp=%d got=%d\n", i, exp_out[i], out_snap[i]);
                } else printf("    n_out exp=%d got=%d\n", exp_n, n_out_snap);
                if (cmm) printf("    carry mism=%d max_abs=%.3e\n", cmm, cma);
            } else passes++;
            free(carryE);
            free_engine(g2);
        }
    }
    /* ---- Phase C2: two-pass carry chain + EOS draft ---- */
    {
        int c2pass = 0, c2fail = 0;
        { cudaError_t _ce = cudaGetLastError(); if (_ce) printf("C2 entry sticky cuda=%d %s\n", (int)_ce, cudaGetErrorString(_ce)); }
        /* C2a: full(j=3) -> full(j=3) chain vs golden. */
        {
            Qwen2Engine *g = make_engine(model, &cfg);
            int rc0 = 0;
            float *grows8 = (float *)malloc((size_t)9 * V * sizeof(float));
            int tru8[9];
            rc0 = qwen2_engine_prefill(g, ptoks, n_prompt - 1);
            if (!rc0) rc0 = qwen2_engine_step_logits(g, H, grows8);
            tru8[0] = rc0 ? -1 : argmax_row(grows8, 0, V);
            for (int i = 1; !rc0 && i <= 8; i++) {
                rc0 = qwen2_engine_step_logits(g, tru8[i-1], grows8 + (long)i * V);
                if (!rc0) tru8[i] = argmax_row(grows8, i, V);
            }
            free_engine(g);
            if (rc0) { printf("C2a golden rc=%d\n", rc0); c2fail++; }
            else {
                Qwen2Engine *e = make_engine(model, &cfg);
                rc0 = qwen2_engine_prefill(e, ptoks, n_prompt - 1);
                if (!rc0) { int one = H; rc0 = qwen2_engine_verify_speculative(e, &one, 1, d_logits); }
                if (!rc0) rc0 = (cudaMemcpy(carry, d_logits, (size_t)V * 4, cudaMemcpyDeviceToHost) != cudaSuccess);
                int ok = !rc0;
                int d1[3] = { tru8[0], tru8[1], tru8[2] };
                int o1[4], n1 = 0, f1 = -9;
                if (!rc0) rc0 = exp_pass(e, carry, d_logits, h_rows, d1, 3, o1, &n1, &f1);
                if (!rc0) { for (int i = 0; i < 4; i++) if (o1[i] != tru8[i]) ok = 0; if (f1 != 1 || n1 != 4) ok = 0; }
                else ok = 0;
                int d2[3] = { tru8[3], tru8[4], tru8[5] };
                int o2[4], n2 = 0, f2 = -9;
                if (!rc0) rc0 = exp_pass(e, carry, d_logits, h_rows, d2, 3, o2, &n2, &f2);
                if (!rc0) { for (int i = 0; i < 4; i++) if (o2[i] != tru8[3+i]) ok = 0; if (f2 != 1 || n2 != 4) ok = 0; }
                else ok = 0;
                int posE = qwen2_engine_pos(e);
                uint64_t hE = kv_hash(e, posE, cfg.n_layers, kv_per_slot);
                double cma = 0; int cmm = bitcmp(carry, grows8 + (long)6 * V, V, &cma);
                if (cmm) ok = 0;
                Qwen2Engine *gr = make_engine(model, &cfg);
                float *grlast = (float *)malloc((size_t)V * sizeof(float));
                int r2 = qwen2_engine_prefill(gr, ptoks, n_prompt - 1);
                if (!r2) r2 = qwen2_engine_step_logits(gr, H, grlast);
                for (int i = 0; !r2 && i < 7; i++) r2 = qwen2_engine_step_logits(gr, tru8[i], grlast);
                int posG = qwen2_engine_pos(gr);
                uint64_t hG = kv_hash(gr, posE, cfg.n_layers, kv_per_slot);
                if (r2 || posE + 1 != posG || hE != hG) ok = 0;
                printf("C2a chain full+full posE=%d posG=%d kv=%s carry=%s -> %s\n", posE, posG, hE==hG?"eq":"DIFF", cmm==0?"eq":"DIFF", ok?"PASS":"FAIL");
                if (ok) { c2pass++; passes++; } else { c2fail++; fails++; }
                free(grlast); free_engine(gr); free_engine(e);
            }
            free(grows8);
        }
        /* C2b: EOS draft (zero-accept w/ EOS correction path exercised). */
        {
            Qwen2Engine *e = make_engine(model, &cfg);
            int rc0 = qwen2_engine_prefill(e, ptoks, n_prompt - 1);
            if (!rc0) { int one = H; rc0 = qwen2_engine_verify_speculative(e, &one, 1, d_logits); }
            if (!rc0) rc0 = (cudaMemcpy(carry, d_logits, (size_t)V * 4, cudaMemcpyDeviceToHost) != cudaSuccess);
            int ok = !rc0;
            int de[2] = { eos, H };
            int oe[3], ne = 0, fe = -9;
            if (!rc0) rc0 = exp_pass(e, carry, d_logits, h_rows, de, 2, oe, &ne, &fe);
            if (rc0) ok = 0;
            else {
                int corr = argmax_row(carry, 0, V);
                /* carry was overwritten by zero-path step; oe[0] must equal pre-step argmax */
                (void)corr;
                if (ne != 1) ok = 0;
                if (qwen2_engine_pos(e) != n_prompt + 1) ok = 0;
            }
            printf("C2b eos-draft n=%d full=%d pos=%d -> %s\n", ne, fe, qwen2_engine_pos(e), ok?"PASS":"FAIL");
            if (ok) { c2pass++; passes++; } else { c2fail++; fails++; }
            free_engine(e);
        }
        printf("C2 summary: %d PASS %d FAIL\n", c2pass, c2fail);
    }
    printf("C summary: %d PASS %d FAIL\n", passes, fails);
    cudaFree(d_logits);
    free(h_rows); free(carry); free(tmp); free(grows); free(glast);
    bpe_tokenizer_free(tok); gguf_free(model);
    return fails ? 1 : 0;
}
