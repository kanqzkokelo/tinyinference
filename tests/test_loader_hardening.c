/* Loader hardening regressions: K-quant alignment, duplicate names,
 * truncated-payload strictness (+ TT_GGUF_ALLOW_PARTIAL opt-out),
 * file-size-relative tensor cap. Self-contained: crafts minimal GGUFs
 * under /tmp (durable scratch lives outside the repo). */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "loader_gguf.h"

#define SCRATCH "/tmp/ti_loader_hard"

static void w8(FILE *f, uint64_t v) { fwrite(&v, 8, 1, f); }
static void w4(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void wname(FILE *f, const char *s) {
    w8(f, (uint64_t)strlen(s));
    fwrite(s, 1, strlen(s), f);
}

typedef struct { const char *name; int ndim; int64_t shape[4]; int type; uint64_t off; } Tent;

/* Write minimal GGUF with given tensor headers + pad payload bytes. */
static void write_gguf(const char *path, uint64_t n_tensors, const Tent *ts,
                       size_t pad) {
    FILE *f = fopen(path, "wb");
    if (!f) { perror("fopen"); exit(2); }
    fwrite("GGUF", 1, 4, f);
    w4(f, 3);            /* version */
    w8(f, n_tensors);
    w8(f, 0);            /* kv_count */
    for (uint64_t i = 0; i < n_tensors; i++) {
        wname(f, ts[i].name);
        w4(f, (uint32_t)ts[i].ndim);
        for (int d = 0; d < ts[i].ndim; d++) w8(f, (uint64_t)ts[i].shape[d]);
        w4(f, (uint32_t)ts[i].type);
        w8(f, ts[i].off);
    }
    for (size_t i = 0; i < pad; i++) fputc(0, f);
    fclose(f);
}

static int fails = 0;
static void check(const char *name, int cond) {
    printf("%s %s\n", cond ? "PASS" : "FAIL", name);
    if (!cond) fails++;
}

int main(void) {
    char cmd[256];
    snprintf(cmd, sizeof cmd, "mkdir -p %s", SCRATCH);
    if (system(cmd) != 0) return 2;
    char p1[256], p2[256], p3[256], p4[256], p5[256];
    snprintf(p1, sizeof p1, "%s/kq.gguf", SCRATCH);
    snprintf(p2, sizeof p2, "%s/dup.gguf", SCRATCH);
    snprintf(p3, sizeof p3, "%s/trunc.gguf", SCRATCH);
    snprintf(p4, sizeof p4, "%s/big.gguf", SCRATCH);
    snprintf(p5, sizeof p5, "%s/ok.gguf", SCRATCH);

    /* T1: K-quant numel not multiple of 256 -> NULL */
    Tent t1[] = {{"w", 1, {1000}, 12, 0}};
    write_gguf(p1, 1, t1, 64);
    GGUFModel *m = gguf_load(p1);
    check("T1 kquant-misaligned-null", m == NULL);
    if (m) gguf_free(m);

    /* T2: duplicate names -> NULL */
    Tent t2[] = {{"w", 1, {4}, 0, 0}, {"w", 1, {4}, 0, 16}};
    write_gguf(p2, 2, t2, 64);
    m = gguf_load(p2);
    check("T2 duplicate-null", m == NULL);
    if (m) gguf_free(m);

    /* T3a: truncated payload, strict default -> NULL */
    Tent t3[] = {{"big", 1, {4096}, 0, 0}};
    write_gguf(p3, 1, t3, 64);
    unsetenv("TT_GGUF_ALLOW_PARTIAL");
    m = gguf_load(p3);
    check("T3a truncated-strict-null", m == NULL);
    if (m) gguf_free(m);

    /* T3b: same file with opt-out -> loads, data NULL */
    setenv("TT_GGUF_ALLOW_PARTIAL", "1", 1);
    m = gguf_load(p3);
    check("T3b partial-allowed-nonnull", m != NULL);
    check("T3b partial-data-null",
          m && m->tensor_count == 1 && m->tensors[0].data == NULL);
    if (m) gguf_free(m);
    unsetenv("TT_GGUF_ALLOW_PARTIAL");
    /* "0" and empty stay strict */
    setenv("TT_GGUF_ALLOW_PARTIAL", "0", 1);
    m = gguf_load(p3);
    check("T3b zero-stays-strict", m == NULL);
    if (m) gguf_free(m);
    unsetenv("TT_GGUF_ALLOW_PARTIAL");

    /* T4: absurd count for tiny file -> NULL before big alloc */
    {
        FILE *f = fopen(p4, "wb");
        fwrite("GGUF", 1, 4, f);
        w4(f, 3); w8(f, 100000); w8(f, 0);
        fclose(f);
        m = gguf_load(p4);
        check("T4 absurd-count-null", m == NULL);
        if (m) gguf_free(m);
    }

    /* T5: valid control loads */
    Tent t5[] = {{"w", 1, {4}, 0, 0}};
    write_gguf(p5, 1, t5, 64);
    m = gguf_load(p5);
    check("T5 valid-nonnull", m != NULL && m->tensors[0].data != NULL);
    if (m) gguf_free(m);

    if (fails) { printf("FAIL: test_loader_hardening (%d)\n", fails); return 1; }
    printf("ALL PASS\n");
    return 0;
}
