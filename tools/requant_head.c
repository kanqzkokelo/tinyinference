// requant_head: offline GGUF tensor rewrite Q6_K -> Q4_0 (head byte cut).
// Usage: requant_head SRC DST [TENSOR]
// Copies the GGUF header verbatim, patches type+offsets, converts one
// tensor by dequant->requant streaming. All other payloads are memcpy.
// PPL/greedy gate runs on DST with zero engine changes.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "loader_gguf.h"
#include "dequant_ref.h"
#include "quant_ref.h"
typedef struct { const uint8_t *p; const uint8_t *end; int err; } Cur;
static uint32_t ru32(Cur *c) {
    uint32_t v = 0;
    if (c->p + 4 > c->end) { c->err = 1; return 0; }
    memcpy(&v, c->p, 4); c->p += 4; return v;
}
static uint64_t ru64(Cur *c) {
    uint64_t v = 0;
    if (c->p + 8 > c->end) { c->err = 1; return 0; }
    memcpy(&v, c->p, 8); c->p += 8; return v;
}
static void rskip(Cur *c, size_t n) {
    if (c->p + n > c->end) { c->err = 1; return; }
    c->p += n;
}
static size_t kvfix(uint32_t t) {
    if (t == 0 || t == 1 || t == 7) return 1;
    if (t == 2 || t == 3) return 2;
    if (t == 4 || t == 5 || t == 6) return 4;
    if (t == 10 || t == 11 || t == 12) return 8;
    return 0;
}
static void kvskip(Cur *c, uint32_t t) {
    size_t f = kvfix(t);
    uint64_t i, n;
    if (f) { rskip(c, f); return; }
    if (t == 8) { n = ru64(c); rskip(c, (size_t)n); return; }
    if (t == 9) {
        uint32_t it = ru32(c);
        n = ru64(c);
        f = kvfix(it);
        if (it == 8) {
            for (i = 0; i < n; i++) { uint64_t L = ru64(c); rskip(c, (size_t)L); }
            return;
        }
        if (it == 9 || it > 12 || !f) { c->err = 1; return; }
        if (n > (uint64_t)(c->end - c->p) / f) { c->err = 1; return; }
        rskip(c, (size_t)(n * f)); return;
    }
    c->err = 1;
}
static size_t al32(size_t v) { return (v + 31) & ~(size_t)31; }
static int cmp_off(const void *a, const void *b) {
    const size_t *x = (const size_t *)a, *y = (const size_t *)b;
    return (*x > *y) - (*x < *y);
}
int main(int argc, char **argv) {
    const char *src, *dst, *tname;
    GGUFModel *m;
    GGUFTensor *tw = NULL;
    int ti = -1, i, fd, out;
    struct stat st;
    const uint8_t *base;
    Cur c;
    uint32_t magic, ver, nkv;
    uint64_t nt, k;
    size_t *e_typeoff, *e_dataoff, *e_size, *e_new, *e_off;
    int *e_istarget, *order;
    int64_t numel;
    size_t newtsz, cursor, dbase, hoff;
    if (argc < 3) { puts("usage: requant_head SRC DST [TENSOR]"); return 1; }
    src = argv[1]; dst = argv[2];
    tname = argc > 3 ? argv[3] : "output.weight";
    /* OPT-IN ONLY: never overwrite the canonical model in data/models.
     * Head-Q4 is an explicit offline conversion; pass a /tmp DST unless
     * TT_HEAD_Q4_ALLOW=1 is set. Engine default model stays untouched. */
    if (!getenv("TT_HEAD_Q4_ALLOW") && strstr(dst, "data/models") != NULL) {
        fprintf(stderr, "REFUSE: DST '%s' is under data/models. "
                "Head-Q4 is opt-in only: write to /tmp (e.g. /tmp/qwen_hq4.gguf) "
                "or set TT_HEAD_Q4_ALLOW=1. SRC untouched.\n", dst);
        return 2;
    }
    m = gguf_load(src);
    if (!m) { puts("FAIL: gguf_load src"); return 2; }
    for (i = 0; i < m->tensor_count; i++)
        if (strcmp(m->tensors[i].name, tname) == 0) { ti = i; tw = &m->tensors[i]; }
    if (ti < 0) { puts("FAIL: tensor not found"); return 2; }
    if (tw->type != 14) { printf("FAIL: type=%d want 14", tw->type); return 2; }
    numel = 1;
    for (i = 0; i < tw->ndim; i++) numel *= tw->shape[i];
    if (numel % 32 != 0) { puts("FAIL: numel not mult of 32"); return 2; }
    newtsz = (size_t)(numel / 32) * 18;
    printf("target %s numel=%lld old=%llu new=%llu", tname, (long long)numel,
        (unsigned long long)tw->size_bytes, (unsigned long long)newtsz);
    fd = open(src, O_RDONLY);
    if (fd < 0) { puts("FAIL: open src"); return 2; }
    if (fstat(fd, &st) < 0) { puts("FAIL: fstat"); return 2; }
    base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { puts("FAIL: mmap"); return 2; }
    c.p = base; c.end = base + (size_t)st.st_size; c.err = 0;
    magic = ru32(&c); ver = ru32(&c); (void)ver;
    nt = ru64(&c); nkv = (uint32_t)ru64(&c);
    if (magic != 0x46554747 || (int)nt != m->tensor_count) { puts("FAIL: hdr"); return 2; }
    for (k = 0; k < nkv; k++) {
        uint64_t L = ru64(&c);
        rskip(&c, (size_t)L);
        kvskip(&c, ru32(&c));
    }
    if (c.err) { puts("FAIL: kv walk"); return 2; }
    e_typeoff = calloc((size_t)m->tensor_count, sizeof(size_t));
    e_dataoff = calloc((size_t)m->tensor_count, sizeof(size_t));
    e_size = calloc((size_t)m->tensor_count, sizeof(size_t));
    e_new = calloc((size_t)m->tensor_count, sizeof(size_t));
    e_off = calloc((size_t)m->tensor_count, sizeof(size_t));
    e_istarget = calloc((size_t)m->tensor_count, sizeof(int));
    order = calloc((size_t)m->tensor_count, sizeof(int));
    for (i = 0; i < m->tensor_count; i++) {
        uint64_t L = ru64(&c);
        uint32_t nd, ty;
        uint64_t off;
        int d;
        rskip(&c, (size_t)L);
        nd = ru32(&c);
        rskip(&c, (size_t)nd * 8);
        e_typeoff[i] = (size_t)(c.p - base);
        ty = ru32(&c);
        e_dataoff[i] = (size_t)(c.p - base);
        off = ru64(&c);
        (void)ty; (void)off;
        e_size[i] = m->tensors[i].size_bytes;
        e_istarget[i] = (i == ti);
        e_new[i] = e_istarget[i] ? newtsz : e_size[i];
        for (d = 0; d < 0; d++) { }
    }
    if (c.err) { puts("FAIL: tensor walk"); return 2; }
    hoff = (size_t)(c.p - base);
    dbase = al32(hoff);
    for (i = 0; i < m->tensor_count; i++) order[i] = i;
    {
        size_t *keys = malloc((size_t)m->tensor_count * sizeof(size_t));
        int *idx = malloc((size_t)m->tensor_count * sizeof(int));
        for (i = 0; i < m->tensor_count; i++) idx[i] = i;
        for (i = 0; i < m->tensor_count; i++) keys[i] = m->tensors[i].offset;
        for (i = 1; i < m->tensor_count; i++) {
            int t = idx[i], j = i - 1;
            size_t kk = keys[t];
            while (j >= 0 && keys[idx[j]] > kk) { idx[j + 1] = idx[j]; j--; }
            idx[j + 1] = t;
        }
        cursor = 0;
        for (k = 0; k < (uint64_t)m->tensor_count; k++) {
            i = idx[k];
            cursor = al32(cursor);
            e_off[i] = cursor;
            cursor += e_istarget[i] ? newtsz : e_size[i];
        }
        for (i = 0; i < m->tensor_count; i++) order[i] = idx[i];
        free(keys); free(idx);
    }
    {
        size_t loff = (size_t)((const uint8_t *)tw->data - (const uint8_t *)m->mmap_addr);
        printf("dbg loff=%llu expect=%llu hoff=%llu dbase=%llu", (unsigned long long)loff, (unsigned long long)(dbase + tw->offset), (unsigned long long)hoff, (unsigned long long)dbase);
        if (loff != dbase + tw->offset) { puts("FAIL: data base mismatch"); return 2; }
    }
    out = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0) { puts("FAIL: open dst"); return 2; }
    {
        uint8_t *hdr = malloc(hoff);
        uint8_t z[32];
        size_t pos = 0;
        uint32_t q40 = 2;
        uint64_t offv;
        if (!hdr) { puts("FAIL: oom hdr"); return 2; }
        memset(hdr, 0, hoff);
        memcpy(hdr, base, hoff);
        for (i = 0; i < m->tensor_count; i++) {
            if (e_istarget[i]) memcpy(hdr + e_typeoff[i], &q40, 4);
            offv = (uint64_t)e_off[i];
            memcpy(hdr + e_dataoff[i], &offv, 8);
        }
        while (pos < hoff) {
            ssize_t w = write(out, hdr + pos, hoff - pos);
            if (w <= 0) { puts("FAIL: write hdr"); return 2; }
            pos += (size_t)w;
        }
        memset(z, 0, sizeof(z));
        while (pos < dbase) {
            size_t n = dbase - pos < sizeof(z) ? dbase - pos : sizeof(z);
            ssize_t w = write(out, z, n);
            if (w <= 0) { puts("FAIL: write pad"); return 2; }
            pos += (size_t)w;
        }
        free(hdr);
    }
    {
        const uint8_t *srcbase = base + dbase;
        size_t wpos = dbase;
        uint8_t z[32];
        float *fb = NULL;
        uint8_t *qb = NULL;
        memset(z, 0, sizeof(z));
        for (k = 0; k < (uint64_t)m->tensor_count; k++) {
            size_t pos2 = 0;
            size_t len;
            i = order[k];
            while (wpos < dbase + e_off[i]) {
                size_t n = dbase + e_off[i] - wpos < sizeof(z) ? dbase + e_off[i] - wpos : sizeof(z);
                ssize_t w = write(out, z, n);
                if (w <= 0) { puts("FAIL: write align"); return 2; }
                wpos += (size_t)w;
            }
            if (e_istarget[i]) {
                const uint8_t *sq = (const uint8_t *)m->tensors[i].data;
                size_t rowbytes = (size_t)(2048 / 256) * 210;
                int64_t done = 0;
                long CH = 1048576L;
                fb = malloc((size_t)CH * 4);
                qb = malloc((size_t)CH / 32 * 18);
                if (!fb || !qb) { puts("FAIL: oom stream"); return 2; }
                while (done < numel) {
                    long n = numel - done > CH ? CH : (long)(numel - done);
                    long got = ttq_dequant(sq + (size_t)(done / 256) * 210, 14, n, fb);
                    long wq;
                    size_t p3 = 0;
                    size_t qlen;
                    if (got != n) { puts("FAIL: dequant"); return 2; }
                    wq = quantize_row_q4_0(fb, qb, n);
                    if (wq != n / 32 * 18) { puts("FAIL: quant"); return 2; }
                    qlen = (size_t)wq;
                    while (p3 < qlen) {
                        ssize_t w = write(out, qb + p3, qlen - p3);
                        if (w <= 0) { puts("FAIL: write q4"); return 2; }
                        p3 += (size_t)w;
                    }
                    done += n;
                }
                free(fb); free(qb); fb = NULL; qb = NULL;
                (void)rowbytes;
                len = newtsz;
            } else {
                const uint8_t *sp = srcbase + m->tensors[i].offset;
                len = e_size[i];
                while (pos2 < len) {
                    ssize_t w = write(out, sp + pos2, len - pos2);
                    if (w <= 0) { puts("FAIL: write payload"); return 2; }
                    pos2 += (size_t)w;
                }
            }
            wpos += len;
        }
    }
    close(out);
    {
        GGUFModel *v = gguf_load(dst);
        GGUFTensor *vt = NULL;
        if (!v) { puts("FAIL: reload dst"); return 2; }
        for (i = 0; i < v->tensor_count; i++)
            if (strcmp(v->tensors[i].name, tname) == 0) vt = &v->tensors[i];
        if (!vt || vt->type != 2 || vt->size_bytes != newtsz) { puts("FAIL: verify"); return 2; }
        printf("OK wrote %s head->Q4_0", dst);
        gguf_free(v);
    }
    gguf_free(m);
    return 0;
}
