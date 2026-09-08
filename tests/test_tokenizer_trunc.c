#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "loader_gguf.h"
#include "tokenizer_bpe.h"

static int failures = 0;
#define CHECK_NULL(expr, name) do { \
    void *r = (void *)(expr); \
    if (r != NULL) { printf("FAIL %s: expected NULL got %p\n", name, r); \
        bpe_tokenizer_free((BPETokenizer *)r); failures++; } \
    else { printf("PASS %s -> NULL\n", name); } \
} while (0)

static void put_u32(uint8_t *b, uint32_t v) { memcpy(b, &v, 4); }
static void put_u64(uint8_t *b, uint64_t v) { memcpy(b, &v, 8); }

int main(void) {
    /* Case A: truncated buffer */
    {
        size_t hdr = 24, tail = 20;
        uint8_t *buf = (uint8_t *)malloc(hdr + tail);
        if (!buf) { printf("OOM\n"); return 1; }
        put_u32(buf + 0, 0x46554747u);
        put_u32(buf + 4, 3u);
        put_u64(buf + 8, 1u);
        put_u64(buf + 16, 50u);
        memset(buf + hdr, 0, tail);
        GGUFModel m;
        memset(&m, 0, sizeof(m));
        m.mmap_addr = buf;
        m.mmap_size = hdr + tail;
        BPETokenizer *tk = bpe_tokenizer_init(&m);
        if (tk != NULL) {
            printf("FAIL truncated: expected NULL\n");
            bpe_tokenizer_free(tk);
            failures++;
        } else {
            printf("PASS truncated -> NULL\n");
        }
        free(buf);
    }

    /* Edge: NULL model */
    CHECK_NULL(bpe_tokenizer_init(NULL), "null-model");

    /* Edge: NULL addr */
    {
        GGUFModel m;
        memset(&m, 0, sizeof(m));
        m.mmap_addr = NULL;
        m.mmap_size = 1024;
        CHECK_NULL(bpe_tokenizer_init(&m), "null-addr");
    }

    /* Edge: tiny header */
    {
        uint8_t *buf = (uint8_t *)malloc(10);
        memset(buf, 0, 10);
        GGUFModel m;
        memset(&m, 0, sizeof(m));
        m.mmap_addr = buf;
        m.mmap_size = 10;
        CHECK_NULL(bpe_tokenizer_init(&m), "tiny-header");
        free(buf);
    }

    /* Edge: bad magic */
    {
        uint8_t *buf = (uint8_t *)malloc(32);
        memset(buf, 0, 32);
        put_u32(buf + 0, 0xdeadbeefu);
        put_u32(buf + 4, 3u);
        put_u64(buf + 8, 0u);
        put_u64(buf + 16, 0u);
        GGUFModel m;
        memset(&m, 0, sizeof(m));
        m.mmap_addr = buf;
        m.mmap_size = 32;
        CHECK_NULL(bpe_tokenizer_init(&m), "bad-magic");
        free(buf);
    }

    /* Case B: real model if present (try known checkout paths) */
    {
        const char *cands[] = {
            "data/models/qwen2.5-0.5b-instruct-q4_0.gguf",
            "data/models/gemma-4-E2B-it-Q4_0.gguf",
            "data/models/SmolLM2-135M-Instruct-Q8_0.gguf",
        };
        FILE *f = NULL;
        for (size_t i = 0; i < sizeof(cands) / sizeof(cands[0]); i++) {
            f = fopen(cands[i], "rb");
            if (f) break;
        }
        if (!f) {
            printf("SKIP caseB: no known model file present\n");
        } else {
            fseek(f, 0, SEEK_END);
            long sz = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (sz <= 0) {
                printf("SKIP caseB: empty file\n");
                fclose(f);
            } else {
                uint8_t *buf = (uint8_t *)malloc((size_t)sz);
                if (!buf) { fclose(f); printf("OOM caseB\n"); return 1; }
                size_t rd = fread(buf, 1, (size_t)sz, f);
                fclose(f);
                if (rd != (size_t)sz) {
                    printf("FAIL caseB: short read\n");
                    free(buf);
                    failures++;
                } else {
                    GGUFModel m;
                    memset(&m, 0, sizeof(m));
                    m.mmap_addr = buf;
                    m.mmap_size = (size_t)sz;
                    BPETokenizer *tk = bpe_tokenizer_init(&m);
                    if (tk == NULL) {
                        printf("FAIL caseB: expected non-NULL\n");
                        failures++;
                    } else {
                        printf("PASS caseB: vocab=%d\n", tk->vocab_size);
                        bpe_tokenizer_free(tk);
                    }
                    free(buf);
                }
            }
        }
    }

    if (failures) { printf("RESULT FAIL (%d)\n", failures); return 1; }
    printf("RESULT OK\n");
    return 0;
}
