/* tests/test_specdec_map.c -- known-answer tests for the hashed multi-window
 * map drafter (tt_ngram_map, ngram-map-k class). Pure C99, no GPU.
 *
 * Build:  gcc -std=c99 -O2 -Wall -Wextra -Isrc -o build/test_specdec_map \
 *             src/specdec.c tests/test_specdec_map.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "specdec.h"

static int failures = 0;

static void check_u32(const char *name, uint32_t got, uint32_t want) {
    if (got != want) {
        printf("FAIL %s: got %u want %u\n", name, got, want);
        failures++;
    } else {
        printf("PASS %s\n", name);
    }
}

static void check_seq(const char *name, const uint32_t *got, uint32_t n,
                      const uint32_t *want, uint32_t wn) {
    if (n != wn || (n > 0 && memcmp(got, want, n * sizeof(uint32_t)) != 0)) {
        printf("FAIL %s: got [", name);
        for (uint32_t i = 0; i < n; i++) printf(" %u", got[i]);
        printf(" ] want [");
        for (uint32_t i = 0; i < wn; i++) printf(" %u", want[i]);
        printf(" ]\n");
        failures++;
    } else {
        printf("PASS %s\n", name);
    }
}

int main(void) {
    /* ---- API guards ---- */
    check_u32("create(max_draft=0) -> NULL",
              (uint32_t)(tt_ngram_map_create(0, 0) == NULL), 1);
    check_u32("feed(NULL) -> -1",
              (uint32_t)(tt_ngram_map_feed(NULL, NULL, 0) == -1), 1);
    {
        tt_ngram_map *m = tt_ngram_map_create(0, 4);
        check_u32("draft on empty history -> 0",
                  (uint32_t)(tt_ngram_map_draft(m, (uint32_t[8]){0})), 0);
        tt_ngram_map_free(m);
    }

    /* ---- 1. cyclic continuation: 1 2 3 4 [1 2 3] -> 4 1 2 3 ----
     * Insert map: [123]->4, [1234]->1, [2341]->2, [3412]->3 (w4 chain),
     * plus w2 pairs. Draft at tail [1,2,3] must chain the cycle back. */
    {
        uint32_t toks[] = { 1, 2, 3, 4, 1, 2, 3 };
        tt_ngram_map *m = tt_ngram_map_create(0, 4);
        tt_ngram_map_feed(m, toks, 7);
        uint32_t out[8] = { 0 };
        uint32_t n = tt_ngram_map_draft(m, out);
        uint32_t want[] = { 4, 1, 2, 3 };
        check_seq("cyclic chain 1234...", out, n, want, 4);
        tt_ngram_map_free(m);
    }

    /* ---- 2. frequency ranking: [5 5 5] -> 9 (seen x2) beats 8 (x1) ---- */
    {
        /* feed 5,5,5,9 x2 then 5,5,5,8 then a final 5,5,5: the tail
         * context [5,5,5] has continuations 9 (x2) and 8 (x1). */
        uint32_t toks[] = { 5, 5, 5, 9,  5, 5, 5, 9,  5, 5, 5, 8,  5, 5, 5 };
        tt_ngram_map *m = tt_ngram_map_create(0, 2);
        tt_ngram_map_feed(m, toks, 15);
        uint32_t out[8] = { 0 };
        uint32_t n = tt_ngram_map_draft(m, out);
        check_u32("frequency: top-1 continuation", n >= 1 ? out[0] : 0, 9);
        tt_ngram_map_free(m);
    }

    /* ---- 3. tie-break: equal counts -> lower token id ---- */
    {
        /* [7,7]->21 and [7,7]->20 at equal counts; tail context is [7,7]
         * (the trailing pair is re-fed so the answer is not in history).
         * 21 is inserted first; deterministic tie-break must emit 20. */
        uint32_t toks[] = { 7, 7, 21,  7, 7, 20,  7, 7 };
        tt_ngram_map *m = tt_ngram_map_create(0, 1);
        tt_ngram_map_feed(m, toks, 8);
        uint32_t out[4] = { 0 };
        uint32_t n = tt_ngram_map_draft(m, out);
        check_u32("tie-break: lower id wins", n >= 1 ? out[0] : 0, 20);
        tt_ngram_map_free(m);
    }

    /* ---- 4. long-window preference: w12 hit (999) beats a MORE FREQUENT
     * w2 hit (20) because longer context is more specific. ---- */
    {
        tt_ngram_map *m = tt_ngram_map_create(0, 1);
        uint32_t A[12];
        for (int i = 0; i < 12; i++) A[i] = 5 + (uint32_t)i;  /* 5..16 */
        /* boost the trailing pair [15,16] -> 20 (count 4) */
        uint32_t boost[3] = { 15, 16, 20 };
        for (int r = 0; r < 4; r++) tt_ngram_map_feed(m, boost, 3);
        /* one occurrence of A followed by 999: w12 key A -> 999 (count 1) */
        uint32_t seq[13];
        memcpy(seq, A, 12 * sizeof(uint32_t));
        seq[12] = 999;
        tt_ngram_map_feed(m, seq, 13);
        /* second occurrence of A, continuation NOT fed yet: tail == A */
        tt_ngram_map_feed(m, A, 12);
        uint32_t out[4] = { 0 };
        uint32_t n = tt_ngram_map_draft(m, out);
        check_u32("long-window preferred over frequent pair", n >= 1 ? out[0] : 0, 999);
        tt_ngram_map_free(m);
    }

    /* ---- 5. ring wrap + forced table rebuilds ---- */
    {
        tt_ngram_map *m = tt_ngram_map_create(8192, 3);
        /* 25k distinct tokens -> ~100k inserts -> >=2 load-factor rebuilds */
        uint32_t *big = (uint32_t *)malloc(25000 * sizeof(uint32_t));
        for (int i = 0; i < 25000; i++) big[i] = (uint32_t)i;
        tt_ngram_map_feed(m, big, 25000);
        free(big);
        /* then a repeating 32-token pattern; tail continuation is 1,4,7 */
        uint32_t buf[32];
        for (int rep = 0; rep < 4; rep++) {
            for (int i = 0; i < 32; i++) buf[i] = (uint32_t)(i * 3 + 1);
            tt_ngram_map_feed(m, buf, 32);
        }
        uint32_t out[8] = { 0 };
        uint32_t n = tt_ngram_map_draft(m, out);
        uint32_t want[] = { 1, 4, 7 };
        check_seq("wrap+rebuild continuation", out, n, want, 3);
        tt_ngram_map_free(m);
    }

    /* ---- 6. determinism: two identical runs -> identical drafts ---- */
    {
        uint32_t toks[] = { 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 3, 1, 4, 1 };
        uint32_t a[8] = { 0 }, b[8] = { 0 };
        tt_ngram_map *m1 = tt_ngram_map_create(0, 4);
        tt_ngram_map *m2 = tt_ngram_map_create(0, 4);
        for (int r = 0; r < 4; r++) {           /* repeat to build counts   */
            tt_ngram_map_feed(m1, toks, 15);
            tt_ngram_map_feed(m2, toks, 15);
        }
        uint32_t na = tt_ngram_map_draft(m1, a);
        uint32_t nb = tt_ngram_map_draft(m2, b);
        check_seq("determinism", a, na, b, nb);
        tt_ngram_map_free(m1);
        tt_ngram_map_free(m2);
    }

    /* ---- 7. max_draft clamp respected ---- */
    {
        uint32_t toks[] = { 8, 8, 8, 1, 8, 8, 8, 1, 8, 8, 8 };
        tt_ngram_map *m = tt_ngram_map_create(0, 100); /* clamped to 8 */
        tt_ngram_map_feed(m, toks, 11);
        uint32_t out[16] = { 0 };
        uint32_t n = tt_ngram_map_draft(m, out);
        if (n > 8) {
            printf("FAIL max_draft clamp: drafted %u > 8\n", n);
            failures++;
        } else {
            printf("PASS max_draft clamp (n=%u)\n", n);
        }
        tt_ngram_map_free(m);
    }

    printf(failures ? "test_specdec_map: %d FAILURES\n" : "test_specdec_map: ALL PASS\n",
           failures);
    return failures ? 1 : 0;
}
