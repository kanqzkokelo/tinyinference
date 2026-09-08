#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include "specdec.h"

static int run_case(const char *name, uint32_t cap, uint32_t window,
                    uint32_t max_draft, const uint32_t *toks, uint32_t n,
                    uint32_t expect_k) {
    tt_ngram *g = tt_ngram_create(cap, window, max_draft);
    assert(g);
    assert(tt_ngram_feed(g, toks, n) == 0);
    uint32_t out[8] = {0xDEADu,0xDEADu,0xDEADu,0xDEADu,0xDEADu,0xDEADu,0xDEADu,0xDEADu};
    uint32_t k = tt_ngram_draft(g, out);
    printf("%s: k=%u expect=%u out=[%u %u %u %u]\n",
           name, k, expect_k, out[0], out[1], out[2], out[3]);
    tt_ngram_free(g);
    if (k != expect_k) { printf("FAIL: %s\n", name); return 1; }
    return 0;
}

int main(void) {
    int fails = 0;
    uint32_t d1[10] = {1,2,3,4,5,6,7,8,9,10};
    fails += run_case("no-match-base0", 64, 4, 4, d1, 10, 0);
    uint32_t d2[5] = {11,12,13,14,15};
    fails += run_case("len-window-plus-1", 64, 4, 4, d2, 5, 0);
    uint32_t d3[1] = {42};
    fails += run_case("len-1", 64, 4, 4, d3, 1, 0);
    uint32_t d4[9] = {1,2,3,4,5,1,2,3,4};
    fails += run_case("true-repeat", 64, 3, 2, d4, 9, 2);
    if (fails) { printf("FAIL: test_specdec_regress (%d)\n", fails); return 1; }
    printf("PASS: test_specdec_regress\n");
    return 0;
}
