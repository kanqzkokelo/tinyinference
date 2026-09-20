#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "loader_gguf.h"
#include "tokenizer_bpe.h"
int main(int argc, char **argv) {
    GGUFModel *m;
    BPETokenizer *tok;
    static int ids[4096];
    int n, i;
    if (argc < 3) { puts("usage: print_ids MODEL TEXTFILE"); return 1; }
    m = gguf_load(argv[1]);
    if (!m) { puts("FAIL load"); return 2; }
    tok = bpe_tokenizer_init(m);
    if (!tok) { puts("FAIL tok"); return 2; }
    {
        FILE *f = fopen(argv[2], "rb");
        static char txt[65536];
        size_t L = 0;
        if (!f) { puts("FAIL text"); return 2; }
        L = fread(txt, 1, sizeof(txt) - 1, f);
        fclose(f);
        txt[L] = 0;
        n = bpe_encode(tok, txt, ids, 4096);
    }
    for (i = 0; i < n; i++) printf("%d,", ids[i]);
    puts(".");
    return 0;
}
