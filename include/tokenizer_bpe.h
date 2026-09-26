#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

#include "loader_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    char **tokens;        // Flat array of string pointers
    int *token_lens;      // Array of token string lengths
    float *scores;        // Token scores for BPE merge ranking
    int bos_id;
    int eos_id;
} BPETokenizer;

BPETokenizer *bpe_tokenizer_init(const GGUFModel *model);
const char *bpe_decode_token(const BPETokenizer *tok, int token_id, int *out_len);
/* Encode text into out_tokens[0..max_tokens). Returns the number of tokens
 * written. `bpe_encode_ex` additionally reports SILENT TRUNCATION: *out_trunc
 * is set to 1 when the input did not fully fit (mid-segment or mid-input),
 * 0 otherwise. Use the _ex form whenever a cut prompt would matter (code
 * review C2: callers could not previously distinguish truncation). */
int bpe_encode_ex(const BPETokenizer *tok, const char *text, int *out_tokens,
                  int max_tokens, int *out_trunc);
/* Back-compat wrapper: bpe_encode_ex(..., NULL). */
int bpe_encode(const BPETokenizer *tok, const char *text, int *out_tokens, int max_tokens);
void bpe_tokenizer_free(BPETokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_BPE_H
