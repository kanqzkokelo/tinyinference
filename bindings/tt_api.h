/* tt_api.h -- small C ABI over the CUDA engine for language bindings.
 *
 * The engine's own headers expose CUDA types and the full TTConfig/QQwen2Engine
 * internals.  This shim keeps the surface stable and dependency-free so a
 * ctypes/cffi caller only needs opaque pointers, ints and char buffers.
 *
 * Every function returns a negative value on error unless documented
 * otherwise.  Buffers are caller-owned; the model owns the GGUF, tokenizer
 * and engine and frees them in tt_api_model_free().
 */
#ifndef TT_API_H
#define TT_API_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define TT_API __declspec(dllexport)
#else
#define TT_API __attribute__((visibility("default")))
#endif

typedef struct TTModel TTModel;

/* Load a GGUF and build the engine.  max_ctx <= 0 uses the model's training
 * context (capped at 8192).  Returns NULL if the file, metadata, tokenizer or
 * CUDA setup fails. */
TT_API TTModel *tt_api_model_load(const char *path, int max_ctx);
TT_API void     tt_api_model_free(TTModel *m);

TT_API int  tt_api_vocab(const TTModel *m);
TT_API int  tt_api_ctx(const TTModel *m);
TT_API int  tt_api_pos(const TTModel *m);
TT_API int  tt_api_eos(const TTModel *m);
TT_API void tt_api_reset(TTModel *m);

/* Tokenize UTF-8 text.  Returns the number of ids written, or the required
 * capacity (negative) when out_cap is too small. */
TT_API int tt_api_encode(TTModel *m, const char *text, int *out, int out_cap);

/* Decode one token to raw UTF-8 bytes (may be a partial code point).  Returns
 * the byte length, or the required capacity (negative) when buf_cap is small. */
TT_API int tt_api_decode(const TTModel *m, int token, char *buf, int buf_cap);

/* Render a chat prompt for the model's architecture.  roles/contents are
 * n NUL-terminated UTF-8 strings.  Returns the formatted length, or the
 * required capacity (negative) when out_cap is too small. */
TT_API int tt_api_chat_format(const TTModel *m, const char *const *roles,
                              const char *const *contents, int n,
                              char *out, int out_cap);

/* Terminating string for this model family ("<|im_end|>", "<end_of_turn>",
 * "<|eot_id|>"), or "" when unknown.  Static storage, do not free. */
TT_API const char *tt_api_stop_string(const TTModel *m);

/* temp <= 0 keeps greedy argmax; topk <= 0 uses the engine default. */
TT_API void tt_api_set_sampling(TTModel *m, float temp, int topk, float penalty);

TT_API int tt_api_prefill(TTModel *m, const int *tokens, int n);

/* Sample one token from the current hidden state and feed it back through
 * the layers.  Returns the token id or a negative value on error. */
TT_API int tt_api_next(TTModel *m);

#ifdef __cplusplus
}
#endif

#endif /* TT_API_H */
