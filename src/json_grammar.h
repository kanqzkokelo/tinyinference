/* json_grammar.h -- incremental JSON validator used to constrain sampling.
 *
 * The state machine accepts any prefix of a well-formed JSON document and
 * answers two questions the decoder needs each step:
 *   - is this token still a valid continuation?  (tt_json_feed on a copy)
 *   - has a complete document been produced?     (tt_json_complete)
 *
 * Byte-oriented on purpose: token bytes come straight from the tokenizer, so
 * no UTF-8 decoding is needed. Structural bytes are ASCII; everything >= 0x20
 * is legal inside a string.
 *
 * State is a small fixed struct (no allocation) so callers can clone it once
 * per candidate token cheaply.
 */
#ifndef TT_JSON_GRAMMAR_H
#define TT_JSON_GRAMMAR_H

#ifdef __cplusplus
extern "C" {
#endif

#define TT_JSON_MAX_DEPTH 64

typedef struct {
    unsigned char s;                     /* current state (see json_grammar.c) */
    unsigned char depth;                 /* open containers                    */
    unsigned char require_object;        /* root must be an object             */
    unsigned char key;                   /* parsing an object key string       */
    unsigned char lit_pos;               /* literal match progress             */
    unsigned char lit_id;                /* 0 true, 1 false, 2 null            */
    unsigned char uhex;                  /* \u hex digits seen                 */
    unsigned char stack[TT_JSON_MAX_DEPTH]; /* 'o' object, 'a' array           */
} tt_json_state;

/* require_object: reject roots that are not `{ ... }` (JSON mode for chat). */
void tt_json_init(tt_json_state *st, int require_object);

/* Feed n bytes. Returns 0 while the prefix stays valid, -1 on the first byte
 * that cannot appear at this point. On -1 the state is left undefined. */
int tt_json_feed(tt_json_state *st, const char *buf, int n);

/* 1 when a complete root value has been consumed (trailing whitespace ok). */
int tt_json_complete(const tt_json_state *st);

/* Convenience for the sampler: does token `buf` keep the document valid? */
int tt_json_accepts(const tt_json_state *st, const char *buf, int n);

#ifdef __cplusplus
}
#endif

#endif
