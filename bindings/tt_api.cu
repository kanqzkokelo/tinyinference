/* tt_api.cu -- implementation of the language-binding ABI in tt_api.h. */
#include "tt_api.h"

#include <stdlib.h>
#include <string.h>

#include "qwen2_engine.h"
#include "loader_gguf.h"
#include "tokenizer_bpe.h"
#include "chat_template.h"

struct TTModel {
    GGUFModel     *gguf;
    TTConfig       cfg;
    Qwen2Engine   *eng;
    BPETokenizer  *tok;
    tt_chat_family fam;
    int           *mask_buf;
    int            mask_cap;
};

static int cfg_from_env_int(const char *name, int fallback) {
    const char *s = getenv(name);
    if (!s || !s[0]) return fallback;
    int v = atoi(s);
    return v > 0 ? v : fallback;
}

TTModel *tt_api_model_load(const char *path, int max_ctx) {
    if (!path || !path[0]) return NULL;

    GGUFModel *gguf = gguf_load(path);
    if (!gguf) return NULL;

    const int want_ctx = max_ctx > 0 ? max_ctx : cfg_from_env_int("TT_MAX_CTX", 8192);
    TTConfig cfg = tt_config_from_gguf(gguf, want_ctx);
    if (cfg.dim <= 0) { gguf_free(gguf); return NULL; }

    Qwen2Engine *eng = qwen2_engine_create(&cfg, gguf);
    if (!eng) { gguf_free(gguf); return NULL; }

    BPETokenizer *tok = bpe_tokenizer_init(gguf);
    if (!tok) { qwen2_engine_free(eng); gguf_free(gguf); return NULL; }

    TTModel *m = (TTModel *)calloc(1, sizeof(*m));
    if (!m) { bpe_tokenizer_free(tok); qwen2_engine_free(eng); gguf_free(gguf); return NULL; }

    m->gguf = gguf;
    m->cfg  = cfg;
    m->eng  = eng;
    m->tok  = tok;
    m->fam  = (tt_chat_family)tt_chat_family_from_arch(gguf->architecture);
    return m;
}

void tt_api_model_free(TTModel *m) {
    if (!m) return;
    qwen2_engine_free(m->eng);
    bpe_tokenizer_free(m->tok);
    gguf_free(m->gguf);
    free(m->mask_buf);
    free(m);
}

int tt_api_vocab(const TTModel *m) { return m && m->tok ? m->tok->vocab_size : -1; }
int tt_api_ctx(const TTModel *m)   { return m ? m->cfg.max_ctx : -1; }
int tt_api_pos(const TTModel *m)   { return m ? qwen2_engine_pos(m->eng) : -1; }
int tt_api_eos(const TTModel *m)   { return m ? m->tok->eos_id : -1; }
void tt_api_reset(TTModel *m)      { if (m) qwen2_engine_reset(m->eng); }

int tt_api_encode(TTModel *m, const char *text, int *out, int out_cap) {
    if (!m || !text) return -1;
    if (!out || out_cap <= 0) return -(int)strlen(text) * 4 - 16;
    int n = bpe_encode(m->tok, text, out, out_cap);
    if (n >= out_cap) return -(n + 1);
    return n;
}

int tt_api_decode(const TTModel *m, int token, char *buf, int buf_cap) {
    if (!m) return -1;
    int len = 0;
    const char *s = bpe_decode_token(m->tok, token, &len);
    if (!s || len < 0) return -1;
    if (!buf || buf_cap <= len) return -(len + 1);
    memcpy(buf, s, (size_t)len);
    buf[len] = '\0';
    return len;
}

int tt_api_chat_format(const TTModel *m, const char *const *roles,
                       const char *const *contents, int n,
                       char *out, int out_cap) {
    if (!m || n < 0 || n > TT_CHAT_MAX_MSGS || (n > 0 && (!roles || !contents)))
        return -1;
    if (m->fam < 0) return -1;
    if (!out || out_cap <= 0) return 4096;

    tt_msg msgs[TT_CHAT_MAX_MSGS];
    for (int i = 0; i < n; i++) {
        msgs[i].role = roles[i] ? roles[i] : "user";
        msgs[i].content = contents[i] ? contents[i] : "";
    }

    tt_chat_opts opts = tt_chat_opts_default();
    /* Gemma sentencepiece tokenizers prepend BOS themselves; a literal
     * "<bos>" in the template would be split into pieces. */
    if (m->fam == TT_CHAT_GEMMA || m->fam == TT_CHAT_GEMMA4)
        opts.add_bos_text = 0;

    int need = tt_chat_format_ex(m->fam, msgs, n, &opts, out, (size_t)out_cap);
    if (need < 0) return -1;
    if (need >= out_cap) return -(need + 1);
    return need;
}

const char *tt_api_stop_string(const TTModel *m) {
    if (!m || m->fam < 0) return "";
    const char *s = tt_chat_stop_string(m->fam);
    return s ? s : "";
}

void tt_api_set_sampling(TTModel *m, float temp, int topk, float penalty) {
    if (m) qwen2_engine_set_sampling(m->eng, temp, topk, penalty);
}

void tt_api_set_token_mask(TTModel *m, const uint8_t *mask) {
    if (!m) return;
    if (!mask) {
        qwen2_engine_set_token_mask(m->eng, NULL);
        return;
    }
    const int vocab = m->tok->vocab_size;
    if (m->mask_cap < vocab) {
        int *grown = (int *)realloc(m->mask_buf, (size_t)vocab * sizeof(int));
        if (!grown) return;
        m->mask_buf = grown;
        m->mask_cap = vocab;
    }
    for (int i = 0; i < vocab; i++) m->mask_buf[i] = mask[i] ? 1 : 0;
    qwen2_engine_set_token_mask(m->eng, m->mask_buf);
}

int tt_api_prefill(TTModel *m, const int *tokens, int n) {
    if (!m || !tokens || n <= 0) return -1;
    return qwen2_engine_prefill(m->eng, tokens, n);
}

int tt_api_next(TTModel *m) {
    if (!m) return -1;
    return qwen2_engine_next(m->eng);
}
