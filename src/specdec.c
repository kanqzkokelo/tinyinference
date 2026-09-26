/* specdec.c -- ngram-simple drafter (M10). See specdec.h. */
#include "specdec.h"

#include <stdlib.h>
#include <string.h>

struct tt_ngram {
    uint32_t *buf;      /* ring buffer of token ids            */
    uint32_t  cap;      /* physical capacity of buf            */
    uint32_t  len;      /* logical tokens stored (<= cap)      */
    uint64_t  total;    /* absolute count fed (for wrap index) */
    uint32_t  window;   /* match window n                      */
    uint32_t  max_draft;
};

/* Token at logical position i lives in slot i % cap. Logical positions
 * are absolute; the oldest surviving token is at total - len. */
static uint32_t ng_at(const tt_ngram *g, uint64_t logical) {
    return g->buf[logical % g->cap];
}

tt_ngram *tt_ngram_create(uint32_t history_cap, uint32_t window,
                          uint32_t max_draft) {
    if (max_draft == 0) return NULL;
    if (window == 0) window = 12;
    if (history_cap == 0) history_cap = 4096;
    if (history_cap <= window + max_draft) return NULL; /* useless config */

    tt_ngram *g = calloc(1, sizeof(*g));
    if (!g) return NULL;
    g->buf = malloc((size_t)history_cap * sizeof(uint32_t));
    if (!g->buf) { free(g); return NULL; }
    g->cap = history_cap;
    g->window = window;
    g->max_draft = max_draft;
    return g;
}

void tt_ngram_free(tt_ngram *g) {
    if (!g) return;
    free(g->buf);
    free(g);
}

int tt_ngram_feed(tt_ngram *g, const uint32_t *toks, uint32_t count) {
    if (!g || (!toks && count > 0)) return -1;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slot = (uint32_t)(g->total % g->cap);
        g->buf[slot] = toks[i];
        g->total++;
        if (g->len < g->cap) g->len++;
    }
    return 0;
}

uint32_t tt_ngram_len(const tt_ngram *g) { return g ? g->len : 0; }

/* First logical index currently resident. */
static uint64_t ng_base(const tt_ngram *g) { return g->total - g->len; }

uint32_t tt_ngram_draft(const tt_ngram *g, uint32_t *out) {
    if (!g || !out || g->len <= g->window) return 0;

    /* needle = last `window` tokens of history:
     * logical [total-window, total). Candidate source occurrence must
     * END strictly before that, i.e. its end e satisfies
     * e <= total - window. Search from most recent backwards. */
    uint64_t base = ng_base(g);
    uint64_t needle_start = g->total - g->window;
    uint64_t cand_end_max = needle_start; /* exclusive upper bound */

    /* e ranges over [base + window, cand_end_max]; written without
     * unsigned subtraction so base==0 terminates instead of wrapping. */
    for (uint64_t e = cand_end_max; e >= base + g->window; ) {
        uint64_t cs = e - g->window;
        uint32_t match = 1;
        for (uint32_t j = 0; j < g->window; j++) {
            if (ng_at(g, cs + j) != ng_at(g, needle_start + j)) {
                match = 0;
                break;
            }
        }
        if (match) {
            /* emit tokens at [e, min(e + max_draft, total)) -- only
             * tokens already in history can be proposed. */
            uint64_t avail = g->total - e;
            uint32_t k = (avail < (uint64_t)g->max_draft)
                       ? (uint32_t)avail : g->max_draft;
            for (uint32_t j = 0; j < k; j++)
                out[j] = ng_at(g, e + j);
            return k;
        }
        if (e == 0) break;
        --e;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* tt_ngram_map -- hashed multi-window map drafter (ngram-map-k class) */
/* ------------------------------------------------------------------ */
/*
 * Design (docs/BEATING_LLAMACPP.md section 5, step 3):
 *  - Ring history of token ids, appended via feed().
 *  - Incremental open-addressing hash map over ALL n-gram contexts seen,
 *    for a multi-scale window ladder. Each entry maps a 64-bit context hash to
 *    its top-4 continuations with frequency counts (Space-Saving sketch).
 *  - Draft = greedy chain: at each step pick the longest window with a hit
 *    and emit its most-frequent continuation (count desc, token id asc).
 *  - Keys are pure hashes: ring wrap-around never corrupts entries, and old
 *    continuations remain draftable (the PLD regime). Rebuild-on-load
 *    refreshes counts from current history only.
 */

#define TT_MAP_K       4                      /* top-k continuations/key   */
/* Multi-scale context ladder (longest-first at draft time). 3-5-grams
 * matter for real BPE text; {2,4,8,12} missed them (the frequency KAT). */
static const int k_map_windows[8] = { 16, 12, 8, 6, 5, 4, 3, 2 };
#define TT_MAP_NWIN    8
/* Power of two. Must satisfy TT_MAP_CAP >= 2 * max_history * TT_MAP_NWIN
 * so a rebuild (reinserts every resident context) can never fill the table
 * and trap linear probing (a 65536 table hung exactly this way). */
#define TT_MAP_CAP     262144u
#define TT_MAP_HIST    8192u                 /* default history_cap       */

typedef struct {
    uint64_t key;                            /* 0 = empty slot            */
    uint32_t next[TT_MAP_K];                 /* continuation token ids    */
    uint16_t cnt[TT_MAP_K];                  /* counts (Space-Saving)     */
} MapEnt;

struct tt_ngram_map {
    uint32_t *buf;                           /* ring history              */
    uint32_t  cap;
    uint32_t  len;
    uint64_t  total;                         /* absolute tokens fed       */
    uint32_t  max_draft;
    MapEnt   *tab;
    uint32_t  tmask;                         /* TT_MAP_CAP - 1            */
    uint32_t  ents_used;
    uint64_t  inserts;                       /* since last rebuild        */
};

static uint64_t map_mix(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static uint32_t map_at(const tt_ngram_map *m, uint64_t logical) {
    return m->buf[logical % m->cap];
}

/* Token at logical position: history ring if fed, else the virtual draft
 * chain being built ([total, total+nchain)). */
static uint32_t map_tok(const tt_ngram_map *m, const uint32_t *chain,
                        uint64_t logical) {
    if (logical < m->total) return map_at(m, logical);
    return chain[logical - m->total];
}

static uint64_t map_hash(const tt_ngram_map *m, const uint32_t *chain,
                         uint64_t end_excl, int n) {
    /* window-length-tagged sequential mix; 0 is reserved for EMPTY */
    uint64_t h = map_mix(0x9E3779B97F4A7C15ULL ^ (uint64_t)(n * 0x100000001B3ULL));
    for (int j = 0; j < n; j++) {
        uint64_t t = (uint64_t)map_tok(m, chain, end_excl - n + j);
        h = map_mix(h ^ (t + 0x9E3779B97F4A7C15ULL));
    }
    return h ? h : 1;
}

static void map_clear(MapEnt *tab, uint32_t cap) {
    memset(tab, 0, (size_t)cap * sizeof(MapEnt));
}

/* Insert (key, next) into the table (Space-Saving top-k update). */
static void map_insert_one(tt_ngram_map *m, uint64_t key, uint32_t next) {
    uint32_t idx = (uint32_t)key & m->tmask;
    MapEnt *tab = m->tab;
    for (uint32_t probe = 0; probe <= m->tmask; probe++) {
        MapEnt *e = &tab[idx];
        if (e->key == 0) {
            e->key = key;
            e->next[0] = next;
            e->cnt[0] = 1;
            for (int k = 1; k < TT_MAP_K; k++) { e->next[k] = 0; e->cnt[k] = 0; }
            m->ents_used++;
            m->inserts++;
            return;
        }
        if (e->key == key) {
            m->inserts++;
            for (int k = 0; k < TT_MAP_K; k++) {
                if (e->cnt[k] > 0 && e->next[k] == next) {
                    if (e->cnt[k] < 0xFFFFu) e->cnt[k]++;
                    return;
                }
            }
            /* free slot? */
            for (int k = 0; k < TT_MAP_K; k++) {
                if (e->cnt[k] == 0) { e->next[k] = next; e->cnt[k] = 1; return; }
            }
            /* Space-Saving: replace the minimum-count slot (deterministic:
             * lowest k wins among equal mins) with (next, min+1). */
            int mn = 0;
            for (int k = 1; k < TT_MAP_K; k++)
                if (e->cnt[k] < e->cnt[mn]) mn = k;
            e->next[mn] = next;
            e->cnt[mn] = (e->cnt[mn] < 0xFFFFu) ? (uint16_t)(e->cnt[mn] + 1) : 0xFFFFu;
            return;
        }
        idx = (idx + 1) & m->tmask;
    }
    /* table full (cannot happen at the sized load factors): drop the insert
     * rather than spin; drafting quality degrades, correctness does not. */
}

/* Rebuild the table from current ring history (drop counts of evicted
 * tokens; keys are hashes so nothing else can go stale). */
static void map_rebuild(tt_ngram_map *m) {
    map_clear(m->tab, TT_MAP_CAP);
    m->ents_used = 0;
    m->inserts = 0;
    const uint64_t base = m->total - m->len;      /* first resident logical */
    for (uint64_t p = base; p < m->total; p++) {
        for (int wi = 0; wi < TT_MAP_NWIN; wi++) {
            int n = k_map_windows[wi];
            if ((uint64_t)n > p - base) continue; /* context not resident   */
            uint64_t key = map_hash(m, NULL, p, n);
            map_insert_one(m, key, map_at(m, p));
        }
    }
}

static void map_insert_for(tt_ngram_map *m, uint64_t p) {
    const uint64_t base = m->total - m->len;
    for (int wi = 0; wi < TT_MAP_NWIN; wi++) {
        int n = k_map_windows[wi];
        if ((uint64_t)n > p - base) continue;
        uint64_t key = map_hash(m, NULL, p, n);
        map_insert_one(m, key, map_at(m, p));
    }
    /* 75% load factor -> rebuild from resident history */
    if (m->ents_used * 4u >= TT_MAP_CAP * 3u) map_rebuild(m);
}

tt_ngram_map *tt_ngram_map_create(uint32_t history_cap, uint32_t max_draft) {
    if (max_draft == 0) return NULL;
    if (max_draft > TT_MAP_MAX_DRAFT) max_draft = TT_MAP_MAX_DRAFT;
    if (history_cap == 0) history_cap = TT_MAP_HIST;
    if (history_cap < 32) return NULL;           /* useless config          */

    tt_ngram_map *m = (tt_ngram_map *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    m->buf = (uint32_t *)malloc((size_t)history_cap * sizeof(uint32_t));
    m->tab = (MapEnt *)malloc((size_t)TT_MAP_CAP * sizeof(MapEnt));
    if (!m->buf || !m->tab) {
        free(m->buf); free(m->tab); free(m);
        return NULL;
    }
    m->cap = history_cap;
    m->tmask = TT_MAP_CAP - 1u;
    m->max_draft = max_draft;
    map_clear(m->tab, TT_MAP_CAP);
    return m;
}

void tt_ngram_map_free(tt_ngram_map *m) {
    if (!m) return;
    free(m->buf);
    free(m->tab);
    free(m);
}

int tt_ngram_map_feed(tt_ngram_map *m, const uint32_t *toks, uint32_t count) {
    if (!m || (!toks && count > 0)) return -1;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t slot = (uint32_t)(m->total % m->cap);
        m->buf[slot] = toks[i];
        m->total++;
        if (m->len < m->cap) m->len++;
        map_insert_for(m, m->total - 1);
    }
    return 0;
}

uint32_t tt_ngram_map_len(const tt_ngram_map *m) { return m ? m->len : 0; }

/* Most frequent continuation for `key` (count desc, id asc). 0 count = none. */
static uint32_t map_top_next(const tt_ngram_map *m, uint64_t key,
                             uint16_t *out_cnt) {
    uint32_t idx = (uint32_t)key & m->tmask;
    const MapEnt *tab = m->tab;
    for (uint32_t probe = 0; probe <= m->tmask; probe++) {
        const MapEnt *e = &tab[idx];
        if (e->key == 0) { if (out_cnt) *out_cnt = 0; return 0; }
        if (e->key == key) {
            int best = -1;
            for (int k = 0; k < TT_MAP_K; k++) {
                if (e->cnt[k] == 0) continue;
                if (best < 0 ||
                    e->cnt[k] > e->cnt[best] ||
                    (e->cnt[k] == e->cnt[best] && e->next[k] < e->next[best]))
                    best = k;
            }
            if (best < 0) { if (out_cnt) *out_cnt = 0; return 0; }
            if (out_cnt) *out_cnt = e->cnt[best];
            return e->next[best];
        }
        idx = (idx + 1) & m->tmask;
    }
    if (out_cnt) *out_cnt = 0;
    return 0;
}

uint32_t tt_ngram_map_draft(const tt_ngram_map *m, uint32_t *out) {
    if (!m || !out || m->len == 0) return 0;
    uint32_t chain[TT_MAP_MAX_DRAFT];
    uint32_t nchain = 0;
    while (nchain < m->max_draft) {
        int got = 0;
        for (int wi = 0; wi < TT_MAP_NWIN && !got; wi++) {
            int n = k_map_windows[wi];
            /* context = n tokens ending at the virtual position
             * (total + nchain); need at least n history-or-chain tokens. */
            if ((uint64_t)n > m->len + nchain) continue;
            uint64_t key = map_hash(m, chain, m->total + nchain, n);
            uint16_t cnt = 0;
            uint32_t nxt = map_top_next(m, key, &cnt);
            if (cnt > 0) {
                chain[nchain++] = nxt;
                got = 1;
            }
        }
        if (!got) break;
    }
    for (uint32_t i = 0; i < nchain; i++) out[i] = chain[i];
    return nchain;
}
