/* json_grammar.c -- see json_grammar.h. */
#include "json_grammar.h"

#include <string.h>

enum {
    JS_VALUE = 0,      /* a value must start here                      */
    JS_ARRAY_START,    /* just after '[': value or ']'                 */
    JS_OBJECT_START,   /* just after '{': key or '}'                   */
    JS_OBJECT_KEY,     /* just after ',': key required                 */
    JS_COLON,          /* after a key: ':'                             */
    JS_COMMA_OR_END,   /* after a value inside a container             */
    JS_STRING,         /* inside a string                              */
    JS_ESC,            /* after '\'                                    */
    JS_UHEX,           /* inside \uXXXX                                */
    JS_NUM_START,      /* after '-': digit required                    */
    JS_NUM_ZERO,       /* after a leading 0                            */
    JS_NUM_INT,        /* integer digits                               */
    JS_NUM_DOT,        /* after '.': digit required                    */
    JS_NUM_FRAC,       /* fractional digits                            */
    JS_NUM_E,          /* after e/E: digit or sign required            */
    JS_NUM_ESIGN,      /* after e+/e-: digit required                  */
    JS_NUM_EXP,        /* exponent digits                              */
    JS_LITERAL,        /* matching true / false / null                 */
    JS_DONE            /* root value complete                          */
};

static const char *const k_literals[3] = { "true", "false", "null" };

static int is_ws(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int is_digit(unsigned char c) { return c >= '0' && c <= '9'; }

static int is_hex(unsigned char c) {
    return is_digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

/* A value just finished: the root is done, otherwise expect ',' or a close. */
static void after_value(tt_json_state *st) {
    st->s = st->depth ? JS_COMMA_OR_END : JS_DONE;
}

static int push(tt_json_state *st, unsigned char kind) {
    if (st->depth >= TT_JSON_MAX_DEPTH) return -1;
    st->stack[st->depth++] = kind;
    return 0;
}

/* Re-processing loop: a byte that ends a number or literal belongs to the
 * next state, so fall through with `goto retry` instead of dropping it. */
static int feed_byte(tt_json_state *st, unsigned char c) {
retry:
    switch (st->s) {
    case JS_VALUE:
        if (is_ws(c)) return 0;
        if (st->depth == 0 && st->require_object && c != '{') return -1;
        switch (c) {
        case '{': if (push(st, 'o')) return -1; st->s = JS_OBJECT_START; return 0;
        case '[': if (push(st, 'a')) return -1; st->s = JS_ARRAY_START; return 0;
        case '"': st->key = 0; st->s = JS_STRING; return 0;
        case '-': st->s = JS_NUM_START; return 0;
        case '0': st->s = JS_NUM_ZERO; return 0;
        case 't': case 'f': case 'n':
            st->lit_id  = (c == 't') ? 0 : (c == 'f') ? 1 : 2;
            st->lit_pos = 1;
            st->s = JS_LITERAL;
            return 0;
        default:
            if (c >= '1' && c <= '9') { st->s = JS_NUM_INT; return 0; }
            return -1;
        }

    case JS_ARRAY_START:
        if (is_ws(c)) return 0;
        if (c == ']') {
            st->depth--;
            after_value(st);
            return 0;
        }
        st->s = JS_VALUE;
        goto retry;

    case JS_OBJECT_START:
        if (is_ws(c)) return 0;
        if (c == '}') {
            st->depth--;
            after_value(st);
            return 0;
        }
        if (c != '"') return -1;
        st->key = 1;
        st->s = JS_STRING;
        return 0;

    case JS_OBJECT_KEY:
        if (is_ws(c)) return 0;
        if (c != '"') return -1;
        st->key = 1;
        st->s = JS_STRING;
        return 0;

    case JS_COLON:
        if (is_ws(c)) return 0;
        if (c != ':') return -1;
        st->s = JS_VALUE;
        return 0;

    case JS_COMMA_OR_END:
        if (is_ws(c)) return 0;
        if (c == ',') {
            st->s = (st->stack[st->depth - 1] == 'o') ? JS_OBJECT_KEY : JS_VALUE;
            return 0;
        }
        if (c == '}' && st->stack[st->depth - 1] == 'o') {
            st->depth--;
            after_value(st);
            return 0;
        }
        if (c == ']' && st->stack[st->depth - 1] == 'a') {
            st->depth--;
            after_value(st);
            return 0;
        }
        return -1;

    case JS_STRING:
        if (c == '"') {
            if (st->key) st->s = JS_COLON;
            else         after_value(st);
            return 0;
        }
        if (c == '\\') { st->s = JS_ESC; return 0; }
        if (c < 0x20) return -1;
        return 0;

    case JS_ESC:
        switch (c) {
        case '"': case '\\': case '/': case 'b':
        case 'f': case 'n': case 'r': case 't':
            st->s = JS_STRING;
            return 0;
        case 'u':
            st->uhex = 0;
            st->s = JS_UHEX;
            return 0;
        default:
            return -1;
        }

    case JS_UHEX:
        if (!is_hex(c)) return -1;
        if (++st->uhex == 4) st->s = JS_STRING;
        return 0;

    case JS_NUM_START:
        if (!is_digit(c)) return -1;
        st->s = (c == '0') ? JS_NUM_ZERO : JS_NUM_INT;
        return 0;

    case JS_NUM_ZERO:
        if (is_digit(c)) return -1;             /* no leading zeros */
        if (c == '.') { st->s = JS_NUM_DOT; return 0; }
        if (c == 'e' || c == 'E') { st->s = JS_NUM_E; return 0; }
        after_value(st);
        goto retry;

    case JS_NUM_INT:
        if (is_digit(c)) return 0;
        if (c == '.') { st->s = JS_NUM_DOT; return 0; }
        if (c == 'e' || c == 'E') { st->s = JS_NUM_E; return 0; }
        after_value(st);
        goto retry;

    case JS_NUM_DOT:
        if (!is_digit(c)) return -1;
        st->s = JS_NUM_FRAC;
        return 0;

    case JS_NUM_FRAC:
        if (is_digit(c)) return 0;
        if (c == 'e' || c == 'E') { st->s = JS_NUM_E; return 0; }
        after_value(st);
        goto retry;

    case JS_NUM_E:
        if (c == '+' || c == '-') { st->s = JS_NUM_ESIGN; return 0; }
        if (!is_digit(c)) return -1;
        st->s = JS_NUM_EXP;
        return 0;

    case JS_NUM_ESIGN:
        if (!is_digit(c)) return -1;
        st->s = JS_NUM_EXP;
        return 0;

    case JS_NUM_EXP:
        if (is_digit(c)) return 0;
        after_value(st);
        goto retry;

    case JS_LITERAL: {
        const char *lit = k_literals[st->lit_id];
        if (c != (unsigned char)lit[st->lit_pos]) return -1;
        if (lit[++st->lit_pos] == '\0') after_value(st);
        return 0;
    }

    case JS_DONE:
        return is_ws(c) ? 0 : -1;
    }
    return -1;
}

void tt_json_init(tt_json_state *st, int require_object) {
    memset(st, 0, sizeof(*st));
    st->s = JS_VALUE;
    st->require_object = require_object ? 1 : 0;
}

int tt_json_feed(tt_json_state *st, const char *buf, int n) {
    for (int i = 0; i < n; i++)
        if (feed_byte(st, (unsigned char)buf[i])) return -1;
    return 0;
}

int tt_json_complete(const tt_json_state *st) {
    if (st->s == JS_DONE) return 1;
    /* A root-level number has no terminator to consume, so it counts as
     * complete as soon as its digits are well formed. */
    if (st->depth == 0 && (st->s == JS_NUM_ZERO || st->s == JS_NUM_INT ||
                           st->s == JS_NUM_FRAC || st->s == JS_NUM_EXP))
        return 1;
    return 0;
}

int tt_json_accepts(const tt_json_state *st, const char *buf, int n) {
    tt_json_state tmp = *st;
    return tt_json_feed(&tmp, buf, n) == 0;
}
