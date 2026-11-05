/* test_json_grammar.c -- prefix acceptance, rejection and completion. */
#include "json_grammar.h"

#include <stdio.h>
#include <string.h>

static int failures;

static void expect(const char *doc, int want_ok, int want_done, int require_object) {
    tt_json_state st;
    tt_json_init(&st, require_object);
    const int ok = tt_json_feed(&st, doc, (int)strlen(doc)) == 0;
    const int done = tt_json_complete(&st);
    if (ok != want_ok || (want_ok && done != want_done)) {
        printf("FAIL %-28s ok=%d want %d, done=%d want %d\n",
               doc, ok, want_ok, done, want_done);
        failures++;
    }
}

/* Feed `prefix`, then check whether `tok` would be accepted. */
static void expect_accepts(const char *prefix, const char *tok, int want) {
    tt_json_state st;
    tt_json_init(&st, 1);
    if (tt_json_feed(&st, prefix, (int)strlen(prefix))) {
        printf("FAIL prefix invalid: %s\n", prefix);
        failures++;
        return;
    }
    const int got = tt_json_accepts(&st, tok, (int)strlen(tok));
    if (got != want) {
        printf("FAIL after %-16s token %-8s accepted=%d want %d\n",
               prefix, tok, got, want);
        failures++;
    }
}

int main(void) {
    /* complete documents */
    expect("{}", 1, 1, 1);
    expect("[]", 1, 1, 0);
    expect("{\"a\":1}", 1, 1, 1);
    expect("{\"a\":[1,2,{\"b\":null}]}", 1, 1, 1);
    expect("  {\"a\" : \"x\\u00e9\\n\" }  ", 1, 1, 1);
    expect("{\"a\":-0.5e+10}", 1, 1, 1);
    expect("{\"a\":true,\"b\":false,\"c\":null}", 1, 1, 1);
    expect("\"plain\"", 1, 1, 0);
    expect("123", 1, 1, 0);

    /* valid prefixes */
    expect("{", 1, 0, 1);
    expect("{\"a", 1, 0, 1);
    expect("{\"a\":", 1, 0, 1);
    expect("{\"a\":[1,", 1, 0, 1);
    expect("{\"a\":-", 1, 0, 1);
    expect("{\"a\":1.", 1, 0, 1);
    expect("{\"a\":tru", 1, 0, 1);

    /* invalid */
    expect("", 1, 0, 1);
    expect("[1]", 0, 0, 1);           /* require_object */
    expect("{,}", 0, 0, 1);
    expect("{\"a\" 1}", 0, 0, 1);
    expect("{\"a\":01}", 0, 0, 1);
    expect("{\"a\":1,}", 0, 0, 1);
    expect("{\"a\":trux}", 0, 0, 1);
    expect("{\"a\":\"\\q\"}", 0, 0, 1);
    expect("{\"a\":\"\\u12g4\"}", 0, 0, 1);
    expect("{\"a\":1} x", 0, 0, 1);
    expect("{\"a\":+1}", 0, 0, 1);

    /* token-level masking: what may follow an opening brace */
    expect_accepts("{", "\"", 1);
    expect_accepts("{", "}", 1);
    expect_accepts("{", " ", 1);
    expect_accepts("{", "a", 0);
    expect_accepts("{", ",", 0);

    /* after a complete member, only , or } (or space) */
    expect_accepts("{\"a\":1", ",", 1);
    expect_accepts("{\"a\":1", "}", 1);
    expect_accepts("{\"a\":1", "\"", 0);
    expect_accepts("{\"a\":1", ":", 0);

    /* multi-byte token crossing a state boundary */
    expect_accepts("{\"a\":1", ",\"b\":2}", 1);
    expect_accepts("{\"a\":1", ",\"b\" 2}", 0);

    /* numbers: leading zero, exponent forms */
    expect_accepts("{\"a\":0", ".5", 1);
    expect_accepts("{\"a\":0", "1", 0);
    expect_accepts("{\"a\":1", "e-3", 1);
    expect_accepts("{\"a\":1", "e", 1);
    expect_accepts("{\"a\":1e", "x", 0);

    /* strings: escapes and control bytes */
    expect_accepts("{\"a\":\"x", "\\n\"", 1);
    expect_accepts("{\"a\":\"x", "\\u00e9\"", 1);
    expect_accepts("{\"a\":\"x", "\x01", 0);

    /* nesting depth cap */
    {
        tt_json_state st;
        tt_json_init(&st, 0);
        int ok = 0;
        for (int i = 0; i <= TT_JSON_MAX_DEPTH; i++)
            if (tt_json_feed(&st, "[", 1)) { ok = -1; break; }
        if (ok == 0) {
            printf("FAIL depth cap not enforced\n");
            failures++;
        }
    }

    if (failures) {
        printf("json_grammar: %d failure(s)\n", failures);
        return 1;
    }
    printf("json_grammar: all checks passed\n");
    return 0;
}
