"""Tests for the JSON grammar and JSON-mode generation."""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tinyinference._json import JSONState  # noqa: E402

VALID = [
    b"{}",
    b"[]",
    b'{"a":1}',
    b'{"a": [1, 2, {"b": null}]}',
    b'"hi"',
    b"-1.5e+10",
    b"0",
    b"true",
    b"false",
    b"null",
    b'{"a":"x\\u00e9"}',
    b'{"k": [ ] }',
    b"123",
    b'{"a":1} ',
    b'"\\ud83d\\ude00"',
]

INVALID = [
    b'{"a": }',
    b'{"a" 1}',
    b"[1,]",
    b"01",
    b"1.",
    b"1e",
    b'"a',
    b"tru",
    b"-",
    b'{"a":1',
    b"}",
    b"[}",
    b'"\\q"',
    b"\x01",
    b'{"a":1}{"b":2}',
]


@pytest.mark.parametrize("doc", VALID)
def test_accepts_valid(doc):
    st = JSONState().accept(doc)
    assert st is not None
    assert st.complete()


@pytest.mark.parametrize("doc", INVALID)
def test_rejects_invalid(doc):
    st = JSONState().accept(doc)
    assert st is None or not st.complete()


def test_incremental_bytewise():
    doc = b'{"name": "tiny", "n": [1, 2.5, -3e2], "ok": true}'
    st = JSONState()
    for i in range(len(doc)):
        st = st.accept(doc[i : i + 1])
        assert st is not None, i
    assert st.complete()


def test_prefix_allows_continuation():
    st = JSONState().accept(b'{"a": 12')
    assert st is not None and not st.complete()
    st2 = st.accept(b"3}")
    assert st2 is not None and st2.complete()


def test_allowed_first_is_a_subset_of_accepting():
    """Every advertised first byte must lead somewhere."""
    st = JSONState().accept(b'{"a": [1')
    for b in st.allowed_first():
        assert st.accept(bytes([b])) is not None


def test_mask_keeps_only_grammar_tokens():
    from tinyinference import Model

    path = os.environ.get("TT_MODEL")
    if not path or not os.path.isfile(path):
        pytest.skip("TT_MODEL not set")
    with Model(path, n_ctx=256) as m:
        st = JSONState().accept(b'{"a": ')
        mask = m.json_mask(st)
        allowed = [i for i, v in enumerate(mask) if v]
        assert allowed
        for tid in allowed[:64]:
            assert st.accept(m.decode(tid)) is not None
        # a token that would close the object is not valid right after ':'
        for tid in range(m.vocab_size):
            if m.decode(tid) == b"}":
                assert mask[tid] == 0
                break


def test_generate_json():
    from tinyinference import Model

    path = os.environ.get("TT_MODEL")
    if not path or not os.path.isfile(path):
        pytest.skip("TT_MODEL not set")
    with Model(path, n_ctx=512) as m:
        text = m.generate_json(
            'Reply with JSON only. {"city": "Paris", "population": ',
            max_tokens=128,
        )
        doc = json.loads(text)
        assert isinstance(doc, dict)
