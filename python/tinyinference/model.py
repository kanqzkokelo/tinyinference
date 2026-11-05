"""High-level Python API over the CUDA engine."""

from __future__ import annotations

import codecs
import ctypes
import os
from typing import Iterable, Iterator, Sequence

from ._ffi import lib

__all__ = ["Model"]

DEFAULT_MAX_TOKENS = 256
_ENC_CHUNK = 4096


class Model:
    """A loaded GGUF model.

    The engine keeps one KV cache, so a Model is not thread-safe; callers that
    need concurrency should hold one Model per thread.
    """

    def __init__(self, path: str | os.PathLike[str], n_ctx: int = 0) -> None:
        self.path = str(path)
        self._tok_bytes: list[bytes] | None = None
        self._by_first: dict[int, list[int]] | None = None
        self._handle = lib.tt_api_model_load(
            os.fsencode(self.path), int(n_ctx)
        )
        if not self._handle:
            raise RuntimeError(f"failed to load model: {self.path}")

    # -- lifecycle -----------------------------------------------------

    def close(self) -> None:
        if self._handle:
            lib.tt_api_model_free(self._handle)
            self._handle = None

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _require(self) -> int:
        if not self._handle:
            raise RuntimeError("model is closed")
        return self._handle

    # -- geometry ------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return int(lib.tt_api_vocab(self._require()))

    @property
    def n_ctx(self) -> int:
        return int(lib.tt_api_ctx(self._require()))

    @property
    def pos(self) -> int:
        """Number of tokens currently in the KV cache."""
        return int(lib.tt_api_pos(self._require()))

    @property
    def eos_token_id(self) -> int:
        return int(lib.tt_api_eos(self._require()))

    @property
    def stop_string(self) -> str:
        raw = lib.tt_api_stop_string(self._require()) or b""
        return raw.decode("utf-8", "replace")

    def reset(self) -> None:
        lib.tt_api_reset(self._require())

    # -- text <-> tokens ----------------------------------------------

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        cap = max(_ENC_CHUNK, len(data) * 4 + 16)
        handle = self._require()
        while True:
            buf = (ctypes.c_int * cap)()
            n = int(lib.tt_api_encode(handle, data, buf, cap))
            if n >= 0:
                return list(buf[:n])
            cap = -n

    def decode(self, token: int) -> bytes:
        handle = self._require()
        cap = 64
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = int(lib.tt_api_decode(handle, int(token), buf, cap))
            if n >= 0:
                return buf.raw[:n]
            cap = -n

    def chat_format(self, messages: Sequence[dict]) -> str:
        roles = [str(m.get("role", "user")).encode() for m in messages]
        contents = [str(m.get("content", "")).encode() for m in messages]
        role_arr = (ctypes.c_char_p * len(roles))(*roles)
        content_arr = (ctypes.c_char_p * len(contents))(*contents)
        handle = self._require()
        cap = 4096
        while True:
            buf = ctypes.create_string_buffer(cap)
            n = int(
                lib.tt_api_chat_format(
                    handle, role_arr, content_arr, len(roles), buf, cap
                )
            )
            if n >= 0:
                return buf.raw[:n].decode("utf-8")
            if n == -1:
                raise ValueError("unsupported architecture or role")
            cap = -n

    # -- generation ----------------------------------------------------

    def set_sampling(
        self, temperature: float = 0.0, top_k: int = 0, repeat_penalty: float = 1.0
    ) -> None:
        lib.tt_api_set_sampling(self._require(), float(temperature), int(top_k), float(repeat_penalty))

    def set_token_mask(self, mask: "ctypes.Array | None") -> None:
        """Restrict the next sample to tokens where mask[i] != 0.

        Passing None clears the constraint.  Enabling a mask permanently
        disables CUDA-graph replay for this model (the sampling kernels are
        baked into the graph).
        """
        if mask is None:
            lib.tt_api_set_token_mask(self._require(), None)
            return
        lib.tt_api_set_token_mask(self._require(), mask)

    def prefill(self, tokens: Iterable[int]) -> None:
        ids = [int(t) for t in tokens]
        if not ids:
            return
        arr = (ctypes.c_int * len(ids))(*ids)
        if int(lib.tt_api_prefill(self._require(), arr, len(ids))) != 0:
            raise RuntimeError("prefill failed")

    def next_token(self) -> int:
        return int(lib.tt_api_next(self._require()))

    def stream_tokens(
        self,
        prompt_tokens: Iterable[int],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> Iterator[int]:
        self.prefill(prompt_tokens)
        eos = self.eos_token_id
        limit = min(max_tokens, max(0, self.n_ctx - self.pos - 1))
        for _ in range(limit):
            tok = self.next_token()
            if tok < 0 or tok == eos:
                break
            yield tok

    def stream_text(
        self,
        prompt_tokens: Iterable[int],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stop: str | Sequence[str] | None = None,
    ) -> Iterator[str]:
        stops = self._stop_list(stop)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        seen = ""
        emitted = 0
        for tok in self.stream_tokens(prompt_tokens, max_tokens):
            chunk = decoder.decode(self.decode(tok))
            if not chunk:
                continue
            seen += chunk
            hit = self._find_stop(seen, stops)
            if hit is not None:
                if hit > emitted:
                    yield seen[emitted:hit]
                return
            safe = len(seen) - self._holdback(seen, stops)
            if safe > emitted:
                yield seen[emitted:safe]
                emitted = safe
        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail

    def generate(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_k: int = 0,
        repeat_penalty: float = 1.0,
        stop: str | Sequence[str] | None = None,
    ) -> str:
        self.set_sampling(temperature, top_k, repeat_penalty)
        return "".join(
            self.stream_text(self.encode(prompt), max_tokens, stop=stop)
        )

    def chat(
        self,
        messages: Sequence[dict],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.7,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        stop: str | Sequence[str] | None = None,
    ) -> str:
        self.set_sampling(temperature, top_k, repeat_penalty)
        prompt = self.chat_format(messages)
        return "".join(
            self.stream_text(self.encode(prompt), max_tokens, stop=stop)
        )

    # -- constrained decoding ------------------------------------------

    def _token_table(self) -> tuple[list[bytes], dict[int, list[int]]]:
        if self._tok_bytes is None:
            vocab = self.vocab_size
            data = [self.decode(i) for i in range(vocab)]
            by_first: dict[int, list[int]] = {}
            for tid, raw in enumerate(data):
                if raw:
                    by_first.setdefault(raw[0], []).append(tid)
            self._tok_bytes = data
            self._by_first = by_first
        assert self._by_first is not None
        return self._tok_bytes, self._by_first

    def json_mask(self, state: "JSONState") -> "ctypes.Array":
        """Allow-mask over the vocabulary for one JSON grammar state."""
        from ._json import JSONState

        assert isinstance(state, JSONState)
        tokens, by_first = self._token_table()
        mask = (ctypes.c_uint8 * self.vocab_size)()
        for first in state.allowed_first():
            for tid in by_first.get(first, ()):
                if state.accept(tokens[tid]) is not None:
                    mask[tid] = 1
        if state.complete():
            mask[self.eos_token_id] = 1
        return mask

    def stream_json(
        self,
        prompt_tokens: Iterable[int],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        state: "JSONState | None" = None,
    ) -> Iterator[str]:
        """Generate text constrained to a single valid JSON value."""
        from ._json import JSONState

        self.prefill(prompt_tokens)
        cur = state or JSONState()
        eos = self.eos_token_id
        limit = min(max_tokens, max(0, self.n_ctx - self.pos - 1))
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            for _ in range(limit):
                if cur.complete():
                    break
                self.set_token_mask(self.json_mask(cur))
                tok = self.next_token()
                if tok < 0 or tok == eos:
                    break
                raw = self.decode(tok)
                nxt = cur.accept(raw)
                if nxt is None:
                    break
                cur = nxt
                text = decoder.decode(raw)
                if text:
                    yield text
                if cur.complete():
                    break
        finally:
            self.set_token_mask(None)

    def generate_json(
        self,
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_k: int = 0,
        repeat_penalty: float = 1.0,
    ) -> str:
        """Generate one JSON value and return the whole document.

        A partial document at the end of *prompt* is continued rather than
        restarted, so callers can pre-fill a schema skeleton.
        """
        from ._json import from_prefix

        seeded = from_prefix(prompt)
        prefix, state = seeded if seeded else ("", None)
        self.set_sampling(temperature, top_k, repeat_penalty)
        return prefix + "".join(
            self.stream_json(self.encode(prompt), max_tokens, state=state)
        )

    # -- stop strings --------------------------------------------------

    def _stop_list(self, stop: str | Sequence[str] | None) -> list[str]:
        if stop is None:
            default = self.stop_string
            return [default] if default else []
        if isinstance(stop, str):
            return [stop] if stop else []
        return [s for s in stop if s]

    @staticmethod
    def _find_stop(text: str, stops: Sequence[str]) -> int | None:
        hits = [text.find(s) for s in stops if s in text]
        return min(hits) if hits else None

    @staticmethod
    def _holdback(text: str, stops: Sequence[str]) -> int:
        """Longest suffix of text that is a prefix of a stop string."""
        best = 0
        for stop in stops:
            for k in range(min(len(stop) - 1, len(text)), best, -1):
                if text.endswith(stop[:k]):
                    best = k
                    break
        return best
