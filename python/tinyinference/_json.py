"""Incremental JSON grammar used to constrain sampling.

The state machine works on bytes, one token at a time, and never needs the
whole document.  `JSONState.accept()` returns a new state, so candidate
tokens can be tried without disturbing the current one.
"""

from __future__ import annotations

__all__ = ["JSONState"]

_WS = frozenset(b" \t\r\n")
_DIGITS = frozenset(b"0123456789")
_HEX = frozenset(b"0123456789abcdefABCDEF")
_VALUE_START = frozenset(b'{["-0123456789tfn')
_ESCAPES = frozenset(b'"\\/bfnrtu')

_WS_STATES = frozenset(
    {
        "value",
        "arr_first",
        "arr_val",
        "obj_first",
        "obj_key",
        "colon",
        "obj_val",
        "obj_comma_or_end",
        "arr_comma_or_end",
        "done",
    }
)


class JSONState:
    """A JSON document prefix, as a pushdown automaton."""

    __slots__ = ("mode", "stack", "lit", "unhex", "num", "key")

    def __init__(
        self,
        mode: str = "value",
        stack: tuple[str, ...] = (),
        lit: str = "",
        unhex: int = 0,
        num: str = "",
        key: bool = False,
    ) -> None:
        self.mode = mode
        self.stack = stack
        self.lit = lit
        self.unhex = unhex
        self.num = num
        self.key = key

    def _copy(self, **kw: object) -> "JSONState":
        st = JSONState(self.mode, self.stack, self.lit, self.unhex, self.num, self.key)
        for name, value in kw.items():
            setattr(st, name, value)
        return st

    # -- public --------------------------------------------------------

    def accept(self, data: bytes) -> "JSONState | None":
        st: JSONState | None = self
        for b in data:
            st = st._feed(b)
            if st is None:
                return None
        return st

    def complete(self) -> bool:
        if self.stack:
            return False
        if self.mode == "number":
            return self._number_ok()
        return self.mode in ("after", "done")

    def allowed_first(self) -> frozenset[int]:
        """Bytes that can begin a valid next token (for candidate lookup)."""
        mode = self.mode
        ws = _WS if mode in _WS_STATES else frozenset()
        if mode in ("value", "arr_val", "obj_val"):
            return frozenset(_VALUE_START | ws)
        if mode == "arr_first":
            return frozenset(_VALUE_START | ws | {ord("]")})
        if mode == "obj_first":
            return frozenset(b'"}' + bytes(ws))
        if mode == "obj_key":
            return frozenset(b'"' + bytes(ws))
        if mode == "colon":
            return frozenset(b":" + bytes(ws))
        if mode == "obj_comma_or_end":
            return frozenset(b",}" + bytes(ws))
        if mode == "arr_comma_or_end":
            return frozenset(b",]" + bytes(ws))
        if mode == "done":
            return ws
        if mode == "after":
            return self._terminators()
        if mode == "string":
            return frozenset(b'"\\' + bytes(range(0x20, 0x100)))
        if mode == "escape":
            return _ESCAPES
        if mode == "unicode":
            return _HEX
        if mode == "number":
            return self._number_first()
        if mode == "literal":
            return frozenset({ord(self.lit[0])}) if self.lit else frozenset()
        return frozenset()

    # -- transitions ---------------------------------------------------

    def _feed(self, b: int) -> "JSONState | None":
        mode = self.mode
        if mode == "literal":
            if not self.lit or b != ord(self.lit[0]):
                return None
            rest = self.lit[1:]
            if rest:
                return self._copy(lit=rest)
            return JSONState("after" if self.stack else "done", self.stack)
        if mode == "number":
            return self._feed_number(b)
        if mode == "string":
            return self._feed_string(b)
        if mode == "escape":
            return self._feed_escape(b)
        if mode == "unicode":
            if b not in _HEX:
                return None
            if self.unhex == 1:
                return self._copy(mode="string", unhex=0)
            return self._copy(unhex=self.unhex - 1)
        if mode == "after":
            return self._after_value(b)
        if mode == "done":
            return self._copy() if b in _WS else None
        if b in _WS:
            return self._copy() if mode in _WS_STATES else None

        if mode in ("value", "arr_first", "arr_val", "obj_val"):
            if b == ord("{"):
                return JSONState("obj_first", self.stack + ("obj",))
            if b == ord("["):
                return JSONState("arr_first", self.stack + ("arr",))
            if b == ord('"'):
                return JSONState("string", self.stack)
            if b == ord("-") or b in _DIGITS:
                return JSONState("number", self.stack, num=chr(b))
            if b == ord("t"):
                return JSONState("literal", self.stack, lit="rue")
            if b == ord("f"):
                return JSONState("literal", self.stack, lit="alse")
            if b == ord("n"):
                return JSONState("literal", self.stack, lit="ull")
            if mode == "arr_first" and b == ord("]"):
                return self._close("arr")
            return None
        if mode == "obj_first":
            if b == ord('"'):
                return JSONState("string", self.stack, key=True)
            if b == ord("}"):
                return self._close("obj")
            return None
        if mode == "obj_key":
            if b == ord('"'):
                return JSONState("string", self.stack, key=True)
            return None
        if mode == "colon":
            return self._copy(mode="obj_val") if b == ord(":") else None
        if mode == "obj_comma_or_end":
            if b == ord(","):
                return self._copy(mode="obj_key")
            if b == ord("}"):
                return self._close("obj")
            return None
        if mode == "arr_comma_or_end":
            if b == ord(","):
                return self._copy(mode="arr_val")
            if b == ord("]"):
                return self._close("arr")
            return None
        return None

    def _close(self, kind: str) -> "JSONState | None":
        if not self.stack or self.stack[-1] != kind:
            return None
        stack = self.stack[:-1]
        if not stack:
            return JSONState("done")
        return JSONState("after", stack)

    def _after_value(self, b: int) -> "JSONState | None":
        """A complete value ended; route to the enclosing container."""
        if not self.stack:
            return JSONState("done") if b in _WS else None
        top = self.stack[-1]
        st = JSONState("obj_comma_or_end" if top == "obj" else "arr_comma_or_end", self.stack)
        if b in _WS:
            return st
        return st._feed(b)

    def _feed_string(self, b: int) -> "JSONState | None":
        if b == ord('"'):
            if self.key:
                return JSONState("colon", self.stack)
            return JSONState("after" if self.stack else "done", self.stack)
        if b == ord("\\"):
            return self._copy(mode="escape")
        if b < 0x20:
            return None
        return self._copy()

    def _feed_escape(self, b: int) -> "JSONState | None":
        if b == ord("u"):
            return self._copy(mode="unicode", unhex=4)
        if b in _ESCAPES:
            return self._copy(mode="string")
        return None

    # -- numbers -------------------------------------------------------

    def _feed_number(self, b: int) -> "JSONState | None":
        num = self.num
        c = chr(b)
        if b in _DIGITS:
            if num == "-":
                return self._copy(num=num + c)
            if num in ("0", "-0"):
                return None
            if num[-1] in "0123456789":
                return self._copy(num=num + c)
            if num[-1] in ".eE" or num[-2:] in ("e+", "e-", "E+", "E-"):
                return self._copy(num=num + c)
            return None
        if b == ord("."):
            return self._copy(num=num + c) if self._int_ok(num) else None
        if b in (ord("e"), ord("E")):
            if not (self._int_ok(num) or self._frac_ok(num)):
                return None
            return self._copy(num=num + c)
        if b in (ord("+"), ord("-")):
            if num and num[-1] in "eE":
                return self._copy(num=num + c)
            return None
        if not self._number_ok():
            return None
        return JSONState("after" if self.stack else "done", self.stack)._feed(b)

    def _number_first(self) -> frozenset[int]:
        num = self.num
        if num == "-" or num[-1] in ".eE" or num[-2:] in ("e+", "e-", "E+", "E-"):
            return frozenset(_DIGITS)
        if not self._number_ok():
            return frozenset()
        out = set()
        if num not in ("0", "-0"):
            out |= _DIGITS
        if self._int_ok(num):
            out |= {ord("."), ord("e"), ord("E")}
        elif self._frac_ok(num):
            out |= {ord("e"), ord("E")}
        out |= self._terminators()
        return frozenset(out)

    def _terminators(self) -> frozenset[int]:
        """Bytes that may follow a finished value in the current container."""
        out = set(_WS)
        if not self.stack:
            return frozenset(out)
        if self.stack[-1] == "obj":
            out |= {ord(","), ord("}")}
        else:
            out |= {ord(","), ord("]")}
        return frozenset(out)

    @staticmethod
    def _int_ok(num: str) -> bool:
        if num.startswith("-"):
            num = num[1:]
        return bool(num) and (num == "0" or (num[0] in "123456789" and num.isdigit()))

    @staticmethod
    def _frac_ok(num: str) -> bool:
        if "." not in num:
            return False
        frac = num.split(".", 1)[1]
        frac = frac.replace("e", "E").split("E", 1)[0]
        return bool(frac) and frac.isdigit()

    def _number_ok(self) -> bool:
        num = self.num
        if not num:
            return False
        has_exp = "e" in num or "E" in num
        base, _, exp = num.replace("E", "e").partition("e")
        if has_exp:
            if exp[:1] in ("+", "-"):
                exp = exp[1:]
            if not exp.isdigit():
                return False
        if "." in base:
            return self._int_ok(base.split(".", 1)[0]) and self._frac_ok(base)
        return self._int_ok(base)


def from_prefix(text: str) -> "tuple[str, JSONState] | None":
    """Split *text* into (json prefix, state).

    Prompts often end with a partial document ("... {\"city\": \"Pa").
    Returns None when *text* holds no JSON value start.
    """
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return None
    raw = text[start:].encode()
    st = JSONState()
    used = 0
    for i, b in enumerate(raw):
        nxt = st._feed(b)
        if nxt is None:
            break
        st = nxt
        used = i + 1
    return raw[:used].decode(), st

