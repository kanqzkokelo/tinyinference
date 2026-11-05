"""ctypes bindings for the tinyinference engine shared library."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

__all__ = ["lib", "library_path"]


def _candidates() -> list[Path]:
    env = os.environ.get("TINYINFERENCE_LIB")
    if env:
        return [Path(env)]

    here = Path(__file__).resolve().parent
    names = (
        ["tinytorch_engine.dll", "libtinytorch_engine.so", "libtinytorch_engine.dylib"]
        if sys.platform == "win32"
        else ["libtinytorch_engine.so", "libtinytorch_engine.dylib", "tinytorch_engine.dll"]
    )

    roots = [here / "bin", here.parent.parent / "build"]
    return [root / name for root in roots for name in names]


def library_path() -> Path:
    for path in _candidates():
        if path.is_file():
            return path
    tried = "\n  ".join(str(p) for p in _candidates())
    raise OSError(
        "tinyinference engine library not found. Build it with `make pylib` "
        "(or `make -f Makefile.win pydll` on Windows), or set "
        f"TINYINFERENCE_LIB. Tried:\n  {tried}"
    )


def _load() -> ctypes.CDLL:
    path = library_path()
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(path.parent))
    return ctypes.CDLL(str(path))


lib = _load()

lib.tt_api_model_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.tt_api_model_load.restype = ctypes.c_void_p
lib.tt_api_model_free.argtypes = [ctypes.c_void_p]
lib.tt_api_model_free.restype = None

lib.tt_api_vocab.argtypes = [ctypes.c_void_p]
lib.tt_api_vocab.restype = ctypes.c_int
lib.tt_api_ctx.argtypes = [ctypes.c_void_p]
lib.tt_api_ctx.restype = ctypes.c_int
lib.tt_api_pos.argtypes = [ctypes.c_void_p]
lib.tt_api_pos.restype = ctypes.c_int
lib.tt_api_eos.argtypes = [ctypes.c_void_p]
lib.tt_api_eos.restype = ctypes.c_int
lib.tt_api_reset.argtypes = [ctypes.c_void_p]
lib.tt_api_reset.restype = None

lib.tt_api_encode.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
]
lib.tt_api_encode.restype = ctypes.c_int

lib.tt_api_decode.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
]
lib.tt_api_decode.restype = ctypes.c_int

lib.tt_api_chat_format.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.POINTER(ctypes.c_char_p),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_int,
]
lib.tt_api_chat_format.restype = ctypes.c_int

lib.tt_api_stop_string.argtypes = [ctypes.c_void_p]
lib.tt_api_stop_string.restype = ctypes.c_char_p

lib.tt_api_set_sampling.argtypes = [
    ctypes.c_void_p,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_float,
]
lib.tt_api_set_sampling.restype = None

lib.tt_api_set_token_mask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
lib.tt_api_set_token_mask.restype = None

lib.tt_api_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.tt_api_prefill.restype = ctypes.c_int
lib.tt_api_next.argtypes = [ctypes.c_void_p]
lib.tt_api_next.restype = ctypes.c_int
