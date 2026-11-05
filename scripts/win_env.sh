#!/usr/bin/env bash
# Source this file to put the MSVC + Windows SDK + CUDA toolchain on PATH.
#
# Works with a normal "x64 Native Tools" shell (nothing to do), a standard
# NVIDIA/MSVC install, or a user-local unpacked toolchain configured through
# TT_MSVC_DIR / TT_WINSDK_DIR / TT_CUDA_DIR.
#
# Usage:  source scripts/win_env.sh

_win_env_add() { # prepend a directory when it exists and is not already there
  [ -d "$1" ] || return 0
  case ":$PATH:" in *":$1:"*) ;; *) PATH="$1:$PATH" ;; esac
}

# mingw-w64 first so make/gcc resolve inside MSYS2.
_win_env_add /mingw64/bin

# --- MSVC ---------------------------------------------------------------
# MSYS2 may report HOME=/home/<user>; the real profile lives at USERPROFILE.
_home="$HOME"
if [ -n "${USERPROFILE:-}" ] && command -v cygpath >/dev/null 2>&1; then
  _home="$(cygpath -u "$USERPROFILE")"
fi
if ! command -v cl.exe >/dev/null 2>&1; then
  MSVC_ROOT="${TT_MSVC_DIR:-}"
  if [ -z "$MSVC_ROOT" ]; then
    for d in "$_home"/msvc/Contents/VC/Tools/MSVC/* \
             "/c/Program Files/Microsoft Visual Studio"/*/*/VC/Tools/MSVC/* \
             "/c/Program Files (x86)/Microsoft Visual Studio"/*/*/VC/Tools/MSVC/*; do
      [ -x "$d/bin/Hostx64/x64/cl.exe" ] && MSVC_ROOT="$d" && break
    done
  fi
  if [ -n "$MSVC_ROOT" ] && [ -d "$MSVC_ROOT/bin/Hostx64/x64" ]; then
    _win_env_add "$MSVC_ROOT/bin/Hostx64/x64"
    _m="$(cygpath -m "$MSVC_ROOT" 2>/dev/null || echo "$MSVC_ROOT")"
    export INCLUDE="$_m/include${INCLUDE:+;$INCLUDE}"
    export LIB="$_m/lib/x64${LIB:+;$LIB}"
  fi
fi

# --- Windows SDK --------------------------------------------------------
if [ -z "${TT_WINSDK_DIR:-}" ]; then
  for d in "$_home"/winsdk/c "/c/Program Files (x86)/Windows Kits/10"; do
    [ -d "$d/Include" ] && TT_WINSDK_DIR="$d" && break
  done
fi
if [ -n "${TT_WINSDK_DIR:-}" ]; then
  for v in "$TT_WINSDK_DIR"/Include/*/; do
    [ -d "$v/ucrt" ] || continue
    ver="$(basename "${v%/}")"
    _win_env_add "$TT_WINSDK_DIR/bin/$ver/x64"
    _s="$(cygpath -m "$TT_WINSDK_DIR" 2>/dev/null || echo "$TT_WINSDK_DIR")"
    _lib="$TT_WINSDK_DIR/Lib/$ver"
    [ -d "$_lib/um/x64" ] || _lib="$TT_WINSDK_DIR"   # flattened unpack
    _l="$(cygpath -m "$_lib" 2>/dev/null || echo "$_lib")"
    export INCLUDE="$_s/Include/$ver/ucrt;$_s/Include/$ver/um;$_s/Include/$ver/shared;$_s/Include/$ver/winrt${INCLUDE:+;$INCLUDE}"
    export LIB="$_l/ucrt/x64;$_l/um/x64${LIB:+;$LIB}"
    break
  done
fi

# --- CUDA ---------------------------------------------------------------
if command -v nvcc >/dev/null 2>&1; then
  CUDA_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
else
  CUDA_ROOT="${TT_CUDA_DIR:-${CUDA_PATH:-}}"
  if [ -z "$CUDA_ROOT" ]; then
    for d in "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA"/* \
             "$_home"/cuda/toolkit; do
      [ -x "$d/bin/nvcc.exe" ] && CUDA_ROOT="$d" && break
    done
  fi
  [ -n "$CUDA_ROOT" ] && _win_env_add "$CUDA_ROOT/bin"
fi
if [ -n "${CUDA_ROOT:-}" ] && [ -d "$CUDA_ROOT/include" ]; then
  export CUDA_BIN="$CUDA_ROOT/bin/"
  export CUDA_INC="$(cygpath -m "$CUDA_ROOT/include" 2>/dev/null || echo "$CUDA_ROOT/include")"
fi
unset -f _win_env_add
