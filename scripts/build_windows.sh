#!/usr/bin/env bash
# Native Windows build (MSYS2 / mingw-w64). Linux users keep using Makefile.
#
# Usage:
#   ./scripts/build_windows.sh            # CPU library, tools, tests
#   ./scripts/build_windows.sh cuda       # also build the CUDA engine
#   ./scripts/build_windows.sh clean
set -euo pipefail
cd "$(dirname "$0")/.."
# MSVC + SDK + CUDA from a dev shell, standard install, or TT_* overrides.
source scripts/win_env.sh

if ! command -v gcc >/dev/null 2>&1 && [ -x /mingw64/bin/gcc.exe ]; then
  export PATH="/mingw64/bin:$PATH"
fi
if ! command -v gcc >/dev/null 2>&1; then
  echo "error: gcc not found; install MSYS2 mingw-w64-x86_64-gcc" >&2
  exit 1
fi

if [ "${1:-}" = "cuda" ]; then
  if ! command -v cl.exe >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
    echo "error: cuda build needs nvcc and the MSVC host compiler on PATH." >&2
    echo "       Run from a 'x64 Native Tools Command Prompt for VS', then" >&2
    echo "       start MSYS2 bash and re-run this script." >&2
    exit 1
  fi
fi

make -f Makefile.win "$@"
