#!/usr/bin/env bash
# Single source of truth. Exit 0 = gate green, nonzero = red.
# Usage: ./scripts/verify.sh [m3|m61|m84|ple|tok|backfill|all]
set -uo pipefail
cd "$(dirname "$0")/.."

GATE="${1:-all}"
FAIL=0

# Windows/MSYS2 builds use Makefile.win and .exe suffixes; Python may be
# named either way. Keep Linux behavior untouched.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) IS_WIN=1; MAKE=(make -f Makefile.win); EXE=.exe ;;
  *)                    IS_WIN=0; MAKE=(make);                EXE= ;;
esac

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
export PYTHONPATH="$PWD/tests"

run() { # run <name> <cmd...>
  local name="$1"; shift
  echo "=== $name ==="
  if "$@"; then echo "PASS: $name"; else echo "FAIL: $name"; FAIL=1; fi
}

case "$GATE" in
  m3)  run "gpu-parity"      "$PY" tests/test_gpu_parity.py ;;
  m61) run "m6-logits-parity" "$PY" tests/gate_m6_logit_parity.py
       run "chat-multiturn"    "$PY" tests/gate_chat.py ;;
  m84) run "m84-gemma4-parity" "$PY" tests/gate_m84_gemma4.py ;;
  ple) run "m6-ple-golden"     "$PY" tests/gate_ple_golden.py ;;
  tok) run "tokenizer-oracle-parity" "$PY" tests/gate_tokenizer.py ;;
  backfill) "${MAKE[@]}" -s test_qcache_backfill &&
            run "qcache-backfill" "build/test_qcache_backfill$EXE" ;;
  all) for g in m3 m61 m84 ple tok backfill; do "$0" "$g"; done ;;
  *) echo "unknown gate: $GATE"; exit 2 ;;
esac

exit $FAIL
