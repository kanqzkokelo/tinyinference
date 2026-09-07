#!/usr/bin/env bash
# Single source of truth. Exit 0 = gate green, nonzero = red.
# Usage: ./scripts/verify.sh [m3|m61|m84|ple|tok|backfill|all]
set -uo pipefail
cd "$(dirname "$0")/.."

GATE="${1:-all}"
FAIL=0

run() { # run <name> <cmd...>
  local name="$1"; shift
  echo "=== $name ==="
  if "$@"; then echo "PASS: $name"; else echo "FAIL: $name"; FAIL=1; fi
}

case "$GATE" in
  m3)  run "gpu-parity"      python3 tests/test_gpu_parity.py ;;
  m61) run "m6-logits-parity" python3 tests/gate_m6_logit_parity.py
       run "chat-multiturn"    python3 tests/gate_chat.py ;;
  m84) run "m84-gemma4-parity" python3 tests/gate_m84_gemma4.py ;;
  ple) run "m6-ple-golden"     python3 tests/gate_ple_golden.py ;;
  tok) run "tokenizer-oracle-parity" python3 tests/gate_tokenizer.py ;;
  backfill) make -s test_qcache_backfill &&
            run "qcache-backfill" build/test_qcache_backfill ;;
  all) for g in m3 m61 m84 ple tok backfill; do "$0" "$g"; done ;;
  *) echo "unknown gate: $GATE"; exit 2 ;;
esac

exit $FAIL
