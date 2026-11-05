#!/usr/bin/env bash
# Windows counterpart of scripts/ci_local.sh: CPU-only sanity in seconds.
#
# Covers: mingw syntax-check of all portable C sources, libtinytorch.dll,
# CPU unit tests, samplers / specdec / chat-template unit gates, cpu_backend
# synthetic run (model-backed part auto-skips without model files), IPC bench.
#
# Usage: ./scripts/ci_windows.sh          (exit 0 = green)
set -uo pipefail
cd "$(dirname "$0")/.."

if ! command -v gcc >/dev/null 2>&1 && [ -x /mingw64/bin/gcc.exe ]; then
  export PATH="/mingw64/bin:$PATH"
fi
export PYTHONUTF8=1
# Some embedded Windows Pythons run in safe-path mode and drop the script
# directory from sys.path; make tests/ importable explicitly.
export PYTHONPATH="$PWD/tests"

T0=$(date +%s)
FAIL=0
LOGDIR=build/ci_logs
mkdir -p "$LOGDIR"

run() { # run <name> <cmd...>
  local name="$1"; shift
  echo "=== $name ==="
  if "$@" 2>&1 | tee "$LOGDIR/$name.log"; then
    echo "PASS: $name"
  else
    echo "FAIL: $name (log: $LOGDIR/$name.log)"
    FAIL=1
  fi
}

CC_BIN="${CC:-gcc}"
PY_BIN="${PYTHON:-python}"

echo "=== compile-syntax ==="
SYN_FAIL=0
for f in src/*.c; do
  if ! "$CC_BIN" -std=c11 -fsyntax-only -DTT_IN_LIB -Iinclude -Isrc "$f" \
      >"$LOGDIR/syntax-$(basename "$f").log" 2>&1; then
    echo "FAIL: syntax $f"; cat "$LOGDIR/syntax-$(basename "$f").log"
    SYN_FAIL=1
  fi
done
if [ "$SYN_FAIL" -eq 0 ]; then
  echo "PASS: compile-syntax ($(ls src/*.c | wc -l) files)"
else
  FAIL=1
fi

run build-gcc make -f Makefile.win all
run test-samplers "$PY_BIN" tests/test_samplers.py
run test-specdec-sim "$PY_BIN" tests/test_specdec_sim.py
run test-chat-template "$PY_BIN" tests/test_chat_template.py
run test-profile-decode "$PY_BIN" tests/test_profile_decode.py
run test-kvcache build/test_kvcache.exe
run test-cpu-backend "$PY_BIN" tests/test_cpu_backend.py
run test-ipc build/bench_ipc.exe

# ---- GPU suite (optional: needs nvcc-built engine + a CUDA device) ----
if [ -x build/run_llm_gpu.exe ] && command -v nvidia-smi >/dev/null 2>&1 \
   && nvidia-smi -L >/dev/null 2>&1; then
  run test-gpu-parity "$PY_BIN" tests/test_gpu_parity.py
  run gate-m6-logits "$PY_BIN" tests/gate_m6_logit_parity.py
  run gate-chat "$PY_BIN" tests/gate_chat.py
  run test-tokenizer-special "$PY_BIN" tests/test_tokenizer_special_tokens.py
  run test-server-minimal "$PY_BIN" tests/test_server_minimal.py
  run test-server-multimodel "$PY_BIN" tests/test_server_multimodel.py
  # Python binding + HTTP API (same engine, via ctypes).
  if [ -f data/models/qwen2.5-0.5b-instruct-q4_0.gguf ]; then
    export TT_TEST_MODEL="$PWD/data/models/qwen2.5-0.5b-instruct-q4_0.gguf"
    export TINYINFERENCE_LIB="$PWD/build/tinytorch_engine.dll"
    export PYTHONPATH="$PWD/python:$PWD/tests"
    run python-tests "$PY_BIN" -m pytest python/tests -q
    unset TT_TEST_MODEL TINYINFERENCE_LIB
    export PYTHONPATH="$PWD/tests"
  else
    echo "=== python-tests ==="; echo "SKIP: qwen2.5 model not present"
  fi
  for t in test_arch_registry test_gemv_typed test_batch4_gemv test_spec_verify \
           test_q8_kvcache test_q4_kvcache test_q4_split_exact test_qcache_backfill \
           test_prefill_gemm test_wmma_prefill_gemm test_logits_q8_v4; do
    [ -x "build/$t.exe" ] && run "$t" "build/$t.exe"
  done
  # Oracle-backed tokenizer parity (needs a built llama.cpp oracle).
  if [ -x oracle/llama.cpp/build/bin/llama-cli.exe ]; then
    run gate-tokenizer "$PY_BIN" tests/gate_tokenizer.py
    if compgen -G "data/golden/*.jsonl" >/dev/null \
       && [ -f data/models/qwen2.5-0.5b-instruct-q4_0.gguf ]; then
      run engine-golden "$PY_BIN" tests/test_engine_golden.py
    else
      echo "=== engine-golden ==="; echo "SKIP: goldens/models not present"
    fi
    GEMMA4=data/models/gemma-4-E2B-it-Q4_0.gguf
    if [ -f "$GEMMA4" ]; then
      run gate-ple-golden "$PY_BIN" tests/gate_ple_golden.py --model "$GEMMA4"
      run gate-m84-gemma4 "$PY_BIN" tests/gate_m84_gemma4.py --model "$GEMMA4"
    else
      echo "=== gate-ple-golden ==="; echo "SKIP: $GEMMA4 not present"
    fi
  else
    echo "=== gate-tokenizer ==="; echo "SKIP: llama.cpp oracle not built"
  fi
else
  echo "=== gpu ==="; echo "SKIP: no CUDA device or CUDA build"
fi

SEC=$(( $(date +%s) - T0 ))
if [ "$FAIL" -eq 0 ]; then
  echo "ci_windows: GREEN (${SEC}s)"
else
  echo "ci_windows: RED (${SEC}s)"
fi
exit $FAIL
