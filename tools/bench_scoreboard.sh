#!/usr/bin/env bash
# tools/bench_scoreboard.sh — wrapper calling authoritative bench/bench_llm.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ $# -eq 0 ]; then
    exec python3 bench/bench_llm.py
elif [ $# -eq 1 ] && [[ "$1" != -* ]]; then
    exec python3 bench/bench_llm.py --model "$1"
else
    exec python3 bench/bench_llm.py "$@"
fi
