#!/usr/bin/env bash
# tinyinference one-line quickstart:
#   curl -sSL https://raw.githubusercontent.com/kanqzkokelo/tinyinference/m6-correctness/scripts/quickstart.sh | bash
set -e
REPO="${TINYINF_REPO:-https://github.com/kanqzkokelo/tinyinference}"
BRANCH="${TINYINF_BRANCH:-m6-correctness}"
DIR="${TINYINF_DIR:-$PWD/tinyinference}"
MODEL_URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q8_0.gguf?download=true"
MODEL_FILE="SmolLM2-135M-Instruct-Q8_0.gguf"

command -v curl >/dev/null || { echo "need curl"; exit 1; }
[ -d "$HOME/mmcuda/bin" ] && export PATH="$HOME/mmcuda/bin:$PATH"
[ -d "$HOME/mmcuda/lib" ] && export LD_LIBRARY_PATH="$HOME/mmcuda/lib:$LD_LIBRARY_PATH"
command -v nvcc >/dev/null || { echo "need CUDA nvcc (set PATH to your CUDA bin)"; exit 1; }
[ -d "$DIR" ] || git clone --depth 1 -b "$BRANCH" "$REPO" "$DIR"
cd "$DIR"
make -j"$(nproc)" build/run_llm_gpu 2>&1 | tail -1
mkdir -p data/models
[ -f "data/models/$MODEL_FILE" ] || curl -sSL -o "data/models/$MODEL_FILE" "$MODEL_URL"
export TT_MODEL="data/models/$MODEL_FILE"
./build/run_llm_gpu "Paris is the capital" 32
