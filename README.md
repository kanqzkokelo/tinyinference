# tinyinference

Multi-architecture LLM inference engine in C + CUDA, built from scratch.
Loads GGUF weights, runs quantized transformer decode on NVIDIA GPUs.
Split out of the `nnfromscratch` monorepo; tensor/autograd/ML code lives
in `tinytorch`.

What works: GGUF loader, arch registry (qwen2/llama/qwen3/gemma/gemma4/
tinyllama/granite/smollm2/mistral/internlm2/xverse/exaone/ernie4_5),
BPE tokenizer, chat templates, top-k/top-p/min-p samplers, paged KV-cache
with Q8 backfill, MoE router, N-gram speculative decode, threaded CPU
GEMV backend (Q4_0/Q8_0/Q4_K/Q5_K/Q6_K), CUDA GEMV/GEMM/flash kernels.

## Build

```
make lib                 # CPU objects -> build/libtinytorch.so
make run_llm_gpu         # needs nvcc + RTX-class GPU
./scripts/ci_local.sh    # 30s CPU sanity, no GPU needed
```

## Quickstart

```
./build/run_llm_gpu data/models/qwen2.5-0.5b-instruct-q4_0.gguf "Hello"
```

## Gates

```
./scripts/ci_local.sh        # samplers, specdec-sim, chat-template, kvcache, cpu_backend
./scripts/verify.sh m61      # greedy-logit parity vs llama.cpp + multiturn chat
./scripts/verify.sh m84      # gemma4 parity
./scripts/verify.sh ple      # PLE-fused golden
./scripts/verify.sh tok      # tokenizer oracle parity (needs oracle build)
./scripts/verify.sh backfill # Q8 KV-cache backfill (needs nvcc)
```

## Numbers

Decode throughput tracks 0.55-0.82x llama.cpp CUDA on RTX 3050 4GB for
qwen2.5-0.5b-q4_0; see `docs/BENCHMARKS.md`. Supported families and
quant tiers: `docs/SUPPORTED_FAMILIES.md`.

## Layout

`src/` engine (loader, arch_registry, tokenizer, samplers, kvcache,
cpu_backend, moe_router, specdec, ngram_lookup, dequant_ref),
`kernels/` CUDA, `tools/` microbenches, `bench/bench_llm.py`
scoreboard, golden logits in `data/golden/`, servers in `examples/`.
