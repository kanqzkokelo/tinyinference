# tinyinference

[![ci](https://github.com/kanqzkokelo/tinyinference/actions/workflows/ci.yml/badge.svg)](https://github.com/kanqzkokelo/tinyinference/actions/workflows/ci.yml)

A from-scratch LLM inference engine in C + CUDA. It loads GGUF weights and
runs quantized transformer decode on NVIDIA GPUs with no Python in the hot
path. Built for fast single-stream generation on small-VRAM machines: the
reference numbers below are from a 4 GB laptop RTX 3050.

- **Fast decode where it is hardest.** Qwen2.5-0.5B Q4_0 at 289 tok/s @ 32
  context, 261 tok/s @ 2048, 103 tok/s @ 30k.
- **Typed kernels, no dequant-to-FP32 round trip.** Q4_0/Q4_K/Q5_K/Q6_K/Q8_0
  GEMV, flash attention, FP32/Q8/Q4 KV caches.
- **One binary.** `run_llm_gpu`, `chat_llm_gpu`, and an OpenAI-compatible
  server, no runtime dependencies beyond the CUDA driver.

## Quickstart

Grab a GGUF (Qwen2.5-0.5B-Instruct Q4_0 is 397 MB) and run:

```
# Linux
./build/run_llm_gpu -m qwen2.5-0.5b-instruct-q4_0.gguf "Explain KV caching." 64

# Windows
run_llm_gpu.exe -m qwen2.5-0.5b-instruct-q4_0.gguf "Explain KV caching." 64
```

`-m` picks the model, the quoted string is the prompt, the last argument is
the token budget. `TT_MODEL` works too. Interactive chat:

```
./build/chat_llm_gpu          # TT_MODEL=... if the model is not in data/models
```

## Server

```
TT_MODEL=qwen2.5-0.5b-instruct-q4_0.gguf TT_LISTEN=127.0.0.1:8080 ./build/server_minimal
```

```
curl -s localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"stream":true}'
```

Routes: `POST /v1/chat/completions` (streaming and non-streaming),
`GET /v1/models`, `GET /healthz`. `server_multimodel` routes by the `model`
field to several GGUF files in one process.

## Python

The wheel bundles the engine `.dll`/`.so` and the CUDA runtime, so `pip install`
needs no compiler:

```
pip install https://github.com/kanqzkokelo/tinyinference/releases/download/v0.9.1/tinyinference-0.9.1-py3-none-win_amd64.whl
```

Use a normal CPython (python.org or a venv), not the MSYS2 build. CUDA 12
driver or newer is required; nothing else.

```
tinyinference chat model.gguf "Explain KV caching in one sentence"
tinyinference serve model.gguf --port 8000
```

```python
from tinyinference import Model

with Model("model.gguf", n_ctx=4096) as m:
    print(m.chat([{"role": "user", "content": "hi"}]))
    for piece in m.stream_text(m.encode("Once upon a time"), 64):
        print(piece, end="", flush=True)
```

`tinyinference serve` speaks the OpenAI chat and completions API. The official
`openai` client works unmodified, streaming included:

```python
from openai import OpenAI

c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
print(c.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "hi"}],
).choices[0].message.content)
```

Requests are serialized: one model, one KV cache.

To build the wheel from source instead:

```
make -f Makefile.win pydll        # or `make pylib` on Linux
bash python/build_wheel.sh        # -> python/dist/tinyinference-0.9.1-*.whl
```

## Build

Linux, CPU library plus the CUDA engine:

```
make lib
make run_llm_gpu
```

Windows, native, no WSL:

```
make -f Makefile.win all      # lib, tools, CPU test binaries -> build/
make -f Makefile.win cuda     # CUDA engine and GPU tools
make -f Makefile.win test     # CPU suite
./scripts/ci_windows.sh       # CPU suite + GPU gates when a device is present
```

The portable C side builds with MSYS2 mingw-w64; the CUDA side needs MSVC
`cl` and `nvcc` on `PATH` (or `NVCC` / `CL` set). See `Makefile.win`.

## What it runs

Architectures: qwen2, qwen3, llama, mistral, gemma, gemma4, tinyllama,
granite, smollm2, internlm2, xverse, exaone, ernie4_5.

Weights: F32, F16, Q4_0, Q4_1, Q4_K, Q5_0, Q5_1, Q5_K, Q6_K, Q8_0.

Engine features: BPE tokenizer, chat templates, top-k/top-p/min-p/temperature
sampling with repetition penalty, paged KV cache with Q8 backfill, MoE
routing, N-gram speculative decoding, CUDA graph replay, batched prefill,
threaded CPU GEMV fallback.

## Numbers

Decode throughput vs llama.cpp CUDA on RTX 3050 Laptop 4 GB, llama.cpp given
the same KV quantization (`-ctk q4_0 -ctv q4_0`). Ratios above 1.0 favor
tinyinference.

| model | 32 | 512 | 2048 | 4096 | 30k |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 1.11x | 1.13x | 1.21x | 1.08x | 1.12x |
| SmolLM2-135M | — | 1.33x | 1.14x | 0.96x | — |
| Llama-3.2-1B | — | 1.13x | 1.08x | — | — |

Absolute Qwen2.5-0.5B: 289.6 / 321.3 / 261.6 / 226.2 / 103.3 tok/s. Prefill
on small batches is 1.54x-2.38x faster. We lose one cell (SmolLM2 @ 4096);
the full tables, hardware, and methodology are in `docs/BENCHMARKS.md`.

## Gates

Correctness is checked against llama.cpp logits and tokenizer output, not
against itself.

```
./scripts/ci_local.sh        # samplers, specdec, chat templates, KV cache, CPU backend
./scripts/ci_windows.sh      # the same plus CUDA gates when a GPU is present
./scripts/verify.sh m61      # greedy logit parity + multiturn chat
./scripts/verify.sh m84      # gemma4 parity
./scripts/verify.sh ple      # PLE-fused golden (pip install gguf)
./scripts/verify.sh tok      # tokenizer oracle parity
./scripts/verify.sh backfill # Q8 KV-cache backfill
```

## Known limits

- NVIDIA only. There is no Vulkan, Metal, or ROCm backend, and generation
  requires a CUDA GPU; the CPU backend covers prefill/GEMV paths used by the
  library and tests, not standalone generation.
- No grammar/JSON-constrained decoding, tool calling, or continuous
  batching. The server handles one request at a time per process.
- No quantization or GGUF conversion. Bring a file produced by llama.cpp.
- Prefix caching (`TT_PREFIX_CACHE`) is opt-in and covers prefill tokens;
  generated tokens are re-fed on the next turn.
- Gemma 4 needs `data/models/gemma-4-E2B-it-Q4_0.gguf` for its gates; the
  file is not redistributed here.

## Layout

`src/` engine (GGUF loader, arch registry, tokenizer, samplers, KV cache,
CPU backend, MoE router, spec decode, n-gram lookup), `kernels/` CUDA,
`examples/` CLIs and servers, `tools/` microbenches, `bench/bench_llm.py`,
`data/golden/` reference logits, `tests/` and `scripts/` gates.

## License

MIT. See [LICENSE](LICENSE).
