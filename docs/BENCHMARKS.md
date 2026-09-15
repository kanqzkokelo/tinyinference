# Benchmarks — fleet vs llama.cpp CUDA, RTX 3050 Laptop (sm_86, 4GB)

Decode scoreboard from `bench/scoreboard_decode.csv` (fleet v9 run on the
OOB-fix binary, 5 measured runs post-warmup, medians). Geomean ratio:
graph 0.980 over 13 cells, eager 0.931 over 13 cells, overall 0.955.
Short ctx beats oracle up to 1.3x. Known gaps: llama-2048 0.85x and
smol-2048 0.72x. Raw per-cell jsonl lives gitignored in `data/bench/`.

| Model | Quant | Mode | Ctx | Ours tok/s | Oracle tok/s | Ratio |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b-instruct | q4_0 | graph | 32 | 300.0 | 253.7 | 1.18 |
| qwen2.5-0.5b-instruct | q4_0 | eager | 32 | 287.7 | 218.9 | 1.31 |
| qwen2.5-0.5b-instruct | q4_0 | graph | 512 | 277.3 | 244.0 | 1.14 |
| qwen2.5-0.5b-instruct | q4_0 | eager | 512 | 263.0 | 249.6 | 1.05 |
| qwen2.5-0.5b-instruct | q4_0 | graph | 2048 | 191.9 | 227.1 | 0.85 |
| qwen2.5-0.5b-instruct | q4_0 | eager | 2048 | 180.9 | 176.9 | 1.02 |
| smollm2-135m-instruct | q4_0 | graph | 32 | 445.2 | 371.0 | 1.20 |
| smollm2-135m-instruct | q4_0 | eager | 32 | 471.6 | 441.5 | 1.07 |
| qwen3-0.6b | q8_0 | graph | 32 | 169.5 | 155.4 | 1.09 |
| qwen3-0.6b | q8_0 | eager | 32 | 163.8 | 180.9 | 0.91 |
| llama-3.2-1b | q4_0 | graph | 32 | 161.2 | 168.6 | 0.96 |
| llama-3.2-1b | q4_0 | eager | 32 | 159.5 | 174.1 | 0.92 |
| gemma-4-E2B-it | q4_0 | graph | 32 | 71.4 | 80.9 | 0.88 |
| gemma-4-E2B-it | q4_0 | eager | 32 | 71.3 | 86.3 | 0.83 |
| llama-3.2-1b | q4_0 | graph | 2048 | 139.5 | 164.4 | 0.85 |
| llama-3.2-1b | q4_0 | eager | 2048 | 134.6 | 160.7 | 0.84 |
| qwen3-0.6b | q8_0 | graph | 2048 | 128.8 | 138.6 | 0.93 |
| qwen3-0.6b | q8_0 | eager | 2048 | 122.5 | 134.0 | 0.91 |
| smollm2-135m-instruct | q4_0 | graph | 2048 | 301.5 | 421.7 | 0.72 |
| smollm2-135m-instruct | q4_0 | eager | 2048 | 285.2 | 426.6 | 0.67 |
| smollm2-135m-instruct | q4_0 | graph | 512 | 493.6 | 487.3 | 1.01 |
| smollm2-135m-instruct | q4_0 | eager | 512 | 460.5 | 492.9 | 0.93 |
| qwen3-0.6b | q8_0 | graph | 512 | 169.8 | 156.8 | 1.08 |
| qwen3-0.6b | q8_0 | eager | 512 | 130.2 | 169.9 | 0.77 |
| llama-3.2-1b | q4_0 | graph | 512 | 155.1 | 156.6 | 0.99 |
| llama-3.2-1b | q4_0 | eager | 512 | 151.3 | 145.0 | 1.04 |

## Older prefill note — Qwen2.5-0.5B-Instruct-Q4_0

HOT medians, 2 warmups + 3 measured. Prefill prompt 752 tokens; decode gen 128.
`verify.sh m61` PASS (7/7 parity + chat), backfill PASS, 5/5 greedy-match default vs best.

| Config | Prefill tok/s | Prefill ms | Decode tok/s |
|---|---|---|---|
| Default | 1416.5 | 530.9 | 224.2 |
| `TT_CUBLAS_PRE=1 TT_FA2_PRE=1` | 6408.3 | 117.3 | — |
| `TT_CUBLAS_FP16=1 TT_FA2_PRE=1` (best) | 7736.6 | 97.2 | 212.8 |

Best = **5.5x** default prefill. Decode flags add small overhead (~5%); prefill flags target prefill only.
DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M: coherent, "The answer is 144."
