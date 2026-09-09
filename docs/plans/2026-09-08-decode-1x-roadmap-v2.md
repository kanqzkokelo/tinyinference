# TinyInference decode parity plan — repository-grounded version

## Objective

Reach reproducible CUDA decode parity with the pinned llama.cpp reference on the RTX 3050 `sm_86` test machine, without weakening numerical or model-semantic gates.

```text
geomean(TinyInference / llama.cpp) >= 1.00x
each primary model >= 0.90x
no existing parity, tokenizer, KV-cache, server, or CPU regressions
short-context and long-context results reported separately
```

This plan supersedes the earlier broad roadmap. It describes work still needed in the current repository, not work that has already landed.

## Status after latest work

| Milestone | Status | Evidence |
|---|---|---|
| M0 scoreboard | Done | `bench/bench_llm.py`; pinned model/oracle/device metadata; 5-run graph/eager output |
| M1 profiling | Done | `tools/profile_decode.py`; deterministic eager-stage parsing; separate CSV |
| M2 GEMV bakeoff | Partial | Q4 V2/V4/R8 evidence plus Q6 V2; broad production-shape table still incomplete |
| M3 dispatch | Done (Audited) | `tt_gemv_layer_dispatch` + `tt_gemv_typed` provide unified layer GEMV & typed routing with `TT_DISPATCH=1` tracing |
| M4 QKV fusion | Done (Audited) | `k_rmsnorm_qkv_q4_0` verified; +2.2% (Qwen2.5) to +8.8% (SmolLM2) decode gain over `TT_NO_FUSE=1`; numerical parity preserved |
| M5 FFN/LM head | Done (Audited) | Q6 V2 shipped; Q4_0 & Q8_0 V4 LM-heads active (1.48-1.7x speedup over scalar, bit-exact); FFN Gate+Up fused; MMQ v1 rejected |
| M6 attention/KV | Done (Audited) | FP32 split-K attention gated for HD<=128; Qwen2.5-0.5B ctx512 decode boosted from 121.9 to 289.0 tok/s (1.09x-1.33x oracle); SmolLM2-135M ctx512 from 101.9 to 367.2 tok/s; zero precision loss |
| M7 graph hardening | Done (Audited) | Graph capture verified bit-exact vs eager across primary models; dynamic device-side token feeding; hybrid KV eager/graph threshold transition guarded; `tests/proto_graph_capture.cu` validated |
| M8 prefill | Existing | keep separate from decode decisions |
| M9 fleet validation | In progress | v4 fleet (2026-09-09, 60W AC): default-safe graph geomean **0.973x** over 8 points (was 0.867x in v3). 5/8 cells >= 1.0x. Remaining gaps: Qwen3-0.6B ctx512 0.719x, Llama-3.2-1B ctx32 0.771x, ctx512 0.729x. See `data/bench/scoreboard_fleet_60w_v4.md` |

Current default-safe result is still below parity. The Q4-KV `0.947x` result is an
opt-in performance experiment, not the acceptance baseline, because short-context
greedy identity diverged on three primary models.

## v4 fleet result (2026-09-09, commit da425fe, 60W AC)

| Model | Ctx | Ours decode | Oracle decode | Ratio | Note |
|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 32 | 303.4 tok/s | 236.5 tok/s | 1.283x | graph_captured |
| Qwen2.5-0.5B Q4_0 | 512 | 297.4 tok/s | 232.6 tok/s | 1.279x | graph_captured |
| Qwen3-0.6B Q8_0 | 32 | 177.4 tok/s | 173.2 tok/s | 1.024x | graph_captured (embed-dyn fix) |
| Qwen3-0.6B Q8_0 | 512 | 123.6 tok/s | 172.0 tok/s | 0.719x | TTFT 2407ms vs oracle ~57ms; prefill 212 vs 8891 tok/s |
| Llama-3.2-1B Q4_0 | 32 | 142.6 tok/s | 185.0 tok/s | 0.771x | |
| Llama-3.2-1B Q4_0 | 512 | 131.6 tok/s | 180.5 tok/s | 0.729x | |
| SmolLM2-135M Q4_0 | 32 | 502.4 tok/s | 441.0 tok/s | 1.139x | |
| SmolLM2-135M Q4_0 | 512 | 485.3 tok/s | 468.4 tok/s | 1.036x | |

Geomean graph: **0.973x**. Definition-of-done needs geomean >= 1.00x AND every
primary model >= 0.90x, so three cells block parity.

### Remaining-gap priority (adjusted plan)

1. Qwen3-0.6B ctx512 (0.719x, biggest lever). TTFT 2407ms / prefill 212 tok/s
   vs oracle 8891 tok/s points at the Q8 prefill path, not steady decode.
   Profile `TT_DISPATCH=1` + `tools/profile_decode.py` on the 509-token prompt:
   suspect per-row Q8 GEMV or scalar fallback where a Q8 GEMM/cuBLAS path
   belongs. Fix prefill routing first, then re-read decode (currently 123.6
   vs 172.0 tok/s) — decode may inherit the same slow kernel.
2. Llama-3.2-1B both contexts (~0.75x). Coalesced Q6_K v2 helped (+60%) but a
   second head remains: profile per-stage shares at ctx32/ctx512, check LM-head
   shape routing (160128-row Q6/Q4 head) and whether down_proj/Gate-Up still
   miss fusion or run scalar rows. One more targeted kernel/routing fix here
   plus the Qwen3 fix mathematically closes geomean to ~1.0x.
3. Re-run v5 fleet to confirm, then claim parity only if all gates pass. Do
   NOT chase Q4-KV quantization (0.947x opt-in, greedy divergence) or speculative
   decoding until the three default-safe cells clear 0.90x.

### Progress 2026-09-09 — decode FP32-scatter skip (worktree, unverified fleet)

- Change (`kernels/qwen2_cuda.cu`, decode single-token path): skip the FP32
  KV scatter when the quantized path is effective for the step
  (`kv_use_q8_eff`/`kv_use_q4_eff` + slab ptrs). Prefill batched path still
  dual-writes (prefill flash reads FP32). `TT_NO_BACKFILL` / missing-slab
  configs keep FP32 via the same predicates as attn dispatch.
- Graph census (llama-3.2-1B, `TT_GRAPH_DUMP=1`): 216 -> 200 nodes (-16,
  one wasted FP32 scatter per layer removed). No other node change.
- Correctness: Q8KV-graph == Q8KV-eager text (identical); Q8KV == FP32-KV
  text on normal prompt (identical); `TT_NO_BACKFILL` == `TT_Q8_KV=0` text
  (identical, fallback path intact). Garbage-prompt Q8-vs-FP32 divergence is
  quantization (pre-existing, B path untouched by this change).
- Bench (single cell, 5-run): llama-3.2-1B ctx512 prompt520 gen32 graph =
  134.65 tok/s ours vs 177.43 oracle = 0.759x, ours variance ~1.1%.
  Raw outputs: /tmp/bench_llama512.{jsonl,md,csv}. NOT a fleet rerun —
  do not cite as new geomean. v5 fleet baseline on same prompt length was
  0.721x (gen48); gen counts differ so the delta mixes configs.
- Still open: rope q/k 2x launches per layer (32 nodes), k_add residuals
  (32 nodes), gumbel/penalty passes when greedy — next fusions in that order.

### Progress 2026-09-09 — bench hardening + attn duel + 60W stage table

- Bench hardening (ccb1918): get_power_info snapshots Current Power Limit
  + AC state into every jsonl row and meta, warns when limit != 60W.
  Oracle stores prompt_n/predicted_n per run. Per-row start/end stamps.
  Empty-prompt oracle call refused (was the 25-min hang).
- Attention duel (bfb3111, tools/micro_duel_attn.cu, Llama shapes):
  A=30.6us B(u32 staging)=29.5us C(+fastexp)=28.7us. +6.7% total, KILLED
  as lever; per-token FMA/shuffle dominates, S-sweep already flat.
- 60W AC stage table (llama ctx512 eager, /tmp/prof60w.csv): qkv 0.75 /
  attn 0.75 / o 0.37 / ffn 4.08 / head 1.61 / other 0.35 = 7.92ms.
  qkv at roofline, head 87%, o+ffn ~70% eff. Gap drivers: small-M GEMV
  occupancy + ~64 fusable graph nodes (rope q/k, k_add) + launch gaps.
- Next: 1) split-K-within-block GEMV prototype in /tmp. 2) rope/k_add
  fusion. 3) v6 fleet on AC + commit per change.

## Current state

Already implemented:

- CUDA graph capture and replay for steady-state decode.
- Eager fallback through `TT_NO_GRAPH=1` and capture-failure fallback.
- Per-stage profiling through `TT_PROFILE=1`.
- Q4_0 V2/V4 GEMV paths, plus experimental R8 and WMMA paths.
- Q8_0 GEMV and specialized LM-head paths.
- RMSNorm→QKV Q4_0 fusion in the engine.
- Q4/Q8 KV cache, backfill, and split-K FlashAttention paths.
- WMMA/cuBLAS prefill paths.
- Extensive correctness and parity tests.

Key files:

```text
kernels/qwen2_cuda.cu
kernels/gemv_q4_cuda.cu
kernels/gemv_typed.cu
kernels/gemm_cuda.cu
kernels/cublas_ref.cu
tools/bench_cuda_vs_cuda.sh
tools/bench_scoreboard.sh
bench/bench_llm.py
tools/profile_decode.py
tools/profile_step.cu
```

At plan creation, local worktree changes included:

```text
bench/scoreboard_decode.csv
kernels/qwen2_cuda.cu
tools/profile_decode.py
```

Always inspect current `git status` before declaring a new baseline; later commits may have resolved or replaced these changes.

## Execution notes (2026-09-08)

- Q6_K two-row GEMV was corrected against the scalar layout and enabled for
  even-M dispatch. Golden coverage is `30/30`; Llama 3.2 tied-head shape
  `M=128256,K=2048` runs under graph replay. `TT_DISABLE_Q6_V2=1` restores the
  scalar control path.
- Controlled Llama eager decode improved from `65.5` to `87.8` tok/s on a
  32-token sample. Authoritative 5-run scoreboard after the change: graph
  `89.3` tok/s at ctx32 (`0.490x` oracle), `66.3` tok/s at ctx512 (`0.380x`).
- Remaining gap is structural: ctx~512 eager-stage profile attributes about
  `5.9 ms` to Q6 LM-head, `4.0 ms` to attention, and `4.0 ms` to FFN per token.
- A direct-layout Q6 four-row warp candidate was measured and rejected:
  `85.7` vs `87.1` tok/s on the same 32-token Llama run (`-1.6%`).
- FFN analysis: Gate+Up SwiGLU is already single-launch fused (`k_fused_swiglu_q4_0`).
  Isolated microbenchmark confirms 18-43% kernel-level speedup vs 3-sequential
  launches across Qwen, Llama, and SmolLM shapes. Down-projection remains a standard GEMV.
  Note on benchmarking: laptop battery status throttles GPU power cap from 60W to 30W,
  halving effective memory clock/throughput; authoritative benchmark comparisons must
  always verify AC power connection (`Current Power Limit == 60.00 W`).
- KV result: Q4 cache + graph capture at `TT_QKV_THRESH=0` reached `0.947x`
  fleet geomean over 8 graph points (Qwen2.5, Qwen3, Llama, SmolLM at
  ctx32/512), versus the prior ~`0.47x` baseline. Q4 remains opt-in because
  greedy output diverged on Qwen2.5/Llama/SmolLM in short-context identity
  checks. Hybrid capture now keeps short prompts FP32/eager and captures the
  quantized path directly for prompts already beyond the threshold.

## Non-goals for the parity critical path

Do not put tokenizer rewrites, sampler optimization, CPU GEMM work, speculative decoding, IPC/frontend cleanup, or broad uncommon-quant support ahead of CUDA decode parity. Keep those on separate tracks.

## M0 — one authoritative scoreboard

### Files

- `tools/bench_cuda_vs_cuda.sh`
- `tools/bench_scoreboard.sh`
- `bench/bench_llm.py`
- `tools/run_live_performance_suite.py`
- `bench/scoreboard_decode.csv`

`bench/bench_llm.py` is the authoritative runner. Other scripts remain wrappers
or diagnostic tools; they must not overwrite its scoreboard.

Every result must record:

```text
TinyInference commit
llama.cpp commit and binary path
model path and file hash
quantization
GPU, SM, CUDA/toolchain identity
context and prompt-token count
generation-token count
graph/eager mode
prefill_us, decode_us, first_token_us, total_us
all samples, median, min, max
```

Use identical model files, prompt tokens, context, generation length, greedy settings, and offload configuration. Missing models or failed oracle runs must be failures, not silent skips.

Primary matrix:

```text
Qwen2.5-0.5B Q4_0
Qwen3-0.6B Q8_0
Llama 3.2-1B Q4_0
SmolLM2-135M — explicitly choose and document F16 or Q4_0
```

Contexts:

```text
short: ~32 tokens
medium: ~512 tokens
long: 2048 and 8192 where the model fits
```

### Acceptance

- [x] One command generates JSONL plus Markdown.
- [x] The runner writes the authoritative schema without profiler overwrite.
- [x] The llama.cpp build is identical across all models.
- [x] At least five measured repetitions follow warmup in normal mode.
- [x] Graph and eager results are reported separately.
- [ ] Regenerate checked-in fleet CSV only after the default-safe configuration is finalized.

## M1 — make profiling truthful

### Files

- `kernels/qwen2_cuda.cu`
- `tools/profile_step.cu`
- `tools/profile_decode.py`

`TT_PROFILE=1` currently forces eager mode. Therefore its stage totals must not be presented as graph-replay stage totals.

Maintain two explicit measurements:

```text
eager-stage: per-stage CUDA-event attribution
graph-total: graph-replay end-to-end timing
```

Keep stable stage names:

```text
qkv-gemv, o-proj, ffn-gateup, ffn-down, attn,
logits-gemv/lm-head, rmsnorm, rope, kv-scatter, argmax, other
```

The profile output must include mode, context, token count, and total stage sum. The parser must select the decode block deterministically.

### Acceptance

- [x] Eager stage sums agree approximately with eager CUDA-event timing.
- [x] Graph replay timing is independently reported by the scoreboard.
- [x] The top two costs are visible at each context.

## M2 — exact-shape GEMV bakeoff

### Files

- `kernels/gemv_q4_cuda.cu`
- `kernels/gemv_typed.cu`
- `tools/micro_v4.cu`
- `tools/micro_gemv_rows.cu`
- `tools/mmq_v1.cu`
- `tools/test_gemv_typed.cu`

Do not begin a generic MMQ rewrite based only on the name “MMQ.” Decode uses an effectively single-vector input, and the repository already records failed shared-memory/double-buffered GEMV experiments caused by synchronization and register pressure.

Build a table for real production shapes:

```text
model, quant, operation, M, K,
V2_us, V4_us, R8_us, MMQ_us,
effective weight GB/s, registers, occupancy, correctness
```

Cover Q/K/V, O, FFN gate/up/down, and LM head. Use per-iteration synchronization, median timing, and explicit L2 labeling.

Decision rules:

- Keep the existing path unless a candidate wins by at least 5% repeatedly.
- Ship R8 only if it wins by at least 10% on a real production shape.
- Ship MMQ only if it wins at `M=1` on an important production shape without regressing the primary fleet.
- Preserve rejected experiments as evidence, not as production candidates.

Measured MMQ v1 outcome (`tools/mmq_v1.cu` on RTX 3050 sm_86):
- Q/K/V (896x896): V2 8.4 us vs MMQ 47.3 us (0.18x, 5.6x slower)
- FFN up/gate (4864x896): V2 35.1 us vs MMQ 96.9 us (0.36x, 2.8x slower)
- FFN down (896x4864): V2 34.8 us vs MMQ 403.4 us (0.09x, 11.6x slower)
- LM head (151936x896): V2 1198.6 us vs MMQ 4299.4 us (0.28x, 3.6x slower)
Conclusion: WMMA m16n16k16 fp16->fp32 for single-token vector decode is a definitive NO-GO
across all shapes (15/16ths of tensor-core compute is wasted on redundant N-dimension broadcast,
plus dequantization to shared memory adds massive latency and register overhead).

Completed evidence: Q6_K V2 is correct on `30/30` typed-GEMV cases and wins
the Llama tied-head runtime path; a Q6 four-row candidate lost by `1.6%` and
was removed. Remaining work is the full production-shape table and any Q8/Q4
gaps it exposes.

## M3 — centralize dispatch without changing behavior

### Files

- `kernels/qwen2_cuda.cu`
- `kernels/gemv_q4_cuda.cu`
- `kernels/gemv_typed.cu`

Create one internal selector:

```c
choice = choose_decode_gemv(sm, dtype, M, K, operation, context);
```

First move existing thresholds into the selector unchanged. Change thresholds only in a separate measured commit.

Expose diagnostic selection records containing operation, shape, dtype, SM, selected path, and fallback reason. Do not put GPU tuning constants in the model architecture registry.

## M4 — validate current QKV fusion

### Files

- `kernels/qwen2_cuda.cu`
- `kernels/gemv_q4_cuda.cu`
- `tests/test_engine_golden.py`
- `tests/test_gpu_parity.py`

RMSNorm→QKV fusion is already wired into the Qwen2 forward path. Treat it as existing functionality, not future work.

Run an A/B matrix:

```text
fused
TT_NO_FUSE=1
```

Measure all primary models and relevant contexts. Validate greedy identity, logit tolerance, graph capture, eager/graph throughput, GQA dimensions, and non-Q4 fallbacks.

Do not fuse attention or RoPE in this milestone.

## M5 — FFN and LM-head optimization

### Files

- `kernels/gemv_q4_cuda.cu`
- `kernels/gemv_typed.cu`
- `kernels/qwen2_cuda.cu`
- `tools/micro_ffn_fused.cu`
- `tools/micro_logits_q8_v4.cu`
- `tests/test_logits_q8_v4.c`

The current live profile still shows FFN and LM-head work as material costs after QKV fusion.

For FFN:

1. Measure gate/up, activation, and down separately.
2. Verify whether the existing fused FFN launcher is used in production.
3. Compare fused/unfused paths at actual model shapes.
4. Keep the change only when end-to-end decode improves.

For LM head:

1. Measure Q4 and Q8 independently.
2. Preserve V4 where it wins.
3. Test vocabulary and K-alignment boundaries.
4. Require end-to-end improvement, not only a microbenchmark gain.

## M6 — context-dependent attention and KV decisions

### Files

- `kernels/qwen2_cuda.cu`
- `src/kvcache.c`
- `include/kvcache.h`
- `tools/bench_fa2.cu`
- `tools/micro_paged_fa2.cu`
- `tools/micro_q8_kvcache.cu`
- `tests/test_flash_multi.cu`
- `tests/test_q8_kvcache.c`
- `tests/test_q4_kvcache.c`

Compare FP32 KV, Q8 KV, Q4 KV, serial attention, split-K, and paged paths at each context. Keep short-context and long-context decisions separate.

All attention benches must use per-iteration synchronization, randomized/paged access where relevant, explicit L2 labeling, and median timing.

Do not make Q4 KV the default merely because it saves bytes. Keep Q8 KV as the quantized reference until accuracy and end-to-end results justify a change.

Thresholds must be derived from measured context/shape data, not universal assumptions.

## M7 — graph hardening after kernel choices stabilize

### Files

- `kernels/qwen2_cuda.cu`
- `tests/proto_graph_capture.cu`
- new or expanded graph parity test

Graph capture already exists. This milestone validates coverage and reduces fallback, rather than creating a graph system from scratch.

Verify:

```text
graph vs eager logits
graph capture success/fallback
graph replay time
host-to-device token update cost
steady-state decode rate
```

Use `TT_NO_GRAPH=1` as the correctness control. Use context buckets or explicit eager fallback when dynamic dispatch cannot be safely captured.

## M8 — keep prefill separate

### Files

- `kernels/gemm_cuda.cu`
- `kernels/cublas_ref.cu`
- `tools/micro_prefill_gemm.cu`
- `tools/micro_wmma_prefill_gemm.cu`
- `tests/test_prefill_gemm.c`
- `tests/test_wmma_prefill_gemm.c`

The repository already has strong prefill work. Measure the crossover among Q4 GEMM, WMMA, and cuBLAS/FP16 at actual prompt sizes, but do not trade decode performance for prefill gains without reporting both.

## M9 — fleet validation and stop conditions

After every performance change, run the authoritative scoreboard and all relevant correctness gates.

Report:

```text
short/medium/long decode
prefill
first-token latency
graph/eager comparison
per-model ratios
geomean ratio
correctness status
```

Use this ladder:

```text
0.80x geomean: usable intermediate baseline
0.90x geomean: decision point
0.95x geomean: final-gap work only
1.00x geomean: parity candidate
```

If exact-shape GEMV and attention work still leave a primary model below `0.70x`, stop broad refactoring and identify the structural mismatch before adding more kernels.

## Definition of done

Parity may be claimed only when:

```text
the benchmark command is reproducible
the llama.cpp commit/build is pinned
all primary models are measured
no model is silently skipped
geomean >= 1.00x
each primary model >= 0.90x
short and long contexts are separate
graph/eager behavior is documented
all existing parity and model-semantic gates pass
```

Speculative decoding may be benchmarked afterward, but it must not disguise a sub-parity single-token decode baseline.

## v5 fleet addendum (2026-09-09, dirty tree on da425fe)

### What landed since v4

1. Q8 prefill GEMM rewrite (`k_gemm_q8_0_prefill`): 6.7-7.8x gap to Q4 closed to 1.76x (byte-ratio floor). Bit-exact 5/5 goldens. Qwen3-512 prefill 212 -> 814 tok/s.
2. Fused rmsnorm+QKV for Q8 (`k_rmsnorm_qkv_q8_0`, `fuse_q8` gate): bit-exact, +1.8% graph decode. Small because graph already hides launch overhead.
3. **Q8 KV-cache default-ON** (`TT_Q8_KV` opt-out, `TT_QKV_THRESH` default 256 -> 0): top-1 stable 5/5 goldens + 32tok text A/B identical, graph-safe static path. Qwen3-512 decode 123.6 -> 159.6 tok/s (+29%).
4. Runner fix: `errors="replace"` on ours/oracle capture (SmolLM2 non-UTF8 bytes aborted the first v5 attempt mid-fleet).

### v5 result (`data/bench/scoreboard_fleet_60w_v5.*`, graph, 5-run medians)

| Model | Ctx | Ours | Oracle | Ratio | v4 |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 | 32 | 321.8 | 282.3 | 1.140x | 1.283x |
| Qwen2.5-0.5B Q4_0 | 512 | 282.8 | 282.6 | 1.001x | 1.279x |
| Qwen3-0.6B Q8_0 | 32 | 189.1 | 189.1 | 1.000x | 1.024x |
| Qwen3-0.6B Q8_0 | 512 | 159.6 | 173.9 | 0.918x | 0.719x |
| Llama-3.2-1B Q4_0 | 32 | 142.3 | 184.7 | 0.770x | 0.771x |
| Llama-3.2-1B Q4_0 | 512 | 129.9 | 180.0 | 0.721x | 0.729x |
| SmolLM2-135M Q4_0 | 32 | 541.6 | 438.5 | 1.235x | 1.139x |
| SmolLM2-135M Q4_0 | 512 | 465.2 | 515.2 | 0.903x | 1.036x |

Geomean **0.947x** (v4 0.973x). Dropped despite the Qwen3 fix because oracle numbers drifted up run-to-run (qwen2.5-512 oracle 233 -> 283, smol-512 468 -> 515: thermal/clock drift, no pinning) and Q8KV is neutral-to-slightly-negative on small Q4 models at ctx512 (-4-5%, inside noise).

### Verification verdict: TRUST-WITH-CAVEATS

- All 8 reported numbers are exact medians of their 5 samples; all `graph_captured`, 5/5 samples everywhere.
- Ours variance <=3.1% (mostly <1%): engine stable. Oracle noisy on smol (17-20% range), qwen2.5-512 (11%), qwen2.5-32 (9%, ramp-up pattern = 1 warmup insufficient for oracle cold start).
- Llama gaps (0.770x/0.721x, both sides <=2.3% variance) are REAL, not noise. Qwen3-512 0.918x credible (oracle ramp-up means true ratio ~0.90-0.92).
- Smol cells sit inside oracle noise bands: do not over-read 1.235x or 0.903x.
- Still missing: per-row timestamps (single timestamp for all rows), oracle `prompt_n`/`predicted_n` not stored (token-count equality unverifiable post-hoc), no power/clock logging (60W filename-only), gen lengths vary by EOS (31-64) across versions.

### Revised priority (only Llama blocks now)

1. **Llama-3.2-1B both ctx (~0.75x, tight variance, real gap).** Traffic math: ~774MB/tok (logits Q6_K 220MB = 28%, FFN Q4_0 554MB); engine realizes ~103 GB/s vs oracle ~140. No single kernel is 2x off under graph (V4 microbench hits 144 GB/s isolated; eager profile overstates per-op cost via `TT_PROFILE` syncs). Needs +35% bandwidth efficiency everywhere: down-proj occupancy shapes, Q6_K logits tune, launch-bubble trim. This is the final grind; expect several small wins, not one fix.
2. **Methodology hardening before claiming anything**: 3 warmups (oracle cold-start ramp is visible in samples), per-row timestamps, store oracle token counts, cooldown between models, log `nvidia-smi` clocks/power. Re-run v6 on a cool machine; confirm smol-512 >= 0.90x (likely noise, not a real regress).
3. **Do NOT**: requant LM heads, speculative decoding, or Q4-KV (greedy divergence) until Llama clears 0.90x default-safe.
- Still open: rope q/k 2x launches per layer (32 nodes), k_add residuals
  (32 nodes), gumbel/penalty passes when greedy — next fusions in that order.
- Still open: rope q/k 2x launches per layer (32 nodes), k_add residuals
  (32 nodes), gumbel/penalty passes when greedy — next fusions in that order.
- Gate Q2-lite (`tests/gate_m6_logit_parity.py`, 2026-09-09 worktree): 0/7 with
  Q8KV default-ON (top-1 all match, median|dlogit| 0.25-1.0 vs 0.15 gate) vs
  7/7 PASS with `TT_Q8_KV=0` (median <= 0.10). Verdict: drift comes from Q8
  KV quantization (pre-existing dirty change), NOT the scatter skip — FP32
  path is bit-stable with the skip in place. Q8KV default-ON needs a gate
  decision (relax median vs keep opt-in) before it can become the baseline.

### 2026-09-09 pm: add-norm fusion NO-GO, Llama gap confirmed real (no commit)

* Tried fusing attn-residual `k_add` + ffn `k_rmsnorm` (`k_add_rmsnorm`,
  bit-exact, TT_NO_FUSE/trace fallback). Nodes 182 -> 166 (-1/layer, Llama).
  Interleaved A/B both orders, both models: fused == old within noise
  (Llama ~144, Qwen3 ~186 tok/s). Reason: dim-sized tensors are L2-resident
  and graph replay already removes launch cost, so the fusion saves ~zero
  DRAM traffic. REVERTED clean (tree matches HEAD for kernels/qwen2_cuda.cu).
  Lesson: only fusions that cut WEIGHT/KV traffic can move decode now.
* Llama-512 gap is NOT power artifact: interleaved ours-vs-oracle (542-tok
  prompt, 3x alternating) gives 134.5 vs 183 tok/s = 0.735x, matching fleet
  0.72x. Real kernel-side deficit ~2.0 ms/tok.
* `TT_NO_FUSE` (qkv rmsnorm+GEMV fused vs separate) is model-dependent:
  Qwen3-Q8 fused wins +1.5% (186 vs 183.5), Llama-Q4 unfused wins ~1.5% at
  short ctx (4/4 triples both orders), ~neutral at ctx512. Do NOT change
  committed qkv gating on this alone; needs its own ctx512 experiment.
* Same-process micro (`/tmp/micro_duel`): ffn fused 138.2us vs unfused
  137.1us (identical); Q6_K head 1913us @112.7 GB/s (prior v2 live in tree).
  Fused epilogue kernels all measure ~= their unfused equivalents.
* Dispatch audit (`TT_DISPATCH=1`, Llama): all Q4 GEMVs take V4, head takes
  q6-k-v2. No slow-path leaks; routing is clean.
* Gate Q2-lite re-verified on live tree: 0/7 default (known Q8KV drift),
  7/7 with `TT_Q8_KV=0`. Prior-session dirt (Q6_K v2, Q8 prefill GEMM,
  dispatch trace) is numerics-clean; Q8KV default still needs YOUR call.
* Red-cell accounting (Llama ctx542, per tok): Q4 stack ~4.1ms + head 1.9ms
  + KV 0.25ms + tiny kernels ~0.7ms ~= 6.9-7.1 vs actual 7.43. Biggest
  quantified levers left: Q6_K tune (-0.4ms), attention kernel (-0.4ms est),
  ~1ms unexplained GEMV in-situ deficit. Next: Q6_K v3 micro sweep, then
  attention t-unroll, both same-process A/B + gate before commit.
