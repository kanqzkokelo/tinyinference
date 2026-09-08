# Decode to 1× llama.cpp — Performance Roadmap Implementation Plan

> **REQUIRED SUB-SKILL:** Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Lift single-stream GPU decode from 0.55–0.82× llama.cpp to ~1.0× on RTX 3050, profiler-led, without regressing the 7.7k tok/s cuBLAS/FP16 prefill path.

**Architecture:** Profile first (existing TT_PROFILE + CUDA events), then attack the measured top-2 ops per milestone: GEMV row-reuse sweep → fusion → LM-head specialization → attention/KV at context → graph coverage → prefill-dispatch crossover → autotuned dispatcher. Prefill and decode stay asymmetric (cuBLAS/FP16 vs custom quantized kernels). Every milestone keeps `bench/bench_llm.py` parity green.

**Tech Stack:** CUDA (nvcc, sm_86/sm_89 via `CUDA_ARCHS`), existing kernels in `kernels/qwen2_cuda.cu`, `kernels/gemv_q4_cuda.cu`, `kernels/gemv_typed.cu`, profiler `qwen2_debug_profile_*`, bench `bench/bench_llm.py`, gates `tests/test_gpu_parity.py`.

**Repo root:** `/home/mitesh/Storage/repos/tinyinference`, branch `m6-correctness`. All commands from root. GPU on this machine: RTX 3050 Laptop (sm_86). Models in checkout: `data/models/qwen2.5-0.5b-instruct-q4_0.gguf`, `data/models/gemma-4-E2B-it-Q4_0.gguf`.

**Ground truths (verified in-tree, do not re-debate):**
- Decode graph capture already exists (M6.3: `graph_exec` ~line 3180, `TT_NO_GRAPH`/`TT_PROFILE` force eager ~line 3765). Roadmap extends coverage, not greenfield graphs.
- Per-stage profiler exists (`qwen2_debug_profile_reset/report` ~line 4018; stage names ~line 3983 include rmsnorm/kv-scatter/logits-gemv/argmax). M1 extends it, not rebuilds it.
- GEMV family exists: V2 2-rows/warp, V4 4-rows/warp (`kernels/gemv_q4_cuda.cu:299,641`), Q8 V4 LM head (`:785`), BATCH4 (`:954`). M2 sweeps R, not new math.
- cuBLAS/FP16 prefill exists (`cublas_build_shadows`, `cublas_fp16_build` ~line 3877). Measured: prefill 1416→7737 tok/s, decode 224.2→212.8 tok/s (decode must NOT move to cuBLAS).
- `prefill_batched_gemm` (`:5192`) + `_dx` (`:5571`) with eager fallback at verify path (`:6064`); fallback is silent. M6 makes dispatch explicit.
- Flash attention exists (`k_flash_gqa`, `_splitk`, `_q8_0`, prefill variants). M5 benchmarks it vs ctx, not a rewrite.
- No `oracle/` checkout here → `tests/test_engine_golden.py verify` cannot run; use `bench/bench_llm.py` parity gate + `tests/test_gpu_parity.py` instead.

**Docs to check before touching code:**
- `bench/bench_llm.py` head (~lines 1-70: median-of-N + parity gate protocol)
- `kernels/qwen2_cuda.cu` TT_PROFILE block (~3969-4030) and graph section (~3179-3200, ~3760-3800)
- `docs/plans/2026-08-27-cuda-graph-design.md` (existing graph design constraints)
- `tools/bench_scoreboard.sh` (scoreboard convention)

---

### Task 1: Per-op decode profiler → CSV scoreboard

**Files:**
- Modify: `kernels/qwen2_cuda.cu:3983-4030` (stage table + report)
- Create: `tools/profile_decode.py`
- Test: manual run on qwen2.5-0.5b, ctx 512

**Step 1: Extend stage names to cover decode ops**

Read lines 3980-4030. Add distinct stages so one decode token reports: `qkv-gemv`, `attn`, `o-proj`, `ffn-gateup`, `ffn-down`, `lm-head`, `rmsnorm`, `rope`, `kv-scatter`, `other`. Keep existing names stable (do not rename `logits-gemv` if the gate greps it — check with `grep -rn logits-gemv tests/ tools/ bench/` first; if referenced, keep it as the LM-head stage name).

**Step 2: Run profiled decode**

Run: `TT_PROFILE=1 ./build/run_llm_gpu "Explain quantum computing in one sentence." 64 2>&1 | tail -30`
Expected: per-stage ms table for decode tokens. If `run_llm_gpu` lacks profile hooks, use the same env with `tools/bench_prefill` or add `qwen2_debug_profile_report()` call at the end of the example's decode loop (one call, no other changes).

**Step 3: Scoreboard script**

Create `tools/profile_decode.py`: runs the profiled binary 3× at ctx ∈ {128, 512, 2048}, parses stage ms, writes `bench/scoreboard_decode.csv` with columns `model,quant,ctx,pp_tps,tg_tps,qkv_ms,attn_ms,o_ms,ffn_ms,lmhead_ms,other_ms`. Baseline row first, commit CSV.

**Step 4: Commit**

```bash
git add kernels/qwen2_cuda.cu tools/profile_decode.py bench/scoreboard_decode.csv
git commit -m "perf: per-op decode profiler plus CSV scoreboard"
```

Gate for all later tasks: no task may regress `tg_tps` on qwen2.5-0.5b ctx512 vs this baseline by >2% without explicit justification in the commit message.

---

### Task 2: GEMV rows-per-warp sweep (R = 1/2/4/8)

**Files:**
- Create: `tools/micro_gemv_rows.cu`
- Modify: none (sweep only; kernel changes are Task 3 candidates)
- Test: `tests/test_gpu_parity.py` (must stay green — you change nothing, proving sweep is measurement-only)

**Step 1: Write the microbenchmark**

```cu
// tools/micro_gemv_rows.cu — time existing entry points per shape class.
// Shape classes (M rows, K width; Q4_0 weights):
//   Q-proj: M=896/1152/1536 (per model), K=dim
//   FFN:    M=4864/5376/4096, K=dim
//   LMhead: M=151936 (qwen vocab), K=dim
// For each class, time tt_gemv_q4_0 (V2 path), tt_gemv_q4_0_v4,
// tt_gemv_q4_0_batch4 (as R=4 batched proxy) with CUDA events, 200 iters,
// report GB/s (weights streamed) + us.
```

Build with the same NVCC flags as the Makefile `cuda` target (copy the exact command, add this file). R=1 has no kernel — emulate with V2 masked to one row? No: measure V2/V4/BATCH4 as R=2/4/4-wide proxies and record launch config + occupancy (`--ptxas-options=-v` output saved to `bench/micro_gemv_rows.txt`).

**Step 2: Run and record**

Run: `./build/micro_gemv_rows 2>&1 | tee bench/micro_gemv_rows.txt`
Expected: table of us + GB/s per (class, kernel). Commit the txt.

**Step 3: Decision record (no code yet)**

Append to `bench/micro_gemv_rows.txt`: for each class, which R wins and by what margin; explicit GO/NO-GO for building an R=8 kernel in Task 3 (GO only if best-vs-second margin >10% on FFN or LM-head class).

**Step 4: Commit**

```bash
git add tools/micro_gemv_rows.cu bench/micro_gemv_rows.txt
git commit -m "perf: GEMV rows-per-warp sweep results"
```

---

### Task 3: LM-head specialized decode kernel

**Files:**
- Modify: `kernels/gemv_q4_cuda.cu` (new kernel `k_logits_q4_0_lm` next to `k_logits_q4_0_v4:313`), dispatch in `kernels/gemv_typed.cu` or `qwen2_cuda.cu` (read current LM-head call site first)
- Test: `tests/test_logits_q8_v4.c`-style check + `tests/test_gpu_parity.py`

**Step 1: Write the failing test**

Extend the existing logits test (read `tests/test_logits_q8_v4.c` build rule in Makefile) or add `tests/test_logits_lm.c`: random Q4_0 vocab×dim matrix (use M=151936,K=896 and M=256000,K=2304 for gemma-scale), compare new kernel vs `k_logits_q4_0_v4` bit-exact (`max|Δ|==0`, same math, different tiling) and vs fp32 reference (`<2e-2`).
Run: expect FAIL (symbol missing).

**Step 2: Implement (only if Task 2 said GO for LM-head, else implement R-winner variant)**

Design: 8 rows/warp if Task 2 GO, else 4 rows/warp with retuned block shape for vocab-M (more blocks in M, x-vector in registers — see `k_logits_q8_0_v4:785` pattern). One kernel, Q4_0 decode only. No prefill changes.

**Step 3: Verify**

Run: new test PASS + `python3 tests/test_gpu_parity.py 2>&1 | tail -2` → GATE A GREEN + `bench/bench_llm.py` decode median improves or holds (record both numbers in commit message).

**Step 4: Commit**

```bash
git add kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu tests/test_logits_lm.c
git commit -m "perf: specialized LM-head decode kernel (R=<n>, <old>-><new> tok/s)"
```

---

### Task 4: Fusion round 1 (RMSNorm→QKV, O/residual, down/residual)

**Files:**
- Modify: `kernels/qwen2_cuda.cu` (new fused kernels beside `k_rmsnorm:173`; call sites ~4252,4473,4489,4521)
- Test: `tests/test_gpu_parity.py` + golden-free parity via `build/dump_logits` ARGMAX compare before/after

**Step 1: Parity harness (before touching kernels)**

Run: `./build/dump_logits --model data/models/qwen2.5-0.5b-instruct-q4_0.gguf "2,2202,1110" /tmp/pre_fusion.bin` → record `ARGMAX <id> <val>` line. This is your bit-exactness anchor for this task (logits must match to <1e-5; fusion changes launch structure, not math).

**Step 2: Fuse RMSNorm→QKV GEMV only**

New kernel: block computes RMSNorm row into registers/shared, then QKV GEMV rows directly (no global round-trip of normalized x). Keep the old path behind `TT_NO_FUSE=1` env for A/B. One fusion, not three — O/residual and down/residual are separate commits after this proves out.

**Step 3: Verify**

Run: `TT_NO_FUSE=1` vs fused `dump_logits` ARGMAX diff <1e-5 + `bench/bench_llm.py` decode delta recorded. `tests/test_gpu_parity.py` GREEN.

**Step 4: Commit**

```bash
git add kernels/qwen2_cuda.cu
git commit -m "perf: fuse RMSNorm-QKV GEMV (dump-parity ok, <old>-><new> tok/s)"
```

Repeat Steps 2–4 for O+residual and down+residual as two more commits (same parity protocol). Do NOT fuse attention (numerically sensitive) in this task.

---

### Task 5: Attention/KV at context (measure, then tune)

**Files:**
- Modify: none until measurement says so; candidates `k_flash_gqa:618`, `k_flash_gqa_q8_0:990`, Q8 backfill path
- Test: `tools/profile_decode.py` (Task 1) extended with ctx sweep

**Step 1: Context scaling measurement**

Run: `python3 tools/profile_decode.py --ctxs 128,512,2048,4096,8192 --model data/models/qwen2.5-0.5b-instruct-q4_0.gguf`
Expected: `bench/scoreboard_decode.csv` rows showing attn_ms share. Decision rule: if attn <15% at 8k, STOP this task after committing the CSV (record NO-GO with numbers); if ≥15%, proceed.

**Step 2 (only on GO): FP16 vs Q8 KV A/B**

Env-gate the KV path (read current backfill select logic first; add `TT_KV_F16=1` force path if missing), benchmark both at 4k/8k ctx, keep winner as default, loser behind env. Accuracy guard: `dump_logits` ARGMAX top-1 must agree between modes on 3 prompts.

**Step 3: Commit**

```bash
git add bench/scoreboard_decode.csv kernels/qwen2_cuda.cu
git commit -m "perf: attention ctx scaling (<share> at 8k); <decision>"
```

---

### Task 6: Prefill dispatch crossover + explicit fallback reporting

**Files:**
- Modify: `kernels/qwen2_cuda.cu` (`prefill_batched_gemm:5192` dispatch, verify fallback `:6064`)
- Test: `tests/test_prefill_gemm.c`, `tests/test_wmma_prefill_gemm.c` (must stay green)

**Step 1: Crossover sweep**

Time Q4 WMMA prefill vs cuBLAS/FP16 path at N ∈ {1,8,16,32,64,128,256,512,1024} on qwen2.5-0.5b (use `tools/bench_prefill.c`, extend its N list if needed). Record table in `bench/prefill_crossover.txt`. Set threshold constant from data (not 32 by fiat), e.g. `#define PREFILL_CUBLAS_MIN_N <measured>`.

**Step 2: Explicit dispatch reporting**

Add one stderr line in benchmark/profile mode only (guard with existing TT_PROFILE or new TT_DISPATCH): `prefill dispatch: <q4-wmma|cublas-fp16|eager-fallback> N=<n>`. Convert the silent verify fallback (`:6064`) to log `[verify] prefill_dx refused (rc=<rc>), eager fallback` at the same verbosity. No behavior change besides logging + threshold.

**Step 3: Verify + commit**

Run: both prefill tests PASS + crossover table committed.

```bash
git add kernels/qwen2_cuda.cu bench/prefill_crossover.txt tools/bench_prefill.c
git commit -m "perf: measured prefill crossover N=<n>, explicit dispatch logging"
```

---

### Task 7: Graph coverage extension + no-graph parity lock

**Files:**
- Modify: `kernels/qwen2_cuda.cu` (graph section ~3179-3200, eager condition ~3765)
- Test: new `tests/test_graph_parity.c` (graph vs TT_NO_GRAPH=1 logits bit-exact on 3 prompts)

**Step 1: Write the parity test**

C probe (link line modeled on Makefile `run_llm_gpu` rule, output to `~/ti-scratch`): prefill 8 tokens, decode 4, capture logits under graph vs `TT_NO_GRAPH=1`, assert max|Δ|==0. Run: expect PASS already (locks current coverage against regressions).

**Step 2: Extend capture to the fused kernels from Task 4**

Each Task-4 fused kernel must be capturable (no disallowed ops: no malloc/free, no sync, host→device only via pre-created buffers). If a fused kernel breaks capture, fix the kernel (stage through device buffers), not the graph. Re-run parity test.

**Step 3: Spec-verify path under graph**

`prefill_batched_gemm_dx` + verify sequence (`:6064`) runs under capture where shapes are static; log when it falls back to eager. No perf claim without measurement: record graph vs eager decode tok/s in scoreboard.

**Step 4: Commit**

```bash
git add kernels/qwen2_cuda.cu tests/test_graph_parity.c bench/scoreboard_decode.csv
git commit -m "perf: graph coverage for fused kernels, parity locked"
```

---

### Task 8: Autotuned dispatcher + final scoreboard

**Files:**
- Create: `kernels/gemv_dispatch.h` (or extend existing dispatch site — read current V4 conditions like `M>=128` first)
- Modify: single call-site file only
- Test: full `scripts/ci_local.sh` + `tests/test_gpu_parity.py`

**Step 1: Centralize choice**

```c
typedef enum { TT_GEMV_Q4_V2, TT_GEMV_Q4_V4, TT_GEMV_Q4_LM, TT_GEMV_Q8, TT_GEMV_CUBLAS } TTKernelChoice;
TTKernelChoice tt_choose_gemv(int M, int K, int dtype, int sm, int is_prefill, int N);
```

Move existing scattered conditions (V4 `M>=128` etc.) into this function verbatim first (no threshold changes), route all decode GEMV calls through it. Thresholds from Task 2/3/6 data go in as a second commit with before/after numbers.

**Step 2: Final scoreboard + targets check**

Run full matrix (both models × ctx 128/512/2048): `bench/bench_llm.py` + profile CSV. Grade vs ladder: 0.80 minimum → 0.90 → 0.95 → 1.00. Whatever the number, commit the CSV with an honest header (no rounding up).

**Step 3: Commit**

```bash
git add kernels/gemv_dispatch.h kernels/qwen2_cuda.cu bench/scoreboard_decode.csv
git commit -m "perf: centralized GEMV dispatcher, final scoreboard <ratio>x llama.cpp"
```

---

**Explicit non-goals (do not touch):** tokenizer, chat templates, speculative decoding, MoE, server code. LM-head work stays decode-only. No decode move to cuBLAS (measured worse: 212.8 vs 224.2 tok/s). No kernel change without a `bench/scoreboard_decode.csv` row proving the delta.

**Reference skills:** implement with `/skill:executing-plans` (parallel session) or `/skill:subagent-driven-development` (this session). Bugs mid-flight: `/skill:systematic-debugging`. Done: `/skill:finishing-a-development-branch`.
