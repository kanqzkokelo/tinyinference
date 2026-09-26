# Beating llama.cpp — exhaustive guide

Date: 2026-09-26 · Companion to `docs/CODE_REVIEW.md` and
`docs/plans/2026-09-20-every-axis-vs-llamacpp.md` (which this guide supersedes
as the master document; that plan's phase numbering is kept).

This is the operational playbook for winning against llama.cpp **on every axis
that can be measured**, honestly. It consolidates: the measured scoreboard
(`docs/BENCHMARKS.md`, `bench/scoreboard_decode.csv`), the decode research log
(`docs/plans/2026-09-10-decode-1x-v3.md` — including its do-not-retry list),
the quant-parity program (`every-axis` plan Parts A/B/C), and the benchmarking
lessons from `data/profile/AUDIT_HONEST_2026-08-29.md`. Everything here either
carries a measurement with provenance or is explicitly marked UNMEASURED.

---

## 0. Rules of engagement

1. **llama.cpp is the oracle, not the enemy.** It defines "the same quant",
   "the same model", and the reference numerics. Every correctness claim is a
   parity test against it; every performance claim is a ratio against its own
   tools.
2. **Pinned everything.** Oracle pinned at `3f545be`
   (`DEFAULT_LLAMACPP_COMMIT` in `bench/bench_llm.py:43`); our commit recorded;
   model file SHA256 recorded; GPU + clocks + power recorded. A ratio without
   provenance is not a claim.
3. **Oracle default AND oracle tuned.** Report llama.cpp both at its defaults
   and at its best fair configuration (e.g. `-ctk q4_0 -ctv q4_0` for memory
   parity, `--spec-type ngram-map-k` for spec-decode). Winning against a
   mistuned oracle is a marketing loss.
4. **5 medians, warmup, cooldowns, both min and max.** This laptop's oracle
   samples swung 136-374 tok/s inside one cell (bimodal clocks); a single
   sample is a lie generator. Rerun any cell whose spread is bimodal; cite
   the rerun, not the lucky median.
5. **UNMEASURED is not WON.** The axis table prints UNMEASURED until the
   matching oracle tool produces a ratio. This rule alone prevents most
   self-deception.
6. **Parity gates precede speed acceptance.** A kernel that fails
   `verify.sh m61` (greedy-logit parity) does not ship, however fast.
7. **Honor the do-not-retry list** (Section 13). Most "new" kernel ideas
   are old killed ideas with fresh paint.

---

## 1. Where we stand (2026-09-26)

Reference box: RTX 3050 Laptop 4 GB (sm_86, 176 GB/s), i5-11400H. Fleet:
qwen2.5-0.5B-q4_0, qwen3-0.6B-q8_0, llama-3.2-1B-q4_0, smollm2-135M-q4_0,
gemma-4-E2B.

| Axis | Oracle tool | Status | Verdict |
|---|---|---|---|
| Decode tok/s, batch 1, ctx 32-512 | `llama-bench -p 0 -n 64` | 1.00-1.33x (fleet, `docs/BENCHMARKS.md`) | **WON** (short ctx) |
| Decode tok/s, batch 1, ctx 2048+ | `llama-bench -p 0 -n 64 -d N` | 1.03-1.21x qwen family; smol 0.96x @4096; llama flat-tax ~0.85x historically | MOSTLY WON, mixed |
| Prefill / TTFT | `llama-bench -p 512,2048` | was 0.06-0.24x; batchn+FA2+WMMA landed since; ratio UNMEASURED post-landing | LIKELY STILL LOSE, closing |
| End-to-end time-to-N tokens | `llama-cli` timing | dominated by prefill; UNMEASURED post-landing | LOSE (follows prefill) |
| Speculative decode | `llama-cli --spec-type ngram-map-k` | ours unproven in e2e; verify path bit-exact; batch4 head exists | UNMEASURED |
| Aggregate throughput (B>1) | `llama-batched-bench` | no continuous batching | LOSE (large project) |
| Tensor quant types consumed | load + compute | 13 vs 43 (16 meaningful missing) | LOSE |
| Quantizer / imatrix | `llama-quantize`, `llama-imatrix` | 1 type (`quantize_row_q4_0`), no imatrix | LOSE |
| KV cache types | `-ctk/-ctv` parity | f32/Q8/Q4(+f16 partial) vs f32/f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1 | LOSE small |
| Arch/model coverage | any GGUF | ~15 families; no MLA/SSM/LN+bias | LOSE (months) |
| Backends | any | CUDA sm_86/89 + AVX2 CPU | LOSE (Vulkan/Metal/ROCm) |
| Quality at equal size | `llama-perplexity` | no eval harness | LOSE |
| Load / warm-start | wall clock | 1.18 GB/s memcpy-bound upload | LOSE, fixable |
| Energy/token | `nvidia-smi` | UNMEASURED (follows tok/s if we win) | UNMEASURED |
| Server features | `llama-server` | minimal N=1 server | LOSE (months) |

The scoreboard is allowed to say we lose. The purpose of this guide is to turn
each LOSE into a WIN **or** into a documented, justified concession.

---

## 2. Why a 15-file engine can beat a 10-year project at all

llama.cpp's design constraint is generality: 43 quant types x ~50 archs x 5
backends x batch 1..512 x CUDA graphs on/off, all from one op graph. That
generality is a tax we do not pay:

1. **Shape specialization.** Our GEMVs know the exact (M, K) of the layer
   they run (`k_gemv_q4_0_v4` 4-rows/warp tuned per shape class; fused QKV;
   fused FFN+SwiGLU). ggml's `mmvq`/`mmq` must be correct for any shape.
2. **State specialization.** Dual lazy CUDA-graph replay of the decode step
   (`TT_GRAPH_SLOTS`) removes launch overhead for *our* dispatch choices;
   batch-1 decode launch overhead is real money at 3-8 ms/step.
3. **Traffic wins for free.** At batch 1 every op is memory-bound; a
   hand-tuned dequant-in-registers GEMV and a Q8/Q4 KV cache cut bytes, which
   is the only thing that matters at the roofline. llama.cpp has these too —
   the win is doing them with fewer abstractions in the way, and matching
   them kernel-for-kernel at the exact model shapes.
4. **Decision speed.** The kill lists show the project already burned months
   of kernel thrash. The specialization advantage only pays if we *stop*
   re-trying dead ends and spend the time on coverage and prefill instead.

Where the tax reverses: coverage (quants, archs, backends), server features,
and community trust. Section 11 is the honest concession plan.

The duel must include op-level parity of method: use llama.cpp's own
`test-backend-ops --mode perf -o MUL_MAT` against our micros at llama shapes
(`tools/micro_gemv_llama.cu` covers shapes but links no ggml — record both
sides before claiming "faster quantized GEMV").

---

## 3. Decode playbook (protect the win, close the long-ctx gap)

**Won already — do not regress:** V4 4-rows/warp q4_0 GEMV (1.5-1.6x on FFN
shapes), fused QKV (beats 3xV4 by 21% on llama shapes), fused FFN+SwiGLU,
V4 down/oproj, Q6_K V2 head, Q8/Q4 KV hybrid with `TT_QKV_THRESH` crossover,
split-K flash + unfused/fused attention branch dispatch, dual-graph replay,
automatic prefix caching. Every op is on its micro-proven best kernel.

**Open levers, in order of expected gain:**

1. **Long-ctx KV traffic (B3, mostly landed as hybrid Q8/Q4).** Keep the
   ladder: FP32 below threshold (bit-exact short-ctx gates), Q8 above, Q4
   with per-head scales + fused dequant-in-attention (`k_q4_fused` already
   prototypes scores-never-materialized). Report every cell at equal KV
   precision vs the oracle (`-ctk q8_0/-ctv q8_0` or q4_0 both sides).
2. **The llama flat-tax (~0.85x at all ctx on llama-1B).** Not one broken
   kernel: larger FFN/logits scale + in-situ GEMV deficit + Q6_K head + small
   attn gap. Grind it with (a) head requant Q6_K->Q4_0 once llama-2048
   clears 0.90x (frozen by kill criteria until then), (b) KV-traffic items
   above, (c) nothing else — kernel swaps are exhausted.
3. **Structural attention rework** (the only +0.5 ms idea still open on
   llama-2048): FlashDecoding-style fused qk+softmax+pv with rescale combine
   for the split-K path at high ctx. Micro first at exact shapes; no engine
   change before the duel is won. (Fused-attn already landed: +2.6% at
   ctx2048 — small because attention is a small slice of the step; expect
   the same dilution.)
4. **Spec decode as a multiplier** (Section 5) — the only remaining way to
   move batch-1 decode by more than ~10%.

**Do not:** per-op kernel swaps, launch-removal fusions (add-norm, fused
SwiGLU V4, R8 GEMV, qkv2+R8, PART sweeps — all killed with evidence), or
optimizing attention fusions further (diluted).

---

## 4. Prefill / TTFT playbook (the war that decides end-to-end)

This is where we currently lose and where 5-7x end-to-end deficits came from.
Landed since the plan was written: `k_gemv_q4_0_batchn` for N<=12
(8-14x vs GEMM at N=1, bit-exact vs sequential), `pf_minn` default 2,
WMMA tensor-core GEMM auto at N>=64, FA2 tensor-core prefill flash
default-ON. Still open:

1. **Default-on the cuBLAS fp16/tf32 shadow GEMM (`TT_CUBLAS_FP16`) with
   gates** (P2, 2-3 days): measured 2.7-5.5x; needs (a) per-shape engagement
   gating, (b) KV-parity gate — FP32 KV below thresh keeps m61 green, (c) a
   documented no-cuBLAS fallback to the Q4 GEMM/WMMA path, (d) VRAM budget
   policy for shadows (fp16 shadows of 1B = ~1 GB; on 4 GB cards engage only
   when KV is Q8/Q4 or model < 1B). Exit: ctx2048 prefill ratio >= 0.55x.
2. **Batched prefill for non-Q4_0/Q8_0 dtypes** (P1 item, `CODE_REVIEW` P1):
   K-quant/F16 models currently prefill one GEMV per row. Generalize the
   batchn single-weight-pass pattern (template nr=1/2/4) to Q8_0 and Q4_K/Q6_K,
   or dequant each tensor once to an F16 shadow and reuse the tensor-core
   GEMM. Required before the quant-parity program is *usable*.
3. **Extend batchn's N range and dtype set.** batchn wins up to N~12 and
   ties GEMM at ~16; shape-table dispatch per (M,K,N) class instead of one
   global `TT_PF_BATCHN_N`. For qwen3-0.6B-q8_0 cells the Q8 GEMM exists
   (`tt_gemm_q8_0_prefill`) — give it the same small-N batchn treatment.
4. **Chunked prefill + decode overlap for the server** (Section 9): chunk
   N=256-512 interleaved with decode steps keeps interactive TTFT low under
   load; requires the batched engine work first.
5. **Prompt caching by default.** Automatic prefix caching exists but is
   env-gated (`TT_PREFIX_CACHE`). Agentic multi-turn workloads are the norm;
   turn it on by default and advertise instant history reuse (it is already
   in the README claims — make the default match the claim).
6. **Small-N chat TTFT** is the user-visible number: every chat turn prefills
   only the delta; with batchn landed the delta is batched from N=2. Measure
   `p1_ttft_ab.py` after each change; target <= 10 ms TTFT at 0.5B.

**Method note:** prefill ratios must quote *prompt tokens actually processed*
(tables in `docs/BENCHMARKS.md` already show "Actual Prompt Tokens" — keep
that), because tokenizers differ slightly per prompt filler.

---

## 5. Speculative decode playbook (the only batch-1 multiplier left)

llama.cpp ships `ngram-simple`/`ngram-map-k`/`ngram-mod` and defaults them
**off** — beating the default oracle here is trivial and worthless. The bar is
`llama-cli --spec-type ngram-map-k` (their strongest n-gram drafter).

Current state: verify path is bit-exact vs sequential (`test_spec_verify.c`),
rollback exists (`qwen2_engine_rollback`), batch4 LM head does one weight pass
over 4 candidates (`tt_logits_q4_0_batch4`), batched small-N verify landed as
`TT_SPEC_BATCH` + `prefill_batched_gemm_dx`. The drafter is the weak link.

Build order (each step gated before the next):

1. **Batch4 GEMV to bandwidth-bound.** Today batch4 runs 16-30 GB/s vs
   100-175 GB/s for single-token V4 at the same shapes
   (`bench/micro_gemv_rows.txt`). Verify cost ratio must be <= 1.2x of a
   single decode step before speculation can win at all. The weight pass is
   shared; the problem is x-reuse and reduction structure — kt4/batch4_kt2
   kernels exist (`gemv_q4_batch4_kt4.cu`), duel them at the exact verify
   shapes (N=4, FFN + head).
2. **Device-side accept mask.** Today a verify copies ~2.4 MB D2H
   (K x vocab floats) + a sync per pass at vocab 152k
   (`tools/spec_expA_e2e.c`). Compare greedy argmax per candidate on device,
   return one u32 accept-count + the bonus token id. This is a small kernel
   and removes the sync tax.
3. **A real drafter.** Ours is a 2-3 token window, last-occurrence scan
   (`ngram_lookup.c`) — strictly weaker than `tt_ngram` (12-gram ring, already
   in-tree and simulated in `test_specdec_sim.py`) and much weaker than
   ngram-map-k's hashed multi-window map. Ship: hashed map with multiple
   window lengths (2..12), frequency-based tie-break, over the KV-mirrored
   token history. Optionally: self-speculative (early-exit draft heads) is a
   later game.
4. **Rollout order:** wire `tt_ngram`+hash map into `spec_llm_gpu` and
   `server_minimal`, keep `verify.sh m61` green (spec must be greedy-
   identical to non-spec: accept/reject must reproduce the target chain).

Acceptance targets (from the plan, keep them): >= 1.5x e2e on 2 of 4
workloads and >= 1.15x vs `--spec-type ngram-map-k`. Kill criteria: if
acceptance < 1.5 tokens/verify with verify cost ratio > 1.5 after step 1-3,
report "spec decode not competitive on this fleet" and stop.

Workload honesty: n-gram speculation wins on repetitive/code/RAG prompts and
loses on creative single-turn chat. The bench must include both
(`tests/test_specdec_sim.py` already has the corpus split — reuse it).

---

## 6. Quant parity program (the part that makes claims survive users)

Two obligations, both required:

### 6.1 Consume: every GGUF tensor type llama.cpp can run

13 types today (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K..Q6_K). P0 set
(they change what fits on 4 GB): **IQ4_XS (23), IQ4_NL (20), IQ3_XXS (18),
IQ3_S (21)**; P1: IQ2_XS/S/XXS, MXFP4, Q1_0/Q2_0; P2: IQ1_*, TQ*; SKIP:
NVFP4 (no Blackwell units), CPU presentation formats, activation-only
intermediates (Q8_K/Q8_1), structural ints.

Per-type recipe (the house pattern; no batching types across steps):
1. Type code + `size_bytes` + name in the dispatch tables
   (`include/dequant_ref.h`, `src/loader_gguf.c:336-360`).
2. Reference dequant ported from `ggml-quants.c` into `src/dequant_ref.c`
   (IQ3_XXS is already written in `tests/proto_iq3xxs.cu:56-140` — reuse).
3. Golden fixture via `tests/fixtures/gen_fixtures.py` (gguf-py) + 1e-6 gate.
4. Kernels in `kernels/gemv_typed.cu`: V2/V4 GEMV + logits variants;
   I-quants need `__constant__` grid LUTs + dequant-in-registers (prototype
   `k_gemv_iq3_xxs`).
5. Engine routing for every weight role (qkv/o/gate/up/down/head) — mixed
   per-tensor types are the normal case (K-M and I-M ftypes).
6. Parity row in the oracle grid (`tests/gate_m7_grid.py`). No type ships
   without one.
7. Speed acceptance: GB/s vs the ggml kernel duel at the same shapes
   (`test-backend-ops`), then fleet scoreboard.

Cost: ~0.5-1 day per block type, 1.5-3 days per I-quant. P0 = 1-2 weeks.

**The competitive twist:** supporting a type is parity, not advantage. The
advantage must come from executing it faster than `mmvq`/`mmq` at batch 1
(dequant-in-registers + rows-per-warp patterns are our edge) and from
prefill batchn working on it (Section 4.2).

### 6.2 Produce: the quantizer llama.cpp users expect

Currently `quantize_row_q4_0` only. Build:
1. **`tt-quantize`**: reference `quantize_row_*` for byte-stable types, then
   ftype mix tables (Q4_K_M/S, Q5_K_M/S, Q3_K_*, Q2_K_S, IQ3_M, IQ2_M,
   IQ3_XS...), per-tensor overrides, `--pure`, `--override-kv` equivalents.
   Reuse the loader with write support (do not build a second GGUF writer).
2. **`tt-imatrix`**: calibration corpus through prefill (the `TT_DUMP_XN`
   hook reaches real hidden states), importance accumulation, and the
   llama.cpp imatrix metadata keys so files interoperate both ways.
3. **Quality harness**: perplexity + KL vs F16 baseline, matching
   `llama-perplexity --kl-divergence`. Without this "same quant" is
   unfalsifiable.

**The anti-faking test:** for every ftype where llama.cpp's quantizer is
deterministic (no threads, no imatrix, reference path), quantize the same F16
model with both tools and require **byte-identical tensor payloads**. This is
achievable for the Q* family (our q4_0 is a stated port) and is the only
definition of "the same quant llama is doing" that cannot be gamed. For
imatrix paths: same ftype + same per-tensor assignment + ppl within declared
tolerance (e.g. +/-0.5% on the same corpus), both recorded.

**KV cache type parity** (cheap, do alongside): add BF16, Q4_1, Q5_0, Q5_1,
IQ4_NL KV types and one knob accepting llama.cpp's `-ctk/-ctv` type names
(mapped onto the existing FP32/Q8/Q4 paths and `TT_QKV_THRESH`), so every
comparison can be run at equal KV precision with one flag.

---

## 7. Long context & memory (where 1B models die on 4 GB cards)

- Q8/Q4 KV hybrid landed; finish **Q4-KV per-head scales + fused
  dequant-in-attention** (scores never materialized: `k_q4_fused` pattern).
- **Paged attention** is prototyped (`tools/micro_paged_fa2.cu`,
  `tests/proto_*`): adopt it when the server needs dynamic KV (Section 9),
  not before — fixed slab wins batch 1.
- SWA models: `kvcache.c` compaction plans are ready for engine integration;
  gemma2/3 SWA layers currently attend with a window mask over the full slab
  (traffic scales with ctx even when the window is 4096). Compaction or
  circular-buffer slots cut that traffic — measured target: gemma-family
  decode flat in ctx beyond the window.
- The 30k-ctx win (qwen2.5-0.5B 103.3 vs 91.9 tok/s) shows long ctx is
  winnable at equal KV precision — keep those cells in every fleet run.

---

## 8. Load / warm-start (the cheap win)

Today: mmap + per-tensor H2D upload, ~1.18 GB/s memcpy-bound
(`bench/RESULTS_S0_vs_S4_real_loader.md`). Options in order of payoff:
1. **Pinned staging + async multi-stream upload** (2x-ish expected vs pageable
   memcpy), overlapped with tokenizer/engine init and first-prompt
   tokenization.
2. **Lazy upload by layer** — first token needs layer 0 weights; stream
   layers in ahead of the compute horizon. TTFT can absorb the rest.
3. **Warm-cache fast path**: kernel page cache makes re-load fast already;
   measure cold and warm separately and report both (the scorecard row
   requires it).
4. FP16-shadow duplication (`TT_CUBLAS_FP16`) must not re-read the file —
   convert from the device copy.

Target: >3 GB/s cold upload on this box, or <500 ms to first token for a
0.5B model warm. Oracle comparison: `llama-cli` load timing, both cold/warm.

---

## 9. Aggregate throughput & server (weeks; do not start early)

llama-server wins today: continuous batching, slots, streaming SSE, OpenAI
API surface, grammars, LoRA, parallel decoding. Ours is a documented N=1
mutex server. The honest sequencing:

1. **Do not** build batching before prefill (Section 4) and spec (Section 5)
   land — those raise single-stream perf and are reused by batching anyway.
2. Then: shared KV arena (`kvcache.c` plans already model it), slot
   scheduler (prompt chunking + decode continuation), batched attention
   (the batched prefill kernels generalize: attention over B slots), and
   step-queue batched GEMMs (M dimension becomes B or B*N — the GEMM story
   from Section 4 covers it).
3. Measure with `llama-batched-bench` (build it first, P0), at equal ctx,
   batch 1..8.
4. Server surface: SSE streaming first (users notice), then grammar, then
   LoRA. On a 4 GB laptop, batch 1-2 is the realistic regime — prefer
   shipping low-TTFT single-stream + spec decode and *documenting* the
   batching gap over a half-built scheduler.

---

## 10. Coverage & backends: where to concede and where to fight

**Fight (high value per week):**
- LN+bias arch family (phi-2/3, falcon, mpt, bloom, gpt2...): one LayerNorm
  kernel + bias handling unlocks ~20 families on the existing trait registry
  pattern. Biggest coverage unlock per unit work.
- The P0/P1 quant set (Section 6) — unlocks 3B-8B models on 4 GB.
- MLA (deepseek2) and SSM (mamba/rwkv) only if the target model mix demands
  them; each is a new attention kernel, months not weeks.

**Concede (document as out of scope, or partner):**
- Vulkan/Metal/ROCm/SYCL backends. A second backend is a second kernel
  corpus. The only honest alternative is a *portable* strategy: keep the
  reference CPU backend (`cpu_backend.c`, AVX2) as the universal fallback and
  state "NVIDIA + CPU" as the supported platform. Revisit only if a
  distribution requirement appears.
- Multimodal, TTS, encoders — out of scope (already documented).
- MoE *execution* (qwen3_moe resolves; expert dispatch engine pending):
  `moe_router.c` plan-builder is done; GPU expert GEMVs need the batching
  work from Section 9 first.

---

## 11. Energy per token

Unclaimed axis, cheap to measure: sample `nvidia-smi --query-gpu=power.draw`
at 10 Hz around each bench cell (the harness already records power info
fields), report mJ/token = mean_power x wall_time / tokens. On a
memory-bound batch-1 workload energy/token tracks 1/tok/s almost exactly, so
this axis converts decode wins into a second win for free — but only report
it with the same provenance rules (locked clocks noted, median of runs).

---

## 12. Benchmarking doctrine (the difference between winning and lying)

Hard-won rules; each has a scar behind it:

1. **Micro benches:** per-iteration event sync (not batch-mean over a fused
   loop), median/p95 over >= 500 samples, L2 flush or weights > L2 (the
   Q2_K "46 GB/s" was L2-hot), shapes that exercise the real code path
   (the K-truncation bug: `nsb=3` silently dropped 128 weights), random
   page permutation for paged experiments, and **explicit non-finite
   counting** (the Q4 fused-attn bug hid behind `max()` over NaN diffs).
2. **E2E cells:** 5 medians post-warmup, 3-10 s cooldown, both engines
   measured back-to-back in the same thermal window, all samples recorded
   (jsonl in `data/bench/`, gitignored), never a single sample.
3. **Oracle flakiness protocol:** if the oracle's samples are bimodal
   (136-374 tok/s happened), rerun the cell with cooldowns and cite the
   rerun. Never cite a ratio computed from a collapsed oracle median
   (the fake 1.565x incident). If the oracle cannot stabilize, report the
   range, not the ratio.
4. **Same-session A/B for kernel changes** (fused vs split, ON vs OFF), then
   a fresh full fleet run before any scoreboard refresh — never mix binary
   generations in one geomean.
5. **Fairness parity:** equal KV precision (match `-ctk/-ctv`), equal ctx
   (report actual prompt tokens), equal batch, graph mode labeled, tuned AND
   default oracle columns.
6. **Noise disclosure:** record neighbor GPU load; the kill-list says do not
   design on noisy data ("Do not start attn rework on noisy data" — it got a
   green light only after a quiet-GPU remeasure).
7. **Clock policy:** no `-lgc` in the automated suite (documented), but any
   published number must state clocks/power mode.
8. **Reproducibility fields** (already in `bench/bench_llm.py`): both commits,
   model SHA256, GPU name/SM, CUDA version, prompt token counts, warmup/runs/
   cooldown, graph status. The scorecard generator (`bench/scorecard_vs_llamacpp.py`,
   planned in Part C of the every-axis plan) must refuse to print a ratio
   missing any field.

---

## 13. Kernel craft: the playbook and the graveyard

**Patterns that paid:**
- 4-rows-per-warp GEMV sharing x across row groups (V4; 1.5-1.6x FFN shapes).
- Batch-K weight pass shared across N prompt tokens (batchn; 8-14x vs GEMM at
  small N) — the prefill workhorse pattern for every dtype that gets it next.
- Word-streaming unaligned quant blocks with `__byte_perm` merge (18-byte
  q4_0 rows) with a proven no-OOB boundary argument.
- Fused QKV (one weight pass, three outputs) and fused FFN+SwiGLU for q4_0.
- Split-K flash attention with online-softmax combine (long-ctx decode);
  fused qk+softmax+pv to avoid materializing scores (2.6% at ctx2048 — real
  but diluted).
- Dequant-in-registers + `__constant__` LUTs for I-quants (proto verified).
- CUDA-graph decode with dual lazy slots keyed on the dispatch plan; all
  per-step state device-resident (pos, recent-tokens ring, sampling flags).
- Hybrid KV precision with a parity-preserving threshold (FP32 below keeps
  short-ctx gates bit-identical).

**Graveyard (do-not-retry without new evidence):**
- Launch-removal fusions under graph replay: add-norm, fused SwiGLU V4,
  Q6-head v3, R=8 GEMV and radix-8 variants, fused-qkv-4row, qkv2-2row+R8,
  V4-b4 occupancy push, attn fast-exp. Under graphs, removing launches saves
  ~zero; only traffic cuts and algorithmic multipliers move decode.
- Register-hungry 4-rows variants that spill (fused FFN gate/up 4-row was 26%
  *worse*): occupancy is not the goal; live-register budget is.
- PART sweeps and attn partition micro-tuning beyond the current gate
  (noisy, no structural gain).
- Any micro that reports batch-mean without per-iter sync, or shapes whose
  weights fit L2 while claiming DRAM GB/s.
- Single-noisy-cell "wins" (the 25.4x that was an OOB bug; the 1.565x that
  was oracle bimodality).

**Standing lessons to write into every new kernel's review checklist:**
bit-exactness declared (vs which reference), non-finite count printed,
weights > L2 or labeled L2-hot, per-iter sync, launch config justified
(live regs > occupancy), and a duel against the *incumbent* kernel at the
exact production shape.

---

## 14. Program plan (updated for what has landed)

| Phase | Content | Status | Exit criterion |
|---|---|---|---|
| P0 | Build missing oracle tools (`llama-quantize`, `llama-imatrix`, `llama-perplexity`, `llama-batched-bench`, `test-backend-ops`); scorecard harness `bench/scorecard_vs_llamacpp.py` -> `docs/SCORECARD_VS_LLAMACPP.md` | **not started** | every axis has a provenance-stamped baseline from one oracle tree |
| P1 | Small-N prefill (batchn + `pf_minn=2`) | **landed** | n<32 >=3x achieved; chat TTFT remeasure pending |
| P2 | Tensor-core prefill default-on: cuBLAS fp16 shadows (FA2 + WMMA auto landed) | **half-landed** | ctx2048 prefill >= 0.55x, m61 green |
| P3 | KV ladder to Q4 + fused dequant-in-attn | **hybrid Q8/Q4 landed**; fused attn landed | decode geomean >= 1.05x at equal KV precision |
| P4 | A1 P0 quant set: IQ4_XS, IQ4_NL, IQ3_XXS, IQ3_S | not started | 4 types load+compute+parity+speed; 3B runs in 4 GB |
| P5 | `tt-quantize` + ftype mix tables | not started | byte-identical for every deterministic ftype |
| P6 | imatrix + perplexity/KL harness | not started | ppl/KL within declared tolerance |
| P7 | Spec decode: verify bandwidth, device accept, real drafter | verify bit-exact; batch4 weak; drafter weak | >=1.5x on 2/4 workloads AND >=1.15x vs `ngram-map-k` |
| P8 | Batching/server; KV types BF16/Q4_1/Q5_0/Q5_1/IQ4_NL; load-time work | not started | `llama-batched-bench` parity or better |

P1-P3 (speed) and P4-P6 (quant) are independent; P7 depends on P1 (landed)
and P5's plumbing; P8 depends on everything.

**Kill criteria (report honestly, stop claiming):**
- after P2, prefill still < 0.5x -> position is "decode-only wins".
- after P3, long-ctx decode < 1.0x at equal KV precision -> "short-ctx wins only".
- after P7, acceptance < 1.5 with verify ratio > 1.5 -> "spec not competitive".
- after P5, byte-identical quantizer for < half the deterministic ftypes ->
  "our quantizer is llama.cpp-inspired, not llama.cpp-compatible".
If two or more fire, the published position is exactly what they imply.

---

## 15. Preconditions from the code review (items that gate claims)

Claims are only as good as the harness and error paths behind them
(`docs/CODE_REVIEW.md`):
- **C1** (prefill failure fallback corrupts KV) must land before any e2e
  timing claim on long prompts — a silent re-feed would poison timing AND
  parity.
- **C2/C3** (silent prompt truncation, hardcoded Qwen stop ids) must land
  before any chat/TTFT or "quality" claim cross-family.
- **S1** (de-hardcode bench/build paths) must land before any external party
  can reproduce a ratio — i.e. before the scorecard is published.
- **T1** (GPU nightly) should gate every scoreboard refresh.
- **P1** (batched prefill for non-Q4/Q8) before the P4 quant types are
  benchmarked (they would otherwise lose prefill cells for free).

---

## 16. One-page checklist for every "we beat llama.cpp" claim

- [ ] Both commits + model SHA256 + GPU/clocks recorded
- [ ] Oracle pinned; oracle default AND tuned columns present
- [ ] Equal KV precision (`-ctk/-ctv` matched), equal ctx (actual tokens
      printed), equal batch
- [ ] 5 medians post-warmup, cooldowns, all samples in jsonl, min/max shown
- [ ] Oracle samples not bimodal (else rerun cited)
- [ ] Same binary generation on both sides of the A/B
- [ ] Parity gate green for the config being timed (`verify.sh m61` etc.)
- [ ] Micros: per-iter sync, non-finite count zero, L2 status labeled
- [ ] Ratio computed by the scorecard harness (which refuses missing fields)
- [ ] If UNMEASURED anywhere, the table says UNMEASURED

Beat llama.cpp the only way that counts: same quant, same model, same box,
same commit discipline — and a scorecard anyone can rerun.
