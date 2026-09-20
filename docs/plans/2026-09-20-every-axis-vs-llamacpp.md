# Beat llama.cpp on every axis + full quant parity (2026-09-20)

Supersedes the decode-only framing of `docs/plans/2026-09-10-decode-1x-v3.md`
(that plan's objective is kept as Phase 1, not as the end state).

Reference: llama.cpp pinned at `3f545be` (`/home/mitesh/Storage/llama.cpp/build_cuda/bin/llama-cli`,
`DEFAULT_LLAMACPP_COMMIT` in `bench/bench_llm.py:44`). Box: RTX 3050 Laptop 4GB
(sm_86, no FP4 hardware), i5-11400H, 15GB RAM.

## TL;DR

1. **Decode at batch 1 is already won at short ctx** (up to 1.31x) and lost at
   long ctx (0.71-0.93x). Prefill/TTFT is the real hole: **0.06-0.24x**, i.e.
   end-to-end we are **5-7x slower** today. That is a code-path defect
   (`n < 32` prefill runs one full forward per token) plus an opt-in fast path
   that is 2.7-5.5x better and left disabled.
2. **Quant parity is a coverage program, not a benchmarking trick.** tinyinference
   consumes 13 tensor types; llama.cpp ships 43 (16 meaningful ones missing).
   tinyinference can also only *produce* one type (`quantize_row_q4_0`) and has
   no imatrix path, while llama.cpp's quantizer is the thing that defines
   "the same quant".
3. **"Every way" is only credible if it is auditable.** llama.cpp ships the
   oracles (`llama-bench`, `llama-batched-bench`, `llama-perplexity`,
   `llama-quantize`, `llama-imatrix`, `llama-server`), and every axis claim must
   be a ratio produced by the matching tool, on the same box, at a pinned commit.
   Anything not measured is marked UNMEASURED, not "won".
   **Blocker to clear first:** the local oracle build only has `llama-bench`,
   `llama-cli` and `llama-server` compiled
   (`/home/mitesh/Storage/llama.cpp/build_cuda/bin`); `llama-quantize`,
   `llama-imatrix`, `llama-perplexity` and `llama-batched-bench` must be built
   before the quantizer, quality and throughput rows can exist at all.

## Axis scorecard (what "every way" means, measured)

| Axis | Oracle tool | Status today | Verdict |
|---|---|---|---|
| Decode tok/s, batch 1, ctx 32-512 | `llama-bench -p 0 -n 64` | 1.05-1.31x (`docs/BENCHMARKS.md:12-40`) | **WON** |
| Decode tok/s, batch 1, ctx 2048 | `llama-bench -p 0 -n 64 -d 2048` | 0.71-0.93x | LOSE, fixable (KV traffic) |
| Prefill / TTFT | `llama-bench -p 512,2048` | **0.064-0.235x** | LOSE badly, fixable |
| End-to-end time-to-N tokens | `llama-cli` timing | **5-7x slower** at ctx 2048 | LOSE, follows prefill |
| Speculative decode | `llama-cli --spec-type ngram-simple` | off by default in both; ours unproven | UNMEASURED |
| Aggregate throughput (batch >1) | `llama-batched-bench` | not implemented (no continuous batching) | LOSE, large project |
| Tensor quant types | `llama-quantize` + load | 13 vs 43 types | LOSE, this plan |
| Quantizer / imatrix | `llama-imatrix` + `llama-quantize` | 1 type, no imatrix, no tool | LOSE, this plan |
| KV cache types | `-ctk/-ctv` | FP32/Q8/Q4(+F16 path); missing BF16, Q4_1, Q5_0, Q5_1, IQ4_NL | LOSE small |
| Model/arch coverage | any GGUF | ~15 families; no MLA, no GatedDeltaNet, no SSM, no LayerNorm+bias | LOSE (months) |
| Backends | any | CUDA (sm_86/89) + AVX2 CPU | LOSE (Vulkan/Metal/ROCm/SYCL/RPC) |
| Quality at equal size | `llama-perplexity` | no eval harness at all | LOSE, this plan |
| Load/warm-start time | wall clock | 1.18 GB/s memcpy-bound (`bench/RESULTS_S0_vs_S4_real_loader.md`) | LOSE, fixable |
| Energy per token | `nvidia-smi` power x time | unmeasured | UNMEASURED (winnable if tok/s wins) |
| Server features (OpenAI API, grammars, LoRA, slots) | `llama-server` | `examples/server_*.c` minimal | LOSE (months) |

Rules for this table: a row may be marked WON only with a ratio from the oracle
tool named in column 2, same GPU, same model file, same commit, 5 medians after
warmup with cooldowns. Thermal noise on this laptop is real (oracle samples
swung 136-374 tok/s inside one cell, `docs/plans/2026-09-10-decode-1x-v3.md:646-653`).

---

## Part A — "do the same quant llama is doing"

Two different obligations hide in that sentence. Both are required for the claim
to survive contact with a llama.cpp user:

- **A1 Consume**: load and compute on every GGUF tensor type llama.cpp can emit,
  at speed parity or better.
- **A2 Produce**: quantize F16/F32 models into the same ftypes, with the same
  per-tensor mix recipes and the same importance-matrix weighting, and prove it
  bit-for-bit where llama.cpp is deterministic.

### A1 coverage matrix

tinyinference today (`include/loader_gguf.h:13-26`, `include/dequant_ref.h:22-34`,
dispatch in `kernels/gemv_typed.cu:1264-1400` and `:1532-1595`):

```
F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K   (13)
```

llama.cpp `3f545be` (`ggml/include/ggml.h`, dequant in `ggml/src/ggml-quants.c`,
CUDA in `ggml/src/ggml-cuda/convert.cu` + `mmq.cu`/`mmvq.cu`): 43 enum values.
Missing from tinyinference, split by what they are actually for:

| Type | id | bpw | Why it matters on this box | Priority |
|---|---|---|---|---|
| `IQ4_XS` | 23 | 4.25 | Best quality/byte near Q4_K; smaller than Q4_K_M | **P0** |
| `IQ4_NL` | 20 | 4.50 | Also a legal **KV cache** type in llama.cpp | **P0** |
| `IQ3_XXS` | 18 | 3.06 | 3B-8B fits 4GB; GEMV prototype already exists | **P0** |
| `IQ3_S` | 21 | 3.44 | Better-quality IQ3 tier | **P0** |
| `IQ2_XS` | 17 | 2.31 | 7B-13B into 4GB (tight) | P1 |
| `IQ2_S` | 22 | 2.50 | Ditto, safer quality | P1 |
| `IQ2_XXS` | 16 | 2.06 | Extreme capacity tier | P1 |
| `IQ1_S` / `IQ1_M` | 19 / 29 | 1.56 / 1.75 | Capacity of last resort; quality cliff | P2 |
| `MXFP4` | 39 | 4.25 | Needed to open GPT-OSS-class GGUFs; sm_86 has no FP4 hw so it is dequant-then-FP16 math | **P1** |
| `Q1_0` / `Q2_0` | 41 / 42 | 1.125 / 2.25 | Emitted by current `llama-quantize` (`tools/quantize/quantize.cpp:35-36`) | P1 |
| `TQ1_0` / `TQ2_0` | 34 / 35 | 1.69 / 2.06 | Ternary tiers; CPU-oriented upstream | P2 |
| `NVFP4` | 40 | 4.0 | Needs Blackwell FP4 units; **skip on sm_86** | SKIP |
| `Q4_0_4_4/4_8/8_8`, `IQ4_NL_*` | 31-33, 36-38 | repack | CPU presentation formats, never stored in a GGUF file | SKIP |
| `Q8_K`, `Q8_1` | 15 / 9 | — | Intermediates (activation-side), not weights | SKIP |
| `I8/I16/I32/I64/F64` | 24-28 | — | Structural tensors, not compute | SKIP |

Reality check on the P0 set: it makes a 4GB laptop run 3B-8B models instead of
1B, but llama.cpp runs the same types, so support alone is **parity, not
advantage**. The advantage has to come from executing those types with our
kernels more efficiently than llama.cpp's `mmvq`/`mmq` at batch 1. The ggml side
of that duel must come from llama.cpp's own op-level benchmark
(`test-backend-ops --mode perf -o MUL_MAT`, target `test-backend-ops`), because
`tools/micro_gemv_llama.cu` only times **our** kernels at llama shapes
(M=8192 K=2048 / M=2048 K=8192 / V=128256 K=2048) and links no ggml code.
Recording both sides is Phase 0 work; every "we beat llama at equal quant" claim
depends on it.

### KV cache type parity

llama.cpp allows `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`
(`common/arg.cpp:304-314`). tinyinference has FP32 KV, Q8 KV, Q4 KV and a partial
F16 path (`include/qwen2_engine.h:162-185`). Missing: **BF16, Q4_1, Q5_0, Q5_1,
IQ4_NL**, and there is no user-facing `-ctk/-ctv` equivalent — the hybrid
threshold is an env var (`TT_QKV_THRESH`, `kernels/qwen2_cuda.cu:125-135`).
Target: one CLI/env knob that takes the same type names as llama.cpp, mapped onto
the KV paths, so any comparison can be run at equal KV precision.

### A1 onboarding recipe — one type at a time, 7 steps, no exceptions

This is the existing house pattern (Q8_KV, PLE, gemma4 all followed a version of
it). Run it per type; do not batch types across steps.

1. **Type code + size.** Add the `TTQ_*` code (`include/dequant_ref.h:22-34`) and
   the `size_bytes` entry (`src/loader_gguf.c:336-360`). Add the name to the
   type-name switch (`src/dequant_ref.c:421`) so `TT_DISPATCH` logging stays
   readable.
2. **Reference dequant.** Port `dequantize_row_<type>` from
   `ggml/src/ggml-quants.c` into `src/dequant_ref.c` (the file already states it
   is a port of the ggml references). Grid LUTs go in the same TU; `IQ3_XXS` is
   already written out in `tests/proto_iq3xxs.cu:56-140`, reuse it verbatim.
3. **Golden fixture.** Extend `tests/fixtures/gen_fixtures.py` (gguf-py) and add
   `tests/fixtures/gold_smollm_<TYPE>.npy`; the harness is
   `tests/test_dequant_golden.py` with the manifest in
   `tests/fixtures/dequant_gold_manifest.json`. Gate: max abs diff vs numpy
   golden < 1e-6.
4. **Kernels.** Add the type to `kernels/gemv_typed.cu` in the variants the
   dispatchers use: V2 and V4 for GEMV (`:1264-1400`) and for logits
   (`:1532-1595`), plus batch4 only if the type is a realistic speculative-verify
   candidate. I-quant decode needs the grid LUT in `__constant__` memory and the
   dequant-in-registers layout that `k_gemv_iq3_xxs` already prototypes
   (`tests/proto_iq3xxs.cu:239-300`).
5. **Engine routing.** Make sure every weight role can use it: qkv, oproj, ffn
   gate/up/down, lm head. Mixed-type models are the normal case (K-quant and
   I-quant "M" ftypes are per-tensor mixes), so the engine must not assume one
   type per model.
6. **Parity gate.** Extend the logits-vs-oracle grid (`tests/gate_m7_grid.py`,
   `tests/fixtures/grid_baseline.json`, driven by `./scripts/verify.sh m61`) so
   each new type gets a greedy-logit parity row against llama.cpp for the same
   file. No type ships without a passing parity row.
7. **Speed acceptance.** Record the type in a micro table (pattern:
   `bench/micro_gemv_rows.txt`) as GB/s and per-token us against the equivalent
   ggml kernel duel, and only then run it in the fleet scoreboard.

Cost estimate: ~0.5-1 day per simple type (Q1_0/Q2_0/MXFP4/TQ*: simple block
layouts), ~1.5-3 days per I-quant (grid LUT + two-level scales + sign nibbles),
plus the fixture/gate work which is shared. P0 set = roughly 1-2 weeks of
sustained work including gates.

### A2 — produce the same quant (quantizer + imatrix + quality)

This is the part that currently does not exist at all. tinyinference has
`quantize_row_q4_0` (`include/quant_ref.h:1-11`) and nothing else: no
`llama-quantize` equivalent, no ftype mix tables, no calibration, no perplexity
harness. llama.cpp's side (`tools/quantize/quantize.cpp`):
`--imatrix`, `--include-weights`/`--exclude-weights`, `--output-tensor-type`,
`--token-embedding-type`, `--tensor-type`/`--tensor-type-file`, `--pure`,
`--allow-requantize`, `--prune-layers`, `--keep-split`, `--override-kv`,
`--dry-run`, and the ftype list spanning Q1_0 .. F32 plus IQ*/TQ*/MXFP4_MOE.

Build order:

1. **`tt-quantize` (offline tool, C):** port the `quantize_row_*_reference`
   functions for the types already byte-stable upstream, then the ftype mix
   tables (Q4_K_M/S, Q5_K_M/S, Q3_K_S/M/L, Q2_K_S, IQ3_M, IQ2_M, IQ3_XS...), then
   the per-tensor overrides. Reuse the loader with write support instead of
   building a second GGUF writer.
2. **`tt-imatrix` (calibration):** run a calibration corpus through prefill,
   accumulate per-tensor activation importance (the `TT_DUMP_XN` hook in
   `kernels/qwen2_cuda.cu` is the existing pattern for reaching real hidden
   states), write the imatrix file with llama.cpp's metadata keys
   (`quantize.imatrix.file/dataset/entries_count/chunks_count`,
   `tools/quantize/quantize.cpp:77-80`) so files interoperate both ways.
3. **Quality harness:** perplexity + KL-divergence vs an F16/F32 baseline in the
   same shape as `llama-perplexity --kl-divergence`. Without this, "same quant"
   is unfalsifiable: two quantizers can be byte-different and both defensible,
   but only if quality is measured.

**The parity test that makes this objective:** for every ftype where llama.cpp's
quantizer is deterministic (no threads, no imatrix, reference path), quantize the
same F16 model with `llama-quantize` and with `tt-quantize`, then require
**byte-identical tensor payloads**. `quantize_row_q4_0` in this repo is already a
stated port of the ggml reference, so this is achievable for the Q* family and is
the cheapest possible definition of "the same quant llama is doing". For
imatrix/I-quant paths, where upstream fuses FMA ordering, require instead:
same ftype + same per-tensor type assignment + KL and perplexity within a
declared tolerance (e.g. ±0.5% ppl on the same corpus), and record both.

Scope honesty: quantizer + imatrix + quality parity is the single largest item in
this document (weeks, not days) and it is the item that can be *partially* faked.
The byte-identical test on the Q* family is the anti-faking device; if that test
is skipped, the claim becomes marketing.

---

## Part B — the performance axes (where the wins actually are)

Ordered by measured magnitude. Phase numbers match Part C.

**B1 Prefill cliff at `n < 32` (Phase 1, biggest single win, low risk).**
`kernels/qwen2_cuda.cu:6834-6866` routes `n >= 32` to batched GEMM and `n < 32` to
one full `advance()` forward per token. `bench/prefill_crossover.txt:8-16`: N=16 =
430 tok/s vs N=32 = 1322 vs fp16 N=128 = 7303. Every chat turn prefills only the
delta (`examples/chat_llm_gpu.c:235-271`), so interactive TTFT is 2.4 ms/token vs
oracle ~0.35. Fix: batch N in [2,31] (generalise the batch-K GEMV weight pass),
keep sequential only for N=1. Target: 4-8x on small N; chat TTFT <=10 ms at 0.5B.

**B2 Default-on the fp16/cuBLAS + FA2 prefill path (Phase 2).** Measured
2.7-5.5x (`bench/prefill_crossover.txt:19-22`; `docs/BENCHMARKS.md` flags row:
1416 -> 7736 tok/s). Greedy-match was verified 5/5 when it was written. Needs
per-arch/per-shape gating, a KV-parity gate (FP32 KV below the threshold), and a
documented fallback when cuBLAS is absent. Target: ctx2048 prefill ratio
0.076 -> >=0.55, which alone moves e2e from 5.8x behind to ~1.3x.

**B3 Long-ctx decode = KV traffic (Phase 3).** Finish the Q8-KV hybrid
(`docs/plans/2026-09-10-decode-1x-v3.md:52-73`; measured qwen3-2048 attn 11.2 ->
4.0 ms), then Q4-KV with per-head scales and a fused dequant-in-attention path so
scores are never materialised. Target: smol-2048 0.71 -> >1.0, llama-2048
0.85 -> >1.0, decode geomean > 1.05x. Fairness: also report the oracle at
`-ctk q8_0 -ctv q8_0`.

**B4 Speculative decode as a multiplier (Phase 7).** llama.cpp already ships
`ngram-simple`/`ngram-map-k`/`ngram-mod` and defaults them **off**
(`common/common.h:357-372`, `common/arg.cpp:4204-4277`, `:364`), so beating the
*default* oracle is trivial and worthless. The honest bar is
`llama-cli --spec-type ngram-map-k`. Four prerequisites in order:
(i) small-N verify (B1 gives most of it); (ii) batch4 GEMV made bandwidth-bound -
today it runs 16-30 GB/s vs 100-175 GB/s for the single-token V4 at the same
shapes (`bench/micro_gemv_rows.txt:30,45,52`); (iii) **device-side accept mask**,
one word D2H per pass instead of K x vocab floats
(`tools/spec_expA_e2e.c:110-116` copies ~2.4 MB plus a sync per verify at vocab
152k); (iv) a drafter at least as strong as upstream's - ours is a 2-3 token
window, last-occurrence scan, K <= 4 (`include/ngram_lookup.h:7-17`,
`src/ngram_lookup.c:4-32`) vs their 12-gram / 48-m-gram hashed map. Target:
>=1.5x e2e on 2 of 4 workloads and >=1.15x vs llama.cpp with
`--spec-type ngram-map-k`.

**B5 Aggregate throughput / continuous batching (Phase 8, optional).** Only if the
target metric becomes server tok/s. Oracle: `llama-batched-bench`. Weeks-long
project (shared KV arena, slot scheduler, batched attention) with a hard VRAM
ceiling on a 4GB laptop. Deferred until B1-B4 land.

---

## Part C — making "every axis" auditable

Two new artifacts, because a ratio nobody can reproduce is not a win:

1. **`bench/scorecard_vs_llamacpp.py`** - runs the oracle's *own* tools and ours on
   identical files, and writes `docs/SCORECARD_VS_LLAMACPP.md`:
   - decode/prefill by ctx and batch -> `llama-bench` vs `build/run_llm_gpu`
   - aggregate throughput -> `llama-batched-bench` vs ours (skipped until B5)
   - quality -> `llama-perplexity [--kl-divergence]` vs ours (needs A2 item 3)
   - quantizer -> `llama-quantize` vs `tt-quantize`, byte-diff report
   - load/warm-start -> wall clock both sides, cold and warm page cache
   - energy -> `nvidia-smi` power sampled per run, reported as mJ/token
   Every row records both commits, both configs (oracle **default** and oracle
   **tuned**), prompt token counts, 5 medians with min/max, clocks/power, and a
   `graph_status` column, following the existing field set in `bench/bench_llm.py`.
   Prerequisite: build the missing oracle tools against the same pinned commit
   (`cmake --build ... --target llama-quantize llama-imatrix llama-perplexity
   llama-batched-bench`) so every compared number comes from one tree.
2. **`docs/SCORECARD_VS_LLAMACPP.md`** - the axis table from the top of this
   document, generated. One rule: a row without an oracle ratio prints as
   UNMEASURED, never as a win.

### Phases

| Phase | Content | Exit criterion |
|---|---|---|
| P0 (1-2 d) | build missing oracle tools (llama-quantize, llama-imatrix, llama-perplexity, llama-batched-bench) + `test-backend-ops` for op-level duel; scorecard harness; record small-N prefill micro (`tools/micro_prefill_smalln.cu`) and ggml-vs-ours GEMV duel | every axis has a provenance-stamped baseline from the same oracle tree |
| P1 (2-4 d) | B1 small-N prefill | n<32 >=3x, chat TTFT <=10 ms @0.5B, m61 PASS |
| P2 (2-3 d) | B2 fp16/FA2 prefill default + gates | ctx2048 prefill >=0.55x, greedy-match unchanged |
| P3 (3-5 d) | B3 KV ladder to Q4 + fused attn | decode geomean >=1.05x, all primaries >=0.95x |
| P4 (3-4 d) | A1 P0 quant set: IQ4_XS, IQ4_NL, IQ3_XXS, IQ3_S | 4 types load + compute + parity + speed; a 3B runs in 4GB |
| P5 (1-2 wk) | A2 `tt-quantize` + per-tensor ftype mixes | byte-identical output for every deterministic ftype |
| P6 (1-2 wk) | A2 imatrix + perplexity/KL harness | ppl/KL within declared tolerance of `llama-perplexity` |
| P7 (1-2 wk) | B4 spec decode (verify kernel, device accept, real drafter) | >=1.5x on 2/4 workloads, >=1.15x vs `--spec-type ngram-map-k` |
| P8 (optional) | B5 batching; KV types BF16/Q4_1/Q5_0/Q5_1/IQ4_NL; load-time work | `llama-batched-bench` parity or better |

P4-P6 (quant) and P1-P3 (speed) are independent and can run in parallel; P7
depends on P1, P5 depends on P4's type plumbing.

### Gates and kill criteria

Gates green at every phase: `./scripts/ci_local.sh`, `./scripts/verify.sh m61`,
`m84`, `ple`, `tok`, `backfill`, plus `tests/test_dequant_golden.py`. Each new
tensor type adds a row to the oracle logit grid; **a type that cannot pass parity
does not ship, however fast it is**.

Kill criteria (when to stop claiming and report honestly):
- after P2, prefill still below 0.5x;
- after P3, long-ctx decode below 1.0x at equal KV precision;
- after P7, acceptance below 1.5 tokens/verify with verify cost ratio above 1.5;
- after P5, byte-identical quantizer output for under half the deterministic ftypes.

If two or more fire, the position is "parity at short ctx, behind at
prefill/long ctx" and the scorecard must say exactly that.

### Do-not-retry list (carried from v3, still binding)

Per-op kernel swaps are exhausted ("every op is on its micro-proven best kernel",
`docs/plans/2026-09-10-decode-1x-v3.md:680-684`); launch-removal fusions save ~0
under CUDA graph replay (add-norm fusion, fused SwiGLU V4, Q6 head v3, R=8
LM-head, PART/attn sweeps all NO-GO, `:40-48`, `:664-731`). Only traffic cuts
(weights, KV) and algorithmic multipliers (spec decode, type choice) move decode.
Micro harnesses must explicitly count non-finite outputs: the Q4 fused-attn bug
hid behind `max()` over NaNs for a whole session (`:708-716`). Never cite a single
noisy cell: v8's 25.4x was an OOB bug and oracle bimodality faked a 1.565x
(`:646-653`).


