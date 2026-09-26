# Code Review — tinyinference

Date: 2026-09-26 · Tree: `arena/01a0dcb6-tinyinference` @ `4037b7f`
Scope: every C/CUDA/Python source, headers, Makefile, CI workflows, test
harness, and benchmark tooling (~55k LOC across 15 `src/` files, 6 kernel
files, 40+ micro benches, 60+ test files). CPU-side gates were executed in
this review environment (`scripts/ci_local.sh`); CUDA code was reviewed
statically against its own micro results and parity gates.

Severity: **Critical** (wrong results likely) · **Major** (wrong results on
real paths, or blocks adoption) · **Minor** (edge paths, cost, drift risk) ·
**Nit** (hygiene).

---

## 1. Executive summary

This is an unusually well-run from-scratch inference engine. The things that
usually kill hobby GPU engines — unverified numerics, hand-waved benchmarks,
undocumented kernel thrash — are exactly the things this repo has institutional
answers for: golden-logit parity gates against a pinned llama.cpp oracle, a
kill-list that stops kernel-space thrash, a benchmark honesty audit that
retracted its own inflated claims (`data/profile/AUDIT_HONEST_2026-08-29.md`),
and micro harnesses that count non-finite outputs after a NaN-masked bug.
The GGUF loader is genuinely hardened against hostile input.

The defects found are real but concentrated in **error paths, app glue, and
structure** rather than in the numerics:

| Area | Grade | One-line assessment |
|---|---|---|
| Loader (`loader_gguf.c`) | A | Bounds-checked cursor, malloc-bomb caps, dup-name reject; one metadata-size nit |
| Quant/dequant reference (`dequant_ref.c`, `quant_ref.c`) | A- | Faithful ggml ports with golden fixtures; only Q4_0 quantizer exists |
| CUDA GEMV/GEMM kernels | A- | Hand-tuned, bit-exactness documented, honest micros; remaining levers correctly identified |
| Decode engine (`qwen2_cuda.cu`) | B | Correct and fast, but a 7.7k-line monolith with duplicated dispatch predicates |
| Prefill | B- | Small-N fixed (batchn, `pf_minn=2`), but non-Q4_0/Q8_0 dtypes prefill one row at a time |
| Samplers / chat template / specdec | A- | Deterministic, conformance-tested; two competing n-gram drafters |
| Tokenizer | A- | Full SP+BPE emulation with oracle gate; silent truncation API |
| Servers (`server_*.c`) | B | Honest P1 scope; hand-rolled JSON has documented limits |
| Build / CI | C+ | Works on exactly one machine layout; zero CI coverage for CUDA |
| Bench harness | A- | Best-in-class methodology; author-machine hardcodes block portability |
| Docs | A | Kill lists, plans, and scorecards that keep the project honest |

**Top findings to fix before the next round of "we beat llama.cpp" claims:**
C1 (prefill failure path corrupts KV), C2 (silent prompt truncation), C3
(hardcoded Qwen stop IDs cross-family), S1 (bench/build author-machine
assumptions), M1/M2 (monolith + env-var sprawl), T1 (no CUDA CI).

**Status update (2026-09-26, campaign drop 1):** C1 fixed (eager fallback
resumes at `eager_from`, with `TT_FAULT_INJECT_PF` hatch), C2 fixed
(`bpe_encode_ex` truncation flag; chat buffer sized to `MAX_CTX`), C3 fixed
(`tt_chat_stop_ids` per family, used by chat + both servers), C7 fixed
(`spec_llm_gpu` / `spec_expA_e2e` now wire the `tt_ngram_map` map drafter),
S1 fixed (bench scratch/oracle paths env-overridable; `LLAMA_CPP_DIR` for the
oracle target). P2 default-on (`TT_CUBLAS_FP16=0` opt-out + VRAM pre-check)
and P5 prefix-cache default-on (`TT_NO_PREFIX_CACHE=1` opt-out) landed with
the house kill-switch pattern. Remaining: M1/M2/T1/T3, C4-C6, C8-C10, P1,
P3-P6.

---

## 2. What is excellent (keep doing this)

1. **Loader hardening** (`src/loader_gguf.c:20-100`): `GCursor` bounds-checked
   cursor with an error latch; `MAX_TENSORS` and per-file-size caps against
   malloc bombs; overflow-checked shape products; duplicate tensor names
   rejected (not silently shadowed); truncated payload strict by default with
   `TT_GGUF_ALLOW_PARTIAL` opt-out. This is production-grade parsing.

2. **Bit-exactness as a contract.** Kernels document their equivalence to
   reference paths (`k_gemv_q4_0`: "V4 == V2 == scalar up to FMA order";
   `kernels/gemv_q4_cuda.cu:3006`: "same terms, same order, same rounding"),
   and the micro harnesses print `maxabs` + `nonfinite` counts
   (`bench/micro_batchn.txt`). After the NaN-masked Q4 fused-attn bug
   (`docs/plans/2026-09-10-decode-1x-v3.md:708-716`), this discipline is
   load-bearing.

3. **Do-not-retry lists.** `docs/plans/2026-09-10-decode-1x-v3.md` records
   killed kernel ideas with their reasons (launch-removal fusions save ~0
   under CUDA graph replay; register-pressure kills 4-row FFN variants).
   This prevents the #1 failure mode of perf projects: re-trying exhausted
   ideas.

4. **Samplers** (`src/samplers.c`): documented pipeline order matching
   llama.cpp conventions, sign-aware repeat penalty, caller-owned xorshift64*
   with a repaired zero seed, deterministic tie-breaks, `tt_sample_candidates`
   for speculative decoding, and a conformance JSONL
   (`tests/fixtures/sampler_conformance.jsonl`).

5. **KV cache management** (`src/kvcache.c`): device-pointer-agnostic plan
   language (offsets, not pointers), SWA compaction plans, session
   serialization with FNV-1a checksums, and memory accounting that mirrors
   the engine formula.

6. **Benchmark methodology** (`bench/bench_llm.py`): pinned oracle commit
   (`DEFAULT_LLAMACPP_COMMIT`, hard-fail if unpinned), model SHA256 recorded,
   5 medians post-warmup with cooldowns, graph/eager separated, both commits
   stamped, `--runs >= 5` enforced as an "M0 protocol". The 2026-08-29 honest
   audit shows the culture works: it caught L2-hot micros, truncation bugs in
   bench shapes, and single-sample thermal lies.

7. **Parity-first gating** (`scripts/verify.sh`, `tests/gate_m6_logit_parity.py`,
   golden logits in `data/golden/`, dequant goldens vs gguf-py fixtures): a
   kernel that fails parity does not ship, however fast.

8. **Comment culture in kernels.** The q4_0 nibble-pairing history
   (`kernels/gemv_q4_cuda.cu:1-8`) documents *why* the old pairing was wrong
   and what garbage it produced. Future readers will not re-introduce it.

---

## 3. Correctness findings

### C1 (Major) — prefill failure fallback re-feeds the whole prompt
`kernels/qwen2_cuda.cu:6941-6978` (`qwen2_engine_prefill`). The chunked
batched loop calls `prefill_batched_gemm` per chunk; on the first non-zero
return it sets `failed = 1` and breaks. Earlier chunks have already advanced
`e->pos` and written KV slots. The fallback then runs
`for (int i = 0; i < rem_n; i++) advance(e, rem_toks[i]);` — **from token 0**,
not from `offset`. If a mid-stream failure ever happens (arena cudaMalloc,
a GEMM rc), the prompt prefix is duplicated into the KV cache and every
subsequent position is shifted. Fix: resume at `offset`, and only if
`prefill_batched_gemm` guarantees no state mutation before its failure point.
Add a test that injects a chunk failure.

### C2 (Major) — `bpe_encode` truncates silently
`src/tokenizer_bpe.c:1115-1195`: the encode loop is
`while (pos < len && n_out < max_tokens)` and returns `n_out` — a full buffer
and a truncated prompt are indistinguishable to callers. `examples/chat_llm_gpu.c`
pairs this with `int prompt_tokens[512]` on the stack (`examples/chat_llm_gpu.c:255`), so any turn
that tokenizes to >512 ids is silently cut mid-prompt. Fix: negative sentinel
(or out-param flag) on truncation; size the chat buffer from `MAX_CTX`; report
truncation to the user. Add a unit test.

### C3 (Major) — hardcoded Qwen stop-token ids applied to every family
`examples/chat_llm_gpu.c:313-314`: the decode loop stops on
`next_tok == tok->eos_id || next_tok == 151643 || next_tok == 151645`.
The literals are Qwen `endoftext` / `im_end` ids, evaluated for every family.
On Llama-3.2 the real eot is 128009 and 151643/151645 are ordinary mid-vocab
tokens that can legitimately appear in generated text — output would be cut
mid-sentence. The stop-string machinery partially compensates, but id checks
must come from the tokenizer (`tok->eos_id`, `eot_id`) per family. Also
`gemma`'s 106 `end_of_turn` is handled only via stop strings.

### C4 (Minor) — GGUF metadata array element-size math is wrong for some types
`src/loader_gguf.c:226-235` (dim-key array branch): `esz` is computed as
1 for itype<=1 and 7, 4 for itype 2..6, 8 otherwise. But `kv_fixed_size`
(correctly) says u16/i16 (types 2,3) are 2 bytes. A u16-array metadata key
matching a `dim_keys` name would advance the cursor by the wrong stride and
desynchronize the rest of the metadata parse (recovered only by the error
latch eventually). The branch also does not reject nested-array or string
item types the way `skip_kv_value` does. Fix: route through `kv_fixed_size`
and reject item types it returns 0 for.

### C5 (Minor) — dispatch predicates duplicated: engine vs graph-slot key
`kernels/qwen2_cuda.cu:4018-4065` (`graph_slot_key_at`) states it "mirrors
forward_layers exactly" (attention branch numbering 1-8, split-S formulas,
env kill switches) and `forward_layers` (~:5290-5350) independently re-derives
the same choices. If one side changes and the other does not, a key hit will
replay a graph whose baked launch params no longer match the eager choice —
silently wrong (e.g. attention reads the wrong KV format). The key also calls
`getenv()` per decode step while sibling predicates cache statically.
Fix: one `tt_attn_dispatch_plan(e, pos)` function returning the branch + S,
consumed by both capture and eager launch.

### C6 (Minor) — graph-slot exhaustion is expensive, not graceful
`kernels/qwen2_cuda.cu:7395-7413`: when `graph_slot_used >= TT_GRAPH_SLOTS`
(16) the freshly instantiated graph is destroyed and `-1` returned.
`qwen2_engine_next` (~:7452) only handles `crc == -2`; `-1` is ignored, so
each *new* dispatch key (split-S is ctx-bucketed, changing every 32-64
tokens of decode) costs a full stream-capture + `cudaGraphInstantiate` +
destroy, per new key, forever. Mitigation is that regimes are usually few;
still, implement LRU eviction (or latch capture off after N failures) before
long-running server use.

### C7 (Minor) — two N-gram drafters; the demo wires the weaker one
`src/ngram_lookup.c` (linear last-occurrence scan, window 2-3, used by
`examples/spec_llm_gpu.c`) vs `src/specdec.c` (`tt_ngram`: ring buffer,
default 12-gram, bounded history, tested by `tests/test_specdec_sim.py`).
specdec.c is the better and better-tested implementation, but the speculative
demo and `tests/test_spec_verify.c` path use ngram_lookup. Consolidate on
`tt_ngram` and delete the duplicate.

### C8 (Nit) — kvcache slab layout contract enforced only by comment
`src/kvcache.h:30-40`: the CPU management layer "mirrors
kernels/qwen2_cuda.cu exactly" — two sources of truth for slab geometry,
reconciled by human care. Before `kvcache.c` is engine-integrated (its stated
next step), factor the layout formula into one header both sides include.

### C9 (Nit) — duplicate declaration
`kernels/qwen2_cuda.cu:73-83`: `tt_logits_dispatch` is declared twice in the
same `extern "C"` block. Legal C, sloppy.

### C10 (Nit) — `qwen2_engine_create` does not check `cudaStreamCreate`
`kernels/qwen2_cuda.cu:4171` region: failure leaves `e->stream` indeterminate;
later launches would silently run on the default stream. The `CK_CREATE`
macro exists — use it here too.

---

## 4. Performance findings

### P1 (Major, known gap) — non-Q4_0/Q8_0 weights have no batched prefill
`kernels/qwen2_cuda.cu:6228-6245` (`prefill_gemm_dtype`): Q4_0 routes to
batchn / cuBLAS / Q4 GEMM; Q8_0 routes to `tt_gemm_q8_0_prefill`; **everything
else** (F16, BF16, Q4_1, Q5_0/1, Q2_K..Q6_K) runs
`for (i = 0; i < n; i++) tt_gemv_layer_dispatch(...)` — one GEMV per row per
matmul. Any K-quant or F16 model therefore prefill at per-token speed
regardless of N. This is the dominant prefill hole for the quant-parity
program (A1 in the llama.cpp plan) — new types will be slow by default.
Fix: generalize the batchn weight-pass trick (already proven for q4_0,
`k_gemv_q4_0_batchn`, `bench/micro_batchn.txt`) to Q8_0/K-quants, or
dequant-to-F16-shadow once and reuse the WMMA/cuBLAS path per tensor.

### P2 (Major, known gap) — tensor-core prefill GEMM still opt-in
`TT_CUBLAS_FP16` / `TT_CUBLAS_PRE` are env-opt-in (`cublas_fp16_wanted()`,
`kernels/qwen2_cuda.cu:5906-5910`) though measured 2.7-5.5x over the Q4 GEMM
(`bench/prefill_crossover.txt`). Meanwhile FA2 prefill is now default-ON
(`tt_fa2_pre_on`, disabled only by `TT_FA2_PRE=0`) and WMMA auto-enables for
chunks >= 64 (HEAD commit). The remaining work is: default-on with per-shape
gating, a KV-parity gate (FP32 KV below the hybrid threshold keeps logit
gates green), a documented fallback when cuBLAS is absent, and shadow-VRAM
budgeting (fp16 shadows of a 1B model are ~1GB — fine on 4GB only when KV is
Q4/Q8).

### P3 (Minor) — per-token sync H2D in eager `advance()`
`kernels/qwen2_cuda.cu:5868-5880`: a synchronous `cudaMemcpy` of 4 bytes per
token. The comment explains why (pageable async H2D of mutable host memory
read the already-incremented pos — a real bug they fixed). Better: pinned
staging + async copy on the stream, or keep pos device-resident in eager mode
(the graph path already does, `k_pos_inc_recent`). Spec verify N=4 pays N of
these syncs per verify.

### P4 (Minor) — full-vocab logits D2H per token in the chat loop
`examples/chat_llm_gpu.c:301-302`: `qwen2_debug_copy_logits` copies
`vocab * 4` bytes (600 KB at vocab 152k) D2H + a sync each token so the host
sampler can run. Correct for a frontend; but `server_minimal.c` and any
throughput path should sample on device (the greedy/gumbel kernels already
exist in the graph path) and stream only the chosen token + detokenized bytes.

### P5 (Minor) — env-var reads in hot dispatch
`graph_slot_key_at` calls `getenv("TT_NO_UNFUSED")` etc. on every lookup
(`kernels/qwen2_cuda.cu:4025-4029`); `qwen2_engine_prefill` calls
`getenv("TT_PREFIX_CACHE")` per call. Sibling code caches statically. Not a
measurable bottleneck at 1 decode step/lookup, but parse once into a config
struct (see M2).

### P6 (Minor) — Q6_K lm_head remains the long pole on llama-1B
Per `docs/plans/2026-09-10-decode-1x-v3.md` the head is already on its
micro-proven kernel (V2) and is bandwidth-bound; requant-to-Q4 head is
deliberately frozen until llama clears 0.90x. No action now — do not "fix"
this without the documented precondition; note that a batch4 verify head
(`tt_logits_q4_0_batch4`) already exists for the spec path and shares the
right single-weight-pass pattern.

---

## 5. Robustness / security findings

### S1 (Major for adoption) — author-machine assumptions crash the tooling
- `bench/bench_llm.py:39-40`: `SCRATCH = Path("/home/mitesh/ti-scratch")`
  with a module-level `mkdir` — **import fails** on any machine without that
  path's permissions. Default oracle binary is also hardcoded to
  `/home/mitesh/Storage/llama.cpp/...`.
- `Makefile`: `NVCC ?= $(HOME)/mmcuda/bin/nvcc`, CUDA/cuBLAS includes from
  `~/.local/lib/python3.12/site-packages/nvidia/...`, and the `oracle_logits`
  target links `/home/mitesh/Storage/llama.cpp` directly.
Fix: derive from `nvcc` on PATH / `CUDA_HOME`; make scratch and oracle paths
CLI/env args with lazy creation; keep the pinning *policy* (that part is
excellent) while removing the hardcoded geography.

### S2 (Positive + Minor nit) — loader threat model is handled
Hostile GGUF input is the repo's one true attack surface (models come from the
internet). The parser rejects: bad magic, truncated headers, oversized
tensor/kv counts, nested metadata arrays, ndim > 4, overflowing shapes,
K-quant numel not divisible by 256, unknown dtypes, duplicate names, and
out-of-bounds tensor windows (strict by default). Remaining nit: metadata
keys longer than 127 bytes are truncated by `read_string` (values are
consumed correctly, so it's cosmetic). Consider a libFuzzer target on
`gguf_load` — given C4, fuzzing would pay for itself.

### S3 (Minor) — `server_minimal.c` scope-appropriate, a few gaps
Good: 1 MiB body cap, Content-Length parsing with caps, engine serialized by
one mutex (documented N=1 constraint), default bind 127.0.0.1, hand-rolled
JSON with a documented escape subset and explicit limits. Gaps: `strcasestr`
is a GNU extension (needs `_GNU_SOURCE` guard for musl/BSD builds); no socket
read timeout (a stalled client pins a thread + the engine mutex holder can be
starved if lock ordering ever changes); `MAX_CONTENT` 4096/message is small
for OAI-compatible clients (document it in the API response or raise it);
no auth — acceptable for localhost, but say so in `scripts/serve_minimal.sh`
and refuse non-loopback binds without an explicit `TT_ALLOW_REMOTE=1`.

### S4 (Minor) — unchecked CUDA calls on engine teardown paths
Create paths use `CK_CREATE`; free paths and a few mid-flight calls (e.g.
`cudaStreamCreate` at create, `cudaMemcpy`s in `reset`/`rewind`) ignore
errors. Low risk; keep one `TT_CUDA_CHECK` macro and use it in non-hot paths.

---

## 6. Build / CI findings

### T1 (Major) — zero CI coverage for the CUDA code that wins benchmarks
`.github/workflows/ci.yml` is honest about it: `cuda-conditional` is
`if: false` pending a self-hosted runner. The kernels (3.1k + 7.7k + 1.6k
lines) are protected only by manual `scripts/verify.sh` runs. Given the
project's own history (a nibble-pairing bug and a NaN-masked fused-attn bug
both reached the engine), a self-hosted GPU runner running
`verify.sh m61 ple backfill` + a 6-cell smoke bench nightly is the highest-
value CI investment available. At minimum, run the CPU-side golden tests
(`tests/test_dequant_golden.py`, `test_quant_ref`) in CI — they validate the
reference math the GPU kernels claim to match.

### T2 (Minor) — compiler warnings hide future real ones
Under `-Wall -Wextra -std=c11`:
- `src/async_printer.c:53`: implicit declaration of `nanosleep` (needs
  `#define _POSIX_C_SOURCE 200809L` before includes — `cpu_backend.c:25`
  shows the pattern).
- `src/arch_registry.c:51`: `-Wmissing-field-initializers` on every
  `kArchTable` entry (TTraits grew fields). Use designated initializers
  `{ .rope = ROPE_NEOX, ... }`.
CI currently compiles without `-Wall` in `build-gcc` — turn warnings into
errors there once clean.

### T3 (Minor) — `ci_local.sh` goes RED without numpy
`tests/test_cpu_backend.py` imports numpy unconditionally; the GitHub job
installs it, bare checkouts do not (reproduced here: the only RED in an
otherwise green run). Either gate the import (skip synthetic golden parts)
or add a requirements note to README.

### T4 (Nit) — Makefile CUDA include discovery
`CUDA_INC` defaults into a pip site-packages nvidia include dir. Prefer
`$(NVCC)`-adjacent toolkit includes (`$(dir $(NVCC))/../include`) with the
current value as fallback.

---

## 7. Maintainability findings

### M1 (Major) — `kernels/qwen2_cuda.cu` is a 7,729-line monolith
One TU contains: ~40 kernels (norms, RoPE x8, KV scatter x5, flash attention
x10+, prefill flash x5, argmax, embed, sampling), the engine struct, eager
forward, batched prefill, speculative verify, CUDA graph manager, cuBLAS
shadow builders (fp32 + fp16), profiling stages, and env-var parsing. This is
the root cause of C5 (duplicated dispatch) and the main onboarding risk.
Split by concern — `kernels/attn/`, `kernels/kv/`, `kernels/prefill/`,
`src/engine_*.c` for host logic — with the dispatch plan (C5 fix) as the
seam. The file's section comments are good; the file boundaries should match
them.

### M2 (Major) — env-var sprawl with three caching disciplines
30+ `TT_*` knobs: some cached in `static` locals on first use, some read per
call, some per graph-key evaluation. No single document lists them (README
shows a handful; the rest live in code comments). Two changes: (1) a parse-
once `TTEnv` struct populated in `qwen2_engine_create`, read everywhere else;
(2) `docs/ENV_VARS.md` generated from the parse table so the knob surface is
discoverable. The knobs themselves are a *feature* (kill switches saved the
fused-attn rollout); they need one front door.

### M3 (Minor) — duplicated constants and near-duplicated hot paths
- Quant type codes duplicated: `include/dequant_ref.h` enum vs
  `src/cpu_backend.c:36-43` `#define TTQ_Q4_0 2` "duplicated to stay
  header-only" (the header is already includable; drop the defines).
- `BlockQ4_0` defined in both `loader_gguf.h` and `gemv_q4_cuda.cu`;
  `BlockQ8_0`/`BlockQ8KV` in `qwen2_cuda.cu` and elsewhere. One shared
  `quant_blocks.h`.
- `prefill_batched_gemm_dx` is a deliberate standalone copy of
  `prefill_batched_gemm` (its comment says so) to avoid touching the
  verified path — fine as a transitional tactic, but the two will drift;
  parameterize the output sink instead.

### M4 (Nit) — stale version labels in kernel comments
`kernels/gemv_q4_cuda.cu:36-41`: "V2: ... Same nb-even contract as V2" is
self-referential (should reference the original 1-row/warp variant). The
naming (V2/V4/R8/batch4/batchn/kt4) is fine; the comments need a glossary
in the file header.

### M5 (Nit) — three chat frontends
`chat` (bash launcher), `chat.py`, `real_chat.py`, plus `chat_llm_gpu.c`.
Each has its own defaults (e.g. chat.py sampling knobs vs the C binary's
env knobs). Consolidate defaults and document which frontend is canonical
(README already prefers `./chat`).

---

## 8. Testing & verification assessment

**Strong:**
- Greedy-logit parity vs pinned llama.cpp oracle (`gate_m6_logit_parity.py`,
  `gate_m7_grid.py` + `grid_baseline.json`), per-family golden logits in
  `data/golden/`.
- Dequant goldens: gguf-py generated `.npy` fixtures for 12 quant types with
  1e-6 max-abs gate (`test_dequant_golden.py`).
- Sampler conformance JSONL; chat-template goldens; specdec simulator with
  adversarial corpora (incl. false-positive RAG corpus); kvcache
  serialize/round-trip/compaction unit tests; loader hardening test
  (`test_loader_hardening.c`).
- GPU gates with bit-exactness claims where they matter (`test_spec_verify.c`
  compares verify vs sequential step_logits; `test_q4_split_exact.c`;
  `test_prefill_layer_parity.c`).

**Gaps (ordered by value):**
1. No failure-injection tests for error paths (C1 is exactly the untested
   path). A `TT_FAULT_INJECT` env that makes `prefill_batched_gemm` fail at
   chunk k would have caught it.
2. No GPU coverage in CI (T1).
3. No fuzz target for `gguf_load` despite the hardened parser (S2).
4. No test pinning `bpe_encode` truncation semantics (C2).
5. `tests/fixtures/*.npy` are numpy-only artifacts; the gate that consumes
   them is not in CI (needs numpy + no GPU — should be).
6. Bench regression tracking exists as a CI snapshot placeholder
   (`benchmark-snapshot` writes an empty `results: []`) — wire it to a real
   CPU-side micro (e.g. `cpu_backend` GB/s) so trend data accumulates.

---

## 9. Area walkthrough notes (things that are fine, recorded for the record)

- **`src/loader_gguf.c`** — suffix-based metadata key matching (`.embedding_length`
  after the first dot) is the right call for cross-family GGUFs; the u32-array
  dim-key fallback for gemma4 mirrors reality; defaults (eps 1e-6, rope 10000,
  kv_heads=heads, max_seq_len 2048) match ggml.
- **`kernels/gemv_q4_cuda.cu`** — the unaligned 18-byte block streaming via
  word loads + `__byte_perm` merge with the "last block is odd-b under nb-even"
  no-OOB proof is careful work. `k_gemv_q4_0_batchn` templates (nr=1/2/4)
  share the weight pass across N — this is the pattern to extend to other
  dtypes (P1).
- **`src/samplers.c`** — quickselect with duplicate-collapsing for top-k and
  deterministic candidate ordering is well above average. The header's
  workbuf-size query API (`tt_sampler_workbuf_size`) avoids magic numbers.
- **`src/chat_template.c`** — history accumulation + `add_generation_prompt`
  flag + stop-string per family; the chat app's prefix-cursor logic
  (snapshot without generation prompt) is subtle but correct per its comments.
- **`src/tokenizer_bpe.c`** — byte-unicode mapping tables, pre-tokenizer
  splitters per convention (GPT-2 / llama3 / ggml), special-token longest-
  match, BOS rules per mode. The oracle gate (`gate_tokenizer.py`) is the
  right way to keep this honest. (Except C2.)
- **`src/moe_router.c`** — deterministic top-k ties, sum clamp at 2^-14
  matching `llama-graph.cpp`; execution integration still pending (documented).
- **`examples/spec_llm_gpu.c` + `tools/spec_expA_*`** — the honest framing
  (device accept mask still missing, K*vocab D2H per verify) matches the plan.
- **`docs/`** — plans are dated, supersession is explicit, numbers carry
  provenance. The `every-axis-vs-llamacpp` plan's "anything not measured is
  UNMEASURED, not won" rule is the single best sentence in the repo.

---

## 10. Recommended fix order

| # | Item | Why first |
|---|---|---|
| 1 | C1 prefill fallback resume + fault-injection test | Correctness on real error path |
| 2 | C2 bpe_encode truncation signal + chat buffer sizing | Silent data loss today |
| 3 | C3 family-correct stop ids | Wrong output termination cross-family |
| 4 | S1 de-hardcode bench/build paths | Unblocks any external reproduction of claims |
| 5 | C5/C6 dispatch-plan single source + graph slot eviction | Removes the silent-wrong-replay risk |
| 6 | P2 default-on tensor-core prefill with gates | Biggest remaining measured win (2.7-5.5x) |
| 7 | P1 batched prefill for non-Q4/Q8 dtypes | Required before quant-parity program lands |
| 8 | M2 env config struct + `docs/ENV_VARS.md` | Cheap, helps every future change |
| 9 | T1 GPU nightly (self-hosted) + dequant goldens in CI | Protects the crown jewels |
| 10 | M1 split the monolith | Structural; do alongside P1/P2 work, not before |

The performance items are sequenced for the llama.cpp campaign in
`docs/BEATING_LLAMACPP.md`, which treats this review's items as preconditions
where they gate claims.
