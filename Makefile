CFLAGS  ?= -O3 -fopenmp -Wall -Wextra -std=c11 -fPIC -DTT_IN_LIB
# Opt-in AVX2/FMA TU only (scalar default keeps ARM/old-x86 building):
#   make lib AVX2_FLAGS="-O3 -mavx2 -mfma"
AVX2_FLAGS ?= -O3 -mavx2 -mfma -fopenmp -Wall -Wextra -std=c11 -fPIC -DTT_IN_LIB
BUILD   := build
SRCS    := $(wildcard src/*.c)
HDRS    := $(wildcard include/*.h)

# cpu_backend.c kernels carry target("avx2,fma") + runtime cb_using_avx2() dispatch,
# so it is the one TU that benefits from global AVX2 codegen; everything else stays portable.
$(BUILD)/cpu_backend_avx2.o: src/cpu_backend.c $(HDRS) | $(BUILD)
	$(CC) $(AVX2_FLAGS) -Iinclude -c -o $@ $<
$(BUILD)/libtinytorch.so: $(filter-out src/cpu_backend.c,$(SRCS)) $(HDRS) $(BUILD)/cpu_backend_avx2.o | $(BUILD)
	$(CC) $(CFLAGS) -Iinclude -shared -o $@ $(filter-out src/cpu_backend.c,$(SRCS)) $(BUILD)/cpu_backend_avx2.o -lm

$(BUILD):
	mkdir -p $(BUILD)

lib: $(BUILD)/libtinytorch.so

clean:
	rm -rf $(BUILD)

.PHONY: lib clean

# Configurable CUDA archs + toolchain (defaults preserve current behavior):
#   make cuda CUDA_ARCHS="86 89 90" NVCC=/usr/local/cuda/bin/nvcc CUDA_INC_OVERRIDE=/usr/local/cuda/include
CUDA_ARCHS ?= 86 89
NVCC ?= $(HOME)/mmcuda/bin/nvcc
CUDA_INC := $(if $(CUDA_INC_OVERRIDE),$(CUDA_INC_OVERRIDE),$(HOME)/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/include)
NVCC_GENCODE := $(foreach a,$(CUDA_ARCHS),-gencode arch=compute_$(a),code=sm_$(a))

$(BUILD)/libtinytorch_cuda.so: kernels/gemm_cuda.cu kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -shared -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ kernels/gemm_cuda.cu kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -Lbuild -lcudart

cuda: $(BUILD)/libtinytorch_cuda.so

.PHONY: cuda

CUBLAS_INC := $(HOME)/.local/lib/python3.12/site-packages/nvidia/cublas/include
CUBLAS_LIB := $(HOME)/.local/lib/python3.12/site-packages/nvidia/cublas/lib

$(BUILD)/libtt_cublas.so: kernels/cublas_ref.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -I$(CUBLAS_INC) -shared -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib:$(CUBLAS_LIB) \
	  -o $@ kernels/cublas_ref.cu \
	  -Lbuild -L$(CUBLAS_LIB) -lcudart -lcublas

cublas: $(BUILD)/libtt_cublas.so

.PHONY: cublas

$(BUILD)/run_llm_gpu: examples/run_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/async_printer.c src/chat_template.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/run_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/async_printer.c src/chat_template.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

run_llm_gpu: $(BUILD)/run_llm_gpu

$(BUILD)/chat_llm_gpu: examples/chat_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/async_printer.c src/chat_template.c src/samplers.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/chat_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/async_printer.c src/chat_template.c src/samplers.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

chat_llm_gpu: $(BUILD)/chat_llm_gpu

$(BUILD)/server_minimal: examples/server_minimal.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/chat_template.c src/samplers.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/server_minimal.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/chat_template.c src/samplers.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

server_minimal: $(BUILD)/server_minimal

# Universal Speculative Engine orchestrator: N-gram drafter (host) +
# batched verify_speculative() (CUDA). Source list mirrors run_llm_gpu
# plus src/ngram_lookup.c.
$(BUILD)/spec_llm_gpu: examples/spec_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/async_printer.c src/ngram_lookup.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  examples/spec_llm_gpu.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/async_printer.c src/ngram_lookup.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

spec_llm_gpu: $(BUILD)/spec_llm_gpu

.PHONY: spec_llm_gpu
$(BUILD)/spec_expA_ab: tools/spec_expA_ab.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/spec_expA_ab.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

$(BUILD)/spec_expA_e2e: tools/spec_expA_e2e.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/ngram_lookup.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/spec_expA_e2e.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c src/tokenizer_bpe.c src/ngram_lookup.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

# Oracle logits tool against the vendored llama.cpp build (parity fixtures).
$(BUILD)/oracle_logits: tools/oracle_logits.c | $(BUILD)
	gcc -O2 -Wno-deprecated-declarations \
	  -I /home/mitesh/Storage/llama.cpp/include -I /home/mitesh/Storage/llama.cpp/ggml/include \
	  -o $@ tools/oracle_logits.c \
	  -L /home/mitesh/Storage/llama.cpp/build_cuda/bin -lllama \
	  -Wl,-rpath=/home/mitesh/Storage/llama.cpp/build_cuda/bin

oracle_logits: $(BUILD)/oracle_logits

.PHONY: run_llm_gpu chat_llm_gpu

$(BUILD)/dump_logits: tools/dump_logits.c src/loader_gguf.c src/dequant_ref.c src/cpu_backend.c src/arch_registry.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/dump_logits.c src/loader_gguf.c src/dequant_ref.c src/cpu_backend.c src/arch_registry.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

dump_logits: $(BUILD)/dump_logits

.PHONY: dump_logits

$(BUILD)/bench_prefill: tools/bench_prefill.c src/loader_gguf.c src/dequant_ref.c src/arch_registry.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/bench_prefill.c src/loader_gguf.c src/dequant_ref.c src/arch_registry.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

bench_prefill: $(BUILD)/bench_prefill

.PHONY: bench_prefill

$(BUILD)/profile_step: tools/profile_step.cu src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler "-fPIC -fopenmp" \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/profile_step.cu src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lgomp

profile_step: $(BUILD)/profile_step

.PHONY: profile_step

# CPU golden dequant CLI (validates GGUF quant formats against gguf-py goldens)
$(BUILD)/dequant_ref: src/dequant_ref.c src/loader_gguf.c include/dequant_ref.h include/loader_gguf.h | $(BUILD)
	gcc $(CFLAGS) -DTTQ_MAIN -Iinclude -o $@ src/dequant_ref.c src/loader_gguf.c -lm

dequant_ref: $(BUILD)/dequant_ref

.PHONY: dequant_ref
$(BUILD)/bench_ipc: tools/bench_ipc_throughput.c src/tinytorch_ipc.c include/tinytorch_ipc.h | $(BUILD)
	$(CC) $(CFLAGS) -Iinclude -o $@ tools/bench_ipc_throughput.c src/tinytorch_ipc.c -lpthread -lrt

bench_ipc: $(BUILD)/bench_ipc

.PHONY: bench_ipc

# M7 task 2: GPU golden GEMV grid for all Tier-1 quant types
$(BUILD)/test_gemv_typed: tools/test_gemv_typed.cu src/loader_gguf.c src/dequant_ref.c \
		kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tools/test_gemv_typed.cu src/loader_gguf.c src/dequant_ref.c \
	  kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_gemv_typed: $(BUILD)/test_gemv_typed

.PHONY: test_gemv_typed

# Speculative-decode verify test: compares batched verify(N) logits against
# N sequential single-token forwards (bit-exact on qwen2.5-0.5b-q4_0).
$(BUILD)/test_spec_verify: tests/test_spec_verify.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_spec_verify.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c \
	  kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread

test_spec_verify: $(BUILD)/test_spec_verify

.PHONY: test_spec_verify

# M10+ True Batched-4 GEMV (q4_0 + q8_0) randomized correctness test.
# Compares tt_gemv_q4_0_batch4 against 4 sequential tt_gemv_q4_0 calls
# across engine-relevant shapes; same for q8_0.
$(BUILD)/test_batch4_gemv: tests/test_batch4_gemv.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_batch4_gemv.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_batch4_gemv: $(BUILD)/test_batch4_gemv

.PHONY: test_batch4_gemv

# Q8_0 V4 LM Head Launcher bit-exact correctness test.
$(BUILD)/test_logits_q8_v4: tests/test_logits_q8_v4.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_logits_q8_v4.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_logits_q8_v4: $(BUILD)/test_logits_q8_v4

.PHONY: test_logits_q8_v4

# Q8_0 KV Cache Scatter and Flash Attention correctness test.
$(BUILD)/test_q8_kvcache: tests/test_q8_kvcache.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_q8_kvcache.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_q8_kvcache: $(BUILD)/test_q8_kvcache

.PHONY: test_q8_kvcache

$(BUILD)/test_q4_kvcache: tests/test_q4_kvcache.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_q4_kvcache.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_q4_kvcache: $(BUILD)/test_q4_kvcache

.PHONY: test_q4_kvcache

$(BUILD)/test_q4_split_exact: tests/test_q4_split_exact.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_q4_split_exact.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_q4_split_exact: $(BUILD)/test_q4_split_exact

.PHONY: test_q4_split_exact

$(BUILD)/test_qcache_backfill: tests/test_qcache_backfill.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_qcache_backfill.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_qcache_backfill: $(BUILD)/test_qcache_backfill

.PHONY: test_qcache_backfill

# Batched Q4_0 Prefill GEMM multi-boundary correctness test.
$(BUILD)/test_prefill_gemm: tests/test_prefill_gemm.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Xcompiler -fPIC -Xcompiler -fopenmp \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_prefill_gemm.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_prefill_gemm: $(BUILD)/test_prefill_gemm

.PHONY: test_prefill_gemm

# Tensor Core WMMA Q4_0 Prefill GEMM multi-boundary correctness test.
$(BUILD)/test_wmma_prefill_gemm: tests/test_wmma_prefill_gemm.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Xcompiler -fPIC -Xcompiler -fopenmp \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_wmma_prefill_gemm.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_wmma_prefill_gemm: $(BUILD)/test_wmma_prefill_gemm

.PHONY: test_wmma_prefill_gemm

# Layer-0 Parity Diagnostic Test: compares batched prefill GEMM against sequential advance for N=32.
$(BUILD)/test_prefill_layer_parity: tests/test_prefill_layer_parity.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu | $(BUILD)
	$(NVCC) -O3 $(NVCC_GENCODE) \
	  -I$(CUDA_INC) -Iinclude -Isrc -Xcompiler -fPIC \
	  -Xlinker -rpath=$(CURDIR)/build:$(HOME)/mmcuda/lib \
	  -o $@ \
	  tests/test_prefill_layer_parity.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/cpu_backend.c \
	  kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu \
	  -L$(HOME)/mmcuda/lib -lcudart -lpthread -lm

test_prefill_layer_parity: $(BUILD)/test_prefill_layer_parity

.PHONY: test_prefill_layer_parity

# CI: cheap CPU-only sanity (no GPU needed) -- same checks as GitHub CI.
ci:
	./scripts/ci_local.sh

.PHONY: ci

# Q4_0 reference quantizer roundtrip gate (head-requant prerequisite).
$(BUILD)/test_quant_ref: tests/test_quant_ref.c src/quant_ref.c src/dequant_ref.c src/loader_gguf.c | $(BUILD)
	cc -O2 -Iinclude -Isrc -o $@ tests/test_quant_ref.c src/quant_ref.c src/dequant_ref.c src/loader_gguf.c -lm

test_quant_ref: $(BUILD)/test_quant_ref

.PHONY: test_quant_ref
