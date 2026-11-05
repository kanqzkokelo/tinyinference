# Contributing

Bug reports and patches are welcome. This is a small engine, so the bar is
"does it make the code faster, more correct, or smaller" rather than "does it
add a feature".

## Build and test

Linux:

```
make lib && make run_llm_gpu
./scripts/ci_local.sh
```

Windows (MSYS2 mingw-w64 for C, MSVC + nvcc for CUDA):

```
make -f Makefile.win all
make -f Makefile.win cuda
./scripts/ci_windows.sh
```

`ci_local.sh` is the gate that has to pass for every change. If you touch a
kernel, also run the relevant `scripts/verify.sh` target; they compare against
llama.cpp logits, so they need an oracle build (see `docs/`).

## Style

- C11, no external libraries beyond libc/pthreads and CUDA.
- Match the surrounding formatting; the codebase uses 4-space indent and
  `snake_case` for functions, `CamelCase` for types.
- Keep platform code behind `#ifdef _WIN32` in `src/tt_platform.h` /
  `src/tt_pthread.h` rather than spreading it through call sites.
- No new environment variables unless the behavior cannot be a default.

## Pull requests

- One logical change per PR.
- Say what you measured if it is a performance change: model, quant, context
  length, GPU, and the llama.cpp version used for comparison.
- Correctness changes need a test that fails before the patch.

## Reporting a bug

Include the model and quantization, GPU and driver, the command line, and the
first lines of output. If it is a wrong-output bug, say whether
`TT_NO_GRAPH=1` changes it; that separates graph capture from kernel math.
