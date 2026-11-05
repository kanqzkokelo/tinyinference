# tinyinference (Python)

ctypes bindings for the tinyinference CUDA engine, plus an OpenAI-compatible
HTTP server. The wheel ships the engine shared library; you only need an
NVIDIA driver (no CUDA toolkit).

```python
from tinyinference import Model

with Model("Qwen2.5-0.5B-Instruct-Q4_0.gguf", n_ctx=2048) as m:
    print(m.generate("The capital of France is", max_tokens=16))
```

Chat and streaming:

```python
for piece in m.stream_text(m.encode(m.chat_format([{"role": "user", "content": "hi"}])), 128):
    print(piece, end="", flush=True)
```

Serve the OpenAI API:

```bash
tinyinference serve model.gguf --port 8000
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

## Building the wheel

```bash
python/build_wheel.sh          # builds the engine library, then the wheel
pip install python/dist/*.whl
```

On Windows the script calls `make -f Makefile.win pydll`; elsewhere `make pylib`.
Set `TINYINFERENCE_LIB` to point at a prebuilt `tinytorch_engine` library if
you do not want the bundled copy.
