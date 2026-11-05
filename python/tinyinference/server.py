"""OpenAI-compatible HTTP server backed by one loaded model.

Endpoints:
    GET  /v1/models
    POST /v1/chat/completions   (stream and non-stream)
    POST /v1/completions        (stream and non-stream)

The engine holds a single KV cache, so requests are serialized.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .model import Model

__all__ = ["serve", "make_server"]


def _now() -> int:
    return int(time.time())


def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "tinyinference/0.9"

    # Set by make_server
    model: Model
    model_name: str
    lock: threading.Lock

    def log_message(self, fmt: str, *args: Any) -> None:
        pass

    # -- plumbing ------------------------------------------------------

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._send_json({"error": {"message": message, "type": "invalid_request_error"}}, status)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            return None

    def _start_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # Close after [DONE] so clients that read until EOF (urllib, requests
        # without stream=True) do not hang waiting for more bytes.
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def _sse(self, payload: dict | str) -> None:
        data = payload if isinstance(payload, str) else json.dumps(payload)
        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
        self.wfile.flush()

    # -- routes --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            self._send_json(
                {
                    "object": "list",
                    "data": [
                        {
                            "id": self.model_name,
                            "object": "model",
                            "created": _now(),
                            "owned_by": "tinyinference",
                        }
                    ],
                }
            )
        elif self.path.rstrip("/") in ("/health", "/"):
            self._send_json({"status": "ok"})
        else:
            self._error("not found", 404)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.rstrip("/")
        if path in ("/v1/chat/completions", "/chat/completions"):
            self._chat()
        elif path in ("/v1/completions", "/completions"):
            self._completion()
        else:
            self._error("not found", 404)

    # -- handlers ------------------------------------------------------

    def _chat(self) -> None:
        req = self._read_json()
        if req is None:
            return self._error("invalid JSON body")
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._error("messages must be a non-empty array")

        try:
            prompt = self.model.chat_format(messages)
        except ValueError as exc:
            return self._error(str(exc))

        self._generate(
            prompt=prompt,
            max_tokens=int(req.get("max_tokens") or 256),
            temperature=float(req.get("temperature", 0.7)),
            top_k=int(req.get("top_k") or 40),
            repeat_penalty=float(req.get("repeat_penalty", 1.1)),
            stream=bool(req.get("stream", False)),
            chat=True,
        )

    def _completion(self) -> None:
        req = self._read_json()
        if req is None:
            return self._error("invalid JSON body")
        prompt = req.get("prompt")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not isinstance(prompt, str):
            return self._error("prompt must be a string")

        self._generate(
            prompt=prompt,
            max_tokens=int(req.get("max_tokens") or 256),
            temperature=float(req.get("temperature", 0.7)),
            top_k=int(req.get("top_k") or 40),
            repeat_penalty=float(req.get("repeat_penalty", 1.1)),
            stream=bool(req.get("stream", False)),
            chat=False,
        )

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
        repeat_penalty: float,
        stream: bool,
        chat: bool,
    ) -> None:
        model = self.model
        tokens = model.encode(prompt)
        cid = _completion_id()
        created = _now()

        if stream:
            self._start_stream()
            self._sse(
                self._chunk(cid, created, chat, delta={"role": "assistant", "content": ""})
            )
            self.lock.acquire()
            try:
                model.set_sampling(temperature, top_k, repeat_penalty)
                for piece in model.stream_text(tokens, max_tokens):
                    self._sse(self._chunk(cid, created, chat, delta={"content": piece}))
            finally:
                self.lock.release()
            self._sse(self._chunk(cid, created, chat, finish="stop"))
            self._sse("[DONE]")
            return

        self.lock.acquire()
        try:
            model.set_sampling(temperature, top_k, repeat_penalty)
            text = "".join(model.stream_text(tokens, max_tokens))
        finally:
            self.lock.release()

        if chat:
            payload = {
                "id": cid,
                "object": "chat.completion",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(tokens),
                    "completion_tokens": len(model.encode(text)),
                    "total_tokens": len(tokens) + len(model.encode(text)),
                },
            }
        else:
            payload = {
                "id": cid,
                "object": "text_completion",
                "created": created,
                "model": self.model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": len(tokens),
                    "completion_tokens": len(model.encode(text)),
                    "total_tokens": len(tokens) + len(model.encode(text)),
                },
            }
        self._send_json(payload)

    def _chunk(
        self,
        cid: str,
        created: int,
        chat: bool,
        delta: dict | None = None,
        finish: str | None = None,
    ) -> dict:
        if chat:
            choice = {"index": 0, "delta": delta or {}, "finish_reason": finish}
            obj = "chat.completion.chunk"
        else:
            choice = {"index": 0, "text": (delta or {}).get("content", ""), "finish_reason": finish}
            obj = "text_completion"
        return {
            "id": cid,
            "object": obj,
            "created": created,
            "model": self.model_name,
            "choices": [choice],
        }


def make_server(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    n_ctx: int = 0,
) -> ThreadingHTTPServer:
    model = Model(model_path, n_ctx)
    handler = type(
        "Handler",
        (_Handler,),
        {
            "model": model,
            "model_name": model_path.replace("\\", "/").rsplit("/", 1)[-1],
            "lock": threading.Lock(),
        },
    )
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    n_ctx: int = 0,
) -> None:
    server = make_server(model_path, host, port, n_ctx)
    print(f"tinyinference serving {model_path} on http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
