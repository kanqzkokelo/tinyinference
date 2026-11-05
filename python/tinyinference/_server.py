"""Python front end for the tinyinference native server.

The engine is C + CUDA. This module does not reimplement it: it starts the
bundled ``server_minimal`` binary on a loopback port and talks to its
OpenAI-compatible API. Requests are serialized because the server handles
one generation at a time.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Iterator, Optional, Sequence

__all__ = ["Server", "Model"]

_EXE = ".exe" if os.name == "nt" else ""


def _bin_dir() -> str:
    env = os.environ.get("TINYINFERENCE_BIN")
    if env:
        return env
    pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin")
    if os.path.isdir(pkg):
        return pkg
    # running from a source checkout: <repo>/python/tinyinference -> <repo>/build
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    build = os.path.join(repo, "build")
    if os.path.isdir(build):
        return build
    raise RuntimeError(
        "tinyinference binaries not found. Install the wheel, set "
        "TINYINFERENCE_BIN, or build with `make -f Makefile.win cuda`."
    )


def _binary(name: str) -> str:
    path = os.path.join(_bin_dir(), name + _EXE)
    if not os.path.isfile(path):
        raise RuntimeError(f"{path} is missing; reinstall the package")
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _tail(path: str, lines: int = 25) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:]).strip()
    except OSError:
        return ""


class Server:
    """A running ``server_minimal`` process."""

    def __init__(
        self,
        model: str,
        ctx: Optional[int] = None,
        port: int = 0,
        timeout: float = 180.0,
        env: Optional[dict] = None,
    ) -> None:
        if not os.path.isfile(model):
            raise FileNotFoundError(model)
        self.model = os.path.abspath(model)
        self.port = port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self._log = tempfile.NamedTemporaryFile(
            prefix="tinyinference-", suffix=".log", delete=False
        )
        child_env = os.environ.copy()
        child_env["TT_MODEL"] = self.model
        child_env["TT_LISTEN"] = f"127.0.0.1:{self.port}"
        if ctx:
            child_env["TT_MAX_CTX"] = str(ctx)
        if env:
            child_env.update({k: str(v) for k, v in env.items()})

        flags = 0
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [_binary("server_minimal")],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=child_env,
            creationflags=flags,
        )
        try:
            self._wait_ready(timeout)
        except BaseException:
            self.close()
            raise

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        url = self.base_url + "/healthz"
        while time.monotonic() < deadline:
            code = self._proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"server exited with code {code}\n{_tail(self._log.name)}"
                )
            try:
                with urllib.request.urlopen(url, timeout=1.0) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.05)
        raise TimeoutError(
            f"server did not become ready in {timeout:.0f}s\n{_tail(self._log.name)}"
        )

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        log = getattr(self, "_log", None)
        if log is not None:
            log.close()
            try:
                os.unlink(log.name)
            except OSError:
                pass
            self._log = None

    def __enter__(self) -> "Server":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    # -- HTTP ------------------------------------------------------------

    def _post(self, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(req, timeout=None)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {e.code}: {body}") from None

    def chat(
        self,
        messages: Sequence[dict],
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
    ) -> dict:
        with self._post(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "stream": False,
            }
        ) as r:
            return json.loads(r.read().decode("utf-8"))

    def chat_stream(
        self,
        messages: Sequence[dict],
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
    ) -> Iterator[str]:
        resp = self._post(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "stream": True,
            }
        )
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    continue
                delta = chunk["choices"][0].get("delta", {})
                text = delta.get("content")
                if text:
                    yield text
        finally:
            resp.close()


class Model:
    """Convenience wrapper: a server plus one-shot generation calls."""

    def __init__(self, model: str, ctx: Optional[int] = None, **server_kw) -> None:
        self._server = Server(model, ctx=ctx, **server_kw)
        self._lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return self._server.base_url

    def close(self) -> None:
        self._server.close()

    def __enter__(self) -> "Model":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
    ) -> str:
        with self._lock:
            out = self._server.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        return out["choices"][0]["message"]["content"]

    def stream(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
    ) -> Iterator[str]:
        # held for the duration of the iteration by the caller's loop
        with self._lock:
            yield from self._server.chat_stream(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

    def chat(self, messages: Sequence[dict], **kw) -> dict:
        with self._lock:
            return self._server.chat(messages, **kw)
