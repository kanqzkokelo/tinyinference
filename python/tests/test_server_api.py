"""HTTP surface of ``python -m tinyinference serve``.

Needs a GGUF model:  TT_MODEL=/path/model.gguf python -m pytest python/tests -q
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

MODEL = os.environ.get("TT_TEST_MODEL", "")
pytestmark = pytest.mark.skipif(not MODEL, reason="TT_TEST_MODEL is not set")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "tinyinference", "serve", MODEL, "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(url + "/health", timeout=0.5).read()
                break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError("server exited during startup")
                time.sleep(0.1)
        else:
            raise RuntimeError("server did not come up")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _post(url: str, body: dict, timeout: float = 120):
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _stream(url: str, body: dict) -> list[str]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    out = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: "):
                out.append(line[6:])
    return out


def test_completion_usage(base_url):
    status, body = _post(
        base_url + "/v1/completions",
        {"model": "m", "prompt": "The capital of France is", "max_tokens": 8},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["text"]
    usage = payload["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_stop_string(base_url):
    status, body = _post(
        base_url + "/v1/chat/completions",
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Count: 1 2 3 4 5 6 7 8"}],
            "max_tokens": 32,
            "temperature": 0,
            "stop": "5",
        },
    )
    assert status == 200
    payload = json.loads(body)
    text = payload["choices"][0]["message"]["content"]
    assert "5" not in text


def test_finish_reason_length(base_url):
    status, body = _post(
        base_url + "/v1/completions",
        {"model": "m", "prompt": "Once upon a time", "max_tokens": 1},
    )
    assert status == 200
    assert json.loads(body)["choices"][0]["finish_reason"] == "length"


def test_stream_with_usage(base_url):
    events = _stream(
        base_url + "/v1/chat/completions",
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 4,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(
        c["choices"][0]["delta"].get("content", "")
        for c in chunks
        if c["choices"]
    )
    assert text
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"]["completion_tokens"] > 0


def test_json_object_response(base_url):
    status, body = _post(
        base_url + "/v1/chat/completions",
        {
            "model": "m",
            "messages": [
                {"role": "user", "content": "Give a JSON object with name and age."}
            ],
            "max_tokens": 48,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
    )
    assert status == 200
    text = json.loads(body)["choices"][0]["message"]["content"]
    assert isinstance(json.loads(text), dict)


def test_json_object_truncated_is_not_an_error(base_url):
    # max_tokens can cut a document short; that is a length stop, not a 500.
    status, body = _post(
        base_url + "/v1/chat/completions",
        {
            "model": "m",
            "messages": [{"role": "user", "content": "Return a JSON object."}],
            "max_tokens": 3,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        },
    )
    assert status == 200
    assert json.loads(body)["choices"][0]["finish_reason"] == "length"


def test_rejects_n_greater_than_one(base_url):
    status, body = _post(
        base_url + "/v1/completions",
        {"model": "m", "prompt": "x", "n": 2, "max_tokens": 1},
    )
    assert status == 400
    assert "n > 1" in json.loads(body)["error"]["message"]
