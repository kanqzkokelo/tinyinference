"""OpenAI endpoint tests. Set TT_TEST_MODEL to a GGUF file."""

from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.request

from tinyinference.server import make_server

MODEL = os.environ.get("TT_TEST_MODEL", "")


@unittest.skipUnless(MODEL and os.path.isfile(MODEL), "TT_TEST_MODEL not set")
class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = make_server(MODEL, port=0, n_ctx=512)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def _post(self, path: str, body: dict) -> bytes:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()

    def test_models(self) -> None:
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models") as resp:
            payload = json.loads(resp.read())
        self.assertEqual(payload["object"], "list")
        self.assertTrue(payload["data"])

    def test_chat(self) -> None:
        payload = json.loads(
            self._post(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "max_tokens": 8,
                    "temperature": 0,
                },
            )
        )
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertGreater(payload["usage"]["completion_tokens"], 0)

    def test_chat_stream(self) -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(
                {
                    "messages": [{"role": "user", "content": "Say hi."}],
                    "max_tokens": 4,
                    "temperature": 0,
                    "stream": True,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            lines = [ln.decode() for ln in resp if ln.strip()]
        self.assertTrue(lines[0].startswith("data: "))
        self.assertEqual(lines[-1].strip(), "data: [DONE]")

    def test_completion(self) -> None:
        payload = json.loads(
            self._post(
                "/v1/completions",
                {"prompt": "The capital of France is", "max_tokens": 4, "temperature": 0},
            )
        )
        self.assertEqual(payload["object"], "text_completion")
        self.assertIn("text", payload["choices"][0])


if __name__ == "__main__":
    unittest.main()
