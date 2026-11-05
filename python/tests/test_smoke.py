"""Smoke tests for the ctypes bindings. Set TT_TEST_MODEL to a GGUF file."""

from __future__ import annotations

import os
import unittest

from tinyinference import Model

MODEL = os.environ.get("TT_TEST_MODEL", "")


@unittest.skipUnless(MODEL and os.path.isfile(MODEL), "TT_TEST_MODEL not set")
class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = Model(MODEL, n_ctx=512)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.model.close()

    def test_geometry(self) -> None:
        self.assertGreater(self.model.vocab_size, 0)
        self.assertEqual(self.model.n_ctx, 512)

    def test_roundtrip(self) -> None:
        ids = self.model.encode("Hello, world!")
        self.assertTrue(ids)
        text = b"".join(self.model.decode(t) for t in ids)
        self.assertIn(b"Hello", text)

    def test_greedy_is_deterministic(self) -> None:
        prompt = self.model.encode("The capital of France is")
        self.model.reset()
        a = list(self.model.stream_tokens(prompt, 8))
        self.model.reset()
        b = list(self.model.stream_tokens(prompt, 8))
        self.assertEqual(a, b)
        self.assertTrue(a)

    def test_chat_format(self) -> None:
        prompt = self.model.chat_format([{"role": "user", "content": "hi"}])
        self.assertIn("hi", prompt)


if __name__ == "__main__":
    unittest.main()
