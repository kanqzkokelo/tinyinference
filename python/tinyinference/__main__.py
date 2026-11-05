"""Command line entry point: `python -m tinyinference`."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .model import Model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tinyinference")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate text from a prompt")
    gen.add_argument("model")
    gen.add_argument("prompt")
    gen.add_argument("-n", "--max-tokens", type=int, default=256)
    gen.add_argument("-t", "--temperature", type=float, default=0.0)
    gen.add_argument("--top-k", type=int, default=0)
    gen.add_argument("--repeat-penalty", type=float, default=1.0)
    gen.add_argument("--n-ctx", type=int, default=0)

    chat = sub.add_parser("chat", help="single-turn chat")
    chat.add_argument("model")
    chat.add_argument("prompt")
    chat.add_argument("-n", "--max-tokens", type=int, default=256)
    chat.add_argument("-t", "--temperature", type=float, default=0.7)
    chat.add_argument("--top-k", type=int, default=40)
    chat.add_argument("--n-ctx", type=int, default=0)

    srv = sub.add_parser("serve", help="OpenAI-compatible HTTP server")
    srv.add_argument("model")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.add_argument("--n-ctx", type=int, default=0)

    args = parser.parse_args(argv)

    if args.command == "serve":
        from .server import serve

        serve(args.model, args.host, args.port, args.n_ctx)
        return 0

    with Model(args.model, args.n_ctx) as model:
        if args.command == "generate":
            text = model.generate(
                args.prompt,
                args.max_tokens,
                args.temperature,
                args.top_k,
                args.repeat_penalty,
            )
        else:
            text = model.chat(
                [{"role": "user", "content": args.prompt}],
                args.max_tokens,
                args.temperature,
                args.top_k,
            )
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
