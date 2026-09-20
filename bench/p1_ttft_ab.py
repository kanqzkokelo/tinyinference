#!/usr/bin/env python3
"""P1 A/B: prefill/TTFT — old routing vs new routing, same binary, same prompts.

Old = pre-2026-09-20 behaviour: the batched prefill path engages only for n >= 32;
      below that the engine runs one full forward per prompt token.
      Restored exactly with TT_PF_MINN=32 TT_PF_BATCHN_N=0.
New = engine defaults (batchn GEMV for n <= TT_PF_BATCHN_N, batched GEMM for
      n >= TT_PF_MINN); override with --batchn-max / --pf-minn.

Prompts use TT_RAW_PROMPT=1 so the CLI does not wrap them in a chat template
(which floors every prompt at 28 tokens). The engine's own "[run] prompt: N
tokens" line and, with TT_DISPATCH=1, its "[prefill] dispatch=..." line are
recorded per row so the routing is visible in the output.

Usage:
  python3 bench/p1_ttft_ab.py [--reps 3] [--model PATH] [--words 1,2,4,...]
Writes nothing; pipe to a file to keep a record (bench/p1_ttft_ab.txt).
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "run_llm_gpu"
DEFAULT_MODEL = ROOT / "data" / "models" / "qwen2.5-0.5b-instruct-q4_0.gguf"

# NOTE (2026-09-20): the previous NEW_ENV was {BATCHN_N=8, MINN=12}, which made the
# batched path engage only for n>=12 while batchn fires only for n<=8 — i.e. batchn
# was DEAD CODE and the A/B never measured the P1 batchn win. MINN=2 passes m61 in full
# (the attention-path changes fixed the small-n divergence), so the shipped config that
# both passes m61 AND exercises batchn is {BATCHN_N=12, MINN=2}. See AUDIT.md.
OLD_ENV = {"TT_PF_MINN": "32", "TT_PF_BATCHN_N": "0"}
NEW_ENV = {"TT_PF_BATCHN_N": "12", "TT_PF_MINN": "2"}

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
    "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee "
    "zulu red green blue cyan magenta yellow black white silver gold copper iron nickel "
    "zinc lead tin argon neon xenon krypton radon helium"
).split()

DISPATCH_RE = re.compile(r"\[prefill\] dispatch=([a-z0-9_>=-]+) N=(\d+)")
PROMPT_RE = re.compile(r"\[run\] prompt: (\d+) tokens")
PREFILL_RE = re.compile(r"prefill_us=(\d+)")


def make_prompt(n_words: int) -> str:
    out = []
    while len(out) < n_words:
        out.extend(WORDS)
    return " ".join(out[:n_words])


def run_once(model: Path, prompt: str, gen: int, env_extra: dict, dispatch: bool = False):
    env = dict(os.environ)
    env["TT_RAW_PROMPT"] = "1"
    # run_llm_gpu takes the PROMPT as positional arg 0; the model comes from
    # -m/--model or TT_MODEL. Passing the model positionally silently makes the
    # model path the prompt (constant ~29 tokens) and atoi("prompt") = 1 gen
    # token, which is a trap this harness fell into once already.
    env["TT_MODEL"] = str(model)
    env.update(env_extra)
    if dispatch:
        env["TT_DISPATCH"] = "1"
    try:
        r = subprocess.run(
            [str(BIN), prompt, str(gen)],
            capture_output=True, text=True, env=env, timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    out = r.stdout + "\n" + r.stderr
    m = PREFILL_RE.search(out)
    if not m:
        return None
    pt = PROMPT_RE.search(out)
    d = DISPATCH_RE.search(out)
    return {
        "prefill_us": int(m.group(1)),
        "prompt_tokens": int(pt.group(1)) if pt else -1,
        "dispatch": f"{d.group(1)} N={d.group(2)}" if d else "?",
    }


def best_of(model: Path, prompt: str, gen: int, env_extra: dict, reps: int):
    best = None
    for _ in range(reps):
        r = run_once(model, prompt, gen, env_extra)
        if r is None:
            continue
        if best is None or r["prefill_us"] < best["prefill_us"]:
            best = r
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--gen", type=int, default=2)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--words", type=str, default="1,2,4,6,8,12,16,24,32,64,128,256,512")
    ap.add_argument("--batchn-max", type=int, default=12,
                    help="TT_PF_BATCHN_N for the new config (0 disables batchn)")
    ap.add_argument("--pf-minn", type=int, default=2,
                    help="TT_PF_MINN for the new config (min n for the batched path)")
    args = ap.parse_args()
    new_env = {"TT_PF_BATCHN_N": str(args.batchn_max), "TT_PF_MINN": str(args.pf_minn)}

    if not BIN.is_file():
        print(f"ERROR: {BIN} missing (make run_llm_gpu)", file=sys.stderr)
        return 2

    print(f"# P1 prefill/TTFT A/B")
    print(f"# binary : {BIN}")
    print(f"# model  : {args.model}")
    print(f"# gen    : {args.gen} tokens   reps={args.reps} (min-of-reps prefill_us)   TT_RAW_PROMPT=1")
    print(f"# old env: {OLD_ENV}")
    print(f"# new env: {new_env}")
    print("#")
    print(f"# {'n_words':>7} {'ptok':>5} {'old_us':>9} {'new_us':>9} {'speedup':>8} "
          f"{'new_tok/s':>10} {'old_dispatch':>14} {'new_dispatch':>14}")
    for nw in [int(x) for x in args.words.split(",")]:
        p = make_prompt(nw)
        o = best_of(args.model, p, args.gen, OLD_ENV, args.reps)
        n = best_of(args.model, p, args.gen, new_env, args.reps)
        if not o or not n:
            print(f"# {nw:>7}  run failed")
            continue
        speedup = o["prefill_us"] / n["prefill_us"] if n["prefill_us"] else 0.0
        tps = (n["prompt_tokens"] / (n["prefill_us"] / 1e6)) if n["prefill_us"] else 0.0
        # One dispatch probe per config tells us which path the engine chose.
        od = run_once(args.model, p, args.gen, OLD_ENV, dispatch=True)
        nd = run_once(args.model, p, args.gen, new_env, dispatch=True)
        print(f"  {nw:>7} {n['prompt_tokens']:>5} {o['prefill_us']:>9} {n['prefill_us']:>9} "
              f"{speedup:>7.2f}x {tps:>10.0f} {od['dispatch'] if od else '-':>14} "
              f"{nd['dispatch'] if nd else '-':>14}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
