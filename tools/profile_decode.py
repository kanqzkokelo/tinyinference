#!/usr/bin/env python3
"""Collect eager per-stage decode profiles.

TT_PROFILE=1 forces eager execution so CUDA events can bracket stages. This
tool must not be used as the graph-replay scoreboard; bench/bench_llm.py owns
that result. Output defaults to bench/profile_decode.csv.
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
DEFAULT_CTXS = (128, 512, 2048)
DEFAULT_PROMPT = "Explain quantum computing in one sentence."
FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
          "while soft rain falls on the quiet village below the hills. ")
STAGE_COLUMNS = ("qkv_ms", "attn_ms", "o_ms", "ffn_ms", "lmhead_ms", "other_ms")


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect eager decode stages without overwriting the scoreboard."
    )
    p.add_argument("--model", default=os.environ.get(
        "TT_MODEL", "data/models/qwen2.5-0.5b-instruct-q4_0.gguf"))
    p.add_argument("--ctxs", default=",".join(map(str, DEFAULT_CTXS)))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--output", default="bench/profile_decode.csv")
    a = p.parse_args()
    a.ctxs = tuple(int(x.strip()) for x in a.ctxs.split(",") if x.strip())
    if not a.ctxs or any(x <= 0 for x in a.ctxs):
        p.error("--ctxs must contain positive integers")
    if a.runs < 1 or a.gen_tokens < 1:
        p.error("--runs and --gen-tokens must be positive")
    if not Path(a.model).is_file():
        p.error(f"model does not exist: {a.model}")
    return a


def pad_prompt(target, base_prompt):
    reps = max(0, round((target - 32) / 24))
    return (FILLER * reps + base_prompt) if reps else base_prompt


def model_quant(model_path):
    stem = Path(model_path).name.removesuffix(".gguf")
    m = re.match(r"^(.*)[-_](q\d.*|Q\d.*|f\d+.*)$", stem)
    return (m.group(1), m.group(2).lower()) if m else (stem, "")


def parse_stats(text):
    lines = [x for x in text.splitlines() if x.startswith("STATS ")]
    if not lines:
        return None
    return dict(item.split("=", 1) for item in lines[-1].split()[1:] if "=" in item)


def parse_last_profile(text):
    blocks, current = [], None
    for line in text.splitlines():
        m = re.match(r"^PROFILE mode=(\S+)", line)
        if m:
            current = {"mode": m.group(1), "stages": {}}
            blocks.append(current)
            continue
        if current is None:
            continue
        m = re.match(r"^PROFILE\s+(\S+)\s+([-+]?\d+(?:\.\d+)?)$", line)
        if m and m.group(1) != "TOTAL(med)":
            current["stages"][m.group(1)] = float(m.group(2))
    for block in reversed(blocks):
        if block["stages"]:
            return block
    return None


def stage_columns(stages):
    def total(*names):
        return sum(stages.get(name, 0.0) for name in names)
    return {
        "qkv_ms": total("qkv-gemv"),
        "attn_ms": total("attn", "flash"),
        "o_ms": total("o-proj"),
        "ffn_ms": total("ffn-gateup", "ffn-down"),
        "lmhead_ms": total("logits-gemv", "lm-head"),
        "other_ms": total("rmsnorm", "rope", "kv-scatter", "embed", "argmax", "other"),
    }


def run_once(args, prompt, max_ctx):
    env = dict(os.environ)
    env.update(TT_MODEL=args.model, TT_PROFILE="1", TT_MAX_CTX=str(max_ctx),
               TT_GREEDY="1", TT_RAW_PROMPT="1")
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        os.path.expanduser("~/mmcuda/lib"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
        env.get("LD_LIBRARY_PATH", "")]))
    r = subprocess.run(["build/run_llm_gpu", prompt, str(args.gen_tokens)],
                       capture_output=True, text=True, timeout=600, env=env)
    combined = r.stdout + "\n" + r.stderr
    if r.returncode:
        raise RuntimeError(f"engine exit {r.returncode}\n{combined[-1000:]}")
    stats = parse_stats(r.stdout)
    profile = parse_last_profile(r.stdout)
    if not stats or not profile:
        raise RuntimeError(f"missing STATS or PROFILE block\n{combined[-1200:]}")
    tokens, prefill = int(stats["tokens"]), int(stats["prefill"])
    decode_us, prefill_us = float(stats["decode_us"]), float(stats["prefill_us"])
    if tokens <= 0:
        raise RuntimeError("engine produced zero decode tokens; cannot profile")
    if decode_us <= 0 or prefill_us <= 0:
        raise RuntimeError("engine returned non-positive timing data")
    stages = stage_columns(profile["stages"])
    stage_sum = sum(stages.values())
    step_ms = decode_us / tokens / 1000.0
    return {
        "mode": "eager-stage", "ctx": prefill, "prompt_tok": prefill,
        "gen_tok": tokens, "pp_tps": prefill / (prefill_us / 1e6),
        "tg_tps": tokens / (decode_us / 1e6), "prefill_us": prefill_us,
        "decode_us": decode_us, "stage_sum_ms": stage_sum,
        "stage_gap_pct": ((stage_sum / step_ms) - 1) * 100 if step_ms else 0,
        **stages,
    }


def median_row(samples, model, quant, target):
    numeric = ("ctx", "prompt_tok", "gen_tok", "pp_tps", "tg_tps",
               "prefill_us", "decode_us", "stage_sum_ms", "stage_gap_pct",
               *STAGE_COLUMNS)
    row = {"model": model, "quant": quant, "mode": "eager-stage",
           "target_ctx": target}
    for key in numeric:
        row[key] = statistics.median(x[key] for x in samples)
    return row


def main():
    args = parse_args()
    model, quant = model_quant(args.model)
    rows = []
    for target in args.ctxs:
        samples = [run_once(args, pad_prompt(target, args.prompt), target + 512)
                   for _ in range(args.runs)]
        row = median_row(samples, model, quant, target)
        rows.append(row)
        print(f"ctx~{target} actual={row['ctx']:.0f} "
              f"pp={row['pp_tps']:.1f} tok/s tg={row['tg_tps']:.1f} tok/s "
              f"stage_sum={row['stage_sum_ms']:.3f} ms "
              f"gap={row['stage_gap_pct']:.1f}%")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("model", "quant", "mode", "target_ctx", "ctx", "prompt_tok",
              "gen_tok", "pp_tps", "tg_tps", "prefill_us", "decode_us",
              "stage_sum_ms", "stage_gap_pct", *STAGE_COLUMNS)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))
