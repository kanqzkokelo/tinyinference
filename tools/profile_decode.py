#!/usr/bin/env python3
"""Decode profiler: TT_PROFILE=1 runs at ~128/512/2048 prompt tokens.

Parses the last PROFILE stage block + STATS line, writes
bench/scoreboard_decode.csv. Stage ms come from whatever PROFILE block the
binary prints (currently the prefill block; once run_llm_gpu reports after
the decode loop the last block is the decode table and these become
per-token decode medians with no script change).
"""
import csv
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

CTXS = (128, 512, 2048)
RUNS = 3
GEN_TOKENS = 64
PROMPT = "Explain quantum computing in one sentence."
FILLER = ("The quick brown fox jumps over the lazy dog near the river bank "
          "while soft rain falls on the quiet village below the hills. ")


def pad_prompt(target):
    reps = max(0, round((target - 32) / 24))
    return (FILLER * reps + "Hi.") if reps else PROMPT


def model_quant():
    stem = os.path.basename(os.environ.get(
        "TT_MODEL", "data/models/qwen2.5-0.5b-instruct-q4_0.gguf"))
    if stem.endswith(".gguf"):
        stem = stem[:-5]
    m = re.match(r"^(.*)[-_](q\d.*|Q\d.*|f\d+.*)$", stem)
    if m:
        return m.group(1), m.group(2).lower()
    return stem, ""


def run_once(prompt, max_ctx):
    env = dict(os.environ)
    env["TT_PROFILE"] = "1"
    env["TT_MAX_CTX"] = str(max_ctx)
    env["LD_LIBRARY_PATH"] = ":".join(filter(None, [
        os.path.expanduser("~/mmcuda/lib"),
        os.path.expanduser("~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"),
        env.get("LD_LIBRARY_PATH", "")]))
    r = subprocess.run(["build/run_llm_gpu", prompt, str(GEN_TOKENS)],
                       capture_output=True, text=True, timeout=600, env=env)
    if r.returncode != 0:
        sys.exit(f"engine exit {r.returncode}\n{r.stderr[-500:]}")
    stats = [l for l in r.stdout.splitlines() if l.startswith("STATS")]
    if not stats:
        sys.exit(f"no STATS line\n{r.stdout[-500:]}")
    f = dict(kv.split("=", 1) for kv in stats[0].split()[1:])
    stages = {}
    for line in r.stdout.splitlines():
        m = re.match(r"PROFILE\s+(\S+)\s+([\d.]+)", line)
        if m and m.group(1) not in ("mode=eager", "TOTAL(med)"):
            stages[m.group(1)] = float(m.group(2))
    return {
        "tokens": int(f["tokens"]),
        "prefill": int(f["prefill"]),
        "tg_tps": int(f["tokens"]) / (float(f["decode_us"]) / 1e6),
        "pp_tps": float(f.get("prefill_tok_s", 0)) or
        (int(f["prefill"]) / (float(f["prefill_us"]) / 1e6)),
        "stages": stages,
    }


def cols(st):
    g = lambda *names: sum(st.get(n, 0.0) for n in names)
    return {
        "qkv_ms": g("qkv-gemv"),
        "attn_ms": g("attn", "flash"),
        "o_ms": g("o-proj"),
        "ffn_ms": g("ffn-gateup", "ffn-down"),
        "lmhead_ms": g("logits-gemv", "lm-head"),
        "other_ms": g("rmsnorm", "rope", "kv-scatter", "embed", "argmax", "other"),
    }


model, quant = model_quant()
rows = []
for target in CTXS:
    outs = [run_once(pad_prompt(target), target + 256) for _ in range(RUNS)]
    ctx = int(statistics.median(o["prefill"] for o in outs))
    row = {
        "model": model, "quant": quant, "ctx": ctx,
        "pp_tps": round(statistics.median(o["pp_tps"] for o in outs), 1),
        "tg_tps": round(statistics.median(o["tg_tps"] for o in outs), 1),
    }
    for k in ("qkv_ms", "attn_ms", "o_ms", "ffn_ms", "lmhead_ms", "other_ms"):
        row[k] = round(statistics.median(cols(o["stages"])[k] for o in outs), 3)
    rows.append(row)
    print(f"ctx~{target} (prefill={ctx}): "
          f"pp={row['pp_tps']} tok/s tg={row['tg_tps']} tok/s stages={cols(outs[-1]['stages'])}")

with open("bench/scoreboard_decode.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["model", "quant", "ctx", "pp_tps", "tg_tps",
                                       "qkv_ms", "attn_ms", "o_ms", "ffn_ms",
                                       "lmhead_ms", "other_ms"])
    w.writeheader()
    w.writerows(rows)

base = min(rows, key=lambda r: abs(r["ctx"] - 512))
print(f"\nBASELINE {base['model']} ctx{base['ctx']}: tg_tps={base['tg_tps']} "
      f"(later tasks must not regress >2%)")
