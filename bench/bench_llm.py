#!/usr/bin/env python3
"""Authoritative LLM Benchmark Runner for TinyInference vs llama.cpp.

Consolidates tools/bench_cuda_vs_cuda.sh, tools/bench_scoreboard.sh, and
tools/run_live_performance_suite.py into one authoritative, reproducible runner.

Records:
- TinyInference git commit & llama.cpp commit and binary path
- Model path, filename, size, SHA256 hash, quantization
- Hardware: GPU name, SM architecture (sm_86), CUDA version (12.4)
- Context targets & actual prompt token counts (short ~32, medium ~512, long 2048)
- Generation token counts (e.g. 64 or 128)
- Graph vs Eager execution modes (benchmarked and reported separately)
- 5 measured repetitions after warmup: all samples, median, min, max for:
  prefill_us, decode_us, first_token_us, total_us, decode_tok_s, prefill_tok_s
- Oracle comparison with llama.cpp (llama-cli -st --no-warmup)
- Outputs:
  - data/bench/results_scoreboard.jsonl
  - data/bench/results_scoreboard.md
  - bench/scoreboard_decode.csv
"""

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path("/home/mitesh/ti-scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)

DEFAULT_ORACLE_BIN = "/home/mitesh/Storage/llama.cpp/build_cuda/bin/llama-cli"
DEFAULT_LLAMACPP_COMMIT = "3f545be"

PRIMARY_MODELS = [
    "data/models/qwen2.5-0.5b-instruct-q4_0.gguf",
    "data/testmodels/qwen3-0.6b-q8_0.gguf",
    "data/testmodels/llama-3.2-1b-q4_0.gguf",
    # SmolLM2-135M explicitly chosen as Q4_0 for parity with other quant baselines
    "data/testmodels/smollm2-135m-instruct-Q4_0.gguf",
]

DEFAULT_CTXS = [32, 512, 2048]
DEFAULT_GEN_TOKENS = 64
DEFAULT_RUNS = 5
DEFAULT_WARMUP = 3
DEFAULT_COOLDOWN_SEC = 10.0

FILLER = (
    "The quick brown fox jumps over the lazy dog near the river bank "
    "while soft rain falls on the quiet village below the hills. "
)  # ~24 tokens
BASE_PROMPT = "Explain quantum computing in one sentence."

_SHA256_CACHE: Dict[str, str] = {}


def get_git_commit(repo_dir: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return "unknown"


def get_short_git_commit(repo_dir: Path) -> str:
    c = get_git_commit(repo_dir)
    return c[:7] if c != "unknown" else "unknown"


def get_llamacpp_info(oracle_bin: str) -> Tuple[str, str]:
    bin_path = os.path.abspath(oracle_bin)
    commit = DEFAULT_LLAMACPP_COMMIT
    parent = Path(bin_path).parent
    while parent != parent.parent:
        if (parent / ".git").exists():
            commit = get_short_git_commit(parent)
            break
        parent = parent.parent
    if commit != DEFAULT_LLAMACPP_COMMIT:
        if os.environ.get("TT_ALLOW_UNPINNED_LLAMA") != "1":
            print(
                f"ERROR: llama.cpp commit '{commit}' does not match pinned commit '{DEFAULT_LLAMACPP_COMMIT}'. "
                f"M0 protocol requires pinned reference commit. Set TT_ALLOW_UNPINNED_LLAMA=1 to override.",
                file=sys.stderr,
            )
            sys.exit(1)
    return bin_path, commit


def get_gpu_info() -> Dict[str, str]:
    gpu_name = "NVIDIA GeForce RTX 3050 Laptop GPU"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            gpu_name = r.stdout.strip().splitlines()[0]
    except Exception:
        pass

    cuda_ver = "12.4"
    try:
        nvcc_bin = os.path.expanduser("~/mmcuda/bin/nvcc")
        if os.path.isfile(nvcc_bin):
            r = subprocess.run(
                [nvcc_bin, "--version"], capture_output=True, text=True, timeout=10
            )
            m = re.search(r"release\s+([\d.]+)", r.stdout)
            if m:
                cuda_ver = m.group(1)
    except Exception:
        pass

    return {
        "gpu": gpu_name,
        "sm": "sm_86",
        "cuda": cuda_ver,
    }


def get_power_info() -> Dict[str, Any]:
    """Snapshot GPU power limit/draw + AC state. Fail-open with warning."""
    info: Dict[str, Any] = {
        "power_limit_w": None,
        "power_draw_w": None,
        "clocks_gr_mhz": None,
        "clocks_sm_mhz": None,
        "clocks_mem_mhz": None,
        "ac_online": None,
    }
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.sm,clocks.mem",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
            vals = [float(x) if x not in ("-", "N/A", "") else None for x in parts]
            while len(vals) < 3:
                vals.append(None)
            info["clocks_gr_mhz"], info["clocks_sm_mhz"], info["clocks_mem_mhz"] = vals[:3]
    except Exception:
        pass
    try:
        r = subprocess.run(
            ["nvidia-smi", "-q", "-d", "POWER"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout:
            m = re.search(r"Current Power Limit\s*:\s*([\d.]+)\s*W", r.stdout)
            if m:
                info["power_limit_w"] = float(m.group(1))
            m2 = re.search(r"Average Power Draw\s*:\s*([\d.]+)\s*W", r.stdout)
            if m2:
                info["power_draw_w"] = float(m2.group(1))
    except Exception:
        pass
    for p in ("/sys/class/power_supply/ACAD/online", "/sys/class/power_supply/AC/online"):
        try:
            with open(p) as f:
                info["ac_online"] = int(f.read().strip())
            break
        except Exception:
            continue
    return info


def compute_sha256(file_path: Path) -> str:
    path_str = str(file_path.resolve())
    if path_str in _SHA256_CACHE:
        return _SHA256_CACHE[path_str]
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    digest = h.hexdigest()
    _SHA256_CACHE[path_str] = digest
    return digest


def detect_quant(filename: str) -> str:
    stem = Path(filename).stem
    m = re.search(r"[-_](q\d+_\d+|q\d+_k(?:_[ms])?|q\d+_\d|f16|f32|q8_0)", stem, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def resolve_model_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p.resolve()
    p_root = ROOT / path_str
    if p_root.is_file():
        return p_root.resolve()
    p_models = ROOT / "data" / "models" / p.name
    if p_models.is_file():
        return p_models.resolve()
    p_alt = Path("/home/mitesh/Storage/repos/nnfromscratch/data/testmodels") / p.name
    if p_alt.is_file():
        return p_alt.resolve()
    p_tm = ROOT / "data" / "testmodels" / p.name
    if p_tm.is_file():
        return p_tm.resolve()

    raise FileNotFoundError(
        f"Required model '{path_str}' not found (searched: {p}, {p_root}, {p_models}, {p_alt})"
    )


def make_prompt_for_ctx(target_ctx: int, custom_prompt: Optional[str] = None) -> str:
    if custom_prompt:
        return custom_prompt
    if target_ctx <= 32:
        return BASE_PROMPT
    reps = max(1, round((target_ctx - 32) / 24))
    return (FILLER * reps) + "Hi. " + BASE_PROMPT


def compute_stats(samples: List[float]) -> Dict[str, Any]:
    if not samples:
        return {"samples": [], "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "samples": [round(x, 3) for x in samples],
        "median": round(statistics.median(samples), 3),
        "min": round(min(samples), 3),
        "max": round(max(samples), 3),
    }


def run_engine_once(
    model_path: Path,
    prompt: str,
    gen_tokens: int,
    ctx_cap: int,
    mode: str,
    raw_prompt: bool = True,
    timeout: int = 180,
) -> Dict[str, Any]:
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            [
                os.path.expanduser("~/mmcuda/lib"),
                os.path.expanduser(
                    "~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
                ),
                env.get("LD_LIBRARY_PATH", ""),
            ],
        )
    )
    env["TT_GREEDY"] = "1"
    if raw_prompt:
        env["TT_RAW_PROMPT"] = "1"
    env["TT_MODEL"] = str(model_path)
    env["TT_MAX_CTX"] = str(ctx_cap)

    if mode == "eager":
        env["TT_NO_GRAPH"] = "1"
    else:
        env.pop("TT_NO_GRAPH", None)

    cmd = [
        str(ROOT / "build" / "run_llm_gpu"),
        "-m",
        str(model_path),
        prompt,
        str(gen_tokens),
    ]

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        cwd=ROOT,
        env=env,
    )

    if r.returncode != 0:
        err_snippet = r.stderr[-500:] if r.stderr else r.stdout[-500:]
        raise RuntimeError(
            f"TinyInference run_llm_gpu failed (rc={r.returncode}) on {model_path.name}:\n{err_snippet}"
        )

    combined = r.stdout + "\n" + r.stderr
    stats_lines = [l for l in r.stdout.splitlines() if l.startswith("STATS")]
    if not stats_lines:
        raise RuntimeError(
            f"No STATS line from TinyInference on {model_path.name}:\n{r.stdout[-500:]}"
        )

    fields = dict(kv.split("=", 1) for kv in stats_lines[0].split()[1:] if "=" in kv)
    tokens_actual = int(fields.get("tokens", gen_tokens))
    prefill_actual = int(fields.get("prefill", 0))
    decode_us = float(fields.get("decode_us", 0.0))
    prefill_us = float(fields.get("prefill_us", 0.0))
    first_token_us = float(fields.get("first_token_us", prefill_us + (decode_us / tokens_actual if tokens_actual > 0 else 0.0)))
    total_us = float(fields.get("total_us", prefill_us + decode_us))

    decode_tok_s = (tokens_actual / (decode_us / 1e6)) if decode_us > 0 else 0.0
    prefill_tok_s = float(
        fields.get(
            "prefill_tok_s",
            (prefill_actual / (prefill_us / 1e6)) if prefill_us > 0 else 0.0,
        )
    )

    graph_status = "unknown"
    if mode == "eager":
        graph_status = "eager_forced"
    elif "decode-step graph captured" in combined or "cudaGraph replay ON" in combined:
        graph_status = "graph_captured"
    elif (
        "graph capture failed" in combined
        or "falling back to eager" in combined
        or "fell back to eager" in combined
    ):
        graph_status = "eager_fallback"

    return {
        "tokens": tokens_actual,
        "prefill": prefill_actual,
        "decode_us": decode_us,
        "prefill_us": prefill_us,
        "first_token_us": first_token_us,
        "total_us": total_us,
        "decode_tok_s": decode_tok_s,
        "prefill_tok_s": prefill_tok_s,
        "graph_status": graph_status,
    }


def run_oracle_once(
    oracle_bin: str,
    model_path: Path,
    prompt: str,
    gen_tokens: int,
    ctx_cap: int,
    timeout: int = 180,
) -> Dict[str, Any]:
    if not os.path.isfile(oracle_bin):
        raise FileNotFoundError(f"Oracle binary not found at '{oracle_bin}'")
    if not prompt or not prompt.strip():
        raise ValueError("Refusing to invoke oracle llama-cli with empty prompt (known hang)")

    cmd = [
        oracle_bin,
        "-m",
        str(model_path),
        "-p",
        prompt,
        "-n",
        str(gen_tokens),
        "-c",
        str(ctx_cap),
        "-st",
        "--no-warmup",
        "--temp",
        "0",
        "-ngl",
        "99",
        "-v",
    ]

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        filter(
            None,
            [
                os.path.expanduser("~/mmcuda/lib"),
                os.path.expanduser(
                    "~/.local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
                ),
                env.get("LD_LIBRARY_PATH", ""),
            ],
        )
    )

    r = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=timeout,
        cwd=ROOT,
        env=env,
    )

    if r.returncode != 0:
        err_snippet = r.stderr[-500:] if r.stderr else r.stdout[-500:]
        raise RuntimeError(
            f"Oracle llama-cli failed (rc={r.returncode}) on {model_path.name}:\n{err_snippet}"
        )

    combined = r.stdout + "\n" + r.stderr
    timings_matches = re.findall(r"\"timings\":\{[^}]*\}", combined)
    if timings_matches:
        try:
            tdata = json.loads("{" + timings_matches[-1] + "}")["timings"]
            prompt_n = int(tdata.get("prompt_n", 0))
            predicted_n = int(tdata.get("predicted_n", gen_tokens))
            prompt_ms = float(tdata.get("prompt_ms", 0.0))
            predicted_ms = float(tdata.get("predicted_ms", 0.0))
            prompt_per_sec = float(tdata.get("prompt_per_second", (prompt_n / (prompt_ms / 1000.0)) if prompt_ms > 0 else 0.0))
            predicted_per_sec = float(tdata.get("predicted_per_second", (predicted_n / (predicted_ms / 1000.0)) if predicted_ms > 0 else 0.0))
            predicted_per_token_ms = float(tdata.get("predicted_per_token_ms", (predicted_ms / predicted_n) if predicted_n > 0 else 0.0))

            prefill_us = prompt_ms * 1000.0
            decode_us = predicted_ms * 1000.0
            first_token_us = (prompt_ms + predicted_per_token_ms) * 1000.0
            total_us = (prompt_ms + predicted_ms) * 1000.0

            return {
                "tokens": predicted_n,
                "prefill": prompt_n,
                "prompt_n": prompt_n,
                "predicted_n": predicted_n,
                "decode_us": decode_us,
                "prefill_us": prefill_us,
                "first_token_us": first_token_us,
                "total_us": total_us,
                "decode_tok_s": predicted_per_sec,
                "prefill_tok_s": prompt_per_sec,
            }
        except Exception:
            pass

    # Fallback regex for stdout
    m = re.search(
        r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]",
        r.stdout,
    )
    if not m:
        raise RuntimeError(
            f"Failed to parse timing from llama-cli output on {model_path.name}:\n{combined[-600:]}"
        )

    pp_tps = float(m.group(1))
    tg_tps = float(m.group(2))
    decode_us = (gen_tokens / tg_tps * 1e6) if tg_tps > 0 else 0.0
    prefill_us = (32 / pp_tps * 1e6) if pp_tps > 0 else 0.0
    return {
        "tokens": gen_tokens,
        "prefill": 32,
        "prompt_n": 32,
        "predicted_n": gen_tokens,
        "decode_us": decode_us,
        "prefill_us": prefill_us,
        "first_token_us": prefill_us + (decode_us / gen_tokens if gen_tokens > 0 else 0.0),
        "total_us": prefill_us + decode_us,
        "decode_tok_s": tg_tps,
        "prefill_tok_s": pp_tps,
    }


def benchmark_case(
    model_path: Path,
    target_ctx: int,
    gen_tokens: int,
    mode: str,
    runs: int,
    warmup: int,
    oracle_bin: Optional[str],
    custom_prompt: Optional[str] = None,
    raw_prompt: bool = False,
) -> Dict[str, Any]:
    prompt = make_prompt_for_ctx(target_ctx, custom_prompt)
    ctx_cap = max(1024, target_ctx + gen_tokens + 256)
    row_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    power = get_power_info()

    # 1. TinyInference Warmup
    for _ in range(warmup):
        run_engine_once(
            model_path, prompt, min(16, gen_tokens), ctx_cap, mode, raw_prompt
        )

    # 2. TinyInference Repetitions
    ours_samples: Dict[str, List[float]] = {
        "decode_us": [],
        "prefill_us": [],
        "first_token_us": [],
        "total_us": [],
        "decode_tok_s": [],
        "prefill_tok_s": [],
    }
    actual_tokens = gen_tokens
    actual_prefill = 0
    graph_statuses = []

    for _ in range(runs):
        out = run_engine_once(
            model_path, prompt, gen_tokens, ctx_cap, mode, raw_prompt
        )
        actual_tokens = out["tokens"]
        actual_prefill = out["prefill"]
        graph_statuses.append(out["graph_status"])
        for k in ours_samples:
            ours_samples[k].append(out[k])

    ours_summary = {k: compute_stats(v) for k, v in ours_samples.items()}
    primary_graph_status = graph_statuses[0] if graph_statuses else "unknown"

    # 3. Oracle Repetitions (if enabled)
    oracle_summary = None
    ratio = None
    if oracle_bin:
        for _ in range(warmup):
            run_oracle_once(
                oracle_bin, model_path, prompt, min(16, gen_tokens), ctx_cap
            )

        oracle_samples: Dict[str, List[float]] = {
            "decode_us": [],
            "prefill_us": [],
            "first_token_us": [],
            "total_us": [],
            "decode_tok_s": [],
            "prefill_tok_s": [],
        }
        oracle_prompt_ns: List[int] = []
        oracle_predicted_ns: List[int] = []
        for _ in range(runs):
            oout = run_oracle_once(
                oracle_bin, model_path, prompt, gen_tokens, ctx_cap
            )
            for k in oracle_samples:
                oracle_samples[k].append(oout[k])
            oracle_prompt_ns.append(int(oout.get("prompt_n", 0)))
            oracle_predicted_ns.append(int(oout.get("predicted_n", 0)))
        oracle_summary = {k: compute_stats(v) for k, v in oracle_samples.items()}
        oracle_summary["prompt_n"] = compute_stats([float(x) for x in oracle_prompt_ns])
        oracle_summary["predicted_n"] = compute_stats([float(x) for x in oracle_predicted_ns])
        oracle_med_dec = oracle_summary["decode_tok_s"]["median"]
        ours_med_dec = ours_summary["decode_tok_s"]["median"]
        ratio = round(ours_med_dec / oracle_med_dec, 3) if oracle_med_dec > 0 else None

    return {
        "target_ctx": target_ctx,
        "prompt_tokens": actual_prefill,
        "generation_tokens": actual_tokens,
        "mode": mode,
        "engine_graph_status": primary_graph_status,
        "repetitions": runs,
        "ours": ours_summary,
        "oracle": oracle_summary,
        "ratio_decode_tok_s": ratio,
        "row_start": row_start,
        "row_end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "power": power,
    }


def write_jsonl(results: List[Dict[str, Any]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, separators=(",", ":")) + "\n")


def write_markdown(results: List[Dict[str, Any]], out_path: Path, meta: Dict[str, Any]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Scoreboard Benchmark: TinyInference vs llama.cpp",
        "",
        f"- **Date**: {meta['timestamp']}",
        f"- **TinyInference Commit**: `{meta['tinyinference_commit']}`",
        f"- **llama.cpp Commit**: `{meta['llamacpp_commit']}` (`{meta['llamacpp_bin']}`)",
        f"- **Hardware**: {meta['gpu']} ({meta['sm']}), CUDA {meta['cuda_version']}",
        f"- **Runs**: {meta['runs']} measured runs post-warmup",
        f"- **Warmup/Cooldown**: {meta['warmup']} warmup, {meta.get('cooldown_sec', 0)}s cooldown between cells",
        "",
        "## Performance Scoreboard",
        "",
        "| Model | Quant | Mode | Ctx | Prompt | Gen | Ours Decode | Oracle Decode | Ratio | Ours Prefill | Oracle Prefill | TTFT (ms) | Status |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    ratios_graph = []
    ratios_eager = []
    all_ratios = []

    for r in results:
        m_name = r["model_name"]
        quant = r["quantization"]
        mode = r["mode"]
        ctx = r["context_target"]
        p_tok = r["prompt_tokens"]
        g_tok = r["generation_tokens"]

        ours_dec = r["ours"]["decode_tok_s"]["median"]
        ours_pf = r["ours"]["prefill_tok_s"]["median"]
        ours_ttft = r["ours"]["first_token_us"]["median"] / 1000.0

        if r["oracle"]:
            orc_dec = r["oracle"]["decode_tok_s"]["median"]
            orc_pf = r["oracle"]["prefill_tok_s"]["median"]
            ratio = r["ratio_decode_tok_s"]
            ratio_str = f"**{ratio:.3f}x**"
            all_ratios.append(ratio)
            if mode == "graph":
                ratios_graph.append(ratio)
            else:
                ratios_eager.append(ratio)
        else:
            orc_dec = "-"
            orc_pf = "-"
            ratio_str = "-"

        status = r["engine_graph_status"]
        lines.append(
            f"| {m_name} | {quant} | {mode} | {ctx} | {p_tok} | {g_tok} | "
            f"{ours_dec:.1f} tok/s | {orc_dec if orc_dec == '-' else f'{orc_dec:.1f} tok/s'} | "
            f"{ratio_str} | {ours_pf:.1f} tok/s | "
            f"{orc_pf if orc_pf == '-' else f'{orc_pf:.1f} tok/s'} | "
            f"{ours_ttft:.2f} | {status} |"
        )

    lines.append("")
    lines.append("## Geomean Summary")
    lines.append("")

    def calc_geomean(vals: List[float]) -> str:
        pos_vals = [x for x in vals if x is not None and x > 0]
        if not pos_vals:
            return "N/A"
        gm = math.exp(sum(math.log(x) for x in pos_vals) / len(pos_vals))
        return f"**{gm:.3f}x**"

    lines.append(f"- **Geomean Ratio (Graph mode)**: {calc_geomean(ratios_graph)}")
    lines.append(f"- **Geomean Ratio (Eager mode)**: {calc_geomean(ratios_eager)}")
    lines.append(f"- **Geomean Ratio (Overall)**: {calc_geomean(all_ratios)}")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def update_scoreboard_csv(results: List[Dict[str, Any]], csv_path: Path):
    fieldnames = [
        "model",
        "quant",
        "mode",
        "ctx",
        "prompt_tok",
        "gen_tok",
        "ours_tg_tps",
        "oracle_tg_tps",
        "ratio",
        "ours_pp_tps",
        "oracle_pp_tps",
        "prefill_us",
        "decode_us",
        "first_token_us",
        "total_us",
    ]

    new_rows = []
    for r in results:
        m_stem = Path(r["model_name"]).stem
        m_clean = re.sub(r"[-_](q\d+.*|f\d+.*)$", "", m_stem, flags=re.IGNORECASE)
        quant = r["quantization"].lower()
        mode = r["mode"]
        ctx = r["context_target"]

        new_rows.append(
            {
                "model": m_clean,
                "quant": quant,
                "mode": mode,
                "ctx": ctx,
                "prompt_tok": r["prompt_tokens"],
                "gen_tok": r["generation_tokens"],
                "ours_tg_tps": r["ours"]["decode_tok_s"]["median"],
                "oracle_tg_tps": r["oracle"]["decode_tok_s"]["median"] if r["oracle"] else "",
                "ratio": r["ratio_decode_tok_s"] if r["ratio_decode_tok_s"] is not None else "",
                "ours_pp_tps": r["ours"]["prefill_tok_s"]["median"],
                "oracle_pp_tps": r["oracle"]["prefill_tok_s"]["median"] if r["oracle"] else "",
                "prefill_us": r["ours"]["prefill_us"]["median"],
                "decode_us": r["ours"]["decode_us"]["median"],
                "first_token_us": r["ours"]["first_token_us"]["median"],
                "total_us": r["ours"]["total_us"]["median"],
            }
        )

    # Read existing rows if any, filtering out legacy/blank rows
    existing_rows = []
    if csv_path.is_file():
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("mode") and row.get("ours_tg_tps"):
                        existing_rows.append(row)
        except Exception:
            existing_rows = []

    # Merge or overwrite by (model, quant, mode, ctx)
    key_fn = lambda r: (
        r.get("model", ""),
        r.get("quant", "").lower(),
        r.get("mode", ""),
        str(r.get("ctx", "")),
    )

    merged = {key_fn(r): r for r in existing_rows}
    for nr in new_rows:
        merged[key_fn(nr)] = nr

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged.values())


def main():
    parser = argparse.ArgumentParser(
        description="TinyInference Authoritative Scoreboard Runner"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Single model path or name to benchmark",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model paths to benchmark",
    )
    parser.add_argument(
        "--fleet",
        action="store_true",
        help="Run full primary fleet matrix",
    )
    parser.add_argument(
        "--ctxs",
        type=str,
        default=",".join(map(str, DEFAULT_CTXS)),
        help="Comma-separated context targets (e.g. 32,512,2048)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=None,
        help="Single target context token count",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_GEN_TOKENS,
        help="Generation tokens (default: 64)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="Repetitions after warmup (default: 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="Warmup runs (default: 3; v5 showed oracle cold-start ramp with 1)",
    )
    parser.add_argument(
        "--cooldown-sec",
        type=float,
        default=DEFAULT_COOLDOWN_SEC,
        help="Sleep between benchmark cells to reduce thermal drift (default: 10s; 0 disables)",
    )
    parser.add_argument(
        "--mode",
        choices=["graph", "eager", "both"],
        default="both",
        help="Execution mode(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--oracle-bin",
        type=str,
        default=DEFAULT_ORACLE_BIN,
        help="Path to llama-cli binary",
    )
    parser.add_argument(
        "--no-oracle",
        action="store_true",
        help="Skip oracle llama.cpp comparison",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt string (overrides ctx generation)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        default=False,
        help="Use raw prompt without chat template (TT_RAW_PROMPT=1)",
    )
    parser.add_argument(
        "--jsonl",
        type=str,
        default="data/bench/results_scoreboard.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--md",
        type=str,
        default="data/bench/results_scoreboard.md",
        help="Output Markdown path",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="bench/scoreboard_decode.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--parity-gate",
        action="store_true",
        help="Run tests/gate_m6_logit_parity.py after benchmarking",
    )
    parser.add_argument(
        "--allow-few-runs",
        action="store_true",
        help="Allow fewer than 5 repetitions (bypasses M0 protocol requirement)",
    )

    args = parser.parse_args()

    if args.runs < 5 and not args.allow_few_runs:
        parser.error(
            "M0 protocol violation: --runs must be >= 5 (minimum 5 repetitions post-warmup). "
            "Pass --allow-few-runs to bypass for quick test runs."
        )

    # Determine models
    if args.model:
        model_paths = [args.model]
    elif args.models:
        model_paths = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        # Default to primary fleet or default model
        model_paths = PRIMARY_MODELS

    # Validate and resolve all models up front (no silent skips!)
    resolved_models: List[Path] = []
    for mp in model_paths:
        try:
            resolved_models.append(resolve_model_path(mp))
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    # Determine contexts
    if args.ctx is not None:
        ctxs = [args.ctx]
    else:
        ctxs = [int(x.strip()) for x in args.ctxs.split(",") if x.strip()]

    # Determine modes
    if args.mode == "both":
        modes = ["graph", "eager"]
    else:
        modes = [args.mode]

    # Check oracle binary
    oracle_bin = None if args.no_oracle else args.oracle_bin
    if oracle_bin and not os.path.isfile(oracle_bin):
        print(f"ERROR: Oracle binary '{oracle_bin}' does not exist!", file=sys.stderr)
        sys.exit(1)

    # Collect metadata
    tt_commit = get_git_commit(ROOT)
    tt_short_commit = get_short_git_commit(ROOT)
    oracle_bin_path, llama_commit = (
        get_llamacpp_info(oracle_bin) if oracle_bin else ("none", "none")
    )
    gpu_info = get_gpu_info()
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    power_info = get_power_info()
    if power_info.get("power_limit_w") is not None and abs(power_info["power_limit_w"] - 60.0) > 0.5:
        print(
            f"WARNING: GPU power limit is {power_info['power_limit_w']}W (expected 60W, AC). "
            f"AC online={power_info.get('ac_online')}. Results NOT citable for parity.",
            file=sys.stderr,
        )

    meta = {
        "timestamp": timestamp,
        "tinyinference_commit": tt_commit,
        "tinyinference_short_commit": tt_short_commit,
        "llamacpp_commit": llama_commit,
        "llamacpp_bin": oracle_bin_path,
        "gpu": gpu_info["gpu"],
        "sm": gpu_info["sm"],
        "cuda_version": gpu_info["cuda"],
        "power": power_info,
        "runs": args.runs,
        "warmup": args.warmup,
        "gen_tokens": args.tokens,
        "cooldown_sec": args.cooldown_sec,
    }

    print("=" * 80)
    print("  TinyInference Authoritative Scoreboard Runner (M0)")
    print("=" * 80)
    print(f"TinyInference commit : {tt_short_commit} ({tt_commit})")
    print(f"llama.cpp commit     : {llama_commit} ({oracle_bin_path})")
    print(f"Device               : {gpu_info['gpu']} ({gpu_info['sm']}), CUDA {gpu_info['cuda']}")
    print(f"Models               : {[m.name for m in resolved_models]}")
    print(f"Contexts             : {ctxs}")
    print(f"Modes                : {modes}")
    print(f"Repetitions          : {args.runs} post-warmup")
    print("=" * 80)

    scoreboard_results: List[Dict[str, Any]] = []

    for model_path in resolved_models:
        model_sha = compute_sha256(model_path)
        model_quant = detect_quant(model_path.name)
        file_size = model_path.stat().st_size

        print(f"\n>>> Model: {model_path.name} ({file_size / (1024*1024):.1f} MB, {model_quant}, SHA256: {model_sha[:12]}...)")

        for ctx in ctxs:
            for mode in modes:
                print(f"  --> Benchmarking ctx ~{ctx} in {mode.upper()} mode... ", end="", flush=True)
                t0_bench = time.time()
                try:
                    case_res = benchmark_case(
                        model_path=model_path,
                        target_ctx=ctx,
                        gen_tokens=args.tokens,
                        mode=mode,
                        runs=args.runs,
                        warmup=args.warmup,
                        oracle_bin=oracle_bin,
                        custom_prompt=args.prompt,
                        raw_prompt=args.raw,
                    )
                except Exception as e:
                    print(f"FAILED!\nERROR: {e}", file=sys.stderr)
                    sys.exit(1)

                elapsed = time.time() - t0_bench
                ours_med = case_res["ours"]["decode_tok_s"]["median"]
                orc_med = (
                    case_res["oracle"]["decode_tok_s"]["median"]
                    if case_res["oracle"]
                    else 0.0
                )
                ratio_str = (
                    f"{case_res['ratio_decode_tok_s']:.3f}x"
                    if case_res["ratio_decode_tok_s"] is not None
                    else "-"
                )
                status = case_res["engine_graph_status"]

                print(
                    f"done ({elapsed:.1f}s) | ours: {ours_med:.1f} tok/s | "
                    f"oracle: {orc_med:.1f} tok/s | ratio: {ratio_str} | status: {status}"
                )

                if args.cooldown_sec and args.cooldown_sec > 0:
                    time.sleep(args.cooldown_sec)

                # Format full result record
                record = {
                    "timestamp": timestamp,
                    "tinyinference_commit": tt_commit,
                    "llamacpp_commit": llama_commit,
                    "llamacpp_bin": oracle_bin_path,
                    "model_path": str(model_path),
                    "model_name": model_path.name,
                    "model_sha256": model_sha,
                    "model_size_bytes": file_size,
                    "quantization": model_quant,
                    "gpu": gpu_info["gpu"],
                    "sm": gpu_info["sm"],
                    "cuda_version": gpu_info["cuda"],
                    "context_target": ctx,
                    "prompt_tokens": case_res["prompt_tokens"],
                    "generation_tokens": case_res["generation_tokens"],
                    "mode": mode,
                    "engine_graph_status": case_res["engine_graph_status"],
                    "repetitions": args.runs,
                    "ours": case_res["ours"],
                    "oracle": case_res["oracle"],
                    "ratio_decode_tok_s": case_res["ratio_decode_tok_s"],
                    "row_start": case_res.get("row_start"),
                    "row_end": case_res.get("row_end"),
                    "power": case_res.get("power"),
                }
                scoreboard_results.append(record)

    # Write output artifacts
    jsonl_path = ROOT / args.jsonl
    md_path = ROOT / args.md
    csv_path = ROOT / args.csv

    print("\n" + "=" * 80)
    print("Writing Authoritative Scoreboard Outputs:")
    print(f"  - JSONL : {jsonl_path}")
    write_jsonl(scoreboard_results, jsonl_path)
    print(f"  - MD    : {md_path}")
    write_markdown(scoreboard_results, md_path, meta)
    print(f"  - CSV   : {csv_path}")
    update_scoreboard_csv(scoreboard_results, csv_path)
    print("=" * 80)

    # Optional parity gate
    if args.parity_gate:
        print("\nRunning parity gate tests/gate_m6_logit_parity.py...")
        g = subprocess.run(
            [sys.executable, "tests/gate_m6_logit_parity.py"],
            capture_output=True,
            text=True,
            timeout=1200,
            cwd=ROOT,
        )
        print(g.stdout.strip())
        if g.returncode != 0:
            print(g.stderr, file=sys.stderr)
            sys.exit(1)

    print("\nScoreboard benchmark complete.")


if __name__ == "__main__":
    main()
