#!/usr/bin/env python3
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import profile_decode


class ProfileDecodeTests(unittest.TestCase):
    def test_selects_last_nonempty_profile_block(self):
        text = """PROFILE mode=eager
PROFILE qkv-gemv 9.000
PROFILE TOTAL(med) 9.000
STATS tokens=8 prefill=32 decode_us=80000 prefill_us=100000
PROFILE mode=eager
PROFILE qkv-gemv 1.250
PROFILE attn 0.500
PROFILE TOTAL(med) 1.750
"""
        block = profile_decode.parse_last_profile(text)
        self.assertEqual(block["mode"], "eager")
        self.assertEqual(block["stages"], {"qkv-gemv": 1.25, "attn": 0.5})

    def test_stage_columns_keep_named_buckets(self):
        cols = profile_decode.stage_columns({
            "qkv-gemv": 1.0,
            "attn": 2.0,
            "flash": 0.5,
            "o-proj": 3.0,
            "ffn-gateup": 4.0,
            "ffn-down": 5.0,
            "logits-gemv": 6.0,
            "rmsnorm": 0.25,
            "other": 0.75,
        })
        self.assertEqual(cols["qkv_ms"], 1.0)
        self.assertEqual(cols["attn_ms"], 2.5)
        self.assertEqual(cols["o_ms"], 3.0)
        self.assertEqual(cols["ffn_ms"], 9.0)
        self.assertEqual(cols["lmhead_ms"], 6.0)
        self.assertEqual(cols["other_ms"], 1.0)

    def test_output_is_separate_from_authoritative_scoreboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "profile.csv"
            row = {
                "model": "test", "quant": "q4_0", "mode": "eager-stage",
                "target_ctx": 32, "ctx": 32, "prompt_tok": 32, "gen_tok": 8,
                "pp_tps": 1.0, "tg_tps": 2.0, "prefill_us": 1.0,
                "decode_us": 1.0, "stage_sum_ms": 1.0, "stage_gap_pct": 0.0,
                "qkv_ms": 0.0, "attn_ms": 0.0, "o_ms": 0.0, "ffn_ms": 0.0,
                "lmhead_ms": 0.0, "other_ms": 0.0,
            }
            fields = tuple(row)
            with out.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)
            with out.open(newline="") as handle:
                parsed = next(csv.DictReader(handle))
            self.assertEqual(parsed["mode"], "eager-stage")
            self.assertNotEqual(out.resolve(), Path("bench/scoreboard_decode.csv").resolve())


if __name__ == "__main__":
    unittest.main()
