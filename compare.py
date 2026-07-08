#!/usr/bin/env python3
"""Compare a version's runs/<ver>/summary.json panel to v0_official baseline.json.
Usage: python compare.py <version> [more versions...]"""
import json, sys, os
BASE = "/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/base_free/researcher/baseline.json"
KEYS = [
    ("overall/ttft_p50_ms", "TTFT p50", 0),
    ("overall/ttft_p99_ms", "TTFT p99 (SLO<8000)", 0),
    ("overall/ttft_mean_ms", "TTFT mean", 0),
    ("overall/req_throughput", "req/s (lambda=3)", 1),
    ("overall/out_tok_s", "out tok/s", 1),
    ("overall/hit_rate", "hit_rate", 1),
    ("overall/host_util", "host_util", 1),
    ("mix_hicache/load_back_tokens", "load_back_tok", 0),
    ("mix_hicache/evict_tokens", "evict_tok", 0),
    ("mix_hicache/hit_device_frac", "hit_device_frac", 1),
]
base = json.load(open(BASE))["panel"]
def load(v):
    p = f"runs/{v}/summary.json"
    return json.load(open(p))["panel"] if os.path.exists(p) else None
vers = sys.argv[1:]
print(f"{'metric':22s} {'v0_official':>14s}" + "".join(f"{v:>16s}" for v in vers))
for k, label, hib in KEYS:
    b = base.get(k)
    row = f"{label:22s} {b if b is None else f'{b:,.2f}':>14s}"
    for v in vers:
        p = load(v)
        val = p.get(k) if p else None
        if val is None:
            row += f"{'—':>16s}"
        else:
            d = (val - b) / b * 100 if (b not in (None, 0)) else 0
            row += f"{val:>10,.2f}({d:+.0f}%)"
    print(row)
