#!/usr/bin/env python3
"""Aggregate the two INDEPENDENT same-node A/B pairs (node 1-2 and node 0-3) into
error bars on the exclusive-tiering vs baseline p99/goodput DELTA. Node variance is
large in absolutes, but the WITHIN-node delta (excl - baseline on the identical node)
is the estimand; two independent nodes give a spread on that delta."""
import json, os, glob, statistics as st

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load(p):
    try: return json.load(open(p))
    except Exception: return None

# (node, baseline_dir, excl_dir, filename_pattern)
PAIRS = [
    ("1-2", "runs/sweep-basesweep/r{R}.json", "runs/sweep-excl/r{R}.json"),
    ("0-3", "runs/paired-eb1/baseline-r{R}.json", "runs/paired-eb1/excl-r{R}.json"),
    ("ondem-3", "runs/paired-eb2/baseline-r{R}.json", "runs/paired-eb2/excl-r{R}.json"),
]
RATES = [3, 4]

def get(pat, R):
    d = load(os.path.join(WORK, pat.format(R=R)))
    if not d: return None
    return d.get("p99_ttft_ms"), d.get("request_throughput"), d.get("median_ttft_ms"), d.get("completed")

print(f"{'node':>5} {'λ':>2} | {'base p99':>9} {'excl p99':>9} {'Δp99%':>7} | {'base rps':>8} {'excl rps':>8} {'Δrps%':>6}")
print("-"*72)
dp99 = {R: [] for R in RATES}; drps = {R: [] for R in RATES}
for node, bpat, epat in PAIRS:
    for R in RATES:
        b = get(bpat, R); e = get(epat, R)
        if not b or not e:
            print(f"{node:>5} {R:>2} | (pending)"); continue
        bp99, brps, _, bc = b; ep99, erps, _, ec = e
        dp = 100*(ep99/bp99 - 1); dr = 100*(erps/brps - 1)
        dp99[R].append(dp); drps[R].append(dr)
        print(f"{node:>5} {R:>2} | {bp99:>9.0f} {ep99:>9.0f} {dp:>+6.1f}% | {brps:>8.2f} {erps:>8.2f} {dr:>+5.1f}%")

print("\n=== ERROR BARS on the excl-vs-baseline delta (across the 2 same-node pairs) ===")
for R in RATES:
    if len(dp99[R]) >= 2:
        print(f"λ={R}: Δp99 = {st.mean(dp99[R]):+.1f}% ± {st.pstdev(dp99[R]):.1f}pp  (n={len(dp99[R])}: {[round(x,1) for x in dp99[R]]})")
        print(f"      Δrps = {st.mean(drps[R]):+.1f}% ± {st.pstdev(drps[R]):.1f}pp  (n={len(drps[R])}: {[round(x,1) for x in drps[R]]})")
    elif dp99[R]:
        print(f"λ={R}: only n={len(dp99[R])} so far: Δp99 {[round(x,1) for x in dp99[R]]}, Δrps {[round(x,1) for x in drps[R]]}")
