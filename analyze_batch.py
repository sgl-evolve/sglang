#!/usr/bin/env python3
"""Quick analysis of batch eval results. Run from workspace root."""
import json, os, glob, sys

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")
BASELINE_HIT = 0.622
EXCL_HIT = 0.733

def load_summary(path):
    d = json.load(open(path))
    p = d.get("panel", {})
    return {
        "version": d.get("version", os.path.basename(os.path.dirname(path))),
        "hit": p.get("overall/hit_rate") or 0,
        "p50": p.get("overall/ttft_p50_ms") or 0,
        "p99": p.get("overall/ttft_p99_ms") or 0,
        "mean": p.get("overall/ttft_mean_ms") or 0,
        "req_s": p.get("overall/req_throughput") or 0,
        "host_util": p.get("overall/host_util") or 0,
        "load_back_ms": p.get("overall/load_back_mean_ms") or 0,
    }

summaries = []
for sf in sorted(glob.glob(os.path.join(RUNS_DIR, "*/summary.json"))):
    try:
        summaries.append(load_summary(sf))
    except Exception as e:
        print(f"WARN: {sf}: {e}", file=sys.stderr)

if not summaries:
    print("No summary.json files found.")
    sys.exit(1)

print(f"{'version':25s} {'hit':>7s} {'Δhit':>7s} {'p50':>7s} {'p99':>7s} {'mean':>7s} {'req/s':>6s} {'host':>6s} {'lb_ms':>6s}")
print("-" * 95)
for s in summaries:
    dhit = s["hit"] - BASELINE_HIT if s["hit"] > 0 else 0
    dhit_s = f"{dhit:+.4f}" if dhit != 0 else "   ---"
    slo = " >SLO!" if s["p99"] > 8000 else ""
    print(f"{s['version']:25s} {s['hit']:7.4f} {dhit_s:>7s} {s['p50']:7.0f} {s['p99']:7.0f}{slo:6s} {s['mean']:7.0f} {s['req_s']:6.2f} {s['host_util']:6.4f} {s['load_back_ms']:6.1f}")

print(f"\nTotal versions with summary.json: {len(summaries)}")
print(f"Baseline reference: hit={BASELINE_HIT}, excl reference: hit={EXCL_HIT}")
