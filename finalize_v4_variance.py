#!/usr/bin/env python3
"""Compute v4-champion 3-sample variance headline (mean/median/tail) from summary panels."""
import json, statistics, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = ["v4-parallel-besteffort", "v4r-repeat", "v4r2-repeat"]

def panel(run):
    p = os.path.join(BASE, "runs", run, "summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["panel"]

rows = []
for s in SAMPLES:
    pan = panel(s)
    if pan is None:
        print(f"{s}: NO SUMMARY YET"); continue
    rows.append((s, pan["overall/ttft_mean_ms"], pan["mix_latency/ttft_median_ms"],
                 pan["overall/ttft_p99_ms"]))

print(f"{'run':<24}{'mean_ttft':>12}{'median':>12}{'p99':>12}")
for s, m, med, p99 in rows:
    print(f"{s:<24}{m:>12.1f}{med:>12.1f}{p99:>12.1f}")

if len(rows) >= 2:
    means = [r[1] for r in rows]
    meds  = [r[2] for r in rows]
    p99s  = [r[3] for r in rows]
    print(f"\nn={len(rows)}")
    print(f"MEAN  ttft: {statistics.mean(means):.1f} +/- {statistics.pstdev(means):.1f} ms "
          f"(sample-std {statistics.stdev(means):.1f}), range [{min(means):.0f}, {max(means):.0f}], "
          f"CV={statistics.pstdev(means)/statistics.mean(means)*100:.1f}%")
    print(f"MEDIAN ttft: {statistics.mean(meds):.1f} +/- {statistics.pstdev(meds):.1f} ms "
          f"(range [{min(meds):.0f}, {max(meds):.0f}], CV={statistics.pstdev(meds)/statistics.mean(meds)*100:.1f}%)")
    print(f"P99  ttft: {statistics.mean(p99s):.1f} +/- {statistics.pstdev(p99s):.1f} ms "
          f"(range [{min(p99s):.0f}, {max(p99s):.0f}])")
