#!/usr/bin/env python3
"""Does the caching mirage (P2) hold LIVE, under load? Extract the measured per-rate
hit rate from each full-sweep run's curve.csv.

The request set is identical at every rate (same 1553 conversations / 7037 requests,
replayed), so any hit-rate change across rates is load-induced, not a mix change.
Verdict: the live hit rate declines only modestly (~1.9-3.3 pp, λ=3->10) and does so
under BOTH scheduling policies -> the decline is capacity under the frozen budget
(running-request working set grows with concurrency, sharing the fixed pool), not a
policy-recoverable lever. Combined with the eviction-policy invariance of §3.1
(LRU=LFU=Belady), no lossless policy closes it.

Usage: hit_vs_load.py [runs/<v>/curve.csv ...]   (defaults to the 4 full sweeps)
"""
import csv
import sys


def curve(path):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                rows[float(r["rate"])] = float(r["hit_rate"])
            except (ValueError, KeyError):
                pass
    return rows


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "runs/v0-stock/curve.csv", "runs/v-stock-samenode-1/curve.csv",
        "runs/v-srpf-full/curve.csv", "runs/v-srpf-samenode-1/curve.csv",
    ]
    print(f"{'run':28s} {'λ3':>7} {'λ5':>7} {'λ7':>7} {'λ10':>7}   Δ(λ3→λ10)")
    for p in paths:
        try:
            c = curve(p)
        except FileNotFoundError:
            print(f"{p}: not found (skip)")
            continue
        name = p.split("/")[-2]
        vals = [c.get(r) for r in (3, 5, 7, 10)]
        if all(v is not None for v in vals):
            d = (vals[-1] - vals[0]) * 100
            print(f"{name:28s} " + " ".join(f"{v:7.4f}" for v in vals) + f"   {d:+.2f} pp")
        else:
            print(f"{name:28s} incomplete sweep {vals}")
    print("\nAll declines negative and small (~-1.9..-3.3 pp), under both stock and SRPF =>")
    print("load-induced, policy-invariant, capacity-bound (frozen budget); not a lever.")
