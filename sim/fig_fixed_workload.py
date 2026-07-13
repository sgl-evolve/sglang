#!/usr/bin/env python3
"""fig6_fixed_workload.svg — the central caution, visualized.

Across all full-throughput λ=3 runs the WORKLOAD is byte-identical (fixed seed=1 + --disable-shuffle;
each completes exactly 7037 requests, Σinput within 0.016%). Yet p99 TTFT swings 5.3x (5.9-31.2 s) while
the median (body) is pinned to ~0.54 s (1.1x) and throughput to 3.02 req/s. The tail is execution-driven,
not workload variance; and the p99 spread is policy-MIXED (whale & LRU interleave), so the metastability
swamps any policy signal -> single/few-run A/Bs are void for this metric.

Data: per-run bench_r3.json (thru>=3.0). Usage: .venv/bin/python sim/fig_fixed_workload.py"""
import os, glob, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "submissions/hicache-goodput-limits")

def policy(n):
    if n.startswith("whale"): return "whale"
    if n.startswith(("lru", "cert-lru")): return "LRU"
    if n.startswith("stag"): return "stagger"
    if "car" in n: return "CAR"
    return "other"

COL = {"whale": "#d62728", "LRU": "#8c8c8c", "stagger": "#1f77b4", "CAR": "#2ca02c", "other": "#bbbbbb"}

runs = []
for jf in glob.glob(os.path.join(ROOT, "runs/*/bench_r3.json")):
    try: d = json.load(open(jf))
    except Exception: continue
    if d.get("completed") != 7037 or d.get("request_throughput", 0) < 3.0:
        continue
    runs.append((os.path.basename(os.path.dirname(jf)), d["median_ttft_ms"]/1000, d["p99_ttft_ms"]/1000))
runs.sort(key=lambda r: r[2])                       # sort by p99
x = list(range(len(runs)))

fig, ax = plt.subplots(figsize=(8.2, 3.9))
# p99 markers colored by policy; median markers small grey
for i, (name, med, p99) in enumerate(runs):
    ax.scatter(i, p99, color=COL[policy(name)], s=58, zorder=3, edgecolor="white", linewidth=.6)
    ax.scatter(i, med, color="#333333", s=16, marker="_", zorder=3)
ax.axhline(8, color="k", ls="--", lw=1.1)
ax.text(len(runs)-0.5, 8.25, "8 s SLO", fontsize=8, ha="right", va="bottom")
ax.set_ylim(0, 33)
ax.set_xlabel("run (byte-identical workload: 7037 reqs, Σinput within 0.016%; sorted by p99)", fontsize=9.5)
ax.set_ylabel("λ=3 TTFT (s)", fontsize=10)
ax.set_title("Fixed workload, 5.3× tail: p99 (colored ●) swings 5.9–31.2 s while median (grey −) is pinned ~0.54 s",
             fontsize=9.6)
ax.grid(alpha=.25, axis="y")
# legend
from matplotlib.lines import Line2D
seen = [p for p in ["whale", "LRU", "stagger", "CAR"] if any(policy(n) == p for n, _, _ in runs)]
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[p], markersize=8, label="%s p99" % p) for p in seen]
handles.append(Line2D([0], [0], marker="_", color="#333333", markersize=10, lw=0, label="median TTFT"))
ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=.9, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_fixed_workload.svg")); plt.close(fig)

meds = [m for _, m, _ in runs]; p99s = [p for _, _, p in runs]
print("N=%d runs" % len(runs))
print("p99: %.1f-%.1f s (%.1fx)" % (min(p99s), max(p99s), max(p99s)/min(p99s)))
print("median: %.3f-%.3f s (%.1fx)" % (min(meds), max(meds), max(meds)/min(meds)))
print("wrote", os.path.join(OUT, "fig6_fixed_workload.svg"))
