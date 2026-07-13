#!/usr/bin/env python3
"""fig4_whale.svg — λ=3 TTFT per-run, whale vs lru vs car, with the 8s SLO line.
Reads bench_r3.json (median_ttft_ms, p99_ttft_ms) from each policy's run dirs so it auto-updates as replicates land.
Usage: .venv/bin/python sim/fig_whale.py"""
import json, os, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "submissions/hicache-goodput-limits")
RUNS = os.path.join(BASE, "runs")

# policy -> list of run dirs (λ=3 measurements)
POLICIES = {
    "lru":   ["screen-v0b", "cert-lru-full", "lru-r3", "lru-r4", "lru-r5"],
    "car90": ["screen-car90", "cert-car90-full"],
    "whale": ["whale-full", "whale-r2", "whale-r3", "whale-r4", "whale-r5", "whale-r6"],
}
COLORS = {"lru": "#8c8c8c", "car90": "#4c72b0", "whale": "#d62728"}
LABELS = {"lru": "LRU (stock)", "car90": "CAR (grace 90s)", "whale": "whale (size-aware)"}

def load(pol):
    med, p99 = [], []
    for d in POLICIES[pol]:
        f = os.path.join(RUNS, d, "bench_r3.json")
        if not os.path.exists(f):
            continue
        try:
            j = json.load(open(f))
            med.append(j["median_ttft_ms"]); p99.append(j["p99_ttft_ms"])
        except Exception:
            pass
    return med, p99

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
order = ["lru", "car90", "whale"]
for ax, key, title, ylab in [(a1, "p99", "p99 TTFT (SLO metric)", "p99 TTFT (ms)"),
                             (a2, "med", "median TTFT (body of dist.)", "median TTFT (ms)")]:
    for i, pol in enumerate(order):
        med, p99 = load(pol)
        ys = p99 if key == "p99" else med
        if not ys:
            continue
        xs = [i + (k - (len(ys)-1)/2) * 0.10 for k in range(len(ys))]
        ax.scatter(xs, ys, color=COLORS[pol], s=42, zorder=3, edgecolor="white", linewidth=.6)
        ax.plot([i-0.22, i+0.22], [sum(ys)/len(ys)]*2, color=COLORS[pol], lw=2.2, zorder=2)  # mean bar
    ax.set_xticks(range(len(order))); ax.set_xticklabels([LABELS[p] for p in order], fontsize=7.5, rotation=12)
    ax.set_ylabel(ylab, fontsize=9); ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=.25, axis="y")
    if key == "p99":
        ax.axhline(8000, color="k", ls="--", lw=1.2)
        ax.text(2.35, 8000, " 8s SLO", va="bottom", ha="right", fontsize=8, color="k")
        ax.set_ylim(0, 12500)
fig.suptitle("Size-aware eviction (whale) at λ=3, same node: crosses the SLO where LRU/CAR do not", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUT, "fig4_whale.svg")); plt.close(fig)
# report the data used
for pol in order:
    med, p99 = load(pol)
    print(f"{pol:6} n={len(p99)}  p99={[round(x) for x in p99]}  median={[round(x) for x in med]}")
print("wrote", os.path.join(OUT, "fig4_whale.svg"))
