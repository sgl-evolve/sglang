#!/usr/bin/env python3
"""fig5_crossnode.svg — the central result: whale-vs-LRU λ=3 p99 on TWO nodes.
nodeset-0 (whale n=6, LRU n=5): whale clusters below LRU (same-node A/B significant, p~0.026).
certified 0-3 (whale n=3, LRU n=3): the separation VANISHES (LRU median below whale) -> node-specific / metastable.
Hardcoded from the measured λ=3 p99 values (report sec50-76). Usage: .venv/bin/python sim/fig_crossnode.py"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statistics as st

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "submissions/hicache-goodput-limits")

# measured λ=3 p99 (ms)
DATA = {
    "nodeset-0 (same-node A/B, p≈0.026)": {
        "whale": [6189, 6574, 6698, 8715, 9287, 26589],
        "LRU":   [10360, 11254, 13991, 17133, 28325],
    },
    "certified 0-3 (whale > LRU: reversed)": {
        "whale": [6794, 11461, 20406],
        "LRU":   [5893, 6129, 31203],
    },
    "certified 1-2 (whale < LRU again)": {
        "whale": [8235, 7088],
        "LRU":   [21907, 8089],
    },
}
COL = {"whale": "#d62728", "LRU": "#8c8c8c"}

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)
for ax, (title, d) in zip(axes, DATA.items()):
    for i, pol in enumerate(["LRU", "whale"]):
        ys = [v/1000.0 for v in d[pol]]              # -> seconds
        xs = [i + (k-(len(ys)-1)/2)*0.11 for k in range(len(ys))]
        ax.scatter(xs, ys, color=COL[pol], s=46, zorder=3, edgecolor="white", linewidth=.6)
        ax.plot([i-0.24, i+0.24], [st.median(ys)]*2, color=COL[pol], lw=2.4, zorder=2)  # median bar
    ax.axhline(8, color="k", ls="--", lw=1.1); ax.text(1.45, 8.2, "8 s SLO", fontsize=7.5, ha="right", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LRU", "whale"], fontsize=9)
    ax.set_title(title, fontsize=9.3); ax.grid(alpha=.25, axis="y")
    ax.set_ylim(0, 33)
axes[0].set_ylabel("λ=3 p99 TTFT (s)", fontsize=9.5)
fig.suptitle("Cross-node non-reproduction: whale's same-node p99 advantage on nodeset-0 vanishes on a certified node",
             fontsize=9.6)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "fig5_crossnode.svg")); plt.close(fig)
for title, d in DATA.items():
    print(title, "| whale median", round(st.median(d["whale"])/1000,1), "s ; LRU median",
          round(st.median(d["LRU"])/1000,1), "s")
print("wrote", os.path.join(OUT, "fig5_crossnode.svg"))
