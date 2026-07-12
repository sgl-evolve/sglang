#!/usr/bin/env python3
"""Generate motivation figures (SVG) for the paper from the workload trace + sim results.
Outputs into submissions/hicache-goodput-limits/. Claim-independent (workload characterization)."""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "submissions", "hicache-goodput-limits"))
os.makedirs(OUT, exist_ok=True)
convs = json.load(open(os.path.join(HERE, "conv_trace.json")))

# per-turn uncached prefill under perfect cache (turn0=doc, turnN=new Q) and no cache (full accumulated)
perfect, nocache, nt = [], [], []
for c in convs:
    acc = 0; nt.append(len(c))
    for i, (inp, out) in enumerate(c):
        perfect.append(inp); nocache.append(acc + inp); acc += inp + out
perfect = np.array(perfect); nocache = np.array(nocache); nt = np.array(nt)

# ---- Fig 1: CCDF of per-turn uncached prefill + SLO budget lines (the P-vs-SLO criterion) ----
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for arr, lab, c in [(perfect, "perfect within-conv cache (irreducible)", "#1f77b4"),
                    (nocache, "no cache (full recompute)", "#d62728")]:
    xs = np.sort(arr); ccdf = 1.0 - np.arange(len(xs)) / len(xs)
    ax.plot(xs, 100 * ccdf, label=lab, color=c, lw=2)
for P, style in [(4000, ":"), (8000, "--"), (15000, "-.")]:
    ax.axvline(8 * P, color="gray", ls=style, lw=1, alpha=.8)
    ax.text(8 * P, 60, f" 8s@P={P//1000}K", rotation=90, va="bottom", fontsize=7, color="gray")
ax.axhline(1.0, color="k", lw=.6, alpha=.5); ax.text(300, 1.2, "p99 line (1%)", fontsize=7)
ax.set_xscale("log"); ax.set_xlabel("per-turn uncached prefill (tokens)")
ax.set_ylabel("% of turns exceeding x (CCDF)")
ax.set_title("Cold-context prefill tail vs 8s-SLO budget", fontsize=10)
ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.25, which="both")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_coldtail.svg")); plt.close(fig)

# ---- Fig 2: turns/conv histogram + single-turn fraction ----
fig, ax = plt.subplots(figsize=(5.2, 3.0))
bins = np.arange(1, 22)
ax.hist(np.clip(nt, 1, 21), bins=bins, color="#4c72b0", edgecolor="white")
frac1 = 100 * np.mean(nt == 1)
ax.axvline(1.5, color="#d62728", ls="--", lw=1)
ax.text(2, ax.get_ylim()[1] * .8, f"{frac1:.0f}% single-turn\n(cached, never reused)", fontsize=8, color="#d62728")
ax.set_xlabel("turns per conversation (clipped at 21)"); ax.set_ylabel("# conversations")
ax.set_title("Multi-turn structure: 39% single-turn cache pollution", fontsize=10)
ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_turns.svg")); plt.close(fig)

# ---- Fig 3: recompute headroom (sim v1/v2 numbers) ----
fig, ax = plt.subplots(figsize=(5.2, 3.0))
labels = ["LRU\n(no protect)", "LRU\n+stock protect", "Belady\n(oracle)"]
vals = [5.76, 2.0, 0.0]  # M tokens (sim v1 no-protect, v2 with-protect, oracle)
bars = ax.bar(labels, vals, color=["#d62728", "#dd8452", "#55a868"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + .1, f"{v:.1f}M", ha="center", fontsize=9)
ax.set_ylabel("avoidable recompute (M tokens)")
ax.set_title("Residency headroom: stock protection + dead-KV abundance", fontsize=10)
ax.grid(alpha=.25, axis="y"); fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_headroom.svg")); plt.close(fig)

print("wrote:", sorted(os.listdir(OUT)))
# stats echo
print(f"perfect p99={np.percentile(perfect,99):.0f} nocache p99={np.percentile(nocache,99):.0f} single-turn%={frac1:.1f}")
