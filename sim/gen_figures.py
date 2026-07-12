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

# ---- Fig 3: Belady liveness headroom vs capacity (live from the sim replay) ----
# Timing-independent fixed-order replay: LRU vs Belady (evict-farthest-future) recompute over the served
# access order at several cache capacities. At the real 2-tier cap the live working set fits -> Belady=0.
import sim.simulate as _S
_convs = _S.load()
_base = _S.Sim(_convs, 3, 10.7e6, "lru", 25000, 4000, record=True); _base.run()
_ao = _base.access_order
caps = [4e6, 6e6, 8e6, 10.7e6, 15e6]
lru_rc, opt_rc = [], []
for _cap in caps:
    hr = _S.trace_headroom(_ao, _convs, _cap)
    lru_rc.append(hr["lru"] / 1e6); opt_rc.append(hr["opt"] / 1e6)
head = [100 * (l - o) / l if l else 0 for l, o in zip(lru_rc, opt_rc)]
xM = [c / 1e6 for c in caps]
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(xM, lru_rc, "o-", color="#d62728", lw=2, label="LRU recompute")
ax.plot(xM, opt_rc, "s-", color="#55a868", lw=2, label="Belady (oracle) recompute")
ax.fill_between(xM, opt_rc, lru_rc, color="#f2c14e", alpha=.35, label="avoidable (liveness headroom)")
ax.axvline(10.7, color="gray", ls="--", lw=1); ax.text(10.7, ax.get_ylim()[1]*.9, " real L1+L2\n cap 10.7M", fontsize=7, color="gray")
for x, h in zip(xM, head):
    ax.text(x, (opt_rc[xM.index(x)]+lru_rc[xM.index(x)])/2, f"{h:.0f}%", ha="center", va="center", fontsize=7, color="#7a5c00")
ax.set_xlabel("cache capacity (M tokens)  [total working set = 20.2M, 1.89× at real cap]")
ax.set_ylabel("recompute (M tokens)")
ax.set_title("Liveness headroom: Belady=0 at real cap (live set fits)", fontsize=10)
ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_headroom.svg")); plt.close(fig)
print("headroom vs cap:", {f"{x:.1f}M": f"{h:.0f}%" for x, h in zip(xM, head)})

print("wrote:", sorted(os.listdir(OUT)))
# stats echo
print(f"perfect p99={np.percentile(perfect,99):.0f} nocache p99={np.percentile(nocache,99):.0f} single-turn%={frac1:.1f}")
