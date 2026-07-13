#!/usr/bin/env python3
"""
EMPIRICAL required-median-of-k for a reliable goodput@SLO (kleinrock), from REAL same-node stock draws.

This is the data-grounded companion to the MODEL required-k in tools/coinflip_sim.py (§5.5). It pools all
same-node (ondem-3), stock, cold-flush lambda=3 p99-TTFT draws we measured and asks, by bootstrap: what is
the smallest median-of-k for which >=95% of independent median-of-k experiments agree on the goodput verdict
(3 if median p99 <= SLO else 0)? If that k is large (or unresolved) at this operating point, it CONFIRMS the
model's prediction that median-of-k does not rescue goodput@SLO near the knee -- on measured data, not a model.

Pooled sources (all ondem-3, stock write_through, byte-identical eval flags, cold /flush per draw):
  runs/v0-medk/summary.json        (K=5 medk)
  runs/v0-medk-bign/summary.json   (K=10 medk-bign, this experiment)
  runs/v0-stock/summary.json       (the original single eval's lambda=3 point)
Deterministic (LCG, no Math.random/Date) so the number is reproducible.
"""
import json, os, statistics as st, math, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLO = 8000.0  # ms

def from_medk(path):
    try:
        s = json.load(open(path))
        return [float(r["ttft_p99_ms"]) for r in s.get("rows", []) if r.get("ttft_p99_ms")]
    except Exception:
        return []

def from_stock_curve(path, node_sub="ondem-3", rate=3):
    """Pull the lambda=rate p99 from a curve-format summary.json IF it was measured on the target node."""
    try:
        s = json.load(open(path))
        node = str(s.get("node", "")) + str(s.get("panel", {}))
        curve = s.get("curve") or s.get("rows") or []
        out = []
        for pt in curve:
            r = pt.get("rate") or pt.get("request_rate")
            p99 = pt.get("ttft_p99_ms") or pt.get("p99_ttft_ms") or pt.get("ttft_p99")
            if r is not None and abs(float(r) - rate) < 1e-6 and p99:
                out.append(float(p99))
        return out
    except Exception:
        return []

class LCG:
    def __init__(self, seed): self.s = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    def u(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF

def required_k(pool, target=0.95, kmax=25, experiments=4000, seed=42):
    rng = LCG(seed); n = len(pool)
    rows = []
    kk = None; verdict = "unresolved@k<=%d" % kmax
    for k in range(1, kmax + 1, 2):
        agree3 = 0
        for _ in range(experiments):
            s = sorted(pool[int(rng.u() * n)] for _ in range(k))
            if s[k // 2] <= SLO:
                agree3 += 1
        f3 = agree3 / experiments
        rows.append((k, f3))
        if kk is None and max(f3, 1 - f3) >= target:
            kk = k; verdict = "goodput=3" if f3 >= 0.5 else "goodput=0"
    return kk, verdict, rows

def report_pool(label, pool):
    pool = [x for x in pool if x and x > 0]
    n = len(pool)
    if n < 3:
        print(f"## {label}: NOT ENOUGH DRAWS (n={n})"); return
    med = st.median(pool); sd = st.pstdev(pool)
    m = abs(SLO - med); npass = sum(1 for x in pool if x <= SLO)
    print(f"## {label}  (stock, n={n} cold lambda=3 draws)")
    print(f"#  p99(ms) sorted = {sorted(round(x) for x in pool)}")
    print(f"#  median={med:.0f}ms  min={min(pool):.0f}  max={max(pool):.0f}  pstdev={sd:.0f}ms  "
          f"spread={max(pool)/min(pool):.1f}x")
    print(f"#  margin m=|SLO-median|={m:.0f}ms  sigma/m={sd/m if m>1 else float('inf'):.2f}  "
          f"pass_frac={npass/n:.2f} ({npass}/{n})")
    kk, verdict, rows = required_k(pool)
    print(f"#  bootstrap median-of-k agreement (target 95%):")
    for k, f3 in rows:
        mark = '  <-- 95% resolved' if (kk == k) else ''
        print(f"     k={k:2d}  P(goodput=3)={f3:.3f}  P(=0)={1-f3:.3f}{mark}")
    print(f"#  REQUIRED-K (95%): {kk if kk else '>%d (UNRESOLVED)'%rows[-1][0]}   verdict={verdict}\n")

if __name__ == "__main__":
    # ondem-3 same-node stock pool (medk K=5 + any bign draws + the original v0-stock lambda=3 point)
    ondem3 = (from_medk(os.path.join(ROOT, "runs", "v0-medk", "summary.json"))
              + from_medk(os.path.join(ROOT, "runs", "v0-medk-bign", "summary.json"))
              + from_stock_curve(os.path.join(ROOT, "runs", "v0-stock", "summary.json")))
    # node1-2 same-node stock pool (medk-n2 K=5)
    node12 = from_medk(os.path.join(ROOT, "runs", "v0-medk-n2", "summary.json"))
    print("# EMPIRICAL required-median-of-k, PER NODE (same-node stock cold lambda=3 draws)\n")
    report_pool("ondem-3", ondem3)
    report_pool("node1-2", node12)
