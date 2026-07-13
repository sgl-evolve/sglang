#!/usr/bin/env python3
"""Fold v9 stock replicates into the noise-floor + tail-variance picture.
Run after job 19662 completes. Prints updated §5.1 noise-floor stats and the
fixed-config p99 spread that backs Table 3's 'the tail swing is pure variance' claim.
Node is read from each run's server.log hostname so we group ondem-2 correctly."""
import csv, os, re, glob, statistics as st

RUNS = "runs"

def hit_p99(v, rate="3"):
    p = f"{RUNS}/{v}/curve.csv"
    if not os.path.exists(p): return None
    for r in csv.DictReader(open(p)):
        if r["rate"] == rate:
            try: return float(r["hit_rate"]), float(r["ttft_p99_ms"])
            except: return None
    return None

def node_of(v):
    for f in (f"{RUNS}/{v}/server.log", f"eval-{v}.log"):
        if os.path.exists(f):
            t = open(f, errors="ignore").read(4000)
            m = re.search(r"a3nodeset[- ]?([\w-]+)", t)
            if m: return m.group(1)
    return "?"

stock = sorted(v.split("/")[1] for v in glob.glob(f"{RUNS}/stock_*"))
print(f"{'run':16} {'node':10} {'hit@3':>7} {'p99@3(s)':>9}")
by_node = {}
for v in stock:
    hp = hit_p99(v)
    if not hp:
        print(f"{v:16} {'(no curve yet)':>28}"); continue
    h, p = hp; n = node_of(v)
    print(f"{v:16} {n:10} {h:7.4f} {p/1000:9.1f}")
    by_node.setdefault(n, {"hit": [], "p99": []})
    by_node[n]["hit"].append(h); by_node[n]["p99"].append(p)

# Explicit same-node (ondem-2) FCFS-stock group — known provenance (all pinned/known ondem-2, FCFS).
# stock_lpm is ondem-2 but LPM (different scheduler) -> excluded from the FCFS noise floor.
ON2 = ["stock_fcfs", "stock_v5", "stock_v6", "stock_v8", "stock_v9a", "stock_v9b"]
h = []; p = []
for v in ON2:
    hp = hit_p99(v)
    if hp: h.append(hp[0]); p.append(hp[1] / 1000)
print(f"\n=== ondem-2 FCFS-stock same-node group (n={len(h)}): {[v for v in ON2 if hit_p99(v)]} ===")
if len(h) >= 2:
    print(f"  hit@3  mean={st.mean(h):.4f}  sd={st.pstdev(h):.4f}  range={max(h)-min(h):.4f}pp ({min(h):.4f}..{max(h):.4f})  <- §5.1 noise floor")
    print(f"  p99@3  mean={st.mean(p):.1f}s  sd={st.pstdev(p):.1f}s  range={min(p):.1f}..{max(p):.1f}s  <- fixed config => pure run variance (backs Table 3)")
