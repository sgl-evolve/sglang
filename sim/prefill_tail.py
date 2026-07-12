#!/usr/bin/env python3
"""Sharp fork criterion: is p99 TTFT irreducibly bounded by the long-context cold-prefill tail?

Under PERFECT within-conv cache, turn-0 uncached = full doc (cold), turn N>0 uncached = new Q only.
If a >~1% fraction of turns have PERFECT-cache uncached-prefill whose compute alone exceeds the 8s SLO
(at the real prefill throughput P), then p99 TTFT > 8s is IRREDUCIBLE and no lossless cache mechanism
can raise goodput@SLO -> bounded-negative. Else the avoidable-recompute headroom (baseline uncached vs
perfect uncached) is a real mechanism target. Calibrate P from the baseline sweep (curve.csv) when it lands.
"""
import json, os, numpy as np
convs = json.load(open(os.path.join(os.path.dirname(__file__), "conv_trace.json")))
perfect, nocache = [], []
for c in convs:
    acc = 0
    for i, (inp, out) in enumerate(c):
        perfect.append(inp); nocache.append(acc + inp); acc += inp + out
perfect = np.array(perfect); nocache = np.array(nocache)
def pct(a, label):
    print(f"{label}: n={len(a)} p50={int(np.percentile(a,50)):,} p90={int(np.percentile(a,90)):,} "
          f"p99={int(np.percentile(a,99)):,} p99.5={int(np.percentile(a,99.5)):,} max={int(a.max()):,}")
pct(perfect, "PERFECT-cache uncached/turn")
pct(nocache, "NO-cache uncached/turn    ")
print("\n8s-SLO irreducibility vs assumed prefill tok/s P:")
for P in [4000, 6000, 8000, 12000, 20000, 30000]:
    b = 8 * P
    print(f"  P={P:>6}: budget={b:>7,} tok | %turns(perfect>8s)={100*np.mean(perfect>b):5.2f}%  "
          f"(no-cache {100*np.mean(nocache>b):4.1f}%)  -> {'IRREDUCIBLE p99 (neg)' if np.mean(perfect>b)>0.01 else 'cache can affect p99'}")
w0 = sum(c[0][0] for c in convs); wr = sum(inp for c in convs for inp, _ in c[1:])
print(f"\nperfect-cache uncached work: turn0={w0:,} ({100*w0/(w0+wr):.1f}%) later={wr:,} ({100*wr/(w0+wr):.1f}%)")
print("baseline(hit0.62): uncached ~37.8M | perfect: ~19.3M | avoidable-recompute headroom ~18.5M tok (~49% of baseline prefill)")
