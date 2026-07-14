#!/usr/bin/env python3
"""Regenerate Paper 2 Figure 1 (capacity law) SVG coords from measured saturated points.
Run after v5_size/v6_flat land to turn the hollow PREDICTED markers into filled MEASURED ones.
Emits ready-to-paste <circle>/<text> SVG for the C vs 1/(1-h) plot (same axes as paper).
"""
import csv, os
RUNS=os.path.join(os.path.dirname(__file__),"..","runs")
# axes: x=1/(1-h) in [1.5,4.0]->px[60,600]; y=C in [0,6]->px[340,40]
def xp(v): return 60+(v-1.5)/2.5*540
def yp(v): return 340-v/6*300
def sat(v):  # saturated C,h from lam=10 (else 7)
    p=os.path.join(RUNS,v,"curve.csv")
    if not os.path.exists(p): return None
    rows={int(float(r["rate"])):r for r in csv.DictReader(open(p))}
    for lam in (10,7):
        if lam in rows: return float(rows[lam]["req_throughput"]),float(rows[lam]["hit_rate"]),lam
    return None
COL={"v1_stock":"#1c4a8a","v3_wb":"#1c4a8a","v4_lpm":"#1a6a1a","v5_size":"#1c4a8a","v6_flat_sweep":"#1c4a8a"}
LAB={"v1_stock":"stock","v3_wb":"write_back","v4_lpm":"lpm(+K)","v5_size":"size","v6_flat_sweep":"flat"}
print("# Fig 1 measured points (saturated). Paste <circle>/<text> into paper.html:")
Ks=[]
for v in ("v6_flat_sweep","v1_stock","v4_lpm","v5_size","v3_wb"):
    s=sat(v)
    if not s: print(f"# {v}: (no saturated point yet)"); continue
    C,h,lam=s; x=1/(1-h); K=C*(1-h)
    if v!="v4_lpm": Ks.append(K)
    print(f'  <circle cx="{xp(x):.0f}" cy="{yp(C):.0f}" r="5" fill="{COL[v]}"/>'
          f'<text x="{xp(x)-30:.0f}" y="{yp(C)-8:.0f}" font-size="12">{LAB[v]} (h={h:.2f})</text>'
          f'   <!-- {v} lam{lam} C={C:.2f} K={K:.3f} -->')
if Ks:
    import statistics
    print(f"# cache-K mean={statistics.mean(Ks):.3f} sd={statistics.pstdev(Ks):.3f} n={len(Ks)} "
          f"CV={100*statistics.pstdev(Ks)/statistics.mean(Ks):.1f}% (line slope for C=K/(1-h))")
