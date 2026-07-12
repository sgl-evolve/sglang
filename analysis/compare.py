#!/usr/bin/env python3
# Compare valiant runs: goodput@SLO + per-rate p99/hit/req_s, with deltas vs a stock baseline.
import json, csv, glob, os, sys
SLO=8000.0
def load(v):
    d=f"runs/{v}"
    if not os.path.isdir(d): return None
    rows=[]
    cf=f"{d}/curve.csv"
    if os.path.exists(cf):
        for r in csv.DictReader(open(cf)):
            if r.get("req_throughput"): rows.append(r)
    g=lambda r,k:(float(r[k]) if r.get(k) not in (None,"") else None)
    per={}
    for r in rows:
        R=r["rate"]; per[R]=dict(req=g(r,"req_throughput"),p99=g(r,"ttft_p99_ms"),p50=g(r,"ttft_p50_ms"),hit=g(r,"hit_rate"),tok=g(r,"out_tok_s"))
    good=max([per[R]["req"] for R in per if per[R]["p99"] and per[R]["p99"]<=SLO] or [0])
    return dict(rows=per, good=good)
order=["v0-baseline","stock_b","v1_b","pc_b","stock_c","v1_c","pc_c"]
data={v:load(v) for v in order if load(v)}
if not data: print("no runs yet"); sys.exit()
rates=sorted({R for v in data for R in data[v]["rows"]}, key=float)
print(f"{'run':12s} {'good@SLO':>8s} | "+" ".join(f"p99@{R:>2s}/hit@{R}" for R in rates))
for v in order:
    if v not in data: continue
    d=data[v]; cells=[]
    for R in rates:
        c=d["rows"].get(R)
        if c and c["p99"] is not None: cells.append(f"{c['p99']/1000:5.1f}s/{c['hit']:.3f}")
        else: cells.append("   -/-   ")
    print(f"{v:12s} {d['good']:8.2f} | "+" ".join(cells))
# deltas: v1_b/pc_b vs stock_b ; v1_c/pc_c vs stock_c
for base,treats in [("stock_b",["v1_b","pc_b"]),("stock_c",["v1_c","pc_c"])]:
    if base not in data: continue
    for t in treats:
        if t not in data: continue
        print(f"\n{t} vs {base}: good {data[t]['good']:.2f} vs {data[base]['good']:.2f}")
        for R in rates:
            b=data[base]["rows"].get(R); x=data[t]["rows"].get(R)
            if b and x and b["p99"] and x["p99"]:
                dp99=100*(x["p99"]-b["p99"])/b["p99"]; dhit=x["hit"]-b["hit"]
                print(f"  @{R}: p99 {b['p99']/1000:.1f}->{x['p99']/1000:.1f}s ({dp99:+.0f}%) hit {b['hit']:.3f}->{x['hit']:.3f} ({dhit:+.3f})")
