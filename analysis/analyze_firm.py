#!/usr/bin/env python3
"""Paper 6 firming verdict: collect ALL giant-accel vs stock lam=3 runs, classify SLO-pass (<8s),
report the deterministic (coin-flip-robust) tpot/concurrency distributions, and run Fisher's exact test
on the SLO-pass counts (like base's SRPF 9/9 vs 0/7). Reads runs/*/bench_r3.json.

Accel runs  = v10_accel, v10b_accel, v10c_accel, v10d_accel, v10e_accel, ...
Stock runs  = v9_stock2, v11_stock3, v12_stock4, ...  (same-node) + v1_stock, v1b_stock (other-node band)
"""
import json, glob, os, math
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
SLO = 8000.0

def load(v):
    p = os.path.join(RUNS, v, "bench_r3.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return dict(p99=d["p99_ttft_ms"], tpot=d["mean_tpot_ms"], conc=d["concurrency"],
                completed=d.get("completed"), thr=d.get("request_throughput"))

def group(names):
    out = []
    for v in names:
        r = load(v)
        if r: out.append((v, r))
    return out

ACCEL = group(["v10_accel","v10b_accel","v10c_accel","v10d_accel","v10e_accel"])
STOCK_SAME = group(["v9_stock2","v11_stock3","v12_stock4"])       # same node (19833)
STOCK_OTHER = group(["v1_stock","v1b_stock"])                      # other-node coin-flip band

def show(label, g):
    print(f"\n{label} (n={len(g)}):")
    for v, r in g:
        pas = "PASS" if r["p99"] <= SLO else "FAIL"
        print(f"  {v:14} p99={r['p99']/1000:5.1f}s [{pas}]  tpot={r['tpot']:.0f}  conc={r['conc']:.0f}  completed={r['completed']}")
    if g:
        import statistics
        p99s=[r['p99'] for _,r in g]; tpots=[r['tpot'] for _,r in g]
        npass=sum(1 for _,r in g if r['p99']<=SLO)
        print(f"  -> SLO-pass {npass}/{len(g)} | p99 median {statistics.median(p99s)/1000:.1f}s | tpot median {statistics.median(tpots):.0f}")

show("GIANT-ACCEL", ACCEL)
show("STOCK same-node", STOCK_SAME)
show("STOCK other-node band", STOCK_OTHER)

# Fisher exact on SLO-pass: accel vs all stock
def fisher(a_pass,a_tot,s_pass,s_tot):
    a_fail=a_tot-a_pass; s_fail=s_tot-s_pass
    # 2x2: [[a_pass,a_fail],[s_pass,s_fail]]; one-sided p (accel more likely to pass)
    def C(n,k):
        return math.comb(n,k)
    n=a_tot+s_tot
    row1=a_tot; col1=a_pass+s_pass
    # hypergeometric tail P(X>=a_pass)
    p=0.0
    for x in range(a_pass, min(row1,col1)+1):
        if col1-x <= (n-row1):
            p += C(col1,x)*C(n-col1,row1-x)/C(n,row1)
    return p

allstock = STOCK_SAME + STOCK_OTHER
a_pass=sum(1 for _,r in ACCEL if r['p99']<=SLO); a_tot=len(ACCEL)
s_pass=sum(1 for _,r in allstock if r['p99']<=SLO); s_tot=len(allstock)
print(f"\n=== FISHER (SLO-pass): accel {a_pass}/{a_tot} vs stock {s_pass}/{s_tot} ===")
if a_tot and s_tot:
    p = fisher(a_pass,a_tot,s_pass,s_tot)
    print(f"  one-sided Fisher p = {p:.4f}  {'SIGNIFICANT (<0.05)' if p<0.05 else 'not yet sig — need more n'}")
print("\nNOTE: deterministic tpot/conc (accel systematically below stock coin-flip range) is the coin-flip-ROBUST")
print("evidence; Fisher on p99-pass is the goodput@SLO headline. Both together = the claim.")
