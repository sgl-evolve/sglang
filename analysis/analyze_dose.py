#!/usr/bin/env python3
"""P6 DOSE-RESPONSE verdict: goodput@SLO and saturated capacity C as a function of the
acceleration factor f. Reads each version's curve.csv (label,rate,req_throughput,...,ttft_p99_ms,...).

f=1   (stock)          : runs/v1_stock          (full-sweep companion baseline)
f=1.5 (v15_accelf15)   : runs/v15_accelf15
f=2   (v13_accelfull)  : runs/v13_accelfull
f=3   (v14_accelf3)    : runs/v14_accelf3

Q1 monotonicity: does goodput@SLO / C rise monotonically with f?
Q2 frontier:     does f=3 push goodput@SLO PAST 5 (i.e. lambda=7 passes)?  => goodput 3->7.
Q3 saturation:   or does C plateau (diminishing returns => a near-hard capacity ceiling)?
"""
import csv, os
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
SLO = 8000.0
ARMS = [("1.0 (stock)", "v1_stock"), ("1.5", "v15_accelf15"), ("2.0", "v13_accelfull"), ("3.0", "v14_accelf3")]

def load_curve(ver):
    p = os.path.join(RUNS, ver, "curve.csv")
    if not os.path.exists(p): return None
    rows = {}
    for r in csv.DictReader(open(p)):
        try:
            lam = int(float(r["rate"]));
            p99 = float(r["ttft_p99_ms"]) if r.get("ttft_p99_ms") else None
            thr = float(r["req_throughput"]) if r.get("req_throughput") else None
        except (ValueError, KeyError):
            continue
        rows[lam] = dict(p99=p99, thr=thr)
    return rows

def goodput_at_slo(rows):
    # highest contiguous lambda whose p99 passes the SLO
    g = 0
    for lam in sorted(rows):
        if rows[lam]["p99"] is not None and rows[lam]["p99"] <= SLO:
            g = lam
        else:
            break
    return g

def sat_cap(rows):
    thrs = [rows[l]["thr"] for l in rows if rows[l]["thr"]]
    return max(thrs) if thrs else None

print(f"{'f':>12} | {'goodput@SLO':>11} | {'sat C':>6} | " + " | ".join(f"λ{l} p99(s)" for l in (3,5,7,10)))
print("-"*80)
results = []
for label, ver in ARMS:
    rows = load_curve(ver)
    if not rows:
        print(f"{label:>12} | {'(pending)':>11} |"); continue
    g = goodput_at_slo(rows); c = sat_cap(rows)
    cells = []
    for l in (3,5,7,10):
        if l in rows and rows[l]["p99"] is not None:
            cells.append(f"{rows[l]['p99']/1000:8.1f}")
        else:
            cells.append(f"{'--':>8}")
    print(f"{label:>12} | {g:>11} | {c or 0:6.2f} | " + " | ".join(cells))
    results.append((label, g, c))

print()
if len(results) >= 2:
    gs = [g for _, g, _ in results]; cs = [c for _, _, c in results if c]
    print(f"goodput@SLO vs f: {[g for _,g,_ in results]}  ({'monotonic↑' if gs==sorted(gs) else 'non-monotonic'})")
    if cs:
        print(f"sat C vs f:       {[round(c,2) for c in cs]}  ({'still rising' if cs==sorted(cs) and len(cs)>1 and cs[-1]>cs[-2]+0.1 else 'plateauing'})")
    f3 = next((g for lbl,g,_ in results if lbl=='3.0'), None)
    if f3 is not None:
        if f3 >= 7:   print("VERDICT: f=3 BREAKS PAST 5 → goodput@SLO frontier extends (headroom remains; adaptive controller = Paper 7).")
        elif f3 == 5: print("VERDICT: f=3 holds goodput@SLO=5 (same as f=2) → capacity ~near-hard ceiling; compose accel+SRPF or characterize the wall = Paper 7.")
        else:         print(f"VERDICT: f=3 goodput@SLO={f3} (< f=2's 5) → over-acceleration HURTS (a bigger chunk stalls decode) → optimal f exists; U-shaped dose = Paper 7 finding.")
