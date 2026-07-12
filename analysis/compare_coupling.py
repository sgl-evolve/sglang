# Compare the coupling experiment runs (same-node ondem-2): stock_fcfs / stock_lpm / v1_lpm / v1_fcfs.
# Reads runs/<ver>/curve.csv (label,rate,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,hit_rate).
# Prints the scheduler-cache COUPLING table: does LPM raise stock hit + change the p99 tail vs FCFS,
# and does the pending-pin gain shrink under LPM (as the simulation predicts)?
import csv, os
BASE=os.path.join(os.path.dirname(__file__),"..","runs")
VERS=["stock_fcfs","stock_lpm","v1_fcfs","v1_lpm"]
def load(v):
    p=os.path.join(BASE,v,"curve.csv")
    if not os.path.exists(p): return None
    rows={}
    for r in csv.DictReader(open(p)):
        try: rows[float(r["rate"])]=r
        except Exception: pass
    return rows or None
data={v:load(v) for v in VERS}
avail=[v for v in VERS if data[v]]
print("available:", avail or "(none yet)")
if not avail: raise SystemExit
rates=sorted(set().union(*[set(data[v]) for v in avail]))
def g(v,rate,k):
    r=data[v] and data[v].get(rate)
    return float(r[k]) if r and r.get(k) not in (None,"","nan") else float("nan")
print(f"\n{'rate':>5} | {'hit_fcfs':>8} {'hit_lpm':>8} {'Δhit(lpm-fcfs)':>14} | {'p99_fcfs':>9} {'p99_lpm':>9} {'Δp99%':>7}  [STOCK: does LPM close the cache gap? at what p99 cost?]")
for rate in rates:
    hf,hl=g("stock_fcfs",rate,"hit_rate"),g("stock_lpm",rate,"hit_rate")
    pf,pl=g("stock_fcfs",rate,"ttft_p99_ms"),g("stock_lpm",rate,"ttft_p99_ms")
    dp=100*(pl-pf)/pf if pf==pf and pf else float("nan")
    print(f"{rate:>5.0f} | {hf:>8.3f} {hl:>8.3f} {100*(hl-hf):>+13.2f}pp | {pf:>9.0f} {pl:>9.0f} {dp:>+6.1f}%")
print(f"\n{'rate':>5} | {'pin gain under FCFS':>20} | {'pin gain under LPM':>20}  [MECHANISM: does the pin's win shrink under LPM, as sim predicts?]")
for rate in rates:
    df=g("v1_fcfs",rate,"hit_rate")-g("stock_fcfs",rate,"hit_rate")
    dl=g("v1_lpm",rate,"hit_rate")-g("stock_lpm",rate,"hit_rate")
    print(f"{rate:>5.0f} | {100*df:>+19.2f}pp | {100*dl:>+19.2f}pp")
print("\nPREDICTIONS (from §2.4 sim): stock LPM hit > stock FCFS hit; pin gain(FCFS) > pin gain(LPM)≈0;")
print("watch p99: if LPM also LOWERS p99, the 'locality-without-reorder' value-prop weakens (see §7 open measurement).")
