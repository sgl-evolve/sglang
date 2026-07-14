#!/usr/bin/env python3
# kleinrock: generic same-node A/B analyzer for runs/*/`*_ab.csv` (SRPF or short-lane). CSV col0 = config
# label (policy/arm), then node,lambda,rep,req_throughput,out_tok_s,ttft_p50_ms,ttft_p99_ms,e2e_p99_ms,
# hit_rate,pass. Per (config,lambda): median-of-k of p99/p50/hit/req + goodput-pass-count. Prints paired
# deltas of the two configs on STABLE metrics (per my methodology: the binary coin-flip can't separate configs,
# but the p99 DISTRIBUTION/median can). Fisher-exact on pass-counts if scipy present. Deterministic, no network.
import csv, os, sys, statistics as st
CSV = sys.argv[1]
if not os.path.exists(CSV): print("NO CSV:", CSV); sys.exit(0)
rows = list(csv.DictReader(open(CSV)))
cols = list(rows[0].keys()) if rows else []
if not cols: print("CSV has no data rows yet:", CSV); sys.exit(0)
gcol = cols[0]  # config label column (policy / arm)
rows = [r for r in rows if r.get("ttft_p99_ms")]
if not rows: print("no completed result rows yet in", CSV); sys.exit(0)
def fnum(r,k):
    try: return float(r.get(k,""))
    except: return None
def med(xs): xs=[x for x in xs if x is not None]; return st.median(xs) if xs else None
groups={}
for r in rows: groups.setdefault((r[gcol], r["lambda"]), []).append(r)
def summarize(rs):
    p99=[fnum(r,"ttft_p99_ms") for r in rs]; p99v=[x for x in p99 if x is not None]
    return dict(n=len(rs), npass=sum(int(r.get("pass","0") or 0) for r in rs),
                p99_med=med(p99), p99_min=min(p99v) if p99v else None, p99_max=max(p99v) if p99v else None,
                p50_med=med([fnum(r,"ttft_p50_ms") for r in rs]), hit_med=med([fnum(r,"hit_rate") for r in rs]),
                req_med=med([fnum(r,"req_throughput") for r in rs]), tok_med=med([fnum(r,"out_tok_s") for r in rs]),
                p99_all=sorted(round(x) for x in p99v))
configs=[];
for (c,l) in groups:
    if c not in configs: configs.append(c)
lams=sorted({l for (_,l) in groups}, key=float)
print(f"\n===== A/B ANALYSIS: {CSV} (config col='{gcol}') =====")
print(f"{gcol:10} {'lam':>4} {'n':>2} {'pass':>5} {'p99_med':>9} {'p99_min':>9} {'p99_max':>9} {'p50_med':>8} {'hit':>6} {'req':>6} {'tok':>7}")
tab={}
for l in lams:
    for c in configs:
        rs=groups.get((c,l));
        if not rs: continue
        s=summarize(rs); tab[(c,l)]=s
        f=lambda v,w=9,d=1:(f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}")
        print(f"{c:10} {l:>4} {s['n']:>2} {s['npass']:>2}/{s['n']:<2} {f(s['p99_med'])} {f(s['p99_min'])} {f(s['p99_max'])} {f(s['p50_med'],8)} {f(s['hit_med'],6,3)} {f(s['req_med'],6,2)} {f(s['tok_med'],7,1)}")
    if tab.get((configs[0],l)): print(f"           p99 samples {configs[0]}={tab[(configs[0],l)]['p99_all']}" + (f"  {configs[1]}={tab[(configs[1],l)]['p99_all']}" if len(configs)>1 and tab.get((configs[1],l)) else ""))

if len(configs)>=2:
    a,b=configs[0],configs[1]
    print(f"\n----- PAIRED {a} -> {b} (SLO=8000ms) -----")
    def fisher(pa,fa,pb,fb):
        try:
            from scipy.stats import fisher_exact; return fisher_exact([[pa,fa],[pb,fb]])[1]
        except Exception: return None
    for l in lams:
        A=tab.get((a,l)); B=tab.get((b,l))
        if not A or not B: continue
        d=(B['p99_med']-A['p99_med']) if (A['p99_med'] and B['p99_med']) else None
        pct=(100*d/A['p99_med']) if (d is not None and A['p99_med']) else None
        print(f"lambda={l}: p99_med {A['p99_med']:.0f}->{B['p99_med']:.0f}ms ({'%+.1f%%'%pct if pct is not None else '?'}); "
              f"pass {a} {A['npass']}/{A['n']} vs {b} {B['npass']}/{B['n']}", end="")
        p=fisher(B['npass'],B['n']-B['npass'],A['npass'],A['n']-A['npass'])
        print(f"; Fisher p={p:.4g}" if p is not None else " (scipy N/A)")
        if A['npass']==0 and B['npass']>=max(1,B['n']-1): print(f"   >>> {b} turns reliable-FAIL->PASS: POSITIVE candidate (verify k, lossless)")
        elif pct is not None and pct<-20: print(f"   >>> {b} cuts p99_med {pct:.0f}%: distribution shift (stable-metric signal)")
        elif pct is not None and pct>20: print(f"   >>> {b} WORSENS p99_med +{pct:.0f}%: measured negative")
        else: print(f"   >>> no material p99 separation (within coin-flip band)")
        for extra in ("p50_med","hit_med","req_med","tok_med"):
            if A.get(extra) and B.get(extra):
                dd=100*(B[extra]-A[extra])/A[extra]
                if abs(dd)>=3: print(f"      {extra}: {A[extra]:.2f}->{B[extra]:.2f} ({dd:+.1f}%)")
print()
