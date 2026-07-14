#!/usr/bin/env python3
# kleinrock: analyze the SRPF A/B (runs/v3-srpf-ab/srpf_ab.csv). Per (policy, lambda): median-of-k of
# p99/p50 TTFT, hit-rate, req throughput, and the goodput-pass-count (#reps with p99<=SLO). Reports the
# paired fcfs-vs-srpf deltas on STABLE metrics (per my own methodology: the coin-flip binary can't separate
# configs, but the p99 DISTRIBUTION / median can). Fisher-exact on pass-counts where scipy is available;
# else prints the 2x2 for manual read. Deterministic; no network.
import csv, os, sys, statistics as st
CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/v3-srpf-ab/srpf_ab.csv"
if not os.path.exists(CSV):
    print("NO CSV yet:", CSV); sys.exit(0)
rows = [r for r in csv.DictReader(open(CSV)) if r.get("ttft_p99_ms")]
def fnum(r, k):
    v = r.get(k, "")
    try: return float(v)
    except: return None
# group by (policy, lambda)
groups = {}
for r in rows:
    key = (r["policy"], r["lambda"])
    groups.setdefault(key, []).append(r)

def med(xs): xs=[x for x in xs if x is not None]; return st.median(xs) if xs else None
def summarize(rs):
    p99=[fnum(r,"ttft_p99_ms") for r in rs]; p50=[fnum(r,"ttft_p50_ms") for r in rs]
    hit=[fnum(r,"hit_rate") for r in rs]; req=[fnum(r,"req_throughput") for r in rs]
    npass=sum(int(r.get("pass","0") or 0) for r in rs)
    p99v=[x for x in p99 if x is not None]
    return dict(n=len(rs), npass=npass,
                p99_med=med(p99), p99_min=min(p99v) if p99v else None, p99_max=max(p99v) if p99v else None,
                p50_med=med(p50), hit_med=med(hit), req_med=med(req),
                p99_all=sorted(p99v))

lams = sorted({k[1] for k in groups}, key=lambda x: float(x))
print(f"\n===== SRPF A/B ANALYSIS ({os.path.basename(os.path.dirname(CSV))}) =====")
print(f"{'pol':5} {'lam':>4} {'n':>2} {'pass':>4} {'p99_med':>9} {'p99_min':>9} {'p99_max':>9} {'p50_med':>8} {'hit':>6} {'req':>6}")
tab={}
for lam in lams:
    for pol in ("fcfs","srpf"):
        rs=groups.get((pol,lam))
        if not rs: continue
        s=summarize(rs); tab[(pol,lam)]=s
        f=lambda v,w=9,d=1: (f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}")
        print(f"{pol:5} {lam:>4} {s['n']:>2} {s['npass']:>2}/{s['n']:<1} {f(s['p99_med'])} {f(s['p99_min'])} {f(s['p99_max'])} {f(s['p50_med'],8)} {f(s['hit_med'],6,3)} {f(s['req_med'],6,2)}")

print("\n----- PAIRED fcfs -> srpf deltas (SLO=8000ms) -----")
def fisher(a,b,c,d):
    # a,b = srpf pass/fail ; c,d = fcfs pass/fail ; try scipy else None
    try:
        from scipy.stats import fisher_exact
        return fisher_exact([[a,b],[c,d]])[1]
    except Exception:
        return None
for lam in lams:
    fc=tab.get(("fcfs",lam)); sr=tab.get(("srpf",lam))
    if not fc or not sr: continue
    dp99 = (sr['p99_med']-fc['p99_med']) if (sr['p99_med'] and fc['p99_med']) else None
    pct = (100*dp99/fc['p99_med']) if (dp99 is not None and fc['p99_med']) else None
    print(f"lambda={lam}: p99_med {fc['p99_med']:.0f} -> {sr['p99_med']:.0f} ms "
          f"({'%+.1f%%'%pct if pct is not None else '?'}); pass fcfs {fc['npass']}/{fc['n']} vs srpf {sr['npass']}/{sr['n']}", end="")
    p=fisher(sr['npass'], sr['n']-sr['npass'], fc['npass'], fc['n']-fc['npass'])
    print(f"; Fisher p={p:.4g}" if p is not None else "  (scipy N/A)")
    # verdict hint
    if fc['npass']==0 and sr['npass']>=max(1,sr['n']-1):
        print(f"   >>> SRPF turns reliable-FAIL -> PASS at lambda={lam}: POSITIVE MECHANISM candidate (verify same-node, k)")
    elif dp99 is not None and pct is not None and pct < -20:
        print(f"   >>> SRPF cuts p99_med {pct:.0f}% at lambda={lam}: distribution shift on stable metric")
    elif dp99 is not None and pct is not None and pct > 20:
        print(f"   >>> SRPF WORSENS p99_med {pct:.0f}% at lambda={lam}: confirms starvation prediction (measured negative)")
    else:
        print(f"   >>> no material p99 separation at lambda={lam} (within coin-flip band)")
print()
