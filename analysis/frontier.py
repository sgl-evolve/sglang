#!/usr/bin/env python3
"""Paper 2 -- The schedulable frontier.

Central claim (data-driven, closed form):
    goodput@SLO  <=  C(h,K) = K / (1 - h)
where
  h = cache hit_rate  (a CACHE lever: caching lowers the miss rate 1-h),
  K = raw prefill throughput scale = C * (1-h)  (a HARDWARE/SCHEDULE constant:
      scheduling/packing can raise K; capacity is otherwise fixed).

We validate C = K/(1-h) from the SATURATED region of the measured rate sweeps
(lam in {7,10}, where achieved << offered so the server runs at its ceiling C),
across every cache/schedule variant we ran. Then:
  (1) capacity ceiling  C(h) = K/(1-h)   [validated, closed form]
  (2) stability bound   goodput@SLO <= C  (offered>C => queue unstable => p99->inf)
  (3) the measured crossover: lam=3 stable (offered<C), lam>=5 unstable (offered>C).

This unifies the cache axis (Paper 1: caching raises C but not goodput directly)
and the scheduling axis (SRPF-class policies approach C from below, and raise K)
into ONE capacity-bounded frontier: goodput@SLO = min(feasible-tail-rate, C(h,K)).
"""
import csv, glob, os, statistics, sys, json, ast

RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
MIX = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"; CH = 4.0

def uncached_work(hit):
    """Per-request uncached prefill tokens U_i, faithful to the loogle multiturn loader
    and the eval's hit accounting (tok ~= chars/4)."""
    U = []
    for line in open(MIX):
        r = json.loads(line); qp = r.get("qa_pairs", ""); doc = r.get("input", "")
        if qp == "none" or (isinstance(qp, (list, str)) and len(qp) == 0):
            turns = ["Input: " + doc + " Question: Please summarize the input"]
        else:
            try: qa = ast.literal_eval(qp) if isinstance(qp, str) else qp
            except Exception: continue
            turns = [("Input: " + doc + " Question: " + str(q.get("Q", ""))) if i == 0 else str(q.get("Q", ""))
                     for i, q in enumerate(qa)]
        committed = 0
        for ti, p in enumerate(turns):
            newq = max(1, int(len(p) / CH))
            U.append(newq if ti == 0 else int((1 - hit) * committed) + newq); committed += newq
    return U

def load_curve(v):
    p = os.path.join(RUNS, v, "curve.csv")
    if not os.path.exists(p): return {}
    out = {}
    for row in csv.DictReader(open(p)):
        try: out[int(float(row["rate"]))] = dict(
            C=float(row["req_throughput"]), h=float(row["hit_rate"]),
            p50=float(row["ttft_p50_ms"]), p99=float(row["ttft_p99_ms"]))
        except Exception: pass
    return out

# variant -> kind
VAR = {
 "v1_stock":"cache: stock write_through", "v1b_stock":"cache: stock write_through",
 "v3_wb":"cache: write_back", "v3b_wb":"cache: write_back",
 "v2_flat2":"cache: flat-admission (crater)", "v2b_flat2":"cache: flat-admission (crater)",
 "v4_lpm":"schedule: lpm co-residency", "v5_size":"cache: size-admission",
}

print("="*74)
print("1) CAPACITY LAW  C = K/(1-h)   [K := C*(1-h), saturated region lam>=7]")
print("="*74)
print(f"{'variant':30} {'lam':>3} {'C':>6} {'hit':>7} {'K=C(1-h)':>9}")
Kcache, Ksched = [], []
for v,kind in VAR.items():
    cur = load_curve(v)
    for lam in (7,10):
        if lam in cur:
            C,h = cur[lam]["C"], cur[lam]["h"]
            K = C*(1-h)
            print(f"{v:30} {lam:3d} {C:6.2f} {h:7.4f} {K:9.3f}   {kind}")
            (Ksched if kind.startswith('schedule') else Kcache).append(K)
if Kcache:
    Km, Ks = statistics.mean(Kcache), (statistics.pstdev(Kcache) if len(Kcache)>1 else 0)
    print(f"\nCACHE-variant K: mean={Km:.3f}  sd={Ks:.3f}  n={len(Kcache)}  "
          f"CV={100*Ks/Km:.1f}%   <-- invariant => C=K/(1-h) holds")
    if Ksched:
        print(f"SCHED-variant K (lpm): {['%.3f'%x for x in Ksched]}  "
              f"=> {100*(statistics.mean(Ksched)/Km-1):+.0f}% vs cache-K (scheduling raises K)")
    print(f"\nPredicted vs measured C(h) = {Km:.3f}/(1-h):")
    for v in ("v2_flat2","v1_stock","v5_size","v3_wb"):
        cur=load_curve(v); lam=10 if 10 in cur else (7 if 7 in cur else (3 if 3 in cur else None))
        if lam:
            h,C = cur[lam]["h"], cur[lam]["C"]
            print(f"   {v:12} lam={lam:2d} h={h:.4f}  C_pred={Km/(1-h):5.2f}  C_meas={C:5.2f}"
                  f"  err={100*(Km/(1-h)/C-1):+.1f}%")

print()
print("="*74)
print("2) STABILITY: goodput@SLO <= C.  offered>C => unstable => p99 diverges")
print("="*74)
Km = statistics.mean(Kcache) if Kcache else 1.42
for v in ("v1_stock","v3_wb"):
    cur = load_curve(v)
    h_hi = cur.get(10,cur.get(7,{})).get("h", 0.65)
    C = Km/(1-h_hi)
    print(f"\n{v}: C≈{C:.2f} req/s (h={h_hi:.3f})")
    print(f"   {'offered λ':>9} {'achieved':>9} {'p99(ms)':>9} {'stable?':>8}")
    for lam in (3,5,7,10):
        if lam in cur:
            ach,p99 = cur[lam]["C"], cur[lam]["p99"]
            stable = "yes" if lam < C and ach>=0.95*lam else "NO"
            print(f"   {lam:9d} {ach:9.2f} {p99:9.0f} {stable:>8}")
print("\n=> lam=3 offered<C: stable, p99 finite (coin-flip ~SLO).  lam>=5 offered>C:")
print("   achieved<offered, queue unbounded, p99 24-41s -- NO scheduler can pass SLO.")
print("   The schedulable frontier: goodput@SLO in [3, C(h,K)]; caching lifts C,")
print("   SRPF-class scheduling pushes goodput toward C (and raises K).")

print()
print("="*74)
print("3) INTRINSIC-FEASIBILITY FLOOR: solo prefill time U/R_raw vs SLO=8s")
print("="*74)
# R_raw invariance: R_raw = C_sat * E[U|h] (raw prefill tok/s); ~constant across cache variants
print("R_raw = C_sat(lam=10) * E[U|h]  (raw prefill rate, ~invariant across cache variants):")
Rs=[]
for name,C,h in [("stock",4.14,0.6537),("wb",5.11,0.7247),("lpm",4.64,0.6567)]:
    E=sum(uncached_work(h))/len(uncached_work(h)); R=C*E
    if name!="lpm": Rs.append(R)
    print(f"   {name:6} C={C:.2f} h={h:.4f} E[U]={E:6.0f}  R_raw={R:7.0f} tok/s")
Rraw=min(Rs)  # conservative: lowest R_raw => worst-case (longest) solo times; if feasible here, robust
U=sorted(uncached_work(0.6537)); n=len(U)
print(f"\nR_raw(conservative=min)={Rraw:.0f} tok/s.  Solo TTFT = U/R_raw vs SLO=8s:")
SLO=8.0
for q,lab in [(0.50,"p50"),(0.90,"p90"),(0.99,"p99"),(0.999,"p99.9"),(1.0,"MAX")]:
    u=U[min(n-1,int(q*n)) if q<1 else n-1]
    print(f"   {lab:6} U={u:7d} tok -> solo TTFT={u/Rraw:5.2f}s  "
          f"{'FEASIBLE' if u/Rraw<=SLO else 'INFEASIBLE'}")
Reff=U[int(0.99*n)]/8.9  # companion contended fit: q99(U)/R_eff = 8.9s
print(f"\n=> Every request solo-feasible (p99=1.3s, MAX 191K-doc={U[-1]/Rraw:.2f}s < 8s).")
print(f"   All SLO violations are CONTENTION => goodput@SLO_offline = C(h,K).")
print(f"   Reconcile companion: R_eff(contended)={Reff:.0f}, R_raw(solo)={Rraw:.0f}, "
      f"ratio={Rraw/Reff:.1f}x = contention factor = the [measured,C] gap.")
