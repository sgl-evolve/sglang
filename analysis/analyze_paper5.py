#!/usr/bin/env python3
"""Paper 5 verdict: does fair-share chunk interleaving (v8_fair) break giant-prefill HOL blocking
and cut the p99 tail, losslessly, without moving capacity? Same-node A/B vs v9_stock2.

TWO evidence tracks:
  (1) DETERMINISTIC mechanism metric (coin-flip-ROBUST): HOL-blocked-step fraction from server.log,
      isolated to the lam=3 window. Fair-share should LOWER it (more new-seq admitted/step, giant no
      longer monopolizes budget). This does NOT depend on the coin-flip.
  (2) OUTCOME (coin-flip-sensitive): lam=3 p99/median TTFT, v8_fair vs same-node v9_stock2 + existing band.
  GUARDS: hit_rate invariant (lossless); C@lam10 unchanged (no de-saturation).
"""
import csv, json, os, re, statistics
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")

def curve(v):
    p = os.path.join(RUNS, v, "curve.csv")
    return {int(float(r["rate"])): r for r in csv.DictReader(open(p))} if os.path.exists(p) else {}

def bench(v, r):
    p = os.path.join(RUNS, v, f"bench_r{r}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def lam3_window(v):
    """(start,end) HH:MM:SS of the lam=3 phase, from the eval stdout log rate markers."""
    for cand in (f"/tmp/turing_{v}.log", os.path.join(RUNS, v, "eval.log")):
        if os.path.exists(cand):
            t3 = t5 = None
            for l in open(cand, errors="ignore"):
                m = re.search(r">>> rate=(\d+) (\d\d:\d\d:\d\d)", l)
                if m:
                    if m[1] == "3": t3 = m[2]
                    elif m[1] == "5": t5 = m[2]
            if t3: return t3, (t5 or "99:99:99")
    return None

PAT = re.compile(r'^\[[\d-]+ (\d\d:\d\d:\d\d).*Prefill batch, #new-seq: (\d+), #new-token: (\d+), '
                 r'#cached-token: \d+, full token usage: [\d.]+, mamba usage: [\d.]+, '
                 r'#running-req: (\d+), #queue-req: (\d+), #pending-token: (\d+)')

def hol_stats(v):
    """HOL-blocked-step fraction in the lam=3 window of v's server.log."""
    sp = os.path.join(RUNS, v, "server.log")
    if not os.path.exists(sp): return None
    win = lam3_window(v)
    rows = []
    for l in open(sp, errors="ignore"):
        m = PAT.search(l)
        if not m: continue
        t = m[1]
        if win and not (win[0] <= t < win[1]): continue
        rows.append((int(m[2]), int(m[3]), int(m[4]), int(m[5]), int(m[6])))  # ns,nt,rr,qr,pt
    if not rows: return None
    budget = max(r[1] for r in rows)
    qpos = [r for r in rows if r[3] > 0]
    # steps where a giant is in flight AND shorts are waiting (the HOL opportunity)
    giant_wait = [r for r in qpos if r[4] > budget]
    # CLEAN mechanism metric: of those, how often is a short admitted alongside the giant (new-seq>=2)?
    # Stock ~0% (giant hogs budget); fair-share >0% (interleave). NOT contaminated by the giant's chunk size.
    interleave = [r for r in giant_wait if r[0] >= 2]
    return dict(steps=len(rows), budget=budget,
                wait_frac=len(qpos) / len(rows),
                giant_wait_frac=len(giant_wait) / len(rows),
                interleave_of_giantwait=(len(interleave) / len(giant_wait)) if giant_wait else 0,
                mean_newseq_when_giantwait=statistics.mean([r[0] for r in giant_wait]) if giant_wait else 0,
                med_queue_when_wait=statistics.median([r[3] for r in qpos]) if qpos else 0)

FAIR, STOCK = "v8_fair", "v9_stock2"
cf, cs = curve(FAIR), curve(STOCK)
print("=== Paper 5 A/B: fair-share chunk interleaving vs same-node stock ===")
if not cf:
    print(f"{FAIR} not landed yet."); raise SystemExit(0)

print("\n(1) DETERMINISTIC mechanism metric — INTERLEAVE-FIRING rate (lam=3, coin-flip-ROBUST):")
print("    = of steps with a giant in flight AND shorts waiting, fraction admitting >=2 new-seq")
print("    (stock ~0%: giant hogs budget; fair-share >0%: short interleaved). Uncontaminated by chunk size.")
hf, hs = hol_stats(FAIR), (hol_stats(STOCK) or hol_stats("v1_stock"))
for nm, h in (("fair(v8)", hf), ("stock", hs)):
    if h:
        print(f"  {nm:10} budget={h['budget']} giant_wait_frac={h['giant_wait_frac']:.1%} "
              f"INTERLEAVE={h['interleave_of_giantwait']:.1%} "
              f"mean_newseq(giant+wait)={h['mean_newseq_when_giantwait']:.2f}")
if hf and hs:
    print(f"  -> interleave rate: stock {hs['interleave_of_giantwait']:.1%} -> fair "
          f"{hf['interleave_of_giantwait']:.1%} "
          f"({'MECHANISM FIRES (shorts interleaved)' if hf['interleave_of_giantwait']>hs['interleave_of_giantwait']+0.05 else 'no meaningful interleave'})")

print("\n(2) OUTCOME — lam=3 TTFT (coin-flip-sensitive; same-node A/B):")
for nm, c in (("fair(v8)", cf), ("stock(v9)", cs)):
    if 3 in c:
        r = c[3]
        print(f"  {nm:10} p99={float(r['ttft_p99_ms'])/1000:.1f}s p50={float(r['ttft_p50_ms'])/1000:.2f}s "
              f"req/s={float(r['req_throughput']):.2f} hit={float(r['hit_rate']):.4f}")
print("  (existing stock band: p99 {6.2, 11.5}s, hit ~0.675)")
if 3 in cf:
    p99f = float(cf[3]['ttft_p99_ms']); hitf = float(cf[3]['hit_rate'])
    print(f"  -> fair p99 {p99f/1000:.1f}s vs SLO 8s: {'PASS' if p99f<=8000 else 'FAIL'}; "
          f"vs stock band: {'BELOW good basin (WIN)' if p99f<6200 else 'within/above band'}")
    print(f"  -> LOSSLESS hit {hitf:.4f}: {'OK' if abs(hitf-0.675)<0.02 else 'CHECK moved'}")

if 10 in cf:
    print(f"\nGUARD C@lam10 fair={float(cf[10]['req_throughput']):.2f} "
          f"(stock 4.14): {'~unchanged' if 3.9<float(cf[10]['req_throughput'])<4.9 else 'CHANGED'}")

print("\nVERDICT: WIN if HOL-frac DROPS (mechanism, robust) AND lam=3 p99 down vs same-node stock AND lossless "
      "AND C@10 unchanged. If HOL drops but p99 unchanged -> the tail is the giant's OWN prefill, not HOL "
      "(refined bound, sharpens P1/P2). If p99 UP -> fair-share slows giants net-negative (bound).")
