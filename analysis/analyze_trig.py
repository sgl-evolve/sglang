#!/usr/bin/env python3
"""P6 trigger-signal ablation verdict. Compares the three accel trigger signals at lam=3 (all GIANT_ACCEL=1,
factor 2.0, same node): occ (device-KV occupancy, the current mechanism), queue (waiting-queue depth, P5's HOL
signal), either (occ OR queue). Reports the DETERMINISTIC (coin-flip-ROBUST) tpot/concurrency per signal plus
SLO-pass, so we can say whether the boost is robust to trigger choice (occ ~= queue) or one signal wins.
Reads runs/v19_<signal><n>/bench_r3.json. No conclusion is baked in — it just tabulates whatever landed.
"""
import json, glob, os, statistics
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
SLO = 8000.0

def load(v):
    p = os.path.join(RUNS, v, "bench_r3.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return dict(p99=d["p99_ttft_ms"], tpot=d["mean_tpot_ms"], conc=d["concurrency"],
                thr=d.get("request_throughput"), completed=d.get("completed"))

def group(signal):
    out = []
    for p in sorted(glob.glob(os.path.join(RUNS, f"v19_{signal}*", "bench_r3.json"))):
        v = os.path.basename(os.path.dirname(p)); r = load(v)
        if r: out.append((v, r))
    return out

def show(label, g):
    print(f"\n{label} (n={len(g)}):")
    for v, r in g:
        pas = "PASS" if r["p99"] <= SLO else "FAIL"
        print(f"  {v:14} p99={r['p99']/1000:5.1f}s [{pas}]  tpot={r['tpot']:.0f}  conc={r['conc']:.0f}  thr={r['thr']}  completed={r['completed']}")
    if g:
        p99s=[r['p99'] for _,r in g]; tp=[r['tpot'] for _,r in g]; cc=[r['conc'] for _,r in g]
        npass=sum(1 for _,r in g if r['p99']<=SLO)
        med=lambda x: statistics.median(x)
        print(f"  -> SLO-pass {npass}/{len(g)} | p99 med {med(p99s)/1000:.1f}s | tpot med {med(tp):.0f} | conc med {med(cc):.0f}")

groups = {s: group(s) for s in ("occ","queue","either")}
for s in ("occ","queue","either"):
    show(f"TRIGGER={s}", groups[s])

print("\n=== READOUT ===")
print("Deterministic tpot/conc is coin-flip-ROBUST; compare medians across signals.")
print("If queue/either tpot & conc ~= occ AND all pass SLO -> boost is robust to trigger signal (generality).")
print("If one signal gives systematically lower conc/tpot -> that signal gates the boost better.")
print("If a signal over-fires (much lower tpot but p99 FAIL) -> over-acceleration tail (like f=3).")
