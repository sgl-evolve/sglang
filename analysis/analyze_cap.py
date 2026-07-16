#!/usr/bin/env python3
"""P7 concurrency-cap axis map. Tabulates lam=3 p99/tpot/conc/SLO-pass vs --max-running-requests cap, from
runs/v20_{stock0,cap192,cap128,cap96,cap64}/bench_r3.json. Shows the threshold: as the cap tightens below the
natural lam=3 concurrency (stock ~74-166), decode SPEEDS UP (tpot down, memory-bound) but p99 TTFT BLOWS UP
(admission queue-wait) => decisive bounded negative. Non-binding cap192 (>166) should ~= stock (sanity). No conclusion
baked in; reads whatever landed."""
import json, os
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")
SLO = 8000.0
# ordered from no-cap to tightest
ORDER = [("stock (no cap)","v20_stock0"), ("cap192","v20_cap192"), ("cap128","v20_cap128"),
         ("cap96","v20_cap96"), ("cap64","v20_cap64")]
def load(v):
    p = os.path.join(RUNS, v, "bench_r3.json")
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return dict(p99=d["p99_ttft_ms"], tpot=d["mean_tpot_ms"], conc=d["concurrency"],
                thr=d.get("request_throughput"), completed=d.get("completed"))
print(f"{'label':16} {'cap':>5} {'p99(s)':>8} {'SLO':>5} {'tpot':>6} {'conc':>6} {'completed':>10}")
for label, v in ORDER:
    r = load(v)
    if not r:
        print(f"{label:16} {'':>5} {'(pending)':>8}")
        continue
    cap = v.replace("v20_cap","").replace("v20_stock0","-")
    pas = "PASS" if r["p99"]<=SLO else "FAIL"
    print(f"{label:16} {cap:>5} {r['p99']/1000:>8.1f} {pas:>5} {r['tpot']:>6.0f} {r['conc']:>6.0f} {str(r['completed']):>10}")
print("\nREADOUT: if p99 rises monotonically as cap tightens while tpot FALLS -> capping speeds decode but starves")
print("admission (queue-wait dominates TTFT) = decisive concurrency-cap negative; non-binding cap192 ~= stock = sanity.")
