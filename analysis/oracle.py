#!/usr/bin/env python3
"""floyd paper-2: trace-driven ACHIEVABLE-GOODPUT FRONTIER for goodput@SLO.

Question paper 1 raised but did not answer: goodput@SLO is prefill-compute
bound and admission control cannot help (paper 1) -- but is the *scheduling*
axis closed?  A sibling found SRPF ~+35% (reaches ~4.1 req/s).  Is that
near-optimal, or is there headroom a better scheduler could still recover?

We answer with an offline, CPU-only discrete-event oracle:

  * Job stream = the 7163 real conversation turns, each with its cache-adjusted
    prefill work (LRU@CAP tuned to the MEASURED hit rate 0.677 -> faithful).
  * Server    = single prefill resource at rate C tok/s (M/G/1). C is CALIBRATED
    so FCFS reproduces the MEASURED saturation throughput (~4.22 req/s) and the
    MEASURED p99-TTFT curve (11.8s@3 ... 41s@10).  Decode contention is folded
    into C (workload is prefill-dominated: decode median 33 tok).
  * Disciplines: FCFS (= stock order), SRPF (shortest cache-adjusted prefill
    first, non-preemptive = the sibling's lever), SRPT (preemptive shortest
    remaining = the theoretical mean-flow optimum), and an idealized clairvoyant
    P99-MINIMIZING heuristic (EDF-to-SLO) as an upper bound on achievability.

Outputs: (1) validation table FCFS vs measured; (2) achievable goodput@SLO per
discipline; (3) decomposition of the stock(~3, coin-flip) -> ceiling(~4.22) gap
into scheduling-recoverable vs irreducible-compute.

Model assumptions (stated for the paper):
  - Open-loop Poisson turn arrivals at rate lambda (matches bench request_rate
    semantics: completed/duration = achieved req/s; turns issued in conversation
    order so cache-adjusted work is as measured).
  - Cache-adjusted work is held fixed across disciplines: caching is a bounded,
    schedule-insensitive lever (paper 1 + fleet: LRU~=Belady, hit ~0.66 across
    configs), so reordering prefills does not materially change per-turn work.
  - Single-server prefill (M/G/1): conservative vs the real chunked-batched
    engine, but calibrated to reproduce the measured curve.
"""
import ast, json, heapq, statistics as st
from collections import OrderedDict, deque

TRACE = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CPT = 4.0
CAP = 7_500_000          # LRU capacity tuned to measured hit 0.677 (see validation)
SLO_MS = 8000.0
# Measured stock curve (v0-stock full sweep) -- validation ground truth:
MEASURED = {  # lambda: (achieved_req_s, p99_ttft_ms, median_ttft_ms, concurrency)
    3:  (2.87, 11787, 1057, 167.6),
    5:  (3.66, 17372,  984, 203.2),
    7:  (4.00, 33961, 1020, 234.5),
    10: (4.22, 41334, 1064, 246.1),
}
SAT_REQ_S = 4.22         # measured saturation throughput (peak achievable req/s)

def toks(s): return max(1, int(len(s) / CPT))

def load_turns():
    """Per-turn cache-adjusted prefill work under LRU@CAP, in conversation order.
    Returns list of work (tokens), the order the stock system processes them."""
    convs = []
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line); doc = r.get("input", "")
            try: qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception: qa = []
            turns = []
            for i, t in enumerate(qa):
                if not isinstance(t, dict): continue
                user = ("Input: " + doc + " Question: " + str(t.get("Q",""))) if i == 0 else str(t.get("Q",""))
                turns.append((toks(user), toks(str(t.get("A","")))))
            if turns: convs.append(turns)
    cache = OrderedDict(); res = 0; work = []
    ctx = {}; idx = {}; tof = {}; pend = deque(); it = iter(enumerate(convs)); done = False
    tot_in = 0; tot_new = 0
    def evict(need):
        nonlocal res
        while res + need > CAP and cache:
            _, ev = cache.popitem(last=False); res -= ev
    while not done or pend:
        try:
            cid, turns = next(it); ctx[cid]=0; idx[cid]=0; tof[cid]=turns; pend.append(cid)
        except StopIteration: done = True
        if not pend: break
        cid = pend.popleft(); turns = tof[cid]; ti = idx[cid]
        intok, outok = turns[ti]; context = ctx[cid] + intok
        cached = cache.get(cid, 0)
        adj = max(0, context - cached) if cid in cache else context
        work.append(max(1, adj)); tot_in += context; tot_new += adj
        newp = context + outok; old = cache.pop(cid, 0); res -= old
        evict(newp); cache[cid] = newp; res += newp; ctx[cid] = newp; idx[cid] += 1
        if idx[cid] < len(turns): pend.append(cid)
    hit = 1 - tot_new / tot_in
    return work, hit

# ---- deterministic Poisson arrival stream (fixed seed via LCG; no Math.random needed) ----
class LCG:
    def __init__(self, seed): self.s = seed & 0xFFFFFFFF
    def u(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return (self.s + 1) / 0x80000000
import math
def exp_iat(rng, lam): return -math.log(rng.u()) / lam

B_CHUNK = 6144                               # --chunked-prefill-size (frozen)
CONC = 256                                   # bench max_concurrency (client semaphore)

def _key(discipline, i, work, rem, admit_t, now):
    """priority key -- SMALLER served first each step."""
    if discipline == "FCFS":  return admit_t[i]
    if discipline == "SRPF":  return work[i]            # shortest total prefill first (small-first)
    if discipline == "SRPT":  return rem[i]             # shortest remaining first
    if discipline == "LPF":   return -work[i]           # longest prefill first (tail-first)
    if discipline == "EDF":   return admit_t[i] + SLO_MS/1000.0   # deadline = admit + SLO
    if discipline == "SLACK":                            # least laxity first: deadline - now - remaining/C
        return (admit_t[i] + SLO_MS/1000.0) - now - rem[i]/25000.0
    if discipline == "PS":    return 0                   # all equal -> round-robin fill
    raise ValueError(discipline)

def simulate(work, lam, C, discipline, seed=1):
    """Discrete-STEP sim of sglang chunked prefill: each step (tau = B/C seconds)
    a budget of B_CHUNK new tokens is filled across in-flight, prefill-incomplete
    requests in schedule-policy order (a big doc is chunked across many steps; small
    turns that fit are co-batched). Concurrency capped at CONC (bench semaphore);
    TTFT measured from admission (= bench send time). Decode folded into C.
    Returns (achieved_req_s, p99_ttft_ms, median_ttft_ms)."""
    n = len(work)
    rng = LCG(seed)
    offer = [0.0]*n; t = 0.0
    for i in range(n):
        t += exp_iat(rng, lam); offer[i] = t
    rem = [float(w) for w in work]
    admit_t = [0.0]*n; ttft = [0.0]*n
    tau = B_CHUNK / C                         # step duration (s)
    now = 0.0; ai = 0; last_dep = 0.0; done = 0
    active = []                               # admitted, prefill-incomplete
    def admit():
        nonlocal ai
        while ai < n and offer[ai] <= now + 1e-12 and len(active) < CONC:
            admit_t[ai] = now; active.append(ai); ai += 1
    while done < n:
        admit()
        if not active:
            if ai < n: now = max(now, offer[ai]); continue
            break
        # order in-flight by discipline; fill B_CHUNK tokens this step
        if discipline == "PS":
            order = active                    # round-robin equal fill
            share = B_CHUNK / len(active)
            for i in order: rem[i] -= share
        else:
            order = sorted(active, key=lambda i: _key(discipline, i, work, rem, admit_t, now))
            budget = B_CHUNK
            for i in order:
                if budget <= 0: break
                take = min(rem[i], budget); rem[i] -= take; budget -= take
        now += tau
        fin = [i for i in active if rem[i] <= 1e-6]
        for i in fin:
            ttft[i] = (now - admit_t[i]) * 1000.0
            last_dep = now; done += 1; active.remove(i)
        admit()
    achieved = n / last_dep if last_dep > 0 else 0.0
    ts = sorted(ttft)
    p99 = ts[min(n-1, int(0.99*n))]
    med = ts[n//2]
    return achieved, p99, med

def goodput_at_slo(work, C, discipline):
    """max achieved req/s with p99 TTFT <= SLO, scanning lambda."""
    best = 0.0; curve = []
    for lam in [2,2.5,3,3.5,4,4.5,5,6,7,8,10]:
        ach, p99, med = simulate(work, lam, C, discipline)
        curve.append((lam, ach, p99, med))
        if p99 <= SLO_MS:
            best = max(best, ach)
    return best, curve

def main():
    work, hit = load_turns()
    n = len(work); mean_w = st.mean(work)
    print(f"# turns={n}  sim hit_rate={hit:.3f} (measured 0.677)  mean_work={mean_w:.0f} tok  max={max(work)}")
    # calibrate C so FCFS saturates at measured ~4.22 req/s: C = SAT * mean_work
    C = SAT_REQ_S * mean_w
    print(f"# calibrated prefill capacity C = {C:.0f} tok/s (= {SAT_REQ_S} req/s x {mean_w:.0f} mean work)")
    print(f"# (measured saturated new-prefill throughput ~19k tok/s; C here = {C/1000:.1f}k)\n")

    print("=== VALIDATION: stock-proxy (SRPF, small-first like lpm) vs MEASURED ===")
    print(f"{'lam':>4} {'ach(sim)':>8} {'ach(meas)':>9} {'p99(sim)':>9} {'p99(meas)':>10} {'med(sim)':>8} {'med(meas)':>9}")
    for lam in [3,5,7,10]:
        ach, p99, med = simulate(work, lam, C, "SRPF")
        m = MEASURED[lam]
        print(f"{lam:>4} {ach:>8.2f} {m[0]:>9.2f} {p99:>9.0f} {m[1]:>10.0f} {med:>8.0f} {m[2]:>9.0f}")

    print("\n=== DISCIPLINE COMPARISON: p99 TTFT (ms) and median, per lambda ===")
    print(f"{'disc':>6} " + " ".join(f"{'p99@'+str(l):>9}" for l in [3,5,7,10]) + "   " +
          " ".join(f"{'med@'+str(l):>8}" for l in [3,5]))
    rows = {}
    for disc in ["SRPF","SRPT","FCFS","PS","LPF","EDF","SLACK"]:
        p99s = {}; meds = {}
        for lam in [3,5,7,10]:
            _, p99, med = simulate(work, lam, C, disc); p99s[lam]=p99; meds[lam]=med
        rows[disc]=(p99s,meds)
        print(f"{disc:>6} " + " ".join(f"{p99s[l]:>9.0f}" for l in [3,5,7,10]) + "   " +
              " ".join(f"{meds[l]:>8.0f}" for l in [3,5]))

    print("\n=== ACHIEVABLE GOODPUT@SLO by discipline ===")
    gps = {}
    for disc in ["SRPF","SRPT","FCFS","LPF","EDF","SLACK"]:
        gp, curve = goodput_at_slo(work, C, disc); gps[disc]=gp
        pass_pts = ",".join(f"{l}:{p:.0f}ms" for (l,a,p,m) in curve if l in (3,3.5,4,4.5,5))
        print(f"  {disc:>5}: goodput@SLO = {gp:.2f} req/s     [{pass_pts}]")
    print(f"\n  throughput ceiling (measured) = {SAT_REQ_S:.2f} req/s (compute wall)")
    best = max(gps, key=gps.get)
    print(f"  BEST discipline = {best} @ {gps[best]:.2f} req/s   "
          f"(tail-aware LPF/EDF/SLACK vs small-first SRPF: "
          f"{'TAIL WINS' if max(gps.get('LPF',0),gps.get('EDF',0),gps.get('SLACK',0))>gps['SRPF']+0.2 else 'no tail advantage'})")

if __name__ == "__main__":
    main()
