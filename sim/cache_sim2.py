#!/usr/bin/env python3
"""Goodput-curve simulator (time-stepped) of the v0.25 2-tier HiCache serving loop.

Adds over cache_sim.py: (1) rate-limited client dispatch of ALL turns (new + re-enqueued) at lambda with a
128 in-flight semaphore (matches bench_serving); (2) shared prefill/decode throughput ceilings so a real
QUEUE forms as lambda rises (to probe the goodput knee); (3) a max_running server cap and a cold-prefill
DEFER mechanism. Purpose: does deferring cache-COLD large-document prefills under load raise hit_rate and
push the p99-TTFT-SLO knee right? Compares policies at lambda=3/4/5. Absolute numbers are approximate
(the real eval is ground truth); the RELATIVE hit_rate / p99 deltas across policies guide mechanism choice.

Cache model = cache_sim.py's (linear per-conv radix chain, LRU tail-trim, both tiers as one C-token pool).
"""
import json, os, argparse
from collections import deque, OrderedDict
import numpy as np

WL = os.path.join(os.path.dirname(__file__), "workload_tokens.json")

def load_workload():
    d = json.load(open(WL)); convs = []
    for turns in d["conversations"]:
        P, meta, cum = [], [], 0
        for t in turns:
            P.append(cum); meta.append((t["new_prompt_tok"], t["output_tok"]))
            cum += t["new_prompt_tok"] + t["output_tok"]
        convs.append((P, meta))
    return convs, d["meta"]

def simulate(convs, lam=3.0, C=7_500_000, in_flight_cap=128, max_running=128,
             P_max=45000.0, D_max=320.0, dt=0.25, seed=0,
             cold_defer=None, cold_frac=0.5, conv_cap=None, evict_finished=False):
    """cold_defer: max # of cache-COLD requests allowed to PREFILL concurrently (None = unlimited).
       conv_cap: max # of simultaneously-ACTIVE (cache-holding or running) conversations. A NEW conv
                 (R==0, not running) is DEFERRED when the active-conv working set is at the cap; continuing
                 turns (R>0) are always admitted. Bounds the doc working set to keep active docs resident.
       A request is COLD if its matched-hit fraction < cold_frac."""
    rng = np.random.default_rng(seed)
    N = len(convs)
    nturns = [len(c[0]) for c in convs]

    R = [0]*N
    lru = OrderedDict()
    resident = 0

    client_q = deque((c, 0) for c in range(N))   # seeded with all turn-0s (disable_shuffle)
    in_flight = 0
    ready = deque()          # dispatched to server, not yet started
    prefilling = []          # dicts: {c,t,rem_pref,P,n,o,hit,t_disp,t_start}
    decoding = []            # dicts: {c,t,rem_dec,...}
    done_ttft = []; done = 0
    t = 0.0
    disp_bucket = 0.0
    cold_running = 0

    def evict(needed, protect):
        nonlocal resident
        while resident + needed > C and lru:
            c0 = next(iter(lru))
            if c0 == protect:
                lru.move_to_end(c0)
                if all(k == protect for k in lru): break
                continue
            deficit = resident + needed - C
            if R[c0] <= deficit:
                resident -= R[c0]; R[c0] = 0; del lru[c0]
            else:
                R[c0] -= deficit; resident -= deficit

    def start_req(c, tt):
        nonlocal resident, cold_running
        P = convs[c][0][tt]; n, o = convs[c][1][tt]
        hit = min(R[c], P)
        pref = (P - hit) + n
        new_total = P + n + o
        grow = new_total - R[c]
        if grow > 0: evict(grow, protect=c)
        resident += (new_total - R[c]); R[c] = new_total
        if c in lru: del lru[c]
        is_cold = (hit / max(1, P)) < cold_frac
        if is_cold: cold_running += 1
        prefilling.append({"c": c, "t": tt, "rem": float(pref), "P": P, "n": n, "o": o,
                           "hit": hit, "cold": is_cold, "t_disp": disp_time[(c, tt)]})

    disp_time = {}
    # metrics
    hit_tok = 0; prompt_tok = 0
    maxN = sum(nturns)

    while done < N:
        # 1) dispatch from client queue at rate lambda, honoring in-flight cap
        disp_bucket += lam * dt
        while disp_bucket >= 1.0 and client_q and in_flight < in_flight_cap:
            c, tt = client_q.popleft()
            disp_bucket -= 1.0
            in_flight += 1
            ready.append((c, tt)); disp_time[(c, tt)] = t
        # 2) start ready reqs into running (respect max_running + cold_defer + conv_cap)
        running = len(prefilling) + len(decoding)
        run_convs = set(r["c"] for r in prefilling) | set(r["c"] for r in decoding)
        def active_convs():
            return len(lru) + len(run_convs)
        while ready and running < max_running:
            picked = None
            for j, (c, tt) in enumerate(ready):
                P = convs[c][0][tt]; hit = min(R[c], P)
                is_new = (R[c] == 0 and c not in run_convs)
                # conv_cap: defer a NEW conv when the active working set is full
                if conv_cap is not None and is_new and active_convs() >= conv_cap:
                    continue
                # cold_defer: defer a cold req when concurrent cold prefills are capped
                if cold_defer is not None:
                    isc = (hit/max(1,P)) < cold_frac
                    if isc and cold_running >= cold_defer:
                        continue
                picked = j; break
            if picked is None:
                break   # nothing admissible right now -> wait
            c, tt = ready[picked]; del ready[picked]
            start_req(c, tt); running += 1; run_convs.add(c)
        # 3) advance prefill (shared P_max)
        if prefilling:
            share = P_max * dt / len(prefilling)
            fin = []
            for r in prefilling:
                r["rem"] -= share
                if r["rem"] <= 0:
                    fin.append(r)
            for r in fin:
                prefilling.remove(r)
                if r["cold"]: cold_running -= 1
                done_ttft.append(t + dt - r["t_disp"])   # TTFT proxy = dispatch->first token
                hit_tok += r["hit"]; prompt_tok += (r["P"] + r["n"])
                r["rem_dec"] = float(r["o"]); decoding.append(r)
        # 4) advance decode (shared D_max)
        if decoding:
            share = D_max * dt / len(decoding)
            fin = []
            for r in decoding:
                r["rem_dec"] -= share
                if r["rem_dec"] <= 0: fin.append(r)
            for r in fin:
                decoding.remove(r)
                c = r["c"]; tt = r["t"]
                # returns to cache as MRU
                if R[c] > 0: lru[c] = True; lru.move_to_end(c)
                in_flight -= 1
                if tt + 1 < nturns[c]:
                    client_q.append((c, tt + 1))   # re-enqueue next turn (client tail, rate-limited)
                else:
                    done += 1
                    if evict_finished and R[c] > 0:   # ORACLE: drop finished conv's dead-weight cache
                        resident -= R[c]; R[c] = 0
                        if c in lru: del lru[c]
        t += dt
        if t > 200000:  # safety
            break
    ttft = np.array(done_ttft) if done_ttft else np.array([0.0])
    return {"lam": lam, "cold_defer": cold_defer, "max_running": max_running,
            "hit_rate": round(hit_tok/max(1,prompt_tok), 4),
            "makespan_s": round(t, 0),
            "ttft_mean": round(float(ttft.mean()), 2),
            "ttft_p99": round(float(np.percentile(ttft, 99)), 2),
            "ttft_p50": round(float(np.percentile(ttft, 50)), 2),
            "n_done_turns": len(done_ttft)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=int, default=7_500_000)
    ap.add_argument("--P_max", type=float, default=45000.0)
    ap.add_argument("--D_max", type=float, default=320.0)
    ap.add_argument("--lams", default="3,4,5")
    args = ap.parse_args()
    convs, meta = load_workload()
    print("workload:", meta["n_conversations"], "convs,", meta["n_turns_total"], "turns")
    print(f"C={args.C} P_max={args.P_max} D_max={args.D_max}\n")
    lams = [float(x) for x in args.lams.split(",")]
    configs = [("fcfs", dict()),
               ("convcap600", dict(conv_cap=600)),
               ("convcap450", dict(conv_cap=450)),
               ("convcap300", dict(conv_cap=300)),
               ("convcap200", dict(conv_cap=200)),
               ("convcap120", dict(conv_cap=120))]
    hdr = f"{'lam':>4} {'config':<14}{'hit':>7}{'makespan':>10}{'ttft_p50':>9}{'ttft_p99':>9}"
    print(hdr)
    import time
    for lam in lams:
        for name, kw in configs:
            t0=time.time()
            r = simulate(convs, lam=lam, C=args.C, P_max=args.P_max, D_max=args.D_max, **kw)
            print(f"{lam:>4} {name:<14}{r['hit_rate']:>7}{r['makespan_s']:>10}{r['ttft_p50']:>9}{r['ttft_p99']:>9}   ({time.time()-t0:.0f}s)")
        print()

if __name__ == "__main__":
    main()
