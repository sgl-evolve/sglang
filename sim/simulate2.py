#!/usr/bin/env python3
"""Sim v2 — does STOCK at-server protection close the LRU-vs-Belady recompute gap?

Key upgrade over v1: (a) processor-sharing (PS) service so concurrency actually builds under load;
(b) explicit client semaphore (256) => reqs beyond it are CLIENT-BLOCKED (invisible to server, prefix
ages); (c) STOCK PROTECTION modeled: any conv with a req currently AT-SERVER (prefilling/decoding) has
its prefix recency continuously refreshed (mirrors fcfs's per-step match_prefix bumping last_access).

Compares recompute under: LRU+stockprotect (realistic stock) vs OPT (Belady over the realized access
order). If OPT ~= LRU+protect across congestion, the concurrent regime does NOT leave capturable residency
headroom (supports the bounded-negative). If OPT << LRU+protect, headroom survives (mechanism hope).

Time-stepped PS. Calibrate P_AGG/D_AGG from the baseline later; here sweep to be robust to P.
Cache model = radix-faithful per-conv resident depth (contiguous prefix; leaf-first eviction).
"""
import json, os, random, numpy as np
from collections import deque, defaultdict

TRACE = os.path.join(os.path.dirname(__file__), "conv_trace.json")

def run(convs, lam, cap, P_agg, D_agg, policy, maxc=256, dt=0.25, seed=1, opt_future=None,
        record=False):
    rng = random.Random(seed)
    rs = lambda ci, k: sum(convs[ci][k])  # run size = in+out tokens of turn k
    depth = defaultdict(int); used = 0
    last = defaultdict(float); atserver = defaultdict(int)  # conv -> #reqs at server
    # requests
    client_q = deque((ci, 0) for ci in range(len(convs)))   # lambda-paced
    blocked = deque()                                        # pulled, semaphore-blocked
    server = []  # list of dicts: {ci,t,phase:'p'/'d',rem, admit}
    t = 0.0; next_pull = 0.0
    recompute = 0; ttfts = []; access_order = []; aseq = 0
    total = sum(len(c) for c in convs); done = 0
    fut_ptr = None
    if policy == "opt":
        fut_ptr = {k: 0 for k in opt_future}

    def evict(need):
        nonlocal used
        while used + need > cap:
            victim = None; best = None
            for cj, d in depth.items():
                if d <= 0: continue
                if policy == "opt":
                    fq = opt_future.get(cj); p = fut_ptr.get(cj, 0)
                    nxt = fq[p] if (fq and p < len(fq)) else 10**12
                    sc = -nxt          # evict farthest-next-use
                else:  # lru with stock protection: at-server convs have last=now (fresh)
                    sc = (t if atserver[cj] > 0 else last[cj])
                if best is None or sc < best: best = sc; victim = cj
            if victim is None: break
            d = depth[victim]; used -= rs(victim, d - 1); depth[victim] = d - 1

    def admit_from_blocked():
        while blocked and len(server) < maxc:
            ci, tt = blocked.popleft(); start_req(ci, tt)

    def start_req(ci, tt):
        nonlocal recompute, aseq, used
        aseq += 1
        if record: access_order.append((ci, tt))
        if policy == "opt":
            fut_ptr[ci] = fut_ptr.get(ci, 0) + 1
        atserver[ci] += 1
        d = depth[ci]; miss = sum(rs(ci, k) for k in range(d, tt)) if d < tt else 0
        recompute += miss
        uncached = convs[ci][tt][0] + miss
        server.append({"ci": ci, "t": tt, "phase": "p", "rem": uncached, "admit": t, "out": convs[ci][tt][1]})

    # main loop
    while done < total:
        # arrivals (Poisson pulls into server or blocked)
        while next_pull <= t and client_q:
            ci, tt = client_q.popleft()
            if len(server) < maxc: start_req(ci, tt)
            else: blocked.append((ci, tt))
            next_pull += rng.expovariate(lam)
        if not client_q and next_pull <= t:
            next_pull = t + 1.0 / lam  # idle; keep clock moving
        # PS service: split throughput among phase peers
        pf = [r for r in server if r["phase"] == "p"]; dc = [r for r in server if r["phase"] == "d"]
        if pf:
            rate = P_agg / len(pf) * dt
            for r in pf:
                r["rem"] -= rate
                if r["rem"] <= 0:
                    ttfts.append(t + dt - r["admit"]); r["phase"] = "d"; r["rem"] = r["out"]
        if dc:
            rate = D_agg / len(dc) * dt
            for r in dc: r["rem"] -= rate
        # completions
        fin = [r for r in server if r["phase"] == "d" and r["rem"] <= 0]
        for r in fin:
            server.remove(r); ci, tt = r["ci"], r["t"]; atserver[ci] -= 1; done += 1
            # materialize KV up to depth tt+1
            add = sum(rs(ci, k) for k in range(depth[ci], tt + 1))
            if add > 0: evict(add)
            used += add; depth[ci] = tt + 1; last[ci] = t
            if tt + 1 < len(convs[ci]):
                client_q.append((ci, tt + 1))
        # refresh protection for at-server convs (stock behavior)
        for r in server: last[r["ci"]] = t
        admit_from_blocked()
        t += dt
        if t > 5e6: break  # safety
    ttfts.sort()
    pct = lambda q: ttfts[min(len(ttfts) - 1, int(len(ttfts) * q))] if ttfts else 0
    return {"policy": policy, "lam": lam, "recompute": recompute,
            "p50": pct(.5), "p90": pct(.9), "p99": pct(.99), "makespan": t, "n": len(ttfts),
            "access_order": access_order if record else None}

def build_future(access_order):
    fut = defaultdict(list)
    for (ci, t) in access_order:
        for k in range(t + 1): fut[(ci, k)].append(len(fut[(ci, k)]))  # placeholder
    # we need seq of accesses per conv (by conv, not node) for eviction farthest-use
    seqfut = defaultdict(list); s = 0
    for (ci, t) in access_order:
        s += 1; seqfut[ci].append(s)
    return seqfut

if __name__ == "__main__":
    convs = json.load(open(TRACE))
    cap = float(os.environ.get("SIM_CAP", 10.7e6))
    print(f"convs={len(convs)} cap={cap:.2e}")
    print(f"{'lam':>4} {'Pagg':>7} {'policy':>14} {'recompute':>12} {'p50':>7} {'p90':>8} {'p99':>9} {'makespan':>9} {'maxc_hit?':>9}")
    for P_agg in [15000, 40000]:
        D_agg = P_agg  # rough; decode shares similarly
        for lam in [3, 5, 10]:
            base = run(convs, lam, cap, P_agg, D_agg, "lru", record=True)
            fut = build_future(base["access_order"])
            opt = run(convs, lam, cap, P_agg, D_agg, "opt", opt_future=fut)
            for r in (base, opt):
                tag = "lru+protect" if r["policy"] == "lru" else "opt(belady)"
                print(f"{lam:>4} {P_agg:>7} {tag:>14} {r['recompute']:>12,} {r['p50']:>7.1f} "
                      f"{r['p90']:>8.1f} {r['p99']:>9.1f} {r['makespan']:>9.0f}")
            gap = 100 * (base["recompute"] - opt["recompute"]) / max(1, base["recompute"])
            print(f"     -> OPT saves {gap:.1f}% recompute vs LRU+stock-protection  (lam={lam}, Pagg={P_agg})")
