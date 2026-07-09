#!/usr/bin/env python3
"""Compute Belady's OPT (furthest-future-use) eviction for the v0.25 workload and compare to LRU.

Under Belady, on eviction we pick the conversation whose next use is furthest in the future
(or never used again). This is the theoretical maximum hit rate for any demand-paged cache
with the given capacity — any practical policy is ≤ OPT.

If LRU ≈ OPT, it proves eviction-order optimization is a dead end at this operating point.
"""
import json, os, heapq
from collections import OrderedDict
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_workload():
    d = json.load(open(os.path.join(os.path.dirname(__file__), "workload_tokens.json")))
    convs = []
    for turns in d["conversations"]:
        P, meta, cum = [], [], 0
        for t in turns:
            P.append(cum); meta.append((t["new_prompt_tok"], t["output_tok"]))
            cum += t["new_prompt_tok"] + t["output_tok"]
        convs.append((P, meta))
    return convs, d["meta"]

def simulate_belady(convs, C=10_200_000, M=128, lam=30.0, A=0.0003, B=0.010, seed=0):
    """Simulate with Belady OPT eviction: on eviction, remove the conversation whose next
    service time is furthest in the future."""
    rng = np.random.default_rng(seed)
    N = len(convs)
    arriv = np.cumsum(rng.exponential(1.0/lam, size=N))
    R = [0]*N
    turn_idx = [0]*N
    nturns = [len(c[0]) for c in convs]

    cached = {}  # conv_id -> True (in cache, not serving)
    serving = set()
    resident_sum = 0

    # Pre-compute the service schedule (FCFS order) to know future use times.
    # The schedule is: arrivals in order, then re-enqueue after completion.
    # Build the full event sequence first.
    events = []  # (time, conv_id, turn_idx)
    evq = [(float(arriv[c]), 0, 'a', c) for c in range(N)]
    heapq.heapify(evq)
    ready_q = []  # (arrival_time, conv_id)
    t_idx = [0]*N
    now = 0.0
    busy = 0
    finished_c = 0
    ctr = 0

    while finished_c < N:
        while busy < M and ready_q:
            _, c = heapq.heappop(ready_q)
            t = t_idx[c]
            P = convs[c][0][t]; n_t, o_t = convs[c][1][t]
            miss = max(0, P + n_t)  # upper bound on service
            events.append((now, c, t))
            busy += 1
            heapq.heappush(evq, (now + A*miss + B*o_t, ctr, 'f', c)); ctr += 1
        if not evq:
            break
        now, _, kind, c = heapq.heappop(evq)
        if kind == 'a':
            heapq.heappush(ready_q, (now, c))
        else:
            busy -= 1
            t_idx[c] += 1
            if t_idx[c] < nturns[c]:
                heapq.heappush(ready_q, (now, c))
            else:
                finished_c += 1

    # Now events[] is the ordered list of (time, conv_id, turn_idx) — the actual service order.
    # Build next_use[c] = index in events[] of conv c's next service from any given position.
    # For Belady: at event i, we want: for each cached conv c, what's the next index j>i where c appears?
    # Pre-build: for each conv, the list of event indices where it's served.
    from collections import defaultdict
    conv_events = defaultdict(list)  # conv_id -> sorted list of event indices
    for i, (_, c, _) in enumerate(events):
        conv_events[c].append(i)

    # For each conv, maintain a pointer into its event list
    conv_ptr = {c: 0 for c in range(N)}

    def next_use_after(c, current_event_idx):
        """Return the event index of conv c's next use after current_event_idx, or inf."""
        ptr = conv_ptr[c]
        evts = conv_events[c]
        while ptr < len(evts) and evts[ptr] <= current_event_idx:
            ptr += 1
        conv_ptr[c] = ptr
        if ptr < len(evts):
            return evts[ptr]
        return float('inf')

    # Now replay with Belady eviction
    R2 = [0]*N
    turn_idx2 = [0]*N
    cached2 = {}
    resident_sum2 = 0
    hit_tokens = 0
    total_prompt = 0

    # Reset conv_ptr for the replay
    conv_ptr = {c: 0 for c in range(N)}

    for evt_i, (_, c, t) in enumerate(events):
        P = convs[c][0][t]; n_t, o_t = convs[c][1][t]
        hit = min(R2[c], P)
        new_total = P + n_t + o_t
        grow = new_total - R2[c]

        # Evict using Belady: remove the conv with the furthest next use
        while resident_sum2 + grow > C and cached2:
            # Find the conv in cached2 with the largest next_use_after
            worst_c = None
            worst_next = -1
            for cc in cached2:
                nu = next_use_after(cc, evt_i)
                if nu > worst_next:
                    worst_next = nu
                    worst_c = cc
            if worst_c is None:
                break
            deficit = (resident_sum2 + grow) - C
            if R2[worst_c] <= deficit:
                resident_sum2 -= R2[worst_c]; R2[worst_c] = 0
                del cached2[worst_c]
            else:
                R2[worst_c] -= deficit; resident_sum2 -= deficit

        hit_tokens += hit
        total_prompt += (P + n_t)
        resident_sum2 += grow; R2[c] = new_total
        cached2[c] = True
        turn_idx2[c] = t + 1

    return hit_tokens / max(1, total_prompt)

def simulate_lru(convs, C=10_200_000, M=128, lam=30.0, A=0.0003, B=0.010, seed=0):
    """Quick LRU sim matching Belady's event sequence."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sim.cache_sim import simulate
    r = simulate(convs, policy="fcfs", C=C, M=M, lam=lam, A=A, B=B, seed=seed)
    return r['hit_rate']

if __name__ == "__main__":
    convs, meta = load_workload()
    print("workload:", meta)
    print()
    print(f"{'C (M)':>8s} | {'LRU':>7s} | {'Belady':>7s} | {'gap':>7s}")
    print("-" * 40)
    import time
    for C_M in [7.8, 10.2, 12.0, 15.0]:
        C = int(C_M * 1_000_000)
        t0 = time.time()
        lru_hr = simulate_lru(convs, C=C, lam=30)
        bel_hr = simulate_belady(convs, C=C, lam=30)
        elapsed = time.time() - t0
        print(f"{C_M:>8.1f} | {lru_hr:.4f} | {bel_hr:.4f} | {(bel_hr-lru_hr)*100:+6.2f}pp  ({elapsed:.1f}s)")
