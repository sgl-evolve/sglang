#!/usr/bin/env python3
"""FAST-SCREEN: is the 0.733->0.806 gap capturable by a REALIZABLE completion-aware
eviction, i.e. WITHOUT an oracle for 'conversation done'?

admission_sim showed: free a conv's KV the instant it completes -> hit hits the 0.806
ceiling (active working set is only 8.63M < 10.7M; the rest of the cache is dead weight
held by completed convs that LRU only evicts after a long lag). But the server has no
conv-id / turn-count -> it cannot know a conv is 'done'. The only observable is the
idle gap since a path was last extended.

Signal separation: an ACTIVE conv is re-accessed ~every full queue cycle (idle ~= 1
cycle in serve-index units, where cycle ~= number of live convs). A COMPLETED conv is
never re-accessed (idle -> infinity). So force-evicting any resident conv idle for more
than T * cycle should reclaim completed convs EARLIER than pure recency-LRU, while
sparing actives (idle ~1 cycle) as long as T > ~1.

We sweep T. cycle is ESTIMATED online as the number of currently-resident convs (a
server can count live radix paths). Compare hit vs pure-LRU (0.733) and oracle (0.806).
If a T-band captures most of the 7pp, the mechanism is realizable -> full eval.
"""
import json, os
from collections import deque

TRACE = os.path.join(os.path.dirname(__file__), "trace.json")
CACHE = 10_700_000

def load():
    convs = json.load(open(TRACE))
    return [[(t["prompt_len"], t["output_len"]) for t in c] for c in convs]

def run(convs, cache_cap, T):
    """T = idle-gap multiple of the estimated cycle beyond which a conv is force-evicted.
    T=inf reproduces pure LRU (no proactive completion eviction)."""
    N = len(convs)
    cum_before = []
    for turns in convs:
        cb = []; c = 0
        for (p, o) in turns:
            cb.append(c); c += p + o
        cum_before.append(cb)

    resident = [0]*N
    last_used = [-1]*N
    idx = [0]*N
    total_resident = 0
    from collections import deque
    q = deque(range(N))
    hits = 0; presented = 0; serve = 0

    def resident_convs():
        return [c for c in range(N) if resident[c] > 0]

    def evict():
        nonlocal total_resident
        rc = resident_convs()
        cycle = max(1, len(rc))           # online estimate of the queue cycle length
        # 1) proactive completion eviction: drop convs idle > T*cycle (predicted done)
        if T != float("inf"):
            thresh = T * cycle
            for c in rc:
                if (serve - last_used[c]) > thresh:
                    total_resident -= resident[c]; resident[c] = 0
        # 2) capacity LRU for the rest
        if total_resident > cache_cap:
            for c in sorted((c for c in range(N) if resident[c] > 0), key=lambda c: last_used[c]):
                if total_resident <= cache_cap: break
                total_resident -= resident[c]; resident[c] = 0

    while q:
        c = q.popleft()
        i = idx[c]
        p, o = convs[c][i]
        H = cum_before[c][i]
        reuse = min(resident[c], H)
        hits += reuse; presented += H + p
        resident[c] = H + p + o
        last_used[c] = serve
        total_resident = sum(resident)
        evict()
        serve += 1
        idx[c] += 1
        if idx[c] < len(convs[c]):
            q.append(c)
    return hits/presented if presented else 0

if __name__ == "__main__":
    convs = load()
    print(f"convs={len(convs)}  cache={CACHE/1e6:.1f}M   pure-LRU=0.733  oracle=0.806")
    print(f"{'T (idle/cycle)':>16} | {'hit':>7} | {'capture of 7.3pp gap':>22}")
    print("-"*54)
    base, ceil = 0.7334, 0.8062
    for T in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, float("inf")]:
        h = run(convs, CACHE, T)
        cap = 100*(h-base)/(ceil-base) if T != float("inf") else 0
        tag = "  (=pure LRU)" if T==float("inf") else ""
        print(f"{str(T):>16} | {h:>7.4f} | {cap:>19.0f}%{tag}")
