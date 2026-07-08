"""FAST-SCREEN: reuse-gated eviction. 39% of convs are single-turn (never reused) ->
their docs are pure cache pollution. The server CAN observe reuse (a prefix-match hit
means this path was seen before). So: evict 'unproven' convs (never yet re-accessed)
BEFORE 'proven' ones (re-accessed >=1). This is realizable (unlike completion oracle).

Risk: a multi-turn conv is 'unproven' until its FIRST reuse (turn1), which arrives ~1
full cycle after turn0. If the cache evicts the unproven doc during that cycle, turn1
MISSES -> we lose the very reuse we wanted. So unproven-first eviction reclaims 1-turn
pollution but may sacrifice multi-turn first-reuse. Net is unclear -> sim it.

Compare: pure LRU 0.733, oracle-completion 0.806.
"""
import json, os
from collections import deque

TRACE = os.path.join(os.path.dirname(__file__), "trace.json")
CACHE = 10_700_000

def load():
    convs = json.load(open(TRACE))
    return [[(t["prompt_len"], t["output_len"]) for t in c] for c in convs]

def run(convs, cache_cap, mode="reuse_gated"):
    N = len(convs)
    cum_before = []
    for turns in convs:
        cb = []; c = 0
        for (p, o) in turns:
            cb.append(c); c += p + o
        cum_before.append(cb)
    resident = [0]*N
    last_used = [-1]*N
    served_cnt = [0]*N       # how many turns of this conv have been served (proven if >=2)
    idx = [0]*N
    total_resident = 0
    q = deque(range(N))
    hits = 0; presented = 0; serve = 0

    def evict():
        nonlocal total_resident
        if total_resident <= cache_cap: return
        rc = [c for c in range(N) if resident[c] > 0]
        if mode == "reuse_gated":
            # unproven (served once, never reused) first, then proven; each by LRU
            rc.sort(key=lambda c: (served_cnt[c] >= 2, last_used[c]))
        else:  # pure LRU
            rc.sort(key=lambda c: last_used[c])
        for c in rc:
            if total_resident <= cache_cap: break
            total_resident -= resident[c]; resident[c] = 0

    while q:
        c = q.popleft(); i = idx[c]
        p, o = convs[c][i]; H = cum_before[c][i]
        reuse = min(resident[c], H)
        hits += reuse; presented += H + p
        resident[c] = H + p + o
        last_used[c] = serve; served_cnt[c] += 1
        total_resident = sum(resident)
        evict()
        serve += 1; idx[c] += 1
        if idx[c] < len(convs[c]): q.append(c)
    return hits/presented if presented else 0

if __name__ == "__main__":
    convs = load()
    print(f"cache={CACHE/1e6:.1f}M  pure-LRU=0.733  oracle=0.806")
    for mode in ["lru", "reuse_gated"]:
        h = run(convs, CACHE, mode)
        print(f"  {mode:>12}: hit={h:.4f}")
