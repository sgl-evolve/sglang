#!/usr/bin/env python3
"""Corrected cache sim with the ACTUAL bench_serving multiturn arrival model:
each conversation turn is served then the conversation is RE-QUEUED to the END of
a FIFO queue (bench_serving.py:187-190), and turns are pulled FIFO. So a
conversation's consecutive turns are separated by a FULL QUEUE CYCLE (~all other
active convs), giving a LONG reuse distance that can exceed the cache.

If this reproduces the baseline hit ~0.62 at cache=10.7M tokens, the reuse loss is
capacity-limited by the client-imposed reuse distance (NOT a server mechanism) ->
baseline near-optimal within budget; server scheduling can't change client order.

Token-granularity LRU over per-conversation linear radix paths. Timing-free: the
FIFO order IS the access order (semaphore concurrency only interleaves; the cycle
length is what sets reuse distance).
"""
import json, os

TRACE = os.path.join(os.path.dirname(__file__), "trace.json")
CACHE = 10_700_000  # L1(2.35M) + L2(8.4M) tokens

def load():
    convs = json.load(open(TRACE))
    return [[(t["prompt_len"], t["output_len"]) for t in c] for c in convs]

def run(convs, cache_cap, order="fifo"):
    # per-conv cumulative-before-turn and full-seq-after-turn
    N = len(convs)
    cum_before = []
    for turns in convs:
        cb = []; c = 0
        for (p, o) in turns:
            cb.append(c); c += p + o
        cum_before.append(cb)
    # LRU cache of token "pages" keyed by (conv, pos). We model per-conv a
    # resident depth with global LRU eviction by last-access "time" (serve index).
    # Simpler exact LRU via reuse-distance would need a stack; we approximate with
    # a global ordered dict of conv->last_serve_idx and evict least-recent convs'
    # tails until fit. Track resident_depth per conv.
    resident = [0]*N          # resident prefix tokens (contiguous from root)
    last_used = [-1]*N        # serve index of last access
    total_resident = 0

    # build FIFO schedule: start turn0 of all convs in order, re-queue after each turn
    from collections import deque
    idx = [0]*N
    q = deque(range(N))       # conv ids, turn0 first (arrival order = trace order)

    hits = 0; presented = 0; serve = 0
    # eviction: when over cap, evict whole conversations' resident tails by LRU
    def evict_to_fit():
        nonlocal total_resident
        if total_resident <= cache_cap: return
        # LRU order of resident convs
        order_ids = sorted((c for c in range(N) if resident[c] > 0), key=lambda c: last_used[c])
        for c in order_ids:
            if total_resident <= cache_cap: break
            total_resident -= resident[c]
            resident[c] = 0

    while q:
        c = q.popleft()
        i = idx[c]
        p, o = convs[c][i]
        H = cum_before[c][i]          # reusable history before this turn
        IN = H + p                    # presented prompt tokens
        # reuse = min(resident, H); the request re-accesses the whole prefix
        reuse = min(resident[c], H)
        hits += reuse
        presented += IN
        # after serving, full seq up to end of this turn is resident (locked then cached)
        resident[c] = H + p + o
        last_used[c] = serve
        total_resident = sum(resident)  # recompute (O(N), fine for 1553)
        evict_to_fit()
        serve += 1
        idx[c] += 1
        if idx[c] < len(convs[c]):
            q.append(c)               # RE-QUEUE at END (the key: full-cycle gap)

    return hits/presented if presented else 0

if __name__ == "__main__":
    convs = load()
    for cap in [CACHE, 5_000_000, 20_000_000, 40_000_000, 10**12]:
        h = run(convs, cap)
        tag = "INFINITE" if cap>10**11 else f"{cap/1e6:.1f}M"
        print(f"cache={tag:>8}  hit={h:.4f}")
    print("baseline actual hit = 0.62")
