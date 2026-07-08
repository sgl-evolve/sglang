#!/usr/bin/env python3
"""FAST-SCREEN (free, no GPU): does conversation co-residency / admission control
capture the 0.733->0.806 hit gap between exclusive tiering (10.7M cache) and the
infinite-cache ceiling?

Hypothesis: the reuse loss is a LONG reuse distance (bench re-queues each conv turn
to the FIFO tail -> a conv's turns are separated by a full ~1553-conv cycle). If the
SERVER admits only K conversations at a time (co-residency: keep a small active set,
defer the rest), the active set's reuse distance shrinks to a K-cycle. If K active
convs' working set fits in 10.7M, their intra-conv reuse is fully captured -> hit
should climb toward the 0.806 ceiling.

Counter-hypothesis (the capacity insight): admission does NOT reduce the working set
over the reuse distance -- a conv still needs its full doc resident from turn 1 to its
last turn, and while it is active that doc occupies cache. With big LooGLE docs, even a
handful of concurrent convs' docs can exceed cache, so K-limiting either (a) barely
moves hit (docs still thrash within the active set) or (b) requires K so small that
throughput collapses. Either way it's a rigorous NEGATIVE for goodput.

This sim measures ONLY hit (best case for admission). If hit rises materially we then
pay for a full eval to measure the p99/throughput cost; if not, admission is screened out.

Model: active set of <=K convs cycle FIFO (re-queue to active tail after each turn).
When a conv completes all turns, admit the next waiting conv. Same whole-conv LRU cache.
"""
import json, os
from collections import deque

TRACE = os.path.join(os.path.dirname(__file__), "trace.json")
CACHE = 10_700_000  # L1(2.35M)+L2(8.4M) = exclusive-tiering effective cache

def load():
    convs = json.load(open(TRACE))
    return [[(t["prompt_len"], t["output_len"]) for t in c] for c in convs]

def run(convs, cache_cap, K):
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

    def evict_to_fit():
        nonlocal total_resident
        if total_resident <= cache_cap: return
        order_ids = sorted((c for c in range(N) if resident[c] > 0), key=lambda c: last_used[c])
        for c in order_ids:
            if total_resident <= cache_cap: break
            total_resident -= resident[c]; resident[c] = 0

    waiting = deque(range(N))      # not-yet-admitted convs (arrival order)
    active = deque()              # currently admitted (<=K), cycling
    def admit():
        while len(active) < K and waiting:
            active.append(waiting.popleft())
    admit()

    hits = 0; presented = 0; serve = 0
    peak_active_ws = 0
    while active:
        c = active.popleft()
        i = idx[c]
        p, o = convs[c][i]
        H = cum_before[c][i]
        IN = H + p
        reuse = min(resident[c], H)
        hits += reuse; presented += IN
        resident[c] = H + p + o
        last_used[c] = serve
        total_resident = sum(resident)
        # track active-set working-set pressure (docs of active convs)
        aws = sum(resident[a] for a in active) + resident[c]
        peak_active_ws = max(peak_active_ws, aws)
        evict_to_fit()
        serve += 1
        idx[c] += 1
        if idx[c] < len(convs[c]):
            active.append(c)          # re-queue within the active set (short cycle)
        else:
            resident[c] = 0           # conv done -> free its doc, admit next
            total_resident = sum(resident)
            admit()
        if not active:
            admit()
    return hits/presented if presented else 0, peak_active_ws

if __name__ == "__main__":
    convs = load()
    print(f"convs={len(convs)}  cache={CACHE/1e6:.1f}M")
    print(f"{'K (concurrency)':>16} | {'hit':>7} | {'peak active WS':>14}")
    print("-"*46)
    for K in [4, 8, 16, 32, 64, 128, 256, 512, 1553]:
        h, pws = run(convs, CACHE, K)
        note = "  <- baseline/max-conc" if K in (128, 1553) else ""
        print(f"{K:>16} | {h:>7.4f} | {pws/1e6:>12.2f}M{note}")
    print("\nReference: exclusive tiering (full concurrency, 10.7M) measured 0.733;")
    print("infinite-cache ceiling 0.806. If small K does NOT lift hit >> 0.733,")
    print("admission cannot capture the gap at fixed budget -> rigorous negative.")
