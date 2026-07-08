#!/usr/bin/env python3
"""Discrete-event HiCache 2-tier simulator for the loogle-multiturn workload.

Purpose: quantify recoverable prefix-reuse headroom and RANK candidate mechanisms
cheaply (no GPU) before spending a scarce full eval.

Models the pieces that drive eviction of about-to-be-reused KV:
  - L1 device pool (2.35M tok) + L2 host pool (8.4M tok), write-through, LRU.
  - running requests' KV is LOCKED (not evictable) during prefill+decode.
  - chunked prefill (6144 tok/iter): a big turn-0 doc holds partial KV locked
    across many iters -> concurrent big docs create acute device pressure.
  - 128-way conversation concurrency; a conv holds its slot across ALL its turns;
    turn i+1 becomes ready when turn i's decode completes.
  - Poisson(lambda) conversation starts into the semaphore.

Per-conv tiered residency: dev_depth[c] <= host_depth[c] = prefix tokens resident on
device / host (write-through => host>=device). A turn hits the covered prefix.
"""
import json, os, argparse, random
from collections import deque

TRACE = os.path.join(os.path.dirname(__file__), "trace.json")

CAP_DEVICE = 2_350_000
CAP_HOST   = 8_400_000
CHUNK      = 6144
CONCURRENCY = 128
LAMBDA     = 3.0
ITERS_PER_SEC = 40


def load_trace():
    convs = json.load(open(TRACE))
    return [[(t["prompt_len"], t["output_len"]) for t in c] for c in convs]


def structural(convs):
    total_in = reuse_full = own_prompt = reuse_doconly = 0
    for turns in convs:
        c = 0
        doc_len = turns[0][0]
        for i, (p, o) in enumerate(turns):
            H = c
            total_in += H + p
            own_prompt += p
            if i > 0:
                reuse_full += H
                reuse_doconly += min(doc_len, H)
            c += p + o
    print("==== STRUCTURAL (no eviction) ====")
    print(f"total IN (presented prompt tok)  = {total_in:,}   (baseline=99,908,618)")
    print(f"own new prompt tok (always miss) = {own_prompt:,}")
    print(f"ceiling hit (full-history reuse) = {reuse_full/total_in:.4f}  ({reuse_full:,})")
    print(f"ceiling hit (doc-only reuse)     = {reuse_doconly/total_in:.4f}  ({reuse_doconly:,})")
    print(f"baseline actual hit              = 0.6217")
    print(f"=> recoverable(full-actual)      = {reuse_full/total_in-0.6217:.4f} "
          f"(~{int((reuse_full/total_in-0.6217)*total_in):,} tok/run)")
    return total_in


class Conv:
    def __init__(self, cid, turns):
        self.cid = cid
        self.turns = turns
        self.cum_before = []
        c = 0
        for (p, o) in turns:
            self.cum_before.append(c); c += p + o
        self.idx = -1
        self.state = "new"
        self.lru = 0
        self.prefill_left = 0
        self.resident_at_admit = 0
        self.decode_left = 0


def admit_order(waiting, C, dev, host, policy):
    if policy == "fcfs":
        return list(waiting)
    # warm_first / lpm: continuations (large resident host prefix) first
    def resident(cid):
        co = C[cid]; H = co.cum_before[co.idx]
        return min(host[cid], H)
    return sorted(waiting, key=lambda cid: -resident(cid))


def simulate(convs, policy="fcfs", cold_throttle=None, tau=2048, seed=0, verbose=True):
    rng = random.Random(seed)
    N = len(convs)
    C = [Conv(i, t) for i, t in enumerate(convs)]
    dev = [0]*N          # device-resident prefix tokens
    host = [0]*N         # host-resident prefix tokens (>= dev)

    # Poisson arrival iter per conv
    t = 0.0; arrival_iter = []
    for _ in range(N):
        arrival_iter.append(int(t*ITERS_PER_SEC)); t += rng.expovariate(LAMBDA)
    start_q = deque(range(N))

    active = []; waiting = []; running = set()
    it = 0
    hits = misses = dev_hits = host_hits = recompute = 0
    completed = 0; total_turns = sum(len(c) for c in convs)
    evicted_host = 0

    def evict_device(need):
        cur = CAP_DEVICE - sum(dev)
        if cur >= need: return True
        cand = [cid for cid in range(N) if dev[cid] > 0 and cid not in running]
        cand.sort(key=lambda cid: C[cid].lru)
        for cid in cand:
            cur += dev[cid]; dev[cid] = 0
            if cur >= need: return True
        return cur >= need

    def evict_host(need):
        nonlocal evicted_host
        cur = CAP_HOST - sum(host)
        if cur >= need: return True
        cand = [cid for cid in range(N) if host[cid] > 0 and cid not in running]
        cand.sort(key=lambda cid: C[cid].lru)
        for cid in cand:
            evicted_host += host[cid]
            cur += host[cid]; host[cid] = 0; dev[cid] = 0
            if cur >= need: return True
        return cur >= need

    guard = 0
    while completed < total_turns and guard < 20_000_000:
        guard += 1
        # 1) start arrived convs into semaphore
        while start_q and len(active) < CONCURRENCY and arrival_iter[start_q[0]] <= it:
            cid = start_q.popleft(); co = C[cid]
            co.idx = 0; co.state = "waiting"; active.append(cid); waiting.append(cid)

        # 2) admit waiting -> prefill (policy order), subject to device budget + throttle
        order = admit_order(waiting, C, dev, host, policy)
        admitted_cold = 0
        for cid in order:
            co = C[cid]; i = co.idx
            p, o = co.turns[i]; H = co.cum_before[i]
            resident = min(host[cid], H)
            new_tok = (H + p) - resident       # tokens to (re)compute
            is_cold = resident < tau
            if cold_throttle is not None and is_cold and admitted_cold >= cold_throttle:
                continue
            if not evict_device(new_tok):
                continue
            running.add(cid); waiting.remove(cid)
            co.state = "prefill"; co.prefill_left = new_tok
            co.resident_at_admit = resident
            if is_cold: admitted_cold += 1

        # 3) one prefill iteration: CHUNK tokens across running prefills (round-robin)
        budget = CHUNK
        prefillers = [cid for cid in running if C[cid].state == "prefill"]
        for cid in prefillers:
            if budget <= 0: break
            co = C[cid]
            step = min(co.prefill_left, budget)
            co.prefill_left -= step; budget -= step; dev[cid] += step
            co.lru = it
            if co.prefill_left <= 0:
                i = co.idx; p, o = co.turns[i]; H = co.cum_before[i]
                seq = H + p
                hit = min(co.resident_at_admit, H); miss = H - hit
                hits += hit; misses += miss + p; recompute += miss
                if dev[cid] >= H and H > 0: dev_hits += hit
                else: host_hits += hit
                dev[cid] = seq; host[cid] = max(host[cid], seq)
                co.state = "decode"; co.decode_left = o
                evict_device(0); evict_host(0)

        # 4) decode: 1 tok per decoding conv
        for cid in list(running):
            co = C[cid]
            if co.state != "decode": continue
            co.decode_left -= 1; dev[cid] += 1
            host[cid] = max(host[cid], dev[cid]); co.lru = it
            if co.decode_left <= 0:
                completed += 1; running.discard(cid); i = co.idx
                if i + 1 < len(co.turns):
                    co.idx = i + 1; co.state = "waiting"; waiting.append(cid)
                else:
                    co.state = "done"; active.remove(cid)
        evict_device(0); evict_host(0)
        it += 1

    presented = hits + misses
    hitrate = hits/presented if presented else 0
    res = dict(policy=policy, throttle=cold_throttle, tau=tau,
               hit=round(hitrate,4),
               dev_frac=round(dev_hits/hits,4) if hits else 0,
               host_frac=round(host_hits/hits,4) if hits else 0,
               recompute=recompute, iters=it, evicted_host=evicted_host,
               completed=completed)
    if verbose:
        print(f"[{policy:11s} throttle={str(cold_throttle):4s} tau={tau:5d}] "
              f"hit={hitrate:.4f} dev/host={res['dev_frac']:.2f}/{res['host_frac']:.2f} "
              f"recompute={recompute:,} iters={it:,}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters-per-sec", type=int, default=ITERS_PER_SEC)
    ap.add_argument("--only-structural", action="store_true")
    a = ap.parse_args()
    ITERS_PER_SEC = a.iters_per_sec
    convs = load_trace()
    structural(convs)
    if a.only_structural:
        raise SystemExit
    print("\n==== DES (finite cache) ====")
    simulate(convs, policy="fcfs")
    simulate(convs, policy="warm_first")
    simulate(convs, policy="warm_first", cold_throttle=4)
    simulate(convs, policy="warm_first", cold_throttle=1)
