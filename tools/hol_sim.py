#!/usr/bin/env python3
"""kleinrock: first-principles generality check for Paper 2 (prefill-hol-defer).

Is the FCFS head-of-line tail (and its removal by deferral) an sglang artifact, or a GENERIC consequence of a
heavy-tailed prefill-size distribution under single-server, one-chunked-request-at-a-time serialization? This
is a minimal discrete-event simulator of exactly that queue, driven ONLY by the MEASURED doc-token distribution
(tools/trace_stats.json) + Poisson(lambda) arrivals + a single calibrated prefill rate P (tok/s). No engine, no
caching model, no sglang. We calibrate P so FCFS lambda=5 p99 matches the measured ~24 s, then PREDICT the SRPF
(deferral) p99 with the SAME P (params independent of the A/B) and compare to the measured ~7 s.

Model (matches Paper 2 sec 2.2): one prefill server; budget 6144 tok/iter; at most one chunked request in
flight; a request of T tokens occupies the server ceil(T/6144) iters * (6144/P) s (i.e. T/P s), NON-preemptive
at the request level (once it starts it runs to completion — the sglang invariant). FCFS picks earliest arrival;
SRPF picks smallest remaining prefill among WAITING requests when the server frees. TTFT = server-free-and-picked
-> prefill-done, minus arrival = queue wait + own prefill. We report p99 TTFT over the TINY requests (<1K tok),
the HOL victims. Deterministic (seeded LCG); GPU-free.
"""
import json, sys, statistics as st

S = json.load(open("tools/trace_stats.json"))
doc = S["doc_toks"]; nturns = S["nturns"]; Qm = S["q_p50"]; Am = S["a_p50"]
P = float(sys.argv[1]) if len(sys.argv) > 1 else 20500.0   # calibrated prefill rate tok/s
CHUNK = 6144

# deterministic LCG for reproducible Poisson (Date/random unavailable in some envs; keep it self-contained)
class LCG:
    def __init__(s, seed): s.x = seed & 0xFFFFFFFF
    def u(s):
        s.x = (1103515245 * s.x + 12345) & 0x7FFFFFFF
        return (s.x % 1_000_000 + 0.5) / 1_000_000

def build_jobs(rng):
    """Per-conversation: turn-0 prefill = full doc (cold, unique -> uncacheable, Paper 2 sec 3); follow-up turns
    = tiny prefill (only the new question Qm; prefix cached). Job size in tokens."""
    jobs = []
    for d, n in zip(doc, nturns):
        jobs.append(d + Qm)                 # turn 0: the cold document (the HOL cause when large)
        for _ in range(max(0, n - 1)):
            jobs.append(Qm)                 # follow-up turns: tiny (cached prefix)
    return jobs

def simulate(lam, policy, seed=12345):
    rng = LCG(seed)
    jobs = build_jobs(rng)
    # Poisson arrivals: inter-arrival ~ Exp(lam). arrival[i] cumulative.
    import math
    t = 0.0; arr = []
    for _ in jobs:
        t += -math.log(rng.u()) / lam
        arr.append(t)
    order = sorted(range(len(jobs)), key=lambda i: arr[i])
    # event loop: single server, non-preemptive at request level
    server_free = 0.0
    waiting = []          # indices admitted (arrived) but not yet served
    ttft = [None] * len(jobs)
    ai = 0                # next-to-arrive pointer (in arrival order)
    done = 0
    # process by advancing: when server frees, admit all arrivals <= server_free, pick next
    # simpler: iterate; maintain a frontier
    import heapq
    while done < len(jobs):
        # admit all arrivals up to max(server_free, next arrival)
        if not waiting:
            # jump to next arrival if server idle & queue empty
            if ai < len(order):
                server_free = max(server_free, arr[order[ai]])
        # admit arrivals that have come by server_free
        while ai < len(order) and arr[order[ai]] <= server_free + 1e-9:
            waiting.append(order[ai]); ai += 1
        if not waiting:
            continue
        # pick next job
        if policy == "fcfs":
            waiting.sort(key=lambda i: arr[i]); pick = waiting.pop(0)
        else:  # srpf: smallest remaining prefill
            waiting.sort(key=lambda i: jobs[i]); pick = waiting.pop(0)
        start = max(server_free, arr[pick])
        finish = start + jobs[pick] / P
        ttft[pick] = finish - arr[pick]
        server_free = finish
        done += 1
    tiny = [ttft[i] for i in range(len(jobs)) if jobs[i] < 1000]
    tiny.sort(); n = len(tiny)
    p99 = tiny[min(n - 1, int(0.99 * n))]; p50 = tiny[n // 2]
    return p50, p99, n

if __name__ == "__main__":
    print(f"HOL sim: P={P:.0f} tok/s, chunk={CHUNK}, jobs from measured doc dist (1553 convs)")
    print(f"{'lam':>4} {'policy':>6} {'tiny_p50_s':>11} {'tiny_p99_s':>11}  (tiny = <1K-tok HOL victims)")
    for lam in [3, 5, 7]:
        for pol in ["fcfs", "srpf"]:
            p50, p99, n = simulate(lam, pol)
            print(f"{lam:>4} {pol:>6} {p50:>11.2f} {p99:>11.2f}   (n_tiny={n})")
