#!/usr/bin/env python3
"""kleinrock: EXPLORATORY (not in any paper yet) — characterize a candidate NOVEL mechanism before building it.

Question: is there a scheduling mechanism that clears the HOL tail like deferral (SRPF) but is NOT classic
SJF/SRPF (charter-disqualified), NOT budget-throttling (my reserved lane, failed), and NOT reorder-by-wrong-key
(LOF, failed)? Candidate: GUARANTEED-PROGRESS CHUNK-INTERLEAVING. A mega-doc in flight yields to waiting SHORT
requests between its 6144-token chunks (bounding short-req HOL to ~one chunk), while a progress guarantee forces
a mega-doc chunk at least every N iterations (bounded deferral -> no starvation, unlike SRPF's unbounded defer;
full-budget each iteration -> no throttle, unlike the reserved lane).

This chunk-level discrete-event sim compares FCFS / SRPF(whole-req defer) / INTERLEAVE on the SAME measured doc
distribution. Decision rule: if INTERLEAVE clears the short-req tail like SRPF AND improves the mega-doc's own
TTFT (progresses vs deferred), it is a genuinely distinct mechanism worth an engine build + GPU A/B. If it merely
matches SRPF on everything, the eval doesn't exercise the distinction -> neutral-vs-SRPF -> fold as a closed lead.
Deterministic, GPU-free. NOTE: separate from tools/hol_sim.py (whole-req model cited by Paper 2 §5.4); do not
conflate their absolute numbers.
"""
import json, sys, math
S = json.load(open("tools/trace_stats.json"))
doc = S["doc_toks"]; nturns = S["nturns"]; Qm = S["q_p50"]
P = float(sys.argv[1]) if len(sys.argv) > 1 else 14000.0
CHUNK = 6144
DT = CHUNK / P                 # seconds per iteration (one chunk)
import os
N_PROGRESS = int(os.environ.get("NPROG","2"))   # interleave: force a mega-doc chunk at least every N iters

class LCG:
    def __init__(s, seed): s.x = seed & 0xFFFFFFFF
    def u(s):
        s.x = (1103515245 * s.x + 12345) & 0x7FFFFFFF
        return (s.x % 1_000_000 + 0.5) / 1_000_000

def build(rng, lam):
    jobs = []                                  # (tokens, arrival)
    t = 0.0
    stream = []
    for d, n in zip(doc, nturns):
        stream.append(d + Qm)
        for _ in range(max(0, n - 1)): stream.append(Qm)
    for tok in stream:
        t += -math.log(rng.u()) / lam
        jobs.append([tok, t])
    jobs.sort(key=lambda j: j[1])
    return jobs

def run(policy, lam, seed=12345):
    jobs = build(LCG(seed), lam)
    m = len(jobs)
    rem = [j[0] for j in jobs]                  # remaining tokens
    arr = [j[1] for j in jobs]
    ttft = [None] * m
    started = [False] * m
    ai = 0
    now = 0.0
    waiting = []            # indices arrived, not done
    active = None           # index of in-flight chunked mega-doc (interleave/fcfs/srpf all serve one at a time)
    iters_since_mega = 0
    done = 0
    SHORT = CHUNK           # "short" = fits in one chunk (<=6144 tok)
    while done < m:
        while ai < m and arr[ai] <= now + 1e-9:
            waiting.append(ai); ai += 1
        if not waiting and active is None:
            if ai < m: now = max(now, arr[ai]); continue
            else: break
        # choose which request gets this iteration's chunk
        pick = None
        if policy == "fcfs":
            if active is not None: pick = active
            elif waiting: waiting.sort(key=lambda i: arr[i]); pick = waiting[0]
        elif policy == "srpf":
            if active is not None: pick = active           # once started, finish (whole-req defer at admission)
            elif waiting: waiting.sort(key=lambda i: rem[i]); pick = waiting[0]
        elif policy == "interleave":
            shorts = [i for i in waiting if rem[i] <= SHORT and i != active]
            must_progress = (active is not None and iters_since_mega >= N_PROGRESS)
            if shorts and not must_progress:
                shorts.sort(key=lambda i: arr[i]); pick = shorts[0]   # yield to a waiting short req
            elif active is not None:
                pick = active                                          # advance the mega-doc one chunk
            elif waiting:
                waiting.sort(key=lambda i: rem[i]); pick = waiting[0]  # start smallest if nothing active
        if pick is None:
            if ai < m: now = max(now, arr[ai]); continue
            else: break
        if not started[pick]:
            started[pick] = True
        # serve one chunk of pick
        served = min(CHUNK, rem[pick])
        now += served / P
        rem[pick] -= served
        # track mega-doc activeness
        big = jobs[pick][0] > SHORT
        if policy == "interleave":
            if pick == active: iters_since_mega = 0
            else: iters_since_mega += 1
        if rem[pick] <= 0:
            ttft[pick] = now - arr[pick]      # prefill done => first token
            waiting.remove(pick) if pick in waiting else None
            if active == pick: active = None
            done += 1
        else:
            # still has tokens => it's a chunked (big) req in flight
            if big:
                active = pick
                if pick in waiting: waiting.remove(pick)
    tiny = sorted(ttft[i] for i in range(m) if jobs[i][0] < 1000)
    big  = sorted(ttft[i] for i in range(m) if jobs[i][0] >= 40000)   # mega-docs
    def p(xs, q): return xs[min(len(xs)-1, int(q*len(xs)))] if xs else float('nan')
    return dict(tiny_p99=p(tiny,0.99), tiny_p50=p(tiny,0.5), big_p50=p(big,0.5), big_p99=p(big,0.99), nbig=len(big))

if __name__ == "__main__":
    print(f"chunk-level sim P={P:.0f} DT={DT:.3f}s/iter N_progress={N_PROGRESS}")
    print(f"{'lam':>4} {'policy':>10} {'tiny_p99':>9} {'tiny_p50':>9} {'megadoc_p50':>11} {'megadoc_p99':>11}")
    for lam in [3,5,7]:
        for pol in ["fcfs","srpf","interleave"]:
            r = run(pol, lam)
            print(f"{lam:>4} {pol:>10} {r['tiny_p99']:>9.2f} {r['tiny_p50']:>9.2f} {r['big_p50']:>11.2f} {r['big_p99']:>11.2f}")
