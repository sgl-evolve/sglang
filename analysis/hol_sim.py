#!/usr/bin/env python3
"""floyd: head-of-line (HOL) blocking test — does the λ3 p99 come from big cold docs
blocking small ones (reducible by interleaving), or from big docs' own prefill (a floor)?

Faithful model of the sglang prefill mechanism (verified from server.log + scheduler.py):
  - Prefill serves ONE new sequence's chunk of B=6144 tokens per prefill step.
  - STOCK: a chunked big doc runs its ceil(work/B) chunks BACK-TO-BACK (get_new_batch_prefill
    continues chunked_req every step; scheduler.py:2838-2841) — it monopolizes prefill until done.
  - INTERLEAVE (the proposed mechanism): after each 6144-chunk, the scheduler may serve a
    different waiting request's chunk — so a small doc need not wait for a big doc's full prefill.
  - SHORTEST-FIRST (SRPF, sibling's lever): serve waiting docs shortest-total-work first (no
    mid-doc interruption).
Step time tau = B / C_step, C_step = per-step new-token prefill rate (~39k tok/s from logs).
TTFT(turn) = (last-chunk-done time) - admit time. Report p99 TTFT vs the 8s SLO.
Arrivals: Poisson(λ) over the real per-turn cache-adjusted work (LRU@CAP, conversation order).
"""
import ast, json, heapq, math, statistics as st
from collections import OrderedDict, deque

TRACE = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CPT = 4.0; CAP = 9_000_000; B = 6144; C_STEP = 39000.0
TAU = B / C_STEP  # ~0.157s per prefill step

def load_work():
    convs = []
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line); doc = r.get("input", "")
            try: qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception: qa = []
            turns = []
            for i, t in enumerate(qa):
                if not isinstance(t, dict): continue
                u = ("Input: "+doc+" Question: "+str(t.get("Q",""))) if i == 0 else str(t.get("Q",""))
                turns.append((max(1, int(len(u)/CPT)), max(1, int(len(str(t.get("A","")))/CPT))))
            if turns: convs.append(turns)
    cache = OrderedDict(); res = 0; work = []
    ctx = {}; idx = {}; tof = {}; pend = deque(); it = iter(enumerate(convs)); done = False
    def evict(need):
        nonlocal res
        while res+need > CAP and cache:
            _, ev = cache.popitem(last=False); res -= ev
    while not done or pend:
        try:
            cid, turns = next(it); ctx[cid]=0; idx[cid]=0; tof[cid]=turns; pend.append(cid)
        except StopIteration: done = True
        if not pend: break
        cid = pend.popleft(); turns = tof[cid]; ti = idx[cid]
        intok, outok = turns[ti]; context = ctx[cid]+intok
        cached = cache.get(cid, 0); adj = max(0, context-cached) if cid in cache else context
        work.append(max(1, adj))
        newp = context+outok; old = cache.pop(cid, 0); res -= old
        evict(newp); cache[cid]=newp; res += newp; ctx[cid]=newp; idx[cid]+=1
        if idx[cid] < len(turns): pend.append(cid)
    return work

class LCG:
    def __init__(s, x): s.s = x & 0x7fffffff
    def u(s): s.s = (1103515245*s.s+12345) & 0x7fffffff; return (s.s+1)/0x80000000

def simulate(work, lam, policy, seed=1):
    """Per prefill STEP, B=6144 new tokens are prefilled, filled from the active set in
    POLICY order (small turns pack; a big doc takes a 6144 chunk). One step = TAU seconds."""
    n = len(work); rng = LCG(seed)
    arr = [0.0]*n; t = 0.0
    for i in range(n): t += -math.log(rng.u())/lam; arr[i] = t
    rem = [float(w) for w in work]; ttft = [0.0]*n
    now = 0.0; ai = 0; done = 0
    active = []           # admitted, prefill-incomplete
    CONC = 256; cur = None; rr = 0
    def admit():
        nonlocal ai
        while ai < n and arr[ai] <= now+1e-9 and len(active) < CONC:
            active.append(ai); ai += 1
    while done < n:
        admit()
        if not active:
            if ai < n: now = max(now, arr[ai]); continue
            break
        # order in which docs get the step's 6144-token budget
        if policy == "stock":            # FCFS + back-to-back: oldest first; a big doc at the
            order = list(active)         # head takes the whole budget each step until it finishes
        elif policy == "srpf":           # shortest-remaining first (small turns drain first)
            order = sorted(active, key=lambda j: rem[j])
        elif policy == "interleave":     # round-robin start point → rotate who gets budget first
            k = rr % len(active); order = list(active)[k:] + list(active)[:k]; rr += 1
        elif policy == "lpm":            # DEFAULT: reuse turns (≤1 chunk cold work) first, then cold docs FCFS
            reuse = [j for j in active if work[j] <= B]
            cold  = [j for j in active if work[j] > B]
            order = reuse + cold          # reuse-first; cold docs remain in arrival order (like lpm among 0-prefix)
        else: raise ValueError(policy)
        budget = B; now += TAU; finished = []
        for i in order:
            if budget <= 0: break
            take = min(rem[i], budget); rem[i] -= take; budget -= take
            if rem[i] <= 1e-6:
                ttft[i] = (now - arr[i]) * 1000.0; finished.append(i)
        for i in finished:
            active.remove(i); done += 1
        admit()
    ts = sorted(ttft)
    return ts[int(0.99*n)], ts[n//2]

def main():
    work = load_work(); n = len(work)
    print(f"turns={n} tau={TAU*1000:.0f}ms/step  max-doc chunks={max(math.ceil(w/B) for w in work)} (~{max(math.ceil(w/B) for w in work)*TAU:.1f}s prefill)")
    print(f"{'lam':>4} {'policy':>10} {'p99_ttft':>9} {'p50':>7}")
    for lam in [3, 5]:
        for pol in ["stock", "srpf", "interleave"]:
            p99, p50 = simulate(work, lam, pol)
            print(f"{lam:>4} {pol:>10} {p99:>8.0f}ms {p50:>6.0f}ms")
    print("\nIf interleave/srpf p99 << stock p99 at λ3 => λ3 p99 is HEAD-OF-LINE blocking (reducible).")
    print("If all ~equal => λ3 p99 is the big docs' own prefill floor (irreducible).")

if __name__ == "__main__":
    main()
