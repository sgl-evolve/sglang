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
        elif policy == "srpf":           # PREEMPTIVE shortest-remaining (re-sorted every step)
            order = sorted(active, key=lambda j: rem[j])
        elif policy == "srpf_np":        # NON-PREEMPTIVE srpf (= my --schedule-policy srpf impl):
            # reorder the WAITING set shortest-first, but an in-progress (mid-chunk) doc must
            # continue (chunked_req back-to-back). New small arrivals wait behind an in-progress big doc.
            inprog = cur if (cur is not None and cur in active and 0 < rem[cur] < work[cur]) else None
            order = [inprog] if inprog is not None else sorted(active, key=lambda j: work[j])
        elif policy == "interleave":     # round-robin start point → rotate who gets budget first
            k = rr % len(active); order = list(active)[k:] + list(active)[:k]; rr += 1
        elif policy == "lpm":            # DEFAULT: reuse turns (≤1 chunk cold work) first, then cold docs FCFS
            reuse = [j for j in active if work[j] <= B]
            cold  = [j for j in active if work[j] > B]
            order = reuse + cold          # reuse-first; cold docs remain in arrival order (like lpm among 0-prefix)
        elif policy.startswith("fl"):     # FAST-LANE (my mechanism): FCFS ordering (NO reorder), but reserve
            order = list(active)          # R tokens/step for shortest waiting SMALL turns so they co-run with
        elif policy.startswith("rpb"):    # floyd RPB: non-preempt in-progress big doc CAPPED at (1-r)*B,
            order = list(active)          # reserve r*B for the SRPF-sorted waiting turns (handled below).
        else: raise ValueError(policy)    # the in-progress big cold doc instead of waiting for it to finish.
        budget = B; now += TAU; finished = []
        def _fill(i, cap):
            nonlocal budget
            take = min(rem[i], cap, budget); rem[i] -= take; budget -= take
            if rem[i] <= 1e-6:
                ttft[i] = (now - arr[i]) * 1000.0; finished.append(i)
        if policy.startswith("fl"):
            R = int(policy[2:])           # reserve size in tokens (e.g. fl2048)
            smalls = sorted([j for j in active if work[j] <= B], key=lambda j: rem[j])
            bigs   = [j for j in active if work[j] > B]        # keep FCFS order (no reorder among cold docs)
            sb = min(R, budget)
            for i in smalls:              # phase 1: small turns get up to R (the fast lane)
                if sb <= 0: break
                pre = budget; _fill(i, sb); sb -= (pre - budget)
            for i in bigs:                # phase 2: big cold docs get the rest (B-R + any unused reserve)
                if budget <= 0: break
                _fill(i, budget)
            for i in smalls:             # phase 3: spill leftover budget back to remaining small turns
                if budget <= 0: break
                if rem[i] > 1e-6: _fill(i, budget)
        elif policy.startswith("rpb"):
            r = int(policy[3:]) / 100.0   # reserve fraction, e.g. rpb25 = 0.25 of the chunk budget
            inprog = cur if (cur is not None and cur in active and 0 < rem[cur] < work[cur]) else None
            waiting = sorted([j for j in active if j != inprog], key=lambda j: rem[j])
            if inprog is not None:        # capped in-progress big doc gets (1-r)*B; rest reserved for waiting
                _fill(inprog, int((1.0 - r) * B))
            for i in waiting:             # reserved remainder -> SHORTEST waiting turns first (srpf order)
                if budget <= 0: break
                _fill(i, budget)
        else:
            for i in order:
                if budget <= 0: break
                _fill(i, budget)
        for i in finished:
            active.remove(i); done += 1
        if policy == "srpf_np":  # track the in-progress (mid-chunk) doc so it continues next step
            cur = next((i for i in order if i in active and 0 < rem[i] < work[i]), None)
        elif policy.startswith("rpb"):  # in-progress big doc = largest-work mid-chunk doc (the monopolizer)
            midchunk = [i for i in active if 0 < rem[i] < work[i]]
            cur = max(midchunk, key=lambda j: work[j]) if midchunk else None
        admit()
    ts = sorted(ttft)
    return ts[int(0.99*n)], ts[n//2]

def main():
    work = load_work(); n = len(work)
    print(f"turns={n} tau={TAU*1000:.0f}ms/step  max-doc chunks={max(math.ceil(w/B) for w in work)} (~{max(math.ceil(w/B) for w in work)*TAU:.1f}s prefill)")
    SEEDS = [1, 2, 3, 4, 5]
    lam = 3
    print(f"\n=== FAST-LANE screen @λ{lam} (5 seeds); SLO=8000ms; fl<R>=reserve R tok/step for small turns ===")
    print(f"{'policy':>12} {'p99 per seed (ms)':>44} {'mean':>7} {'pass/5':>7}")
    for pol in ["stock", "srpf_np", "fl1024", "fl2048", "fl3072", "srpf"]:
        p99s = [simulate(work, lam, pol, seed=s)[0] for s in SEEDS]
        passes = sum(1 for x in p99s if x <= 8000)
        mean = sum(p99s)/len(p99s)
        print(f"{pol:>12} {str([round(x) for x in p99s]):>44} {mean:>6.0f} {passes:>5}/5")
    print("\nfl* = FAST-LANE (fcfs, reserve R for small turns, NO reorder). srpf_np = my non-preempt SRPF.")
    print("Screen verdict: adopt fast-lane ONLY if it clears the coin-flip (≥ stock pass-rate, ideally 5/5)")
    print("AND is not strictly dominated by srpf_np. Sim reliable ONLY @λ3 (see caveat below).")
    print("★CAVEAT: this single-server sim is CALIBRATED ONLY AT λ3 (matches the measured coin-flip).")
    print("  ABOVE λ3 it over-serializes and is UNRELIABLE — it wrongly fails non-preempt srpf at λ3.5-4,")
    print("  contradicting base's GPU result that non-preempt SRPF PASSES λ5 (9/9). So the apparent")
    print("  'chunk-preemption beats non-preempt srpf at λ3.5' is a SIM ARTIFACT, NOT a real lever.")
    print("  Trust ONLY the λ3 comparison: stock coin-flips (3/5), srpf stabilizes it (5/5).")

if __name__ == "__main__":
    main()
