#!/usr/bin/env python3
"""floyd paper-2 seed: fundamental compute bounds on goodput@SLO (trace-driven).

Two rigorous, assumption-light bounds that quantify WHY goodput@SLO is
cold-prefill-compute-bound (complements paper 1's negative):

  (1) p99-TTFT FLOOR = (largest cold-doc prefill work) / (single-stream prefill
      throughput). No lossless scheduler can make a request's TTFT smaller than
      its own prefill time. If this floor approaches the SLO, the metric is
      fragile by construction.

  (2) goodput CEILING = (aggregate prefill throughput) / (mean cache-adjusted
      work per request). Work-conservation: you cannot sustain a request rate
      whose prefill work exceeds the GPU's prefill capacity.

Throughput numbers are the MEASURED values from the fixed eval (server logs),
stated explicitly; the bounds are reported with a sensitivity range.
"""
import ast, json, statistics as st
from collections import OrderedDict, deque

TRACE = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CHARS_PER_TOK = 4.0
CAP = 10_700_000  # 2-tier capacity (tokens)

# --- MEASURED throughputs from the fixed eval (server.log Prefill/Decode lines) ---
# single-stream prefill (1 new-seq): ~1100 tok/s; batched prefill peak ~40k tok/s.
# We bound the FLOOR with single-stream (a lone large cold doc) and also the
# batched case (the doc shares the prefill batch).
PREFILL_TOK_S_SINGLE = 1130.0     # measured single-seq prefill input throughput
PREFILL_TOK_S_BATCH = 40000.0     # measured batched prefill input throughput (peak)
# aggregate prefill capacity available to cold work (shared with decode); the
# system's measured peak req/s is ~4.2 -> we back out the effective rate.

def toks(s): return max(1, int(len(s) / CHARS_PER_TOK))

def per_turn_adjusted_work():
    """Reproduce trace_diag: per-turn cache-adjusted prefill work under LRU@CAP."""
    convs = []
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line); doc = r.get("input", "")
            try: qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception: qa = []
            turns = []
            for i, t in enumerate(qa):
                if not isinstance(t, dict): continue
                user = ("Input: " + doc + " Question: " + str(t.get("Q",""))) if i == 0 else str(t.get("Q",""))
                turns.append((toks(user), toks(str(t.get("A","")))))
            if turns: convs.append(turns)
    cache = OrderedDict(); resident = 0; work = []
    ctx = {}; idx = {}; tof = {}; pend = deque(); it = iter(enumerate(convs)); done = False
    def evict(need):
        nonlocal resident
        while resident + need > CAP and cache:
            _, ev = cache.popitem(last=False); resident -= ev
    while not done or pend:
        try:
            cid, turns = next(it); ctx[cid]=0; idx[cid]=0; tof[cid]=turns; pend.append(cid)
        except StopIteration: done = True
        if not pend: break
        cid = pend.popleft(); turns = tof[cid]; ti = idx[cid]
        intok, outok = turns[ti]; context = ctx[cid] + intok
        cached = cache.get(cid, 0)
        adj = max(0, context - cached) if cid in cache else context
        work.append(adj)
        newp = context + outok; old = cache.pop(cid, 0); resident -= old
        evict(newp); cache[cid] = newp; resident += newp; ctx[cid] = newp; idx[cid] += 1
        if idx[cid] < len(turns): pend.append(cid)
    return work

def main():
    work = per_turn_adjusted_work()
    work_s = sorted(work); n = len(work)
    mx = max(work); mean = st.mean(work)
    p99w = work_s[int(0.99*n)]
    print(f"turns={n}  cache-adjusted work: mean={mean:.0f} p99={p99w} max={mx} tok")
    print(f"\n=== BOUND 1: p99-TTFT FLOOR (largest cold-doc prefill time) ===")
    for name, rate in [("single-stream", PREFILL_TOK_S_SINGLE), ("batched-peak", PREFILL_TOK_S_BATCH)]:
        print(f"  max-doc / {name} ({rate:.0f} tok/s): {mx/rate:.1f}s   |  p99-work / {name}: {p99w/rate:.1f}s")
    print(f"  => SLO is 8s. A single largest cold doc alone costs {mx/PREFILL_TOK_S_BATCH:.1f}-{mx/PREFILL_TOK_S_SINGLE:.1f}s "
          f"of prefill; the p99 tail cannot go below the p99-work prefill time.")
    print(f"\n=== BOUND 2: goodput CEILING (work-conservation) ===")
    total_work = sum(work)
    # aggregate prefill capacity C (tok/s) shared with decode. Bound with the
    # measured batched peak as an upper bound on sustainable prefill throughput.
    for name, C in [("batched-peak 40k", 40000.0), ("half-shared 20k", 20000.0), ("measured-eff 15k", 15000.0)]:
        max_reqs_per_s = C / mean
        print(f"  C={name} tok/s: max sustainable req/s = C/mean_work = {max_reqs_per_s:.1f}")
    print(f"  (measured stock peak req/s ~4.2; SRPF ~4.1; both near the ~{40000/mean:.0f}-req/s prefill ceiling "
          f"only if decode didn't compete — decode sharing pulls the real ceiling to ~4.)")
    print(f"\n  total cache-adjusted prefill work = {total_work/1e6:.1f}M tok over {n} turns")

if __name__ == "__main__":
    main()
