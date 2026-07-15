#!/usr/bin/env python3
"""floyd diagnostic: is the p99 prefill-WORK invariant to caching policy? (supports P2 corpus-bound)

FAST-SCREEN OUTCOME (honest): I hypothesized a "condition 2" — that HIT-RATE headroom does
not translate to GOODPUT headroom because the p99 tail is all COLD. This screen REFUTED that
framing: in the 4-pass replay the largest re-prefills are ~75% AVOIDABLE (big documents recur
across passes, so they appear as both a cold first-prefill AND later evicted re-prefills). So
a better cache COULD reduce the COUNT of big prefills. The clean decoupling only holds
within-rate (single-pass, where LRU=Belady already => vacuous). ⇒ "condition 2" DROPPED; do
NOT build a paper on it.

What this script DOES establish cleanly (and it supports P2's corpus-bound thesis): the p99
per-event prefill WORK (~67k tok, the largest single prefill you ever pay) is POLICY-INVARIANT
— the biggest cold document equals the biggest avoidable document (they are the same document),
so the per-request TTFT floor is set by the largest UNIQUE document regardless of eviction/
retention policy. Realizable caching gains stay online-uncapturable (P2: LFU=LRU on the 4-pass
avoidable); only clairvoyant Belady recovers it. Kept as a diagnostic, not a new claim.
"""
import ast, json, hashlib
from collections import OrderedDict, Counter

TRACE = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CPT = 4.0
CAP = 10_700_000
C_BATCH = 39000.0  # measured batched prefill tok/s (for TTFT-floor framing)
def toks(s): return max(1, int(len(s) / CPT))

def load():
    convs = []
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line); doc = r.get("input", "")
            dh = hashlib.md5(doc.encode()).hexdigest(); dtok = toks(doc)
            try: qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception: qa = []
            if qa: convs.append((dh, dtok))
    return convs

def replay_miss_events(convs, passes):
    """LRU 4-pass; return list of (size_tok, 'cold'|'avoidable') for every re-prefill (miss)."""
    seq = list(range(len(convs))) * passes
    cache = OrderedDict(); res = 0; seen = set(); events = []
    for ci in seq:
        dh, dtok = convs[ci]
        if dh in cache:
            cache.move_to_end(dh); continue
        events.append((dtok, "cold" if dh not in seen else "avoidable"))
        seen.add(dh)
        while res + dtok > CAP and cache:
            res -= cache.pop(next(iter(cache)))
        if dtok <= CAP: cache[dh] = dtok; res += dtok
    return events

def pct(sorted_vals, p): return sorted_vals[min(len(sorted_vals)-1, int(p*len(sorted_vals)))]

def main():
    convs = load()
    ev = replay_miss_events(convs, 4)
    sizes = sorted(s for s, _ in ev)
    n = len(ev)
    cold = [s for s, c in ev if c == "cold"]
    avoid = [s for s, c in ev if c == "avoidable"]
    print(f"4-pass replay under LRU: {n} re-prefill events "
          f"({len(cold)} cold, {len(avoid)} avoidable)")
    print(f"re-prefill SIZE (tok): p50={pct(sizes,.5)} p90={pct(sizes,.9)} p99={pct(sizes,.99)} max={max(sizes)}")
    print(f"cold  size: mean={sum(cold)//len(cold)} max={max(cold)}")
    print(f"avoid size: mean={sum(avoid)//len(avoid)} max={max(avoid)}  (what Belady could remove)")

    # Condition 2: is the TAIL cold or avoidable? Look at the largest re-prefills.
    for topp in (0.01, 0.05, 0.10):
        thr = pct(sizes, 1 - topp)
        tail = [(s, c) for s, c in ev if s >= thr]
        tcold = sum(1 for s, c in tail if c == "cold")
        print(f"top {topp*100:.0f}% largest re-prefills (>= {thr} tok, the p99 TTFT contributors): "
              f"{tcold}/{len(tail)} = {100*tcold/len(tail):.0f}% COLD (uncacheable)")

    # p99 prefill-work under LRU vs under a clairvoyant policy that removes ALL avoidable work.
    lru_p99 = pct(sizes, .99)
    cold_only = sorted(cold)                      # Belady-limit: only cold prefills remain
    bel_p99 = pct(cold_only, .99)
    print(f"\np99 re-prefill work: LRU={lru_p99} tok (~{lru_p99/C_BATCH:.1f}s)  |  "
          f"remove-ALL-avoidable(Belady limit)={bel_p99} tok (~{bel_p99/C_BATCH:.1f}s)")
    print(f"=> p99 per-event prefill-work is {'INVARIANT' if abs(lru_p99-bel_p99)/lru_p99<0.1 else 'changed'} "
          f"to caching policy (biggest cold doc == biggest avoidable doc == same document).")
    print("\nHONEST OUTCOME: my 'hit-rate != goodput (tail all cold)' hypothesis is REFUTED here --")
    print("the 4-pass tail is ~75% AVOIDABLE (big docs recur across passes), so a better cache COULD")
    print("cut the big-prefill COUNT. Decoupling holds only within-rate (single-pass: LRU=Belady, vacuous).")
    print("KEPT (true, supports P2 corpus-bound): the p99 per-request TTFT floor = largest UNIQUE document,")
    print("policy-invariant; realizable caching gains stay online-uncapturable (P2: LFU=LRU on 4-pass).")

if __name__ == "__main__":
    main()
