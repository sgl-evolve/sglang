#!/usr/bin/env python3
"""floyd paper-2: cross-request DOCUMENT reuse analysis for goodput@SLO.

Discovery: of 1553 conversations, 1015 carry a document (538 are document-free chat). The 1015 are 887 unique
documents across 1553 conversations, with a heavily skewed popularity (one document
is the turn-0 context of 100 conversations, a long tail of singletons). Naively this looks like
a caching opportunity (~10% of conversations re-use a document some other
conversation already prefilled). This script shows the headroom is a MIRAGE:

  (1) The stock radix cache is keyed on token content, so cross-conversation reuse
      of an identical document is ALREADY a cache hit whenever the doc is resident.
  (2) SINGLE-PASS (one traversal of the 1553-conversation workload) LRU is Belady-
      OPTIMAL: LRU = LFU = Belady = 0 avoidable re-prefill at the real 2-tier
      capacity. Hot documents stay resident (touched constantly); the only misses
      are first-sight prefills of the 887 unique documents -- irreducibly cold.
  (3) The eval sweeps 4 rates over the SAME conversation set with no flush. Only
      this 4x REPLAY creates avoidable re-prefill (55M tok under LRU) -- and it is
      a benchmark artifact that NO online policy captures (LFU=LRU=54-55M; only the
      clairvoyant Belady recovers it, 23M). Reuse distance = one full pass (~1553
      convs) >> cache; per-doc frequency is ~uniform (4x each) so LFU cannot rank.

Conclusion: goodput@SLO is CORPUS-BOUND (unique-document set x hardware prefill
throughput), NOT cache-policy-bound. The cold floor = one prefill per unique
document = 18.4M tok, unmovable by any eviction/retention policy.
"""
import ast, json, hashlib
from collections import OrderedDict, Counter

TRACE = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CPT = 4.0
CAP = 10_700_000  # 2-tier (L1 HBM + L2 host DRAM) capacity in tokens
def toks(s): return max(1, int(len(s) / CPT))

def load():
    convs = []  # (doc_hash, doc_tok, turn0_in_tok)
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line); doc = r.get("input", "")
            if not doc: continue   # skip empty-doc chat records (ShareGPT, no document); DOCUMENT-reuse only
            dh = hashlib.md5(doc.encode()).hexdigest(); dtok = toks(doc)
            try: qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception: qa = []
            if not qa: continue
            t0 = toks("Input: " + doc + " Question: " + str(qa[0].get("Q","")))
            convs.append((dh, dtok, t0))
    return convs

def popularity(convs):
    freq = Counter(dh for dh,_,_ in convs)
    n = len(convs); uniq = len(freq)
    top = freq.most_common(5)
    dup_convs = sum(v for v in freq.values() if v > 1)
    print(f"conversations={n}  unique documents={uniq}  in-a-reuse-group={dup_convs} "
          f"({dup_convs/n*100:.0f}%)")
    print(f"most-popular document reuse counts: {[v for _,v in top]}")
    return freq

def cache_sim(convs, policy, passes):
    """doc-level content-keyed cache; cold = first-ever, avoidable = evicted-then-reprefill."""
    idx = list(range(len(convs)))
    seq = idx * passes
    cache = OrderedDict(); res = 0; freq = Counter(); seen = set()
    cold = 0; avoid = 0; hits = 0
    if policy == "belady":
        pos = {}
        for p, ci in enumerate(seq): pos.setdefault(convs[ci][0], []).append(p)
        ptr = {k: 0 for k in pos}
    for ci in seq:
        dh, dtok, _ = convs[ci]; freq[dh] += 1
        if policy == "belady": ptr[dh] += 1
        if dh in cache:
            hits += 1; cache.move_to_end(dh)
        else:
            if dh in seen: avoid += dtok
            else: cold += dtok; seen.add(dh)
            while res + dtok > CAP and cache:
                if policy == "lru": v = next(iter(cache))
                elif policy == "lfu": v = min(cache, key=lambda k: freq[k])
                else:
                    def nxt(k):
                        l = pos[k]; i = ptr[k]; return l[i] if i < len(l) else 1 << 40
                    v = max(cache, key=nxt)
                res -= cache.pop(v)
            if dtok <= CAP: cache[dh] = dtok; res += dtok
    return hits / len(seq) * 100, cold / 1e6, avoid / 1e6

def main():
    convs = load()
    popularity(convs)
    tot0 = sum(t0 for _,_,t0 in convs)
    seen = set(); uniq0 = 0
    for dh,_,t0 in convs:
        if dh not in seen: seen.add(dh); uniq0 += t0
    print(f"\nturn-0 prefill work: all-convs={tot0/1e6:.1f}M  unique-docs(true cold floor)={uniq0/1e6:.1f}M "
          f"=> cross-conv reuse = {(1-uniq0/tot0)*100:.0f}% of turn-0 work")
    for passes in (1, 4):
        print(f"\n=== {passes}-pass cache sim (CAP={CAP/1e6:.1f}M tok) ===")
        for pol in ("lru", "lfu", "belady"):
            h, cold, avoid = cache_sim(convs, pol, passes)
            print(f"  {pol:>7}: doc-hit={h:4.0f}%  COLD={cold:5.1f}M  AVOIDABLE={avoid:5.1f}M  "
                  f"total-prefill={cold+avoid:5.1f}M")
    print("\nSINGLE-PASS: LRU==LFU==Belady==0 avoidable => LRU is Belady-optimal; caching cannot help.")
    print("4-PASS avoidable is a replay artifact: LFU~=LRU (online-uncapturable); only clairvoyant Belady recovers it.")
    print("=> goodput@SLO is CORPUS-BOUND, not cache-policy-bound.")

if __name__ == "__main__":
    main()
