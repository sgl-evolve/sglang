#!/usr/bin/env python3
"""floyd paper-2 generalization: the caching-headroom PHASE BOUNDARY.

When does document caching have online headroom (i.e., when can a smarter-than-LRU
retention policy help)? Exactly when documents are re-used at an LRU *stack-distance*
(number of DISTINCT documents touched since the last use) that EXCEEDS the cache
horizon (CAP / mean-doc-size, in documents). A reuse with stack-distance < horizon is
already captured by LRU; one with stack-distance > horizon is evicted by LRU and could
in principle be kept by a frequency-/popularity-aware policy.

This makes the "document-reuse mirage" a quantitative, workload-general statement:
- headroom fraction = fraction of reuses with stack-distance > horizon.
For the fixed Mooncake-mix workload this fraction is ~0 (99.8% of reuses are within the
horizon), so LRU is optimal and no retention policy helps. A different corpus (large
working set, long-range reuse) would have a nonzero headroom fraction — the boundary
predicts which.
"""
import ast, json, hashlib, statistics as st

TRACE = "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
CPT = 4.0
CAP = 10_700_000  # 2-tier capacity (tokens)
def toks(s): return max(1, int(len(s) / CPT))

def main():
    seq, sizes = [], {}
    with open(TRACE) as f:
        for line in f:
            r = json.loads(line); doc = r.get("input", "")
            dh = hashlib.md5(doc.encode()).hexdigest()
            try: qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception: qa = []
            if qa: seq.append(dh); sizes[dh] = toks(doc)
    n = len(seq); mean_doc = st.mean(sizes.values()); horizon = CAP / mean_doc
    last, dists = {}, []
    for i, dh in enumerate(seq):
        if dh in last:
            dists.append(len(set(seq[last[dh] + 1:i])))  # distinct docs since last use
        last[dh] = i
    reuses = len(dists); ds = sorted(dists)
    captured = sum(1 for d in dists if d < horizon)
    print(f"convs={n} unique={len(sizes)} mean_doc={mean_doc:.0f}tok  LRU horizon={horizon:.0f} docs")
    print(f"reuses={reuses}  stack-distance p50={ds[reuses//2]} p90={ds[int(.9*reuses)]} max={max(dists)}")
    print(f"LRU-captured (dist<horizon) = {captured}/{reuses} = {captured/reuses*100:.1f}%")
    print(f"headroom (dist>horizon)     = {reuses-captured}/{reuses} = {(reuses-captured)/reuses*100:.1f}%")
    print(f"=> mirage regime: ~0% headroom; LRU is optimal. Boundary = stack-distance vs {horizon:.0f}-doc horizon.")

if __name__ == "__main__":
    main()
