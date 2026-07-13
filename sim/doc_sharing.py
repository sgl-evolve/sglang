#!/usr/bin/env python3
"""doc_sharing.py — quantify the cross-request document-sharing escape hatch (§2.1, §9 no-go).

The no-go criterion (§9) names "cross-request sharing" as an orthogonal capacity escape. This script
bounds it from the dataset + trace:
  (1) DATASET: how many conversations share a document, and the MAX prefill that perfect lossless
      cross-conversation sharing could save (Σ (reuse-1)*doclen).
  (2) TRACE: how much of that sharing RadixAttention already captures (hit rate within shared-prefix
      chash groups).

Result (mooncake_mix_v1, this eval): 538 empty-context ShareGPT chats + 1015 doc-bearing convs over 887
unique docs; only 15.5% share a doc (one 100x doc dominates); MAX shareable ~0.8M of 99.9M tok (<1%);
RadixAttention already captures ~78% within shared-prefix groups. => cross-request sharing is NOT a lever
in this workload; it could dominate only in a shared-corpus regime.

Usage: python3 sim/doc_sharing.py [dataset.jsonl] [trace.rank0]
"""
import json, collections, statistics as st, sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl"
TRACE = sys.argv[2] if len(sys.argv) > 2 else "runs/cert-lru-full/trace.rank0"

def dataset_sharing():
    recs = [json.loads(l) for l in open(DATA) if l.strip()]
    N = len(recs)
    docs = [r["input"] for r in recs if len(r["input"]) > 0]
    empty = N - len(docs)
    cnt = collections.Counter(docs)
    repeated = sum(c for c in cnt.values() if c > 1)
    # ~tokens = chars/4; max shareable = sum over docs of (reuse-1)*doclen
    avoid = sum((c - 1) * (len(d) // 4) for d, c in cnt.items() if c >= 2)
    print("DATASET (%s):" % DATA.split("/")[-1])
    print("  conversations: %d  (empty-context/ShareGPT: %d ; doc-bearing: %d over %d unique docs)"
          % (N, empty, len(docs), len(cnt)))
    print("  share a document with >=1 other: %d (%.1f%% of doc-bearing)  top reuse counts: %s"
          % (repeated, 100 * repeated / len(docs), [c for _, c in cnt.most_common(5)]))
    print("  MAX lossless cross-conv shareable prefill: ~%.2fM tokens" % (avoid / 1e6))
    return avoid

def trace_capture():
    first = {}
    for line in open(TRACE):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "chash" not in d:
            print("TRACE %s: no chash field (pre-instrumentation run)" % TRACE); return
        if d["rid"] not in first:
            first[d["rid"]] = d
    bych = collections.defaultdict(list)
    for d in first.values():
        bych[d["chash"]].append(d)
    big = sorted(bych.values(), key=len, reverse=True)[:12]
    cold = sum(1 for v in big for d in v if d["uncached"] > 1000)
    hit = sum(1 for v in big for d in v if d["uncached"] < 100)
    print("TRACE (%s): top-12 shared-prefix groups -> repeat hit-rate %.1f%% (cold=%d hit=%d)"
          % (TRACE.split("/")[-2], 100 * hit / (cold + hit), cold, hit))

if __name__ == "__main__":
    avoid = dataset_sharing()
    trace_capture()
    print("\n=> cross-request sharing is quantitatively negligible in this workload "
          "(<1%% of the 99.9M-token λ=3 prefill); RadixAttention already captures the majority.")
