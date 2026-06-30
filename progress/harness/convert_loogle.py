#!/usr/bin/env python3
"""Reformat the cached LooGLE longdep_qa.jsonl (new HF schema: context/question/answer/doc_id)
into the OLD schema (input/qa_pairs) that bench_serving's sample_loogle_requests expects.

This changes ONLY the data layout, never the benchmark code: same contexts, questions, answers.
Questions are grouped by document (doc_id) so the (unmodified) loader reconstructs LooGLE's
multiturn-over-shared-context workload exactly as it would for old-format LooGLE.

Usage: convert_loogle.py <src longdep_qa.jsonl> <dst oldfmt.jsonl>
"""
import collections, json, sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    groups = collections.OrderedDict()  # deterministic by first appearance (protocol uses --disable-shuffle)
    n = 0
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = r.get("doc_id") or r.get("title") or str(n)
            groups.setdefault(key, []).append(r)
            n += 1
    with open(dst, "w") as f:
        for key, recs in groups.items():
            ctx = recs[0].get("context", "")
            qa = [{"Q": str(r.get("question", "")), "A": str(r.get("answer", ""))} for r in recs]
            rec = {"input": ctx, "qa_pairs": repr(qa), "title": recs[0].get("title", "")}
            f.write(json.dumps(rec) + "\n")
    print(f"converted {n} records -> {len(groups)} multiturn docs -> {dst}")


if __name__ == "__main__":
    main()
