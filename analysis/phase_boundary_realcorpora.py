#!/usr/bin/env python3
"""floyd (P3/P2 §6 generality): the caching phase boundary across MULTIPLE REAL long-document corpora.

Reviewer objection: §6's phase boundary (LRU captures a reuse iff stack-distance < H = CAP/mean-doc) is
validated on constructed corpora (a definitional/Mattson check) plus ONE real workload — so "long-document
QA sits in the mirage regime" rests on n=1 real corpus. Here we compute the criterion on FOUR real corpora
(the eval mix and its constituents at their native reuse structure): for each, the reuse stack-distance
distribution vs its horizon H, and the online caching headroom = frac(reuses with stack-distance > H).
Offline trace replay; no GPU. If all real long-doc QA corpora land at ~0 headroom, the mirage generalizes
across the class (not just the eval workload).
"""
import ast, json, hashlib, statistics as st, os

CPT = 4.0; CAP = 10_700_000
CORPORA = {
    "mooncake_mix (eval)": "/rmeng_data/junyanch-data/datasets/mooncake_mix_v1.jsonl",
    "leval":               "/rmeng_data/junyanch-data/datasets/leval_v1.jsonl",
    "loogle_longdep_qa":   "/rmeng_data/junyanch-data/datasets/loogle_longdep_qa_v1.jsonl",
    "loogle_full":         "/rmeng_data/junyanch-data/datasets/loogle_full_v1.jsonl",
}
def toks(s): return max(1, int(len(s)/CPT))

def analyze(path):
    seq, sizes = [], {}
    with open(path, errors="ignore") as f:
        for line in f:
            try: r = json.loads(line)
            except Exception: continue
            doc = r.get("input", "")
            if not doc: continue
            dh = hashlib.md5(doc.encode()).hexdigest()
            try: qa = ast.literal_eval(r["qa_pairs"]) if r.get("qa_pairs") else []
            except Exception: qa = []
            if not qa: continue
            seq.append(dh); sizes[dh] = toks(doc)
    n = len(seq)
    if n == 0 or not sizes: return None
    mean_doc = st.mean(sizes.values()); H = CAP / mean_doc
    last, dists = {}, []
    for i, dh in enumerate(seq):
        if dh in last: dists.append(len(set(seq[last[dh]+1:i])))
        last[dh] = i
    reuses = len(dists)
    if reuses == 0:
        return dict(n=n, uniq=len(sizes), mean_doc=mean_doc, H=H, reuses=0,
                    p50=0, p90=0, mx=0, headroom=0.0, note="no cross-record reuse (all singletons)")
    ds = sorted(dists)
    over = sum(1 for d in dists if d > H)
    return dict(n=n, uniq=len(sizes), mean_doc=mean_doc, H=H, reuses=reuses,
                p50=ds[reuses//2], p90=ds[int(.9*reuses)], mx=max(dists),
                headroom=over/reuses, note="")

def main():
    print(f"{'corpus':>22} {'records':>8} {'uniq':>6} {'mean_doc':>9} {'H(docs)':>8} "
          f"{'reuses':>7} {'sd_p90':>7} {'sd_max':>7} {'headroom':>9} {'regime':>8}")
    for name, path in CORPORA.items():
        if not os.path.exists(path): print(f"{name:>22}  (missing)"); continue
        r = analyze(path)
        if r is None: print(f"{name:>22}  (empty)"); continue
        regime = "MIRAGE" if r["headroom"] < 0.05 else "caching"
        print(f"{name:>22} {r['n']:>8} {r['uniq']:>6} {r['mean_doc']:>8.0f}t {r['H']:>8.0f} "
              f"{r['reuses']:>7} {r['p90']:>7} {r['mx']:>7} {r['headroom']*100:>8.1f}% {regime:>8}"
              + (f"  [{r['note']}]" if r['note'] else ""))
    print("\n=> If every real long-doc QA corpus lands at ~0 headroom (reuse stack-distance << H, or no reuse),")
    print("   the document-reuse mirage generalizes across the CLASS, not just the eval workload — the")
    print("   criterion predicts it from corpus statistics alone (no GPU, no policy tuning).")

if __name__ == "__main__":
    main()
