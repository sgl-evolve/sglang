#!/usr/bin/env python3
"""Analyze a per-prefill trace (SGLANG_WILKES_TRACE .rank0). Dedup by rid (chunked prefills log multiple
entries; first entry per rid = true match at admission). CORRECTED classification using `chash` (conv id =
hash of first 48 origin tokens; a conversation's turns share it):

  COLD-TURN0  : first request seen with this chash  -> genuine cold document, IRREDUCIBLE (never cacheable)
  WARM-HIT    : later turn, mostly cached (uncached < 10% of prompt)
  EVICTED     : later turn, prefix largely evicted (uncached >= 10% of prompt) -> AVOIDABLE recompute
                (this is the class my earlier match-only tool wrongly folded into "COLD")

The p99-TTFT tail's composition across these classes decides the fork: EVICTED-dominated tail => cache can
help (mechanism); COLD-TURN0-dominated tail => cold-document-bound (bounded-negative).

Usage: python3 trace_analyze.py runs/<ver>/trace.rank0"""
import sys, json, numpy as np

def load_dedup(path):
    first = {}
    order = []
    for l in open(path, errors="ignore"):
        l = l.strip()
        if not l:
            continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        rid = r.get("rid")
        if rid not in first:
            first[rid] = r
            order.append(rid)
    return [first[r] for r in order]  # in first-seen (prefill) order

def classify(rows):
    seen = set()
    cls = []
    for r in rows:
        ch = r.get("chash")
        plen = max(1, r["plen"]); unc = r["uncached"]
        if ch is None:
            cls.append("UNK"); continue
        if ch not in seen:
            seen.add(ch); cls.append("COLD-TURN0")
        else:
            cls.append("WARM-HIT" if unc < 0.10 * plen else "EVICTED")
    return cls

def main(path):
    rows = load_dedup(path)
    if not rows:
        print("(no trace rows)"); return
    has_chash = rows[0].get("chash") is not None
    plen = np.array([r["plen"] for r in rows], float)
    unc = np.array([r["uncached"] for r in rows], float)
    n = len(rows)
    print(f"unique requests (rid-dedup): {n}  | chash present: {has_chash}")
    q = lambda a, p: np.percentile(a, p)
    print(f"plen p50={q(plen,50):.0f} p90={q(plen,90):.0f} p99={q(plen,99):.0f} max={plen.max():.0f}")
    print(f"uncached p50={q(unc,50):.0f} p90={q(unc,90):.0f} p99={q(unc,99):.0f} max={unc.max():.0f}")
    print(f"aggregate hit-rate = {1 - unc.sum()/max(1,plen.sum()):.4f}")
    if not has_chash:
        print("!! trace lacks chash -> cannot separate genuine-turn0 from evicted-later-turn. Re-run with"
              " the chash-enabled trace. (old match-only classification is UNSAFE.)")
        return
    cls = np.array(classify(rows))
    tail = unc >= q(unc, 99)  # top ~1% by uncached = p99-tail requests
    print("\nclass         %reqs   %uncached-work   %of-p99-tail")
    for c in ["COLD-TURN0", "WARM-HIT", "EVICTED"]:
        m = cls == c
        pr = 100 * m.mean()
        pw = 100 * unc[m].sum() / max(1, unc.sum())
        pt = 100 * (m & tail).sum() / max(1, tail.sum())
        print(f"  {c:<12} {pr:6.1f}   {pw:12.1f}   {pt:11.1f}")
    ev = cls == "EVICTED"
    print(f"\nAVOIDABLE (EVICTED) = {100*unc[ev].sum()/max(1,unc.sum()):.1f}% of uncached work, "
          f"{100*(ev&tail).sum()/max(1,tail.sum()):.1f}% of the p99 tail")
    print("VERDICT: EVICTED dominates tail => CACHE-AFFECTABLE (mechanism); COLD-TURN0 dominates => cold-bound (negative)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1])
