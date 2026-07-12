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
    print("\noverall class    %reqs   %uncached-work")
    for c in ["COLD-TURN0", "WARM-HIT", "EVICTED"]:
        m = cls == c
        print(f"  {c:<12} {100*m.mean():6.1f}   {100*unc[m].sum()/max(1,unc.sum()):12.1f}")
    # SLO-breaching lens: a request breaches the SLO iff its uncached prefill exceeds ~8s*P_eff tokens.
    # cold docs are simply the biggest prefills; the goodput question is the CLASS MIX among big-enough
    # (SLO-threatening) prefills. Report composition above several uncached thresholds.
    print("\nclass mix among requests with uncached >= T (the SLO-breaching-size candidates):")
    print(f"{'T(tok)':>8} {'#reqs':>6} {'%COLD-TURN0':>12} {'%EVICTED(avoid)':>16} {'%WARM-HIT':>10}")
    for T in [20000, 40000, 60000, 100000]:
        big = unc >= T
        nb = int(big.sum())
        if nb == 0:
            print(f"{T:>8} {nb:>6}   (none)"); continue
        pc = 100 * (cls[big] == "COLD-TURN0").mean()
        pe = 100 * (cls[big] == "EVICTED").mean()
        pw = 100 * (cls[big] == "WARM-HIT").mean()
        print(f"{T:>8} {nb:>6} {pc:>12.1f} {pe:>16.1f} {pw:>10.1f}")
    ev = cls == "EVICTED"
    print(f"\nAVOIDABLE (EVICTED) overall = {100*unc[ev].sum()/max(1,unc.sum()):.1f}% of uncached work")
    print("VERDICT: if EVICTED is a large share of big/SLO-breaching prefills => CACHE-AFFECTABLE (a residency")
    print("mechanism protecting proven convs cuts them); if big prefills are ~all COLD-TURN0 => cold-doc-bound.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(2)
    main(sys.argv[1])
