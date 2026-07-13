#!/usr/bin/env python3
"""
Analyze the SESSION-VARIANCE experiment (kleinrock): is the goodput@SLO regime a launch-level (session)
property, controlling the cross-allocation node-state confound? Reads runs/v0-sessvar/sessvar.csv (N server
sessions x K stock lambda=3 draws, all in ONE exclusive allocation on one node).

Decomposes total p99 variance into WITHIN-session (run-to-run, server fixed) and ACROSS-session (launch-to-
launch) components, and reports an intraclass-correlation-like ratio. If across-session >> within-session,
launch-level metastability is real (draws cluster by session); if across ~ within, the earlier 2-session
difference (§5.5) was cross-allocation node-state drift, not session-level -> soften the caveat.
Deterministic (no RNG). Also prints per-session medians and pass fractions.
"""
import os, sys, csv, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLO = 8000.0
CSV = os.path.join(ROOT, "runs", "v0-sessvar", "sessvar.csv")

def load():
    sess = {}
    if not os.path.exists(CSV):
        return sess
    for r in csv.DictReader(open(CSV)):
        v = r.get("ttft_p99_ms")
        if not v:
            continue
        try:
            s = int(r["session"]); p = float(v)
        except (ValueError, KeyError):
            continue
        sess.setdefault(s, []).append(p)
    return sess

if __name__ == "__main__":
    sess = load()
    if not sess:
        print("NO DATA yet at %s" % CSV); sys.exit(0)
    print(f"# SESSION-VARIANCE analysis (one allocation, one node; N={len(sess)} sessions)")
    all_draws = []
    means = []
    within_vars = []
    for s in sorted(sess):
        xs = sess[s]; all_draws += xs
        med = st.median(xs); mean = st.mean(xs); npass = sum(1 for x in xs if x <= SLO)
        means.append(mean)
        if len(xs) > 1:
            within_vars.append(st.variance(xs))
        print(f"  session {s}: n={len(xs)}  p99(ms)={[round(x) for x in xs]}  "
              f"median={med:.0f}  mean={mean:.0f}  pass={npass}/{len(xs)}")
    grand = st.mean(all_draws)
    tot_sd = st.pstdev(all_draws)
    # variance components (one-way, small-n descriptive)
    within = st.mean(within_vars) if within_vars else 0.0            # mean within-session variance
    K = st.mean([len(v) for v in sess.values()])
    across_of_means = st.pvariance(means) if len(means) > 1 else 0.0 # variance of session means
    # unbiased-ish between-session variance component: var(means)*K - within  (clamped >=0)
    between_comp = max(0.0, across_of_means * K - within / max(1, K) * 0 + across_of_means * 0)  # keep simple
    print(f"# grand mean={grand:.0f}ms  total pstdev={tot_sd:.0f}ms")
    print(f"# within-session variance (mean over sessions) = {within:.3e}  (sd={within**0.5:.0f}ms)")
    print(f"# across-session variance (of session means)   = {across_of_means:.3e}  (sd={across_of_means**0.5:.0f}ms)")
    if within > 0:
        ratio = across_of_means / within
        print(f"# ratio across/within = {ratio:.2f}  "
              f"(>>1 => launch-level metastability real; ~<=1 => dominated by run-to-run, soften §5.5 caveat)")
    # pooled: does the whole-allocation pool still coin-flip?
    npass = sum(1 for x in all_draws if x <= SLO)
    print(f"# pooled (all sessions): n={len(all_draws)}  median={st.median(all_draws):.0f}ms  pass={npass}/{len(all_draws)}")
