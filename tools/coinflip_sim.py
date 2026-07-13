#!/usr/bin/env python3
"""
Minimal transient-queue Monte-Carlo model of the goodput@SLO coin-flip (kleinrock).

PURPOSE: test whether the coin-flip is a GENERIC consequence of open-loop scheduling of heavy-tailed
cold prefills — not an sglang quirk. This is an ILLUSTRATIVE minimal model, NOT the full server: it strips
away the KV cache, decode interleaving, and batching, keeping only the hypothesized mechanism:
  (i)  heavy-tailed cold turn-0 prefill demand  (measured distribution: tools/trace_stats.json doc_toks)
  (ii) a single FCFS server of rate P            (chunked prefill serializes concurrent heavy docs, §2.4)
  (iii) open-loop Poisson arrivals near the knee
  (iv) a cold start (empty queue at t=0)         (the harness flushes per rate, §2.3)
It is a FORWARD prediction: parameterized ONLY by the independently-measured doc-token distribution and the
calibrated prefill rate P — nothing is fit to the observed coin-flip. If the resulting p99-TTFT distribution
across random arrival realizations straddles the SLO with sigma/m > 1 (like the measurement), the coin-flip
is confirmed as a generic queueing phenomenon; a sigma/m(rho) sweep maps WHEN it occurs.
No GPU. Pure Monte-Carlo.
"""
import json, math, os, statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = json.load(open(os.path.join(ROOT, "tools", "trace_stats.json")))
DOCS = [d for d in TS["doc_toks"] if d and d > 0]      # measured turn-0 cold-doc token counts (heavy-tailed)
P = 20500.0                                             # calibrated prefill rate (tok/s), §5.2
SLO = 8.0                                               # s
N_TURN0 = len(DOCS)                                     # 1553 cold docs

# --- deterministic LCG so results are reproducible without Math.random/Date (frozen-eval spirit) ---
class LCG:
    def __init__(self, seed): self.s = (seed * 2654435761 + 1013904223) & 0xFFFFFFFF
    def u(self):
        self.s = (1103515245 * self.s + 12345) & 0x7FFFFFFF
        return self.s / 0x7FFFFFFF
    def exp(self, rate): return -math.log(max(1e-12, 1.0 - self.u())) / rate   # Exp(rate) inter-arrival

def sim_one(lam0, rng):
    """One realization: Poisson(lam0) arrivals of N_TURN0 cold docs, FCFS server rate P, cold start.
    Service time of a doc = tokens/P (chunked serialization => effectively serial on the prefill engine).
    TTFT of a cold doc = completion - arrival (first token emitted only after its prefill finishes).
    Returns p99 of TTFT over the docs (s)."""
    t_arr = 0.0
    free = 0.0                       # server free-at time (starts idle at 0 => cold)
    ttft = []
    # shuffle doc order per realization (arrival order is random) via rng-driven index walk
    order = list(range(N_TURN0))
    for i in range(N_TURN0 - 1, 0, -1):
        j = int(rng.u() * (i + 1))
        order[i], order[j] = order[j], order[i]
    for k in order:
        t_arr += rng.exp(lam0)                       # next Poisson arrival
        svc = DOCS[k] / P
        start = max(t_arr, free)                     # FCFS
        comp = start + svc
        free = comp
        ttft.append(comp - t_arr)                    # TTFT = wait + own prefill
    ttft.sort()
    idx = min(len(ttft) - 1, int(math.ceil(0.99 * len(ttft)) - 1))
    return ttft[idx]

def study(lam0, K=200):
    p99s = [sim_one(lam0, LCG(seed + 1)) for seed in range(K)]
    med = st.median(p99s); sd = st.pstdev(p99s)
    m = abs(SLO - med); npass = sum(1 for x in p99s if x <= SLO)
    return dict(lam0=lam0, med=med, sd=sd, min=min(p99s), max=max(p99s),
                sigma_over_m=(sd / m if m > 1e-6 else float('inf')),
                pass_frac=npass / K, K=K)

if __name__ == "__main__":
    Emean = sum(DOCS) / len(DOCS)
    print(f"# docs={N_TURN0}  E[doc]={Emean:.0f} tok  E[svc]={Emean/P:.3f}s  P={P:.0f} tok/s  SLO={SLO}s")
    # eval turn-0 rate at overall lambda: turn0 fraction = 1553/7163
    frac = N_TURN0 / sum(TS["nturns"])
    print(f"# turn-0 fraction = {frac:.3f}; eval lambda=3 => turn-0 rate ~ {3*frac:.2f}/s")
    print(f"{'lam0':>6} {'rho':>6} {'p99_med':>8} {'p99_sd':>7} {'min':>6} {'max':>7} {'sigma/m':>8} {'pass_frac':>9}  regime")
    for lam0 in [0.30, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00, 1.20]:
        r = study(lam0)
        rho = lam0 * Emean / P
        regime = ("COIN-FLIP" if (r["sigma_over_m"] >= 1 and 0.02 < r["pass_frac"] < 0.98)
                  else ("reliable-pass" if r["pass_frac"] >= 0.98 else "reliable-fail"))
        print(f"{lam0:6.2f} {rho:6.2f} {r['med']:8.2f} {r['sd']:7.2f} {r['min']:6.2f} {r['max']:7.2f} "
              f"{r['sigma_over_m']:8.2f} {r['pass_frac']:9.2f}  {regime}")


# ---------------------------------------------------------------------------
# EXTENSION: how many median-of-k replicates does a RELIABLE goodput@SLO need?
# For a given load, goodput_est(k) = 3 if median(k p99 draws) <= SLO else 0. We ask: what is the smallest
# k for which >=95% of independent median-of-k experiments agree (same goodput)? This quantifies the LIMIT
# of the median-of-k fix and connects it to the sigma/m diagnostic: required-k blows up as the operating
# median approaches the SLO (the coin-flip band), so no practical k rescues the metric there.
# ---------------------------------------------------------------------------
def required_k(lam0, target=0.95, kmax=31, experiments=400, pool_n=1500, base=7000):
    # Draw a pool of p99 realizations ONCE, then bootstrap the median-of-k distribution from it (fast,
    # statistically equivalent to re-simulating each experiment).
    rng = LCG(base + int(lam0 * 1000))
    pool = [sim_one(lam0, rng) for _ in range(pool_n)]
    for k in range(1, kmax + 1, 2):                 # odd k so median is a single draw
        agree3 = 0
        for _ in range(experiments):
            s = sorted(pool[int(rng.u() * pool_n)] for _ in range(k))
            if s[k // 2] <= SLO:
                agree3 += 1
        frac3 = agree3 / experiments
        if max(frac3, 1 - frac3) >= target:
            return k, ("goodput=3" if frac3 >= 0.5 else "goodput=0")
    return None, "unresolved@k<=%d" % kmax

if os.environ.get("KLEINROCK_REQK") == "1":
    print("\n# required median-of-k for 95%-reliable goodput@SLO (model):")
    print(f"{'lam0':>6} {'p99_med':>8} {'(med-SLO)':>9} {'sigma/m':>8} {'req_k(95%)':>11}  verdict")
    for lam0 in [0.30, 0.45, 0.55, 0.60, 0.65, 0.70, 0.80, 1.00]:
        r = study(lam0, K=120)
        rk, verdict = required_k(lam0)
        rkstr = (str(rk) if rk else ">99")
        print(f"{lam0:6.2f} {r['med']:8.2f} {r['med']-SLO:9.2f} {r['sigma_over_m']:8.2f} {rkstr:>11}  {verdict}")
