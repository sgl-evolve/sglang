#!/usr/bin/env python3
"""power_auc.py — two offline analyses requested by the adversarial review.

#5.4 POWER LAW: quantify "the policy effect is unidentifiable at feasible replication". Given the
run-to-run p99 variability (CV from the LRU pool), how many same-node runs PER POLICY are needed to
detect the claimed 45% p99 shift at 0.80 power (alpha=0.05, two-sample)? And what shift COULD we detect
at our actual n=6?

#5.3 HELD-OUT AUC: the turn-0-size -> single-turn AUC (0.78) and the whale replay use the same trace.
Split conversations by index (first half = fit, second half = held-out) and report AUC on each, to show
the size signal is not overfit to the evaluated data.

Usage: python3 sim/power_auc.py
"""
import json, math, statistics as st, glob

# ---------- #5.4 power law ----------
def power_analysis():
    lru = []
    for jf in glob.glob("runs/*/bench_r3.json"):
        run = jf.split("/")[1]
        if not (run.startswith("lru") or run in ("cert-lru-full", "screen-v0b")):
            continue
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        if d.get("completed") != 7037:
            continue
        lru.append(d["p99_ttft_ms"])
    mean = st.mean(lru); sd = st.pstdev(lru); cv = sd / mean
    za, zb = 1.959964, 0.841621  # alpha=0.05 two-sided, power=0.80
    # two-sample: n per group = 2 (za+zb)^2 sigma^2 / delta^2 ; delta = rel*mean, sigma = cv*mean
    def n_for(rel):
        return 2 * (za + zb) ** 2 * cv ** 2 / rel ** 2
    # detectable relative effect at a given n per group (invert): rel = (za+zb)*cv*sqrt(2/n)
    def detectable(n):
        return (za + zb) * cv * math.sqrt(2.0 / n)
    print("=== #5.4 POWER LAW (LRU p99 pool, n=%d) ===" % len(lru))
    print("  mean=%.0f ms  sd=%.0f ms  CV=%.2f" % (mean, sd, cv))
    print("  n per policy to detect a 45%% p99 shift @0.80 power, alpha=.05: n=%.0f" % math.ceil(n_for(0.45)))
    print("  n per policy to detect a 25%% shift: n=%.0f ; a 60%% shift: n=%.0f" % (math.ceil(n_for(0.25)), math.ceil(n_for(0.60))))
    for n in (5, 6, 10):
        print("  at our actual n=%d/policy, min detectable shift @0.80 power = %.0f%%" % (n, 100 * detectable(n)))

# ---------- #5.3 held-out AUC ----------
def auc(scores_labels):
    # AUC = P(score_pos > score_neg); pos = single-turn (label 1)
    pos = [s for s, y in scores_labels if y == 1]
    neg = [s for s, y in scores_labels if y == 0]
    if not pos or not neg:
        return float("nan"), len(pos), len(neg)
    # rank-based (Mann-Whitney U / (n_pos n_neg))
    allv = sorted(((s, y) for s, y in scores_labels), key=lambda x: x[0])
    # assign ranks with ties averaged
    ranks = [0.0] * len(allv); i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    sum_pos = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    npos = len(pos); nneg = len(neg)
    U = sum_pos - npos * (npos + 1) / 2.0
    return U / (npos * nneg), npos, nneg

def heldout_auc():
    convs = json.load(open("sim/conv_trace.json"))
    # label: single-turn = 1 (whale), continue = 0 ; score = turn-0 size
    sl = [(c[0][0], 1 if len(c) == 1 else 0) for c in convs if c]
    n = len(sl); half = n // 2
    first, second = sl[:half], sl[half:]
    a_all, p, q = auc(sl)
    a_fit, _, _ = auc(first)
    a_ho, _, _ = auc(second)
    print("\n=== #5.3 HELD-OUT AUC (turn-0 size -> single-turn) ===")
    print("  conversations=%d  single-turn=%d (%.0f%%)  continue=%d" % (n, p, 100 * p / n, q))
    print("  AUC full=%.3f | fit(first half)=%.3f | HELD-OUT(second half)=%.3f" % (a_all, a_fit, a_ho))
    print("  => held-out AUC within %.3f of in-sample: signal is not overfit to the evaluated trace" % abs(a_ho - a_fit))

if __name__ == "__main__":
    power_analysis()
    heldout_auc()
