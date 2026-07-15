#!/usr/bin/env python3
"""floyd: is "scheduling is the goodput lever" robust to the SLO threshold, or an artifact of 8s?

goodput@SLO(T) = max sustained rate with p99 TTFT <= T. We sweep T and compute goodput for stock
(FCFS) and SRPF from the measured same-node p99 curves, to find the WINDOW of SLO thresholds where
scheduling is decisive (SRPF > stock). Below the window both fail (compute/prefill floor: even the
best schedule can't beat a request's own prefill time); above it stock catches up (both
compute-bound at the ~4.2 ceiling). If 8s sits inside a wide window, the conclusion is robust.

Measured p99 TTFT (ms), same-node where available:
  stock: λ3 7612 (this control; a passing coin-flip draw — λ3 spans {6327..11786}), λ5 17372,
         λ7 33961, λ10 41334  (λ5/7/10 from runs/v0-stock; λ5 stock is stable-fail)
  srpf : λ3 5892, λ5 5850     (runs/v-srpf-r1; λ7/10 not measured — arrival>capacity ~4.2 => overload
         => p99 grows unboundedly => treated as FAIL, conservative for SRPF)
"""
RATES = [3, 5, 7, 10]
# (p99 TTFT ms, achieved req/s) per rate. goodput@SLO = max ACHIEVED req/s over rates that pass.
STOCK = {3: (7612, 3.02), 5: (17372, 3.66), 7: (33961, 4.00), 10: (41334, 4.22)}
SRPF  = {3: (5892, 3.02), 5: (5850, 4.16), 7: (10**9, 4.2), 10: (10**9, 4.2)}  # λ7/10 overload => fail
STOCK_L3_RANGE = (6327, 11786)                      # λ3 coin-flip (n=5)

def goodput(curve, T):
    g = 0.0
    for lam in RATES:
        p99, ach = curve[lam]
        if p99 <= T:
            g = max(g, ach)   # max ACHIEVED req/s among rates meeting the SLO
    return g

def main():
    print(f"{'SLO T(s)':>8} {'stock req/s':>12} {'SRPF req/s':>11} {'gain':>7} {'regime':>26}")
    windows = []
    for T_s in [2, 4, 5, 6, 7, 8, 10, 12, 15, 17, 18, 25, 34, 41, 45]:
        T = T_s * 1000
        gs = goodput(STOCK, T); gp = goodput(SRPF, T)
        gain = f"+{(gp/gs-1)*100:.0f}%" if gs > 0 else ("+inf" if gp > 0 else "—")
        if gp > gs + 1e-9: verdict = "SCHEDULING decisive"; windows.append(T_s)
        elif gp == gs == 0: verdict = "both fail (prefill floor)"
        else: verdict = "both ~compute-ceiling"
        print(f"{T_s:>8} {gs:>12.2f} {gp:>11.2f} {gain:>7} {verdict:>26}")
    print(f"\nScheduling is the lever (SRPF req/s > stock) for SLO in ~[{min(windows)}, {max(windows)}]s.")
    print(f"The fixed 8s SLO sits INSIDE this window => the 'scheduling is the lever' conclusion is NOT an")
    print(f"artifact of the threshold. Below ~6s even SRPF's λ3 (5.9s) fails (a request's own prefill floor,")
    print(f"policy-invariant); above ~{max(windows)}s stock's slow rates also clear the SLO and both are")
    print(f"compute-bound. Note stock λ3 is a coin-flip {STOCK_L3_RANGE[0]/1000:.1f}-{STOCK_L3_RANGE[1]/1000:.1f}s,")
    print(f"so stock's low-T goodput is optimistic here (a passing draw); SRPF's λ3 5.9s is stable.")

if __name__ == "__main__":
    main()
