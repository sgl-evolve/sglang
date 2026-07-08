#!/usr/bin/env python3
"""Workload-axis generalization of exclusive tiering (CPU-only; no GPU).

The HW-ratio table in report.md varies the DEVICE tier at a FIXED workload. This adds the complementary,
more falsifiable axis: for a FIXED HW (host H, device D), the exclusive-over-inclusive benefit equals the
marginal hit-rate gained by the reclaimed capacity band [H, H+D] — i.e. the SLOPE of the reuse-mass CDF
(hit-rate vs capacity) integrated over that band. So the crisp, falsifiable boundary is:

    benefit(H, D) = HitRate(C = H+D) - HitRate(C = H)

which is LARGE iff the band [H, H+D] straddles the STEEP region of the hit-vs-capacity curve (the working
set is not yet resident), and -> 0 once H is past the plateau (working set already fits on host alone).

We (1) trace the hit-vs-capacity curve HitRate(C) for THIS workload, then (2) slide the host tier H and
report the marginal benefit of bolting on the fixed device band D=2.35M at each H. This explains the
HW-ratio table (benefit grows with D/(H+D)) AND generalizes to any workload (benefit = CDF slope over band).
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from cache_sim import load_workload, simulate

DEVICE = 2_350_000      # this HW's L1 (GPU) tier, tokens
HOST   = 7_810_000      # this HW's L2 (host) tier, tokens
# CALIBRATION: the sim's lambda is an EFFECTIVE-CONCURRENCY knob, not the real arrival rate. Real prefill of
# a ~10k-tok doc on the 122B model is far slower than this sim's service model, so the protocol's real lam=3
# induces the concurrency this sim reaches near lam~30. We pin lam so the sim reproduces the MEASURED
# hit-rates (host-only 0.60 vs measured baseline 0.62; host+device 0.72 vs measured exclusive 0.75), then
# read the band structure off the calibrated curve. Below lam~20 the sim is not capacity-bound (flat at the
# 0.806 ceiling) and cannot speak to capacity effects.
LAM    = 30.0           # calibrated effective-concurrency (reproduces measured baseline+exclusive hit)

def hit_at(convs, C):
    return simulate(convs, policy="fcfs", C=int(C), lam=LAM)["hit_rate"]

def main():
    convs, meta = load_workload()
    print(f"workload: {meta['n_conversations']} convs, {meta['n_turns_total']} turns; lam={LAM}")
    print(f"HW: host H={HOST:,}  device D={DEVICE:,}  (D/(H+D)={DEVICE/(HOST+DEVICE):.0%})\n")

    # (1) hit-vs-capacity curve (the reuse-mass CDF)
    caps = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 7.81e6, 9e6, 10.16e6, 12e6, 15e6, 20e6, 40e6]
    print("=== hit-vs-capacity curve (reuse-mass CDF) ===")
    print(f"{'C (tok)':>12}{'hit_rate':>10}")
    curve = {}
    for C in caps:
        h = hit_at(convs, C); curve[C] = h
        print(f"{C:>12,.0f}{h:>10.4f}")

    # (2) slide host H; benefit of bolting on fixed device band D at each H
    print(f"\n=== marginal benefit of the reclaimed device band D={DEVICE:,} vs host H ===")
    print(f"{'host H':>12}{'H+D':>12}{'hit(H)':>9}{'hit(H+D)':>10}{'benefit_pp':>12}{'D/(H+D)':>9}")
    hosts = [2e6, 4e6, 6e6, 7.81e6, 10e6, 14e6, 20e6, 38e6]
    for H in hosts:
        hH  = hit_at(convs, H)
        hHD = hit_at(convs, H + DEVICE)
        ben = (hHD - hH) * 100
        star = "  <-- this HW" if abs(H - HOST) < 1e5 else ""
        print(f"{H:>12,.0f}{H+DEVICE:>12,.0f}{hH:>9.4f}{hHD:>10.4f}{ben:>12.2f}{DEVICE/(H+DEVICE):>9.0%}{star}")

    print("\n=== BOUNDARY (falsifiable): benefit -> 0 once host alone covers the working set ===")
    print("Benefit is the CDF slope over [H, H+D]: maximal on the steep region, ~0 on the plateau.")

if __name__ == "__main__":
    main()
