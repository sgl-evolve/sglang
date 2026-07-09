#!/usr/bin/env python3
"""Workload-axis sensitivity analysis for exclusive tiering.

Extends the capacity-band generalization (generalization_band.py) with two new axes:
1. Document-length composition: how exclusive benefit varies with workload type
2. Concurrency: how benefit scales with load (phase transition at cache-pressure onset)

Key findings:
- Benefit is ZERO when active working set < host capacity (no pressure)
- Benefit ONSET occurs when inclusive host capacity is first exceeded (λ=18 in sim)
- Exclusive DELAYS the pressure onset by ~22% higher load (λ=18→22)
- Under sustained pressure, benefit GROWS with load (+12pp at λ=30 → +19pp at λ=75)
- Short-doc-only and mid-doc-only workloads see ZERO benefit (WS fits in host)
- Long-doc workloads see modest benefit (+3pp); mixed workloads see maximal benefit (+12pp)
  because cross-document eviction pressure amplifies the capacity-band effect.
"""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from cache_sim import load_workload, simulate

HOST = 7_810_000
DEVICE = 2_350_000
LAM_CALIBRATED = 30.0


def build_convs(raw_list):
    convs = []
    for turns in raw_list:
        P, meta, cum = [], [], 0
        for t in turns:
            P.append(cum)
            meta.append((t["new_prompt_tok"], t["output_tok"]))
            cum += t["new_prompt_tok"] + t["output_tok"]
        convs.append((P, meta))
    return convs


def main():
    d = json.load(open(os.path.join(os.path.dirname(__file__), "workload_tokens.json")))
    all_raw = d["conversations"]
    doc_lens = [c[0]["new_prompt_tok"] for c in all_raw]

    short_idx = [i for i, l in enumerate(doc_lens) if l < 4096]
    mid_idx = [i for i, l in enumerate(doc_lens) if 4096 <= l < 16384]
    long_idx = [i for i, l in enumerate(doc_lens) if l >= 16384]

    print("=" * 75)
    print("AXIS 1: Document-length composition (at calibrated λ=30)")
    print("=" * 75)
    segments = {
        "ALL (mixed 1:1:1)": list(range(len(all_raw))),
        "Short-only (<4K)": short_idx,
        "Mid-only (4K-16K)": mid_idx,
        "Long-only (>16K)": long_idx,
        "Short+Mid (<16K)": short_idx + mid_idx,
    }
    print(
        f"{'Workload':<22s} {'N':>5s} {'WS(M)':>7s} {'hit(H)':>8s} {'hit(H+D)':>9s} "
        f"{'Δpp':>7s} {'WS/C':>6s}"
    )
    print("-" * 70)
    for name, idx_list in segments.items():
        raw = [all_raw[i] for i in idx_list]
        convs = build_convs(raw)
        ws = sum(sum(t["new_prompt_tok"] + t["output_tok"] for t in c) for c in raw)
        incl = simulate(convs, policy="fcfs", C=HOST, lam=LAM_CALIBRATED)["hit_rate"]
        excl = simulate(convs, policy="fcfs", C=HOST + DEVICE, lam=LAM_CALIBRATED)[
            "hit_rate"
        ]
        delta = (excl - incl) * 100
        print(
            f"{name:<22s} {len(convs):>5d} {ws / 1e6:>7.1f} {incl:>8.4f} {excl:>9.4f} "
            f"{delta:>+7.2f} {ws / (HOST + DEVICE):>6.2f}"
        )

    print()
    print("=" * 75)
    print("AXIS 2: Concurrency scaling (full mixed workload)")
    print("=" * 75)
    convs = build_convs(all_raw)
    print(f"{'λ_sim':>6s} {'hit(H)':>8s} {'hit(H+D)':>9s} {'Δpp':>7s} {'regime':>25s}")
    print("-" * 60)
    for lam in [5, 10, 15, 17, 18, 19, 20, 22, 25, 30, 40, 50, 75]:
        incl = simulate(convs, policy="fcfs", C=HOST, lam=lam)["hit_rate"]
        excl = simulate(convs, policy="fcfs", C=HOST + DEVICE, lam=lam)["hit_rate"]
        delta = (excl - incl) * 100
        if delta < 0.1:
            regime = "no pressure"
        elif excl > 0.80:
            regime = "onset (excl at ceiling)"
        elif lam == 30:
            regime = "measured operating point"
        else:
            regime = "both capacity-bound"
        print(f"{lam:>6d} {incl:>8.4f} {excl:>9.4f} {delta:>+7.2f} {regime:>25s}")

    print()
    print("=" * 75)
    print("SYNTHESIS")
    print("=" * 75)
    print(
        "Exclusive tiering's benefit is the reuse-mass CDF slope over the reclaimed"
    )
    print("band [H, H+D]. This slope is zero when:")
    print("  (a) WS << H (no capacity pressure, any composition), or")
    print("  (b) low concurrency (active WS < H even if total WS >> H)")
    print("The benefit is maximal when the band straddles the CDF's steep knee,")
    print("which occurs at the ONSET of cache pressure under mixed workloads with")
    print("long-document conversations (the dominant capacity consumers).")


if __name__ == "__main__":
    main()
