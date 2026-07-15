#!/usr/bin/env python3
"""floyd: quantify the RPB reserve-WASTE mechanism from server.log prefill batch sizes.

RPB caps the in-flight big doc at (1-r)*chunk_budget = 4608 tok (r=0.25) and reserves 1536 for the
shortest waiting turn(s). If a fitting small turn uses the reserve, the combined prefill batch is ~6144;
if the shortest waiting turn is bigger than the reserve (common near the knee), the guard refuses it and
the reserve is WASTED -> the batch is ~4608 -> effective prefill capacity drops. This script measures the
big-doc-chunk prefill batch #new-token distribution to expose that waste, split by rate period.

Result (v-rpb25 vs v-srpf-ctl4, node 1-1):
  baseline (no RPB): big-chunk steps mean ~6111 tok, 97% >=6000 (full chunks).
  RPB r=0.25:        big-chunk steps mean ~5320 tok, ~50% capped <=4700 with ~0 small added
                     => ~54% of the 1536-tok reserve WASTED on capped steps.
  => RPB reduces effective prefill capacity ~13% on big-doc steps. At lambda=3 (slack) this is absorbed
     (overall tput unchanged) and small-turn interleave helps (-10% p99); at lambda=5 (near knee) the
     capacity loss crosses saturation -> tput -8.6% -> queue -> p99 +41% (SLO FAIL). This is why fixed
     chunk-level reservation is a NET NEGATIVE on goodput@SLO.
"""
import re, sys, statistics as st

def secs(l):
    m = re.search(r"\[20\d\d-\d\d-\d\d (\d\d):(\d\d):(\d\d)", l)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else None

def big_chunks(path, t0, t1, thr=4000):
    xs = []
    for l in open(path, errors="ignore"):
        s = secs(l)
        if s is None or s < t0 or s >= t1:
            continue
        m = re.search(r"Prefill batch.*#new-token: (\d+)", l)
        if m and int(m.group(1)) >= thr:
            xs.append(int(m.group(1)))
    return xs

def rep(name, xs):
    if not xs:
        print(f"{name}: none"); return
    print(f"{name}: n={len(xs)} mean={st.mean(xs):.0f} "
          f"frac_full(>=6000)={sum(x>=6000 for x in xs)/len(xs):.2f} "
          f"frac_capped(<=4700)={sum(x<=4700 for x in xs)/len(xs):.2f}")

if __name__ == "__main__":
    # periods (UTC seconds-of-day); pass overrides as argv if reused for other runs
    print("=== RPB reserve-waste: big-doc-chunk prefill batch #new-token by rate ===")
    rep("rpb25 λ3", big_chunks("runs/v-rpb25/server.log", 8*3600+51*60, 9*3600+30*60+41))
    rep("rpb25 λ5", big_chunks("runs/v-rpb25/server.log", 9*3600+30*60+41, 11*3600))
    rep("base  λ3", big_chunks("runs/v-srpf-ctl4/server.log", 7*3600+26*60, 8*3600+6*60+1))
    rep("base  λ5", big_chunks("runs/v-srpf-ctl4/server.log", 8*3600+6*60+1, 9*3600+30*60))
    print("reserve ~54% wasted on capped steps => -13% capacity on big-doc steps => fatal at the near-knee λ5.")
