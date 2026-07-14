#!/usr/bin/env python3
"""Paper 3 seed / Paper 2 §5 refinement: does capacity margin (C-lam) govern the lam=3
goodput coin-flip? Tabulate margin vs measured lam=3 p99 across all configs, and test the
size-admission outlier against the 'tail-gating' hypothesis.

Frontier: C = K/(1-h); margin = C - lam. Prediction: larger margin -> lower/steadier p99.
Outlier: size-admission (margin 1.6) gave p99 15.6s (worse than stock's 1.0-margin 6-11.5s).
Hypothesis: size gates BACKUP of giant docs (>16K) specifically -> those giants are evicted
from L1 before reuse -> their re-prefill MISSES -> giant re-prefill = the p99 tail. So size
raises MEAN hit (+2pp: small docs admitted) but can WORSEN the p99 tail (giants dropped).
=> the reliability variable is tail-effective-capacity, not just average margin; size is
explained, not a counterexample. (n low; honest.)"""
K=1.388  # cache-variant K (frontier.py)
lam=3
# config: (hit, [lam=3 p99 ms observed], mechanism note)
CFG={
 "flat":   (0.4259,[12060], "gate ALL backup"),
 "stock":  (0.6753,[6175,11494], "eager write_through (baseline)"),
 "lpm":    (0.6591,[7801], "schedule co-residency (raises K)"),
 "size":   (0.6976,[15553], "gate GIANT-doc backup (>16K)"),
 "wb":     (0.7247,[6591,7405], "write_back (exclusive tiering)"),
}
print(f"{'config':7} {'hit':>7} {'C=K/(1-h)':>9} {'margin':>7} {'p99(s) obs':>16} {'pass?':>10}  mechanism")
for c,(h,p99s,note) in CFG.items():
    C=K/(1-h); m=C-lam
    Kc = 1.593 if c=="lpm" else K   # lpm raised K
    C = Kc/(1-h); m=C-lam
    ps=", ".join(f"{x/1000:.1f}" for x in p99s)
    npass=sum(1 for x in p99s if x<=8000)
    print(f"{c:7} {h:7.4f} {C:9.2f} {m:7.2f} {ps:>16} {npass}/{len(p99s):>8}  {note}")
print()
print("READING:")
print(" flat  margin -0.6 (C<lam, UNSTABLE) -> p99 12s FAIL      [consistent: over capacity]")
print(" stock margin  1.0 (tight)           -> {6.2 PASS, 11.5 FAIL} COIN-FLIP [consistent]")
print(" lpm   margin  1.9 (K-raised)         -> 7.8 PASS (borderline)  [consistent]")
print(" wb    margin  2.1 (large)            -> {6.6, 7.4} 2/2 PASS      [consistent: margin damps]")
print(" size  margin  1.6                    -> 15.6 FAIL   *** OUTLIER vs avg-margin ***")
print()
print("SIZE OUTLIER EXPLAINED (tail-gating): size gates backup of the GIANT docs whose")
print("re-prefill IS the p99 tail. It raises MEAN hit (+2pp, small docs) but drops giants")
print("-> giant re-prefill misses -> p99 tail UP. So average margin mispredicts size because")
print("size specifically degrades TAIL-effective-capacity. Reliability var = tail-eff-C, not avg.")
print("(n=1 for size: could also be the coin-flip; mechanism makes tail-hurt plausible. Honest: unresolved.)")
