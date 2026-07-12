# REFRAME DRAFT (assemble into paper.html when n≥4 whale lands; {{PASSRATE}} placeholders filled then)

## Spine (honest, top-venue-shaped)
1. METHODOLOGY: goodput@SLO is METASTABLE — p99 TTFT is decoupled from recompute volume (whale-r2 recomputed
   MORE than lru yet had far lower p99). ⇒ single-run goodput A/Bs unreliable; must compare DISTRIBUTIONS / median.
2. CHARACTERIZATION: two-bound model + chash tail decomposition (63% of ≥40K tail = avoidable evicted recompute)
   + live-set-fits (peak 9.3M = 0.87× cache, Belady≈0) + 37% cold-doc floor.
3. INSIGHT: turn-0 SIZE is an OBSERVABLE liveness proxy (AUC 0.78; whales median 17,051 tok, continuers 241) —
   refutes "liveness unobservable at turn-0". The OBSERVABILITY AXIS matters: SIZE works, TIME-grace (CAR) doesn't.
4. MECHANISM: whale-first (evict biggest-unproven first). Robust median-TTFT reduction (~550 vs lru 678-876);
   p99 shifted below lru/car; crosses 8s SLO on {{PASSRATE}} of runs. Mechanism = displacement/scheduling, not
   recompute reduction (hit/EVICT-work ~flat vs lru). Bounded: AUC<1 (big continuers), 37% cold floor, metastability.
5. NEGATIVES: CAR time-grace ≈ LRU (grace-trap: cache affords ~90s ≪ 460s gap); SLRU backfires (evicts turn-0 entry pts).

---

## NEW TITLE (pick by outcome)
- WIN (whale ≳50% cross): "Size-Aware Liveness Eviction: the Turn-0 Size Signal Moves Goodput@SLO in a Two-Tier HiCache"
- REFINE/metastable: "The Turn-0 Size Signal: an Observable Liveness Proxy for KV Eviction under a Metastable Tail-SLO"

## NEW SUMMARY BOX (draft)
The p99 tail is avoidable recompute (chash: 63% of the ≥40K-tok tail is evicted proven multi-turn convs, 37% cold
docs) and the live set FITS (peak 9.3M = 0.87× cache) → liveness-bound, Belady≈0 → cache-addressable offline.
Liveness is PARTIALLY OBSERVABLE at turn-0 via SIZE: single-turn "whales" have a big turn-0 (median 17,051 tok)
while continuers start tiny (median 241), AUC 0.78. Size-aware eviction (whale-first: evict biggest-unproven
first) exploits this — robustly lowering median TTFT (~550 vs LRU 678-876 ms) and shifting the p99 distribution
below LRU/CAR, crossing the 8 s SLO on {{PASSRATE}} of same-node runs (LRU/CAR: never). BUT goodput@SLO is
METASTABLE: p99 is decoupled from recompute volume (a whale run that recomputed MORE than LRU still had a lower
p99) — it is a queue-timing phenomenon, so the crossing is probabilistic and single-run A/Bs are unreliable. The
TIME axis of continuation-observability is trapped (CAR completion-grace ≈ LRU: cache affords ~90 s ≪ the 460 s
reuse gap) and the naive SLRU backfires (evicts turn-0 entry points). = the turn-0 size signal is the observable
liveness lever; goodput@SLO characterization under metastability.

## NEW §4.2 (retitle + correct core)
Title: "Liveness is PARTIALLY observable at the decision point — via SIZE (not time)"
- KEEP: live-set-fits (9.3M=0.87×), reuse gap p50 460s, the grace-trap TABLE (holds for the TIME axis).
- CORRECT: retract "liveness is unobservable at turn-0". Turn-0 SIZE predicts single-turn-vs-continue at AUC 0.78
  (whales median 17,051 tok / continuers 241; 70× separation). This is an OBSERVABLE causal proxy for Belady's
  evict-dead-first, available at the decision point.
- FRAME as TWO AXES: (a) TIME (hold turn-0 for grace-G) — TRAPPED (finite cache clips grace>90s ≪ 460s gap →
  CAR≈LRU). (b) SIZE (choose WHICH unproven to evict) — ESCAPES the trap: evict big dead whales, spare small
  continuers, no extra retention time needed. The barrier is not "unobservable" but "the RIGHT observable is SIZE,
  not recency/time".
- Imperfection: AUC 0.78 not 1.0 (big continuers exist, e.g. conv0 31,998-tok/11-turn) — but continuers' turn-1
  is small+immediate (closed-loop) so real continuers get promoted before eviction reaches them; whale-first is
  SOFT so it drops safest victims first. Residual: cold-doc floor (37%) + metastability.

## NEW §5.5 "Size-aware eviction (whale)" (draft — fill n≥4 table)
- TTFT table (§43): median/mean/std/p99 for whale (n=4) vs lru (n=3) vs car90 (n=2), same node λ=3.
- Goodput crossing: lru/car {{fail rows}}; whale crosses on {{PASSRATE}}.
- Mechanism (§46/§47): NOT recompute reduction — EVICT-work ~identical (whale 0.183P vs lru 0.184P), hit ~flat;
  whale-r2 recomputed MORE than lru yet lower p99 ⇒ displacement/queue-timing. whale cut big(≥20K) prefills
  560→504 (~10%). The robust signal is median (SNR ≫ p99).
- Metastability caveat: goodput crossing is probabilistic; report the p99 DISTRIBUTION not a point.

## §7 limitations additions
- whale ceiling: AUC 0.78 imperfect; 37% cold-doc floor; goodput crossing metastable (pass-rate, not deterministic).
- median-TTFT win is robust/low-variance; p99/goodput win is metastable — distinguish clearly.

## §9 conclusion
- The observable liveness lever is turn-0 SIZE (not recency/time). Size-aware eviction robustly improves the body
  of the TTFT distribution and nudges the metastable p99 across the SLO. Effort should exploit observable turn-0
  features for liveness prediction; and goodput@SLO under this regime must be measured as a distribution.
