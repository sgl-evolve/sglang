# Self-PC-review — "The Turn-0 Size Signal: Observable Liveness for KV Eviction under a Metastable Tail-SLO" (wilkes, v0.31)

Adversarial read as a skeptical top-venue PC, for DRAFT v0.2 (reframed from the earlier bounded-impossibility once
the size signal was found). Each point: the attack, then the paper's defense / action.

## Summary judgment
The paper's spine is: (a) the p99 tail is avoidable recompute over a live set that fits (chash decomposition +
peak-live-KV); (b) liveness IS partially observable at the eviction decision point via turn-0 SIZE (AUC 0.78),
and in an offline replay size-aware eviction is the UNIQUE causal policy that captures Belady headroom (−23%;
recency/grace capture 0%) — refuting a natural "unobservable" impossibility; (c) online, the 460 s reuse gap caps
the realized recompute capture ≈0 (grace-trap generalizes to all residency), leaving a SCHEDULING benefit (median
TTFT −25%, p99 distribution shifted below LRU/CAR); (d) goodput@SLO is METASTABLE (p99 decoupled from recompute
volume), so the SLO crossing is probabilistic and must be read distributionally. Contributions that are robust
regardless of the whale replicate outcome: the size-observability analysis + offline victim-choice test, the
chash tail decomposition, the two-bound model, and the metastability characterization. The whale GPU result is
n=4 (same-node λ=3); the median win is the low-variance part, the goodput crossing the metastable part (2/4).

## Major weaknesses (ranked by how much a PC would push)

**W1 — the p99/goodput win is metastable (2/4 cross); is it a real policy effect?**
Attack: whale crosses the SLO on only half its runs — could be luck, not a mechanism.
Defense: (i) all FOUR whale runs lie below BOTH LRU and BOTH CAR runs on median, mean, AND p99 — full separation
(rank-sum p≈0.067); (ii) the MEDIAN win (~555 vs 678–876 ms) has far higher SNR than p99, is tight across all four
runs, and is the primary claim; (iii) we explicitly do NOT claim a deterministic goodput 0→3.02 win — only a robust median shift +
a probabilistic crossing. Status: n=4 complete — the median held tight (550–568 ms across all four) and all four
p99 stayed below the LRU/CAR band (6.2–9.3 s vs 10.4–11.3 s), confirming the distributional shift; the crossing
landed at 2/4 (the metric sits on the SLO boundary). Remaining: n→3 LRU (in flight) and a certified-node
confirmation would tighten the crossing probability, but the median win and the offline size result do not depend
on the crossing count.

**W2 — the p99 improvement is metastable queue-timing, not a real mechanism.**
Attack: if p99 is decoupled from recompute (you say so yourself), how is whale's lower p99 a mechanism and not a
lucky draw?
Defense: this is exactly why we separate the two axes. The median/mean/std reduction (whole-body shift) IS the
mechanism claim and is low-variance. The p99 is a queue draw, but whale shifts the DISTRIBUTION it draws from
(both runs below LRU). We frame goodput@SLO as distributional, not a point — and the metastability itself is a
first-class finding (it makes single-run A/Bs on this metric unreliable, which we demonstrate: whale-r2 recomputed
MORE than LRU yet had a lower p99).

**W3 — the offline −23% recompute does not materialize on GPU; is the size signal actually useless?**
Attack: your headline offline number evaporates online.
Defense: no — the offline replay proves size is the UNIQUE causal signal that CAN capture headroom (LRU/SLRU/CAR
capture 0%), i.e. it establishes the correct residency signal in principle. We are explicit that online the 460 s
reuse gap caps the realized recompute capture (§4.2 synthesis, §5.5): durable recompute wins require closing the
gap, not just a better victim signal. This gap-cap is an honest bound, and the surviving online benefit
(scheduling) is separately evidenced. The offline/online split is a feature (it localizes exactly what blocks the
win), not hand-waving.

**W4 — the "scheduling" mechanism is not fully isolated (and one plausible story was refuted).**
Attack: you attribute the median win to timing but can't say how.
Defense (updated, honest): we establish what it is NOT — p99/median is decoupled from hit rate (a whale run with
lower hit had lower p99) so it is not a residency-hit effect; and it is NOT "shorter queues" — a trace of the
worst whale run shows the waiting-queue depth is unchanged vs LRU (mean 3.7 vs 3.6). What differs is the
running-batch composition at prefill admission (whale p50 21 vs LRU p50 246) and the count of large prefills
(−10%), but we do not claim a definite microscopic pathway; the paper (§4.2/§5.5/§9) now states this explicitly
and scopes the exact timing cause as future work. We deliberately retracted the earlier "reduces queueing" wording
once the wq trace refuted it — the empirical claim (whale robustly lowers median TTFT, losslessly, not via hit
rate) stands independent of the unresolved mechanism.

**W5 — AUC 0.78 is imperfect; size-ranked eviction will wrongly drop big continuers.**
Attack: 22% of the ranking is wrong; you will evict live big-turn-0 conversations.
Defense: whale-first is SOFT (evict LARGEST unproven first, LRU tiebreak) so it drops the safest victims first and
reaches a big continuer only under extreme pressure; and a true continuer's turn-1 is small + near-immediate in
closed-loop, so it is promoted to the protected segment before a size-ranked eviction reaches it. §4.2/§7 state
the imperfection and that a different gap/continuation structure could weaken the signal.

**W6 — screening node, not the certified eval pool.**
Attack: numbers are off the blessed pool.
Defense: identical frozen eval.sh + config on a verified-usable 8×H100 node; "certified" is a node-reliability tag,
not a different measurement; all comparisons are same-node (the dominant confound). Action: a certified-node
confirmation of the whale vs LRU λ=3 comparison is queued.

## Minor
- §5.3 could add a one-line bridge to §5.5 ("the grace-trap motivates changing the SIGNAL, not the retention time
  — §5.5"). (pending)
- §6 related work: add a sentence distinguishing the size-observability result from continuation-predictor /
  TTL-aware works (Continuum, Predictive-Multi-Tier) — we identify a SPECIFIC observable (turn-0 size) and show
  offline it is uniquely sufficient among causal victim rules. (pending)
- Reproducibility §8: add whale run dirs (runs/whale-full, whale-r2..r4) + the simulate.py whale branch commit. (pending)

## What would change the verdict
- If whale replicates (n=4) keep the median low and p99 below the LRU band → the mechanism claim strengthens to a
  robust median win + reliable-enough crossing (positive mechanism paper).
- If whale regresses to LRU at n=4 → the GPU win was n=2 noise; retract the crossing claim, keep (i) the size
  signal + offline unique-capture (the "right signal in principle" result), (ii) the metastability finding, and
  (iii) the gap-cap synthesis — a characterization paper rather than a mechanism paper. Both are falsifiable now.

## Resolution status (v0.2 reframe, 2026-07-12)
- Reframed from bounded-impossibility → size-signal after discovering AUC 0.78 + offline whale unique-capture; the
  earlier "liveness unobservable" premise is RETRACTED (self-caught hole).
- Old W-list (single-run/full-curve/swamp/certified for the impossibility) folded in: full LRU/CAR curves +
  car300 (no swamp) + replicates remain in the record (§5.1/§5.3); they now support the recency/grace-negative,
  not an impossibility.
- OPEN: whale n=4 replicates (W1), certified confirmation (W6), direct scheduling-mechanism trace (W4).
