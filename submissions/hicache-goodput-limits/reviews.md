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
n=6 (same-node λ=3): a ~40% reduction of the typical (median) p99 (Mann-Whitney p≈0.033) whose absolute SLO
crossing is metastable (3/6), with overlapping high tails (not full separation).

## Major weaknesses (ranked by how much a PC would push)

**W1 — the effect is distributional/metastable; is it a real policy effect?**
Attack: whale crosses the SLO on only half its runs and its p99 distribution overlaps LRU's at the top — luck?
Defense: the claim is a shift of the TYPICAL p99, not full separation. (i) whale's median p99 ≈7.7 s vs LRU ≈12.6 s
(~40% lower), 5 of 6 whale runs below LRU's best (10.4 s); Mann-Whitney (n=6 vs n=4) one-sided p≈0.033. (ii) p99
at λ=3 is metastable for BOTH policies (whale once 26.6 s, LRU once 28.3 s) so the top tails overlap — we state
this explicitly and retracted an earlier "fully separated (p≈0.029)" impression that a larger n exposed as an
artifact of not-yet-sampling a whale high draw (a self-correction the added replicates forced). (iii) The reduced
typical p99 sits ON the 8 s boundary → absolute crossing metastable (3/6), LRU/CAR never (0/6): goodput@SLO is a
distributional shift, not a deterministic 0→3.02 win. (iv) We do NOT claim a median-TTFT win: whale's median is
lower-variance (σ≈9 vs LRU σ≈140) but LRU's median is itself highly variable (534–876 ms,
one run below whale), so the median is not a reliable discriminator — a self-correction after lru-r3 (the earlier
"median win" rested on LRU's two high runs). A certified-node confirmation would further tighten the crossing
probability; the p99 reduction and the offline size result are the robust claims and do not depend on it.

**W2 — the p99 improvement is metastable queue-timing, not a real mechanism.**
Attack: if p99 is decoupled from recompute (you say so yourself), how is whale's lower p99 a mechanism and not a
lucky draw?
Defense: the claim is that whale shifts the p99 DISTRIBUTION down (median ~40% lower, Mann-Whitney p≈0.033), not
that any single run is guaranteed lower. p99 is a queue draw, so we characterize it distributionally — and the
metastability itself is a first-class finding (it makes single-run A/Bs on this metric unreliable, which we
demonstrate: a whale run that recomputed MORE than LRU still had a lower p99; and both policies show rare high
draws). The mechanism claim is scoped as "size-ranked victim choice shifts the typical p99," honestly bounded.

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
- Reproducibility §8: add whale run dirs (runs/whale-full, whale-r2..r6) + the simulate.py whale branch commit. (pending)

## Outcome at n=6 whale / n=4 LRU (resolved, two self-corrections)
- The TYPICAL p99 reduction held (median ~40% lower, Mann-Whitney p≈0.033, 5/6 whale below LRU's best) — the
  SLO-relevant, significant result.
- Two self-corrections the added replicates forced: (a) the "median-TTFT win" was retracted after lru-r3 (LRU's
  median is highly variable, one run below whale); (b) the "fully separated p99 (p≈0.029)" was downgraded after
  whale-r6 hit 26.6 s — whale has rare high-metastable draws too, so the p99 distributions overlap at the top.
  The honest claim is a distributional (typical) reduction, not full separation or a per-run guarantee.
- What would still change the verdict: a certified-node whale-vs-LRU pairing failing to reproduce the typical p99
  reduction → GPU effect node-specific; the offline size-signal result (deterministic) would still stand.

## Resolution status (v0.2, updated 2026-07-13)
- Reframed from bounded-impossibility → size-signal after discovering AUC 0.78 + offline whale unique-capture; the
  earlier "liveness unobservable" premise is RETRACTED (self-caught hole).
- W1: n=6 whale / n=4 LRU done; headline is the typical p99 reduction (p≈0.033), overlapping tails stated.
- OPEN: certified confirmation (W6, job queued on contended pool), direct scheduling-mechanism trace (W4).
