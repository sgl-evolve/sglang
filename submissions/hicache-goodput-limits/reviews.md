# Self-PC-review — "The Turn-0 Size Signal: Observable Liveness for KV Eviction under a Metastable Tail-SLO" (wilkes, v0.31)

Adversarial read as a skeptical top-venue PC, for DRAFT v0.2 (reframed from the earlier bounded-impossibility once
the size signal was found). Each point: the attack, then the paper's defense / action.

## Summary judgment
The paper is now an INSIGHT + CAUTION paper (the mechanism claim was retracted under cross-node testing). Spine:
(a) the p99 tail is avoidable recompute over a live set that fits (chash + peak-live-KV); (b) CONSTRUCTIVE,
DETERMINISTIC: liveness is partially observable at the eviction decision point via turn-0 SIZE (AUC 0.78), and in
an offline replay size-among-unproven eviction with proven-protection is the UNIQUE causal victim rule capturing
Belady headroom (−23%; a 7-policy sweep: recency/frequency/grace all 0%, naive size-only −529%); (c) CAUTIONARY,
CROSS-NODE (3 nodes): online, the metric defeats causal residency claims — a same-node whale-vs-LRU p99 reduction
significant on ONE node (nodeset-0: median −45%, p≈0.026, n=6/n=5) has NO STABLE SIGN across nodes: whale median
p99 is lower on 2 of 3 nodes (nodeset-0, 1-2) but REVERSES on the third (0-3: LRU median 6.1 s vs whale 11.5 s),
with per-policy p99 swinging 6–31 s by node/run; and the 460 s reuse gap caps online recompute capture ≈0 regardless. Robust
contributions: the size-observability + offline victim-choice test (deterministic), the chash decomposition, the
two-bound model, and the cross-node metastability result (single-node A/Bs unreliable for tail-SLO residency). The
GPU whale p99-reduction is RETRACTED as a portable win → reported as node-specific.

## Major weaknesses (ranked by how much a PC would push)

**W1 — is the online whale p99-reduction a real, portable policy effect?**
Attack: whale's p99 overlaps LRU's, the crossing is metastable, and it's one node — is this a real win?
Defense (fully honest, retraction included): NO, we do not claim a portable win. On nodeset-0 the same-node
whale-vs-LRU comparison was significant (median p99 −45%, Mann-Whitney n=6/n=5, p≈0.026); across 3 nodes total,
whale's median p99 is lower on 2 of 3 (nodeset-0, 1-2) but the sign REVERSES on the third (0-3: LRU median 6.1 s
vs whale 11.5 s; U=3/9), with per-policy p99 swinging 6–31 s by node/run. We therefore report the online advantage
as weak and node-dependent (no stable sign), NOT a portable win. The metric is metastable to the point that a
same-node replicated A/B does not have a stable sign across nodes — itself the paper's central cautionary result
(single-node A/Bs unreliable; multi-node median-of-k required for tail-SLO residency). The robust claims that
remain are the DETERMINISTIC offline size-signal (unique causal capture) and this cross-node metastability finding. (Chain of self-corrections the replication forced: median-win → full-separation → node-portability, all
retracted as more data arrived.)

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

**W6 — does it hold on the certified eval pool? (resolved — and it changed the verdict.)**
Attack: numbers are off a screening node; confirm on the certified pool.
Defense/outcome: we ran median-of-k on TWO certified nodes (0-3: n=3 each; 1-2: n=2 each), identical frozen config.
The nodeset-0 whale advantage does not hold with a stable sign: whale median p99 is lower on 1-2 (7.7 vs 15 s) —
like nodeset-0 — but REVERSES on 0-3 (whale 11.5 vs LRU 6.1 s, U=3/9), with per-policy p99 swinging 6–31 s across
nodes. This drove the retraction (W1): the online advantage is weak/node-dependent, not portable. Rather than a
weakness we hid, the cross-node test is now load-bearing for the paper's central result — goodput@SLO is so
metastable that a same-node replicated A/B (p≈0.026) has no stable sign across 3 nodes. A larger multi-node
median-of-k would further quantify any residual effect; certified-pool contention limited us to n=2–3/node beyond
nodeset-0, but 3 nodes already establish sign-instability.

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
