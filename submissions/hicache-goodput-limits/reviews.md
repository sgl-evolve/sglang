# Self-PC-review — "What Bounds Goodput@SLO in a Two-Tier HiCache…" (wilkes, v0.31)

Adversarial read as a skeptical top-venue PC. Each point: the attack, then the paper's defense / action.

## Summary judgment
A bounded-impossibility for *causal* lossless residency under a tail-SLO, concurrent, multi-turn, long-context
regime. The core is **analytic and trace-driven** (§4.2): (a) the real-time live set fits (9.3M = 0.87× cache →
Belady≈0), (b) the turn-0→turn-1 reuse gap is p50 460 s, (c) liveness is unobservable at turn-0 (hit_count=0 for
both continuing convs and single-turn whales) ⇒ a quantified grace-trap. Two empirical horns confirm it (SLRU
backfires; CAR grace-90 ≈ LRU). The strength is that the verdict does not hinge on a single A/B — it follows
from measured workload structure. Accept-leaning IF the empirical gaps below are closed.

## Major weaknesses (ranked by how much a PC would push)

**W1 — n=1 per policy; no error bars.**
Attack: goodput@SLO on this workload has been reported metastable/coin-flip near the SLO; a single run is
unreliable.
Defense: (i) the result is *categorical*, not marginal — p99 ≈ 11 s misses the 8 s SLO by ~40%, far outside the
±30% node-variance band, so it is not a boundary flip (the coin-flip regime was runs *straddling* 8 s; ours are
not close). (ii) All rows are same-node (the dominant confound). (iii) The verdict rests on §4.2 (analytic), not
the A/B. Action: add ≥2 replicates of lru and car at λ=3 for error bars (cheap; pending compute).

**W2 — λ=3 only; no full rate curve.**
Attack: goodput@SLO is defined over λ∈{3,5,7,10}; you only show λ=3.
Defense: λ=3 is the *lowest* rate; p99 already > SLO there and is monotone-nondecreasing in λ, so goodput@SLO=0
across the whole sweep — λ=3 is sufficient for a *zero* goodput claim. Action: run one full lru + car sweep for
the curve figure (pending compute); state the monotonicity argument explicitly (done, §5 intro).

**W3 — the "swamp" horn (large grace hurts) is shown via SLRU proxy + analysis, not a direct large-grace CAR.**
Attack: you measured car at grace 90 (≈lru) but not a large grace; the swamp horn is inferred.
Defense: §31 quantifies it (grace 300 → 1.97× cache forced-resident); SLRU is the limiting case (protect proven
indefinitely) and does hurt. Action: car300 (grace 300) is running (job 19502) to show it directly — the one
open empirical point. If car300 ≈ lru rather than hurting, the claim weakens to "no grace *helps*" (still
supports the impossibility); the paper states this contingency honestly.

**W4 — screening node, not the certified eval-on-pool path.**
Attack: numbers are off the blessed pool.
Defense: identical frozen eval.sh + config on a verified-usable 8×H100 node; the "certified" distinction is node
reliability, not a different measurement. Action: one certified lru + car confirmation (pending; result expected
identical — goodput 0).

**W5 — the grace-trap "forced-resident KV" model is a simplification.**
Attack: it assumes a grace-G policy holds *everything* touched in the last G s; CAR's 3-segment structure holds
proven indefinitely and unproven only for G, so the real resident set differs.
Defense: the model is an upper bound on the unproven-hold pressure and the conclusion is robust to it — the live
set alone is 0.87×, so *any* material dead-KV hold pushes over cap; and car90 (fits) ≈ lru empirically validates
the "fits ⇒ no coverage ⇒ ≈lru" horn. Action: soften wording to "resident pressure model" and note CAR's
segmentation only *reduces* the pressure vs the model, so the swamp threshold is a lower bound on grace.

**W6 — Belady 100% headroom came first from a timing-blind serial replay.**
Attack: the replay ignores concurrency/wall-time, so opt=0 could be an artifact.
Defense: independently confirmed by the *real-time* peak-live-KV sweep (9.3–9.4M = 0.87–0.88× cache), which does
account for wall-time + 902-way concurrency. Both agree: live set fits. Action: lead with the real-time number
(done, §4.2); present the replay as corroboration only.

## Minor
- Title is a question; some venues prefer a claim. Optional: subtitle "…is online-unrecoverable despite full
  offline headroom."
- §6 related work could add a sentence distinguishing from Belady-headroom studies in classic caching (ours
  adds the *unobservability-at-decision-point* obstruction, absent in block caches).
- Reproducibility §8 needs commit hashes + run dirs per number (pending final pass).

## What would change the verdict
If car300 (large grace) *improved* goodput (unexpected) → the grace-trap is wrong and CAR is a mechanism win →
rewrite as a positive. If a continuation *predictor* (turn-0 features) could separate live from dead → the
"unobservable" premise weakens → future work, explicitly scoped. Neither is expected given the 460 s gap and the
turn-0 feature poverty, but both are falsifiable — a strength.
