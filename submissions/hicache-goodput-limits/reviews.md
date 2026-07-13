# Self-PC-review — "The Metastable Tail-SLO: A No-Go Criterion for Lossless KV-Cache Optimization" (wilkes, v0.31; paper v0.8)

Adversarial read as a skeptical top-venue PC, for DRAFT v0.3 (reframed twice from the earlier bounded-impossibility once
the size signal was found). Each point: the attack, then the paper's defense / action.

## Summary judgment
The paper is now an INSIGHT + CAUTION + BOUNDED-NEGATIVE paper (the online mechanism claim was retracted under
cross-node testing; v0.4 adds execution-driven metastability + a deterministic refutation of the admission axis, so
BOTH scheduling axes are shown dead-end). v0.5 crystallizes the three established pillars into a single quotable
NO-GO CRITERION (§9, boxed): no lossless residency/admission mechanism yields a certifiable goodput@SLO gain when
(1) reuse is gap-dominated (G/R≈5), (2) prefill has no co-execution contention, (3) the tail is
execution-variance-dominated — remaining lossless levers (capacity so R>G; prefill bound P) are orthogonal to
eviction/admission. Carefully scoped: "certifiable" (condition 3 is about identifiability, not proven non-existence)
and it names the escape levers, so it is a bounded no-go, not an absolute impossibility. Spine:
(a) the p99 tail is avoidable recompute over a live set that fits (chash + peak-live-KV); (b) CONSTRUCTIVE,
DETERMINISTIC: liveness is partially observable at the eviction decision point via turn-0 SIZE (AUC 0.78), and in
an offline replay PROVEN-PROTECTION + a prefix size/cost magnitude signal captures Belady headroom where
recency/frequency capture 0% (11-policy sweep: LRU/SLRU/CAR/LFU/GDSF 0%; proven-protected cost-greedy +48%, size
"whale" +23%; cost-only/size-only w/o protection −311%/−529%). [v0.6 11th self-correction: an earlier draft claimed
size was the UNIQUE such rule; adding recompute-cost eviction on adversarial review showed it captures MORE — see W11.]; (c) CAUTIONARY,
CROSS-NODE (3 nodes): online, the metric defeats causal residency claims — a same-node whale-vs-LRU p99 reduction
significant on ONE node (nodeset-0: median −45%, p≈0.026, n=6/n=5) has NO STABLE SIGN across nodes: whale median
p99 is lower on 2 of 3 nodes (nodeset-0, 1-2) but REVERSES on the third (0-3: LRU median 6.1 s vs whale 11.5 s),
with per-policy p99 swinging 6–31 s by node/run; and the 460 s reuse gap caps online recompute capture ≈0 regardless. Robust
contributions: the size-observability + offline victim-choice test (deterministic), the chash decomposition, the
two-bound model, and the cross-node metastability result (single-node A/Bs unreliable for tail-SLO residency). The
GPU whale p99-reduction is RETRACTED as a portable win → reported as weak/node-dependent (no stable sign across 3 nodes).

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
Defense: we do NOT claim it as a mechanism win. On nodeset-0 whale shifted the p99 distribution down (median
−45%, p≈0.026), but p99 is a queue draw decoupled from recompute (a whale run that recomputed MORE than LRU still
had a lower p99), and across nodes the effect has no stable sign (§5.6). The metastability itself is the
first-class finding — it makes single-run and single-node A/Bs on this metric unreliable — which is exactly why we
report the online effect as weak/node-dependent, not a mechanism win.

**W3 — the offline −23% recompute does not materialize on GPU; is the size signal actually useless?**
Attack: your headline offline number evaporates online.
Defense: no — the offline replay proves proven-protection + a size/cost magnitude signal CAN capture headroom that
recency/frequency cannot (LRU/SLRU/CAR/LFU/GDSF capture 0%), i.e. it establishes the correct residency signal-class
in principle (cost-greedy +48% > size/whale +23%; NOT unique — see W11). We are explicit that online the 460 s
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
once the wq trace refuted it. Moot in the end: since the online p99 effect is itself weak/node-dependent (§5.6),
we make no portable mechanism claim — the robust results are the deterministic offline size-signal and the
metastability caution, neither of which depends on isolating an online scheduling pathway.

**W5 — AUC 0.78 is imperfect; size-ranked eviction will wrongly drop big continuers.**
Attack: 22% of the ranking is wrong; you will evict live big-turn-0 conversations.
Defense: whale-first is SOFT (evict LARGEST unproven first, LRU tiebreak) so it drops the safest victims first and
reaches a big continuer only under extreme pressure; and a true continuer's turn-1 is small + near-immediate in
closed-loop, so it is promoted to the protected segment before a size-ranked eviction reaches it. §4.2/§7 state
the imperfection and that a different gap/continuation structure could weaken the signal.

**W6 — does it hold on the certified eval pool? (resolved — and it changed the verdict.)**
Attack: numbers are off a screening node; confirm on the certified pool.
Defense/outcome: we ran median-of-k on TWO certified nodes (0-3: n=3 each; 1-2: whale n=2, LRU n=3), identical frozen config.
The nodeset-0 whale advantage does not hold with a stable sign: whale median p99 is lower on 1-2 (7.7 vs 21.9 s, LRU n=3) —
like nodeset-0 — but REVERSES on 0-3 (whale 11.5 vs LRU 6.1 s, U=3/9), with per-policy p99 swinging 6–31 s across
nodes. This drove the retraction (W1): the online advantage is weak/node-dependent, not portable. Rather than a
weakness we hid, the cross-node test is now load-bearing for the paper's central result — goodput@SLO is so
metastable that a same-node replicated A/B (p≈0.026) has no stable sign across 3 nodes. A larger multi-node
median-of-k would further quantify any residual effect; certified-pool contention limited us to n=2–3/node beyond
nodeset-0, but 3 nodes already establish sign-instability.

**W7 — is the metastability a real system property or an eval-harness/node-health artifact?**
Attack: nodeset-0 was flagged flaky; maybe the huge p99 swings are throttling / NFS / neighbor noise, not a real
queue property — which would undercut the central caution.
Defense: it is a genuine queue-TAIL phenomenon on a BYTE-IDENTICAL workload. The benchmark fixes the RNG seed
(seed=1) and disables shuffle, so content, order, and nominal arrival schedule are identical every run — verified:
all 22 full-throughput λ=3 runs (every policy, 3 nodes) complete exactly 7037 requests with total input tokens
agreeing to within 0.016%. On this fixed workload p99 spans 5.9–31.2 s (5.3×) while the MEDIAN TTFT is pinned to
528–574 ms (1.1×) and throughput to 3.022–3.024 req/s. Identical input, constant body, constant throughput, 5.3×
tail. A SUSTAINED slowdown (thermal throttle, prolonged I/O contention, a persistently slow node) would inflate the
median and throughput too; that only the tail moves rules THOSE out and points to a few large cold-prefills catching
a good-vs-bad queue moment. HONEST CAVEAT (v0.6, from adversarial review): the body-stable/tail-only signature does
NOT rule out TRANSIENT per-prefill stalls (a brief NFS-metadata stall or kernel-launch jitter hitting a few large
prefills would also move only the tail); we cannot separate intrinsic queue metastability from transient environment
jitter without kernel-level instrumentation. The DECISIVE, airtight claim is the weaker one: the variance is NOT
workload variance (byte-identical workload). And the evaluation consequence is identical either way — single/
same-node A/Bs on this tail metric are unreliable. (§5.6.) This is also why the whale (6.2–26.6 s) and LRU
(5.9–31.2 s) p99 ranges overlap: the policy signal is a small fraction of the run-to-run variance.

**W8 — you localize the tail to large prefills; did you test the obvious fix (prefill/admission scheduling)?**
Attack: if large cold prefills drive the tail, stagger/rate-limit them — a natural mechanism you should evaluate,
not just conjecture (as the earlier draft did).
Defense (tested, refuted deterministically — 9th self-correction): we implemented a lossless admission-staggering
mechanism (cap concurrent large cold prefills per prefill batch, deferring extras; SGLANG_WILKES_MAXBIG, off by
default) and tested its PREMISE from the trace, deterministically (sim/prefill_contention.py, measured in-flight
intervals). It does not work, and we can say exactly why: at λ=3 large (≥20K) prefills almost never co-execute
(peak concurrency 2, only ~15% overlap another, overlapped ones prefill at the solo rate — ≤1.04×), and prefill
throughput is near-constant ~35K tok/s (a mild ≤17% chunk-budget penalty appears only at the rare 3-way overlap).
So a concurrent-count cap has nothing to bite on — the large-prefill effect on the tail is the admission-WAIT it
imposes on OTHER requests (wq 2× at large admissions), not big-vs-big execution contention. A single staggered GPU
run (p99 6.3 s) lands within the same-node plain-LRU band (8.1/21.9/30.0 s), consistent with the no-op; we do not
run a powered A/B because the 5.3× execution variance (§5.6/W7) precludes it. This RETRACTS the earlier draft's
"admission-staggering is the promising lever" conjecture: with residency gap-capped (W3) and admission with no
pileup to remove, BOTH scheduling axes are deterministically-shown dead ends for this metastable tail. (§9d.) Far
from a hole, testing-and-refuting the obvious fix — from our own analysis, before claiming it — is exactly the
rigor the metric demands.

**W9 — the cache looks big enough: capacity/mean-write-rate ≈ 650 s > the 460 s gap, so why is anything evicted?**
Attack: your own numbers (10.7M cache, ~20K tok/s λ=3 mean uncached write rate) imply an average entry survives
~540 s — longer than the median reuse gap. So LRU should already hold continuers across the gap, contradicting the
58%-avoidable-recompute and the gap-cap.
Defense (preempted in §4.2): the mean-time estimate is the WRONG lens — LRU eviction is governed by reuse DISTANCE
(distinct KV touched between two accesses), not mean time. Writes are dominated by large cold prefills, so retaining
every turn-0 across the gap means holding them ALL simultaneously; the resident-demand table shows that is 2.6×
capacity at the median gap, so the LRU tail (old, idle turn-0 prefixes) is clipped in bursts long before reuse. The
reuse-distance replay (§5.2) computes this correctly and confirms 58% of continuer prefixes are evicted before
first reuse — the paper leads with the replay precisely because the mean-rate intuition misleads. (Added a §4.2
clarification paragraph so a reviewer doesn't have to rediscover this.)

**W10 — your no-go names "cross-request sharing" as a capacity escape; how big is it? (quantified — negligible here.)**
Attack: if documents are shared across conversations, a dedup/sharing mechanism could raise effective capacity and
realize the offline win — you can't wave that away.
Defense (quantified, §2.1 + sim/doc_sharing.py): we measured it. Of 1553 conversations, 538 are empty-context
ShareGPT chats (no shared document); the 1015 doc-bearing ones draw on 887 unique documents, and only 15.5% share a
document (a single 100x-reused doc dominates; the rest <=3x). Even LOSSLESS PERFECT cross-conversation sharing would
save <1% of prefill work (~0.8M of 99.9M tokens), and RadixAttention already captures ~78% of shared-prefix reuse.
So in THIS workload the sharing escape is quantitatively negligible and cannot move goodput@SLO — it tightens the
no-go. We are explicit that a shared-corpus regime (e.g. RAG over a common corpus) could make sharing the dominant
lever, but that is a capacity/dedup mechanism orthogonal to the eviction/admission axes this metric probes.

**W11 — "size is the UNIQUE causal victim rule" — did you test recompute-cost-aware eviction? (No; it's better.)**
Attack (from an independent adversarial review): your "unique" claim rests on a 7-policy sweep that omits the obvious
competitor — recompute-cost-aware eviction — which sibling work found wins elsewhere. "Unique among the rules you
happened to try" is not unique.
Defense (CONCEDED — 11th self-correction, v0.6): correct, and we tested it. Added cost_aware / cost_aware_pp /
hit_density / gdsf to sim/simulate.py. Result @10.7M: proven-protected recompute-cost-greedy (evict cheapest-to-
rebuild unproven first) captures +48% of the Belady headroom — roughly TWICE the size-aware whale's +23% (robust:
+61% vs +29% @8M). So size is NOT the unique or best signal. We retracted "unique" throughout and reframed the
constructive result to its correct form: the headroom-capturing signal-CLASS is proven-protection + a prefix
size/cost magnitude signal (recency/frequency capture 0% even with proven-protection, cf. SLRU=0%; without
proven-protection cost-only/size-only are catastrophic −311%/−529%). Important honesty: cost-greedy wins the offline
recompute metric by directly minimizing it, but keeps dead single-turn KV resident (poorer live-set utilization),
whereas whale targets liveness — and BOTH are gap-capped online (§5.5), so neither yields an online goodput gain.
This weakens the size-signal-as-headline and is a further argument for the metastability-forward framing (see below).

## Minor
- §5.3 could add a one-line bridge to §5.5 ("the grace-trap motivates changing the SIGNAL, not the retention time
  — §5.5"). (pending)
- RESOLVED in v0.7 (all the deferred review items): (#6) REPOSITIONED — retitled "The Metastable Tail-SLO: A No-Go
  Criterion for Lossless KV-Cache Optimization"; Summary + Abstract now lead with the no-go + execution-variance
  metastability, with the size/cost signal explicitly framed as constructive-but-offline-only. (#5.4) POWER LAW added
  to §5.6: CV=0.55 → n≈24/policy to certify a 45% p99 shift @0.8 power; at n=6 only a ~89% shift is detectable →
  quantifies "unidentifiable at feasible n" and explains the non-reproduction. (#5.3) HELD-OUT AUC added to §4.2:
  fit 0.775 → held-out 0.783 (within 0.008) → size signal not overfit. (#3.3) tempered the 70× framing (medians;
  AUC 0.78 = 22% misrank). (#4.1) scoped the no-go "criterion" generality to the single-workload evidence + reframed
  as a practitioner checklist. All offline/no-GPU (sim/power_auc.py). Paper v0.7.
- §6 related work: add a sentence distinguishing the size-observability result from continuation-predictor /
  TTL-aware works (Continuum, Predictive-Multi-Tier) — we identify a SPECIFIC observable (turn-0 size) and show
  offline it is uniquely sufficient among causal victim rules. (pending)
- Reproducibility §8: add whale run dirs (runs/whale-full, whale-r2..r6) + the simulate.py whale branch commit. (pending)

## Outcome (resolved; a chain of self-corrections)
- On nodeset-0 the whale p99 reduction was significant (median −45%, Mann-Whitney n=6/n=5, p≈0.026), but the
  three-node cross-node test showed it has NO STABLE SIGN (lower on 2/3 nodes, reversed on 1) → weak/node-dependent,
  not portable. The robust, SLO-relevant contributions are the deterministic offline size-signal and the cross-node
  metastability caution.
- Self-corrections the replication/analysis forced, in order (9 total): bounded-impossibility framing →
  cold-doc-floor prediction → median-TTFT win (lru-r3) → full-separation p≈0.029 (whale-r6 high draw) →
  node-portability (certified 0-3) → node-dependent sign-instability refinement (3-node) → "reduces-queueing"
  mechanism (wq trace) → "bursts of large prefills" (CV≈1 Poisson) → "admission-staggering is the lever" (refuted
  deterministically, W8). Each retracted/refined as more data arrived.

## Resolution status (v0.4, updated 2026-07-13)
- Now an INSIGHT + CAUTION + BOUNDED-NEGATIVE paper. Reframed: bounded-impossibility → size-signal (AUC 0.78 +
  offline unique-capture) → insight+caution (3-node cross-node retracted the portable online win) → v0.4 adds two
  deterministic results that tighten it: (i) the metastability is EXECUTION-driven (byte-identical workload, 22
  runs, 5.3× p99 / 1.1× median); (ii) admission-staggering refuted deterministically (W8) → BOTH scheduling axes
  (residency gap-capped, admission no-pileup) are dead ends, each shown so deterministically.
- Robust contributions (all deterministic / node-independent): the offline size-signal unique-capture (§4.2), the
  chash tail decomposition (§5.2), the two-bound model (§4.1), the execution-driven cross-node metastability
  (§5.6), and the dual-axis deterministic bounded-negative (§9d). The sole constructive positive is the turn-0 size
  signal (offline); online it is gap-capped — stated honestly.
- W1/W6 resolved (online whale advantage weak/node-dependent, not portable; 3-node + N=22 fixed-workload done).
  W8 resolved (admission axis refuted deterministically). Remaining OPEN only as future work: a larger multi-node
  median-of-k (contended pool precludes) and a memory-pressure-gated admission variant — both bounded by the
  execution-variance dominance.

## Second independent adversarial review (v0.7 → v0.8), resolutions
A fresh hostile PC review of the repositioned v0.7. Verdict: still weak-reject as-framed, but confirmed the
reposition fixed the integrity problem (no more advertise-then-withdraw of a false positive). It found concrete
revision seams + one deep issue; all actionable ones fixed in v0.8:
- ★ STRONGEST REJECTION ARG (metastability confounded with node/environment health, flagship A/B on a "flaky" node):
  ADDRESSED with a within-vs-between-node VARIANCE DECOMPOSITION (§5.6, sim/power_auc.py). **95% of the p99 variance
  is WITHIN-node** (between-node η²=0.05); a CERTIFIED node (0-3) alone spans the full 5.3× range at n=3, and the
  screening node nodeset-0 has the SMALLEST within-node spread — so it is run-to-run on a fixed node, not node-choice
  or nodeset-0 flakiness. (Whether the within-node residual is intrinsic vs transient-environment stays unseparated,
  but it is same-node/run-to-run either way → single-node A/Bs void regardless. Honest.)
- O2 (power law used POOLED cross-node CV but labeled it same-node): FIXED — recomputed within-node CV=0.63 →
  n≈31/policy (was 24 mislabeled). E4 (4 dead "§9d" refs): FIXED → §9. E6 (§8 stale "7-policy/whale −23%"): FIXED →
  11-policy/whale +23%/cost-greedy +48% (+sign). E2 (§4.1 stale hit 0.62/R_thru 6.5 vs §5.1's 0.68/7.8): FIXED.
  O4 (+48% skim-bankable before caveat): FIXED — inline gap-capped + near-tautological flag. E5 (single-run 63% vs
  "single runs untrustworthy"): FIXED — clarified the decomposition is a COMPOSITIONAL quantity (stable 58–63%),
  distinct from the metastable p99 TIMING (5.3×).
- DEFERRED (polish/judgment, not integrity): E1 (clarify the 22-run pooled range coincides with LRU's because LRU
  holds both extremes; whale 6.2–26.6), O1 (title "criterion" — box already scopes it to a conjecture+checklist),
  O5 ("5/6 below LRU's best" one-sided phrasing), and collapsing the ~700-word Summary that restates facts 2–3×.
NET: two independent adversarial reviews survived; the honest verdict remains that this is a rigorous bounded-negative
on a single frozen workload with goodput@SLO≡0 — a legitimate but hard-to-place contribution whose ceiling is set by
the single-workload scope the frozen protocol imposes, not by remaining fixable flaws.
