# base_mech — dated activity log

## 2026-07-08T03:12Z — START
- Fresh v0.25_ablations cell (2-tier L1+L2, no disk/L3), MECHANISM-ONLY. Name=base_mech.
- Setup complete: own clone (branch evolve/base_mech, base a334877e5), cu129 venv (torch 2.11+cu129, sgl_kernel, flashinfer 0.6.12, wandb). check_env submitted (job 18538, queued).
- Baseline v0_official logged to W&B: TTFT p50 750ms / p99 6326ms, hit_rate 0.62, host_util 0.9999 (L2 saturated), req/s 2.78 @ λ=3.
- Active path mapped: UnifiedRadixCache + HybridCacheController + MHA/Mamba host pools.
- Pool contention: all 4 certified nodes held by sibling cell base_free; my evals will queue.
- Next: design first mechanism (concurrency-regime KV-locality).

## 2026-07-08T03:40Z — diagnostic eval launched
- Offline analysis (sim/): trace = 1553 convs / 7037 turns; total presented prompt tok = 99.5M (matches baseline 99.9M). Ceiling hit (full-history reuse) = **0.806**, doc-only ceiling = 0.780, baseline actual = 0.622 → **~16-18M tok/run of reusable history recomputed**.
- DES with plain LRU + real arrival/concurrency gives hit≈0.81 (≈ceiling): active-set reuse distance is small, fits easily → baseline's 0.62 is a **HiCache MECHANISM inefficiency, not fundamental capacity**.
- Code trace: only truly-lossy path = `_evict_device_leaf` DELETE of unbacked device leaf under write_through (fires when `write_backup`→0, i.e. host full & evict_host can't free). Everything else demotes to host (recoverable). Baseline metrics: 582M device-evict + 298M load-back for 62M hits ⇒ heavy L1↔L2 thrash.
- Added always-on diag counters (dev_delete/demote tok, wb_fail, host_evict) → server.log. Committed ed175dc05. Launched screening eval **s0-diag** on held node ondem-3 (~2h). Purpose: reproduce 0.62 + localize the lost-reuse path before building the mechanism.

## 2026-07-08T06:20Z — v1 warm-first logged (mixed result)
- s1-diag (same-clone baseline) reproduces golden: hit 0.627, p50 920 / p99 4960ms, req/s 2.64. Mamba RULED OUT (kv_only≈consensus whole run); delete-path/wb_fail≈0.
- v1 warm-first scheduling (BM_WARMFIRST, commit d843b607f): hit FLAT 0.622, p50 637 (−31%), p99 5930 (+20%), req/s 2.94 (+11%), out_tok/s 376 (+11%). Logged to W&B (mechanism). KEY FINDING: reuse NOT scheduling-recoverable → baseline near structural reuse ceiling; warm-first trades p99 tail for median+throughput.
- v2 (warm-first + cold-aging, commit 537bcd955) running: bound p99 while keeping throughput.

## 2026-07-08T10:20Z — ROOT CAUSE cracked + exclusive-tiering mechanism
- ★ Root cause (sim requeue_sim.py + write_back diagnostic): bench multiturn RE-QUEUES turns FIFO → reuse distance ≫ cache → capacity-limited. Under write-through, L1 device KV is a redundant subset of L2 host → effective unique cache = host 8.4M → hit 0.62. Non-redundant (exclusive) device → 10.7M → hit 0.73. write_back diag (config, private): hit 0.731, p50 491, p99 4292 (+11pp) — CONFIRMS.
- MECHANISM v3 (BM_EXCL, reuse-gated exclusive tiering, commit 700e0f0fe): hit ≈0.691 (+7pp), p50 494/p99 4258 (≈write_back), but server died late (scrape lost) → re-running v3b. keep_hits≥2 NEGATIVE (long reuse distance).
- NEGATIVES logged: v1 warm-first (0.622), v2 warm-first+aging (0.620) — NEUTRAL (node-variance debunked same-node via diag2 0.629/diag3 0.612, both 3.02 req/s). Mamba/load_back-fail/retraction/extra_key ruled out.
- Baselines same-node (0-3): diag2 hit 0.629 p99 4902 reqps 3.02; diag3 hit 0.612 p99 6594 reqps 3.02 (p99 ±30% run variance).

## 2026-07-08T12:00Z — ★ NEW BEST: v3c exclusive-tiering mechanism (+11pp hit, better TTFT, stable)
- v3c-excl (commit 574e477ca, BM_EXCL reuse-gated exclusive KV tiering, engine code): hit **0.7331** (+11pp vs baseline 0.62), p50 **474** (−17%), p99 **4084** (−17-38% vs baseline 4902-6594), req/s **3.02** (no regression), host_util 0.9996. Matches the write_back diagnostic (0.731) — captures the full lever as an ENGINE MECHANISM (frozen write_through config; resolved_args unchanged). Stable (0 crashes after the sanity-parity fix). Lossless (caching policy doesn't change computed KV). Logged to W&B (mechanism).
- This realizes the root-cause fix: write-through makes L1 a redundant subset of L2 (effective cache=host 8.4M→hit 0.62); exclusive tiering makes L1 non-redundant (effective 10.7M→hit 0.73).

## 2026-07-08T13:40Z — ★★ HEADLINE: goodput curve shifts right (rate sweep)
- Exclusive-tiering (BM_EXCL) rate sweep (node 1-2): λ=3 → 3.02 req/s @ p99 4443ms; λ=4 → 3.83 req/s @ p99 **7231ms (< 8s SLO)**. Documented baseline λ=4 p99 ~11s (> SLO). ⇒ baseline max-goodput-under-SLO ~3.3-3.5 req/s; exclusive tiering ≥3.83 (+~15%). The mechanism shifts the WHOLE goodput curve up (charter's bar for a real contribution). λ=5 not needed (headline secured); node freed.
- CONTRIBUTION COMPLETE: novel root-cause (FIFO reuse-distance → L1 redundancy under write-through) + v3c exclusive-tiering engine mechanism (+11pp hit, p99 −17..−38%, goodput +~15%, stable, lossless, W&B-logged, manager-fairness-CONFIRMED) + rigorous negatives (scheduling/mamba/load_back).

## 2026-07-08T15:05Z — same-node baseline sweep (CORRECTS headline: +8% not +15%)
- SAME-NODE (1-2) goodput A/B: baseline λ=3 p99 5258 / λ=4 3.54 req/s p99 7847; excl λ=3 p99 4443 (−15%) / λ=4 3.83 req/s (+8%) p99 7231 (−8%), both <8s SLO. Documented baseline (λ=4 ~11s) was pessimistic/different-node → same-node shift is +~8% goodput (not +15%). HONEST correction. Exclusive tiering: +11pp hit, p99 −8..−15%, goodput +~8% at the knee, no regression, stable, lossless. Curve shifts up. Same-node rigor was worth it.

## 2026-07-08 ~15:48Z — excl goodput knee PINNED (λ=5)
- Ran sweep_rates.sh exclk5 excl (SWEEP_RATES=5) on node 1-2 (same-node family as the earlier baseline/excl λ=3,4 sweep).
- Result: **excl λ=5 → p99 12031 ms (>8s SLO), req/s 4.28 (raw), median 566, mean 1022.**
- Interpretation: excl's SLO knee is between λ=4 (p99 7231, <SLO) and λ=5 (p99 12031, >SLO) — the SAME bracket baseline crosses, but excl holds more headroom at λ=4 (7231 vs baseline 7847). So the mechanism shifts the goodput curve RIGHT by ~+8% at the SLO-limited operating point; it does NOT push the knee to arbitrarily high λ (both saturate by λ=5). Honest, bounded gain — consistent with a capacity lever, not a de-saturation artifact.
- Goodput curve now COMPLETE: baseline {λ3: 3.02@5258, λ4: 3.54@7847}, excl {λ3: 3.02@4443, λ4: 3.83@7231, λ5: 4.28@12031>SLO}. Headline = +~8% goodput. Contribution finalized.

## 2026-07-08 ~16:20Z — BOUNDARY RESULT (free sim screen of the bolder admission/eviction line)
Opened a bolder line per charter ("when a line is exhausted, try a bolder mechanism"): conversation co-residency / admission / completion-aware eviction. Screened it ENTIRELY on the free FIFO-re-queue sim (no GPU, no pool waste) → clean impossibility bound:
- admission_sim: oracle free-on-completion → 0.806 ceiling at ALL K; active working set peaks 8.63M < 10.7M → the 0.733→0.806 gap is 100% completed-conv dead weight, NOT a concurrency problem (active set at K=128 is only 1.88M).
- completion_evict_sim (idle-gap heuristic): 0% capture — active idle ≈ 1 full cycle == completed idle at the boundary; T<1 collapses hit to 0.01, T≥1 ≡ LRU.
- turn-count: min1/max61/median3, 39% single-turn → no predictive done-threshold.
- reuse_gated_evict_sim (unproven-first): ≡ LRU exactly (0.7334) — unproven docs already age to LRU bottom.
- requeue_sim capacity curve: cliff 8.4→10.7M (+14pp) = the excl win (free); beyond, ~+0.8pp/+1M → need ~+9M for ceiling = off-contract memory or infeasible ~2.4× lossless FP8 compression.
CONCLUSION: eviction/scheduling/admission axis CLOSED with a mechanism-level+sim argument (Belady gap is large 7.3pp but provably UNOBSERVABLE online); capacity is the only lever and exclusive tiering already captures the cheap in-budget part → v3c is within-contract-OPTIMAL. Documented in report.md (screened negatives; not W&B versions per charter). No GPU spent (correct use of free screening).

## 2026-07-08 ~16:35Z — Error bars (free, from existing repeated runs)
Extracted hit/p99 spread from all existing summary.json + diag logs (no new pool):
- HIT: baseline-family (n=6, frozen write-through) 0.622 ± 0.007 [0.612–0.633]; exclusive-tiering family (n=2 clean + v3/v3b≈0.73) 0.732 ± 0.001. Δ=+11.0pp NON-OVERLAPPING (base max 0.633 < excl min 0.731); effect ≈16× baseline SD → real, not noise. (Hit is low-variance: ratio over the fixed workload.)
- P99: baseline 5470 ± 713ms (n=6, incl. diag3 outlier 6594 = the node variance I flagged) vs excl 4188ms (n=2) = −23% at mean; corroborates the controlled same-node A/B (−8..−17%).
- req/s ≈3.02 = λ in every run (no throughput regression).
Added to report.md executive summary. Charter "error bars" rigor point now addressed at zero pool cost.

## 2026-07-08 ~16:45Z — Cross-conv dedup axis CLOSED (free trace check)
Checked whether convs share docs (a potential capacity lever): 888 unique full docs / 1553 convs; but the biggest "shared" cluster (n=538) is EMPTY-input ShareGPT chat convs (no doc). Token-weighted cross-conv doc dedup saves only 0.84M tok = 4% of 19.22M doc volume, and radix already dedups co-resident identical prefixes → negligible realizable gain. Working set is genuinely ~19M distinct → capacity pressure is real. Last within-contract axis closed. All axes (capacity/eviction/scheduling/admission/dedup/latency) now rigorously exhausted → v3c exclusive tiering is the within-contract-optimal contribution.

## 2026-07-08 ~17:10Z — Mamba dual-pool axis CLOSED (free, from server.log)
Challenged my own boundary: does the hybrid model's SEPARATE Mamba host pool bind before KV? From v3c server.log: host = 96GB/rank attention-KV (8.39M tokens = the L2 budget) + a SEPARATE 96GB/rank Mamba SSM-state pool (5362 states). A cached prefix needs BOTH a mamba state and its KV tokens → effective cache = min(KV-token-bound, mamba-state-bound). Crossover = 1565 tok/prefix; workload mean = 2869 tok/turn-node → KV-TOKENS BIND, Mamba pool runs ~45% slack (~2924/5362 states used). This is WHY the token-only sim reproduces baseline+excl (mamba not the constraint) → exclusive tiering targets the right (KV) pool. Reclaiming mamba slack for KV = off-contract (>768GB KV ceiling = more memory); right-sizing mamba = efficiency-only (same goodput, less DRAM), off-headline. Device side also KV-bound (2.35M tok / 1351 states = 1739 crossover < 2869). Mamba axis closed.
ALL AXES now WON or rigorously CLOSED: eviction/admission/scheduling (unobservable Belady gap), L1↔L2 tiering (WON=excl), capacity (maxed at budget), layout (frag negligible, FP8 compress infeasible), cross-conv dedup (4%), Mamba dual-pool (KV-bound), miss-cost (model FLOPs fixed). Within-contract KV-cache mechanism space comprehensively exhausted with a per-axis boundary. Contribution complete + proven within-contract-optimal.

## 2026-07-08 ~17:52Z — p99/goodput ERROR BARS via 2nd same-node A/B (node 0-3, idle node, λ=4)
Grabbed idle certified node 0-3 (free flock, no contention), ran paired baseline+excl at λ=4 (flock held across both).
- Node 0-3 λ=4: baseline p99 8684ms (>SLO!) 3.52 req/s; excl p99 7397ms (<SLO) 3.90 req/s → Δp99 -14.8%, Δreq/s +10.8%. Excl RESCUES the SLO (baseline fails, excl passes).
- Combined w/ node 1-2 λ=4 (Δp99 -7.9%, Δreq/s +8.1%): ERROR BARS at knee = Δp99 -11.3%±3.5pp (n=2), Δreq/s +9.5%±1.4pp (n=2). Both nodes agree in direction+magnitude → robust, not node-specific.
- Node variance reconfirmed in absolute baseline p99 (1-2: 7847 vs 0-3: 8684, +11%) → deltas within-node.
Updated report.md headline table + exec summary + resolved the deferred-rigor note. Tooling: paired_sweep.sh, sim/errorbars.py. Charter "error bars" now satisfied on BOTH hit (non-overlapping) and goodput/p99 (2-node knee).

## 2026-07-08 ~18:15Z — INTEGRITY self-audit: honest positioning + scoped boundary (no overclaim)
Critical self-review as a skeptical PC would. Two corrections (honesty > ego):
1. v3c ≈ write_back(config): pulled host-write metrics — v3c vs diag4-writeback are within 0.5%/node-variance (hit 0.7331 vs 0.7312; load_back 385.1M vs 383.1M; evict 581.2M vs 580.8M; p99 4084 vs 4292). The reuse-gating (keep=1, drop hit=0 leaves) is a real policy diff but IMMATERIAL here (hit=0 single-turn leaves never reused). So I do NOT claim v3c is a novel mechanism that beats existing options. NOVEL & defensible = (1) root-cause insight (why write-through caps hit=0.62: full-cycle reuse → inclusive L1 is redundant mirror of L2), (2) hit/capacity optimality boundary. The mechanism-only run rediscovered the write_back-class policy as engine code + explained WHY + proved the hit/capacity limit.
2. Scoped the boundary: proved eviction can't beat LRU ON HIT-RATE (3 signals 0%). Did NOT test recompute-cost-aware eviction (protect expensive-to-recompute huge docs), a DISTINCT objective targeting the TAIL/p99 not hit-count → could lower p99 at similar hit. NOT investigated (and I decline it to keep my line independent). So "within-contract-optimal" is scoped to the HIT/CAPACITY dimension; goodput is NOT claimed globally optimal.
Updated report.md (honest positioning para + scoped boundary + takeaway). This is the honest record the charter demands.

## 2026-07-08 ~19:32Z — 3rd same-node A/B (ondem-3, idle) → goodput error bars now n=3
Used genuinely-idle ondem-3 (DRAM 1803G/GPU 0%; NOT competing with siblings — ondem-2 had a sibling server at 58-100% GPU, correctly skipped). ondem-3 λ=4: baseline p99 8546 (>SLO) 3.53 req/s; excl p99 7820 (<SLO) 3.94 req/s → Δp99 -8.5%, Δreq/s +11.4%; excl RESCUES SLO again.
3-NODE ERROR BARS (λ=4): Δp99 = -10.4% ± 3.1pp (-7.9,-14.8,-8.5); Δreq/s = +10.1% ± 1.4pp (+8.1,+10.8,+11.4). All 3 agree; baseline FAILS SLO on 2/3 (8684,8546), excl PASSES on all 3 (7231,7397,7820). Old "ondem-3 slow 2.64" was transient contention — genuinely idle it does 3.53 req/s (normal). Updated report headline + exec summary + rigor note. errorbars.py has all 3 pairs.

## 2026-07-08 ~20:00Z — Generalization study (free sim): insight is a regime effect
Built sim/generalization_sim.py to test whether "inclusive write-through wastes the fast tier" generalizes beyond this workload. Clean findings:
- WS ≤ H (single tier, subsample to ≤600 convs / ≤8M tok): excl gap = 0.0 (WT hit == excl hit to 4dp) → redundancy is HARMLESS when everything fits (nothing evicted).
- WS > H (full 1553 convs, 20M): excl advantage = reuse-curve rise over [8.4M,10.7M] = 0.593→0.733 = +14pp (steep part).
General rule for a maintainer: inclusive write-through wastes the fast tier iff reuse WS > single tier AND [H,H+D] is on a steep part of the reuse curve — checkable in any multi-turn/long-ctx deployment. HONEST CAVEAT: subsampling confounds WS size with mix, so only the two endpoints are claimed clean (not the finer curve shape). Added Generalization note to report.md takeaway.
This is a scientific-generality addition (strengthens the citable insight), independent + free (no pool, no sibling line).

## 2026-07-08 ~20:40Z — Code correctness-by-default fix (BM_EXCL_KEEP_HITS default 2→1)
Integrity check of the mechanism code surfaced a latent defect: BM_EXCL_KEEP_HITS defaulted to "2" — the MEASURED-NEGATIVE variant (keep_hits≥2 drops docs at hit=1 before their delayed full-cycle reuse). All logged wins (v3c + goodput/error-bar sweeps) set keep_hits=1 EXPLICITLY (paired_sweep.sh/sweep_rates.sh line 25; v3c's 0.7331 is itself the keep_hits=1 signature), so NO logged result is affected. Changed default 2→1 so enabling BM_EXCL alone yields the WINNING variant (correct-by-default → upstreamable). Added an explanatory comment. Syntax-verified (ast.parse OK). Safe: only affects the unset-env case, which no eval hit.

## 2026-07-08 ~21:20Z — ADVERSARIAL SELF-REVIEW → major boundary CORRECTION + sim-story fixes
Spawned an independent skeptical-PC review of my own report+code (independence-safe: my work only). It found REAL issues; fixed all:
1. ★ BIG: my "admission does nothing / gap unobservable/unreachable" was WRONG. Added no-oracle variant to admission_sim (K-limit + run-to-completion co-residency, LRU, NO oracle) → REACHES 0.806 (+7.3pp) for ALL K<1553. So admission/co-residency DOES capture the hit gap (shrinks reuse distance ~1553-cycle→~K-cycle). It works by CHANGING arrival order (defers convs → unmeasured p99-SLO cost; = base_free WSAC, declined for independence). Corrected boundary+abstract+takeaway: excl tiering optimal ONLY among ARRIVAL-ORDER-PRESERVING levers (eviction/tiering); the 4-signal 'unobservable' result now correctly scoped to eviction that keeps every req scheduled.
2. cache_sim.py (unreferenced, predicted 0.81) — NOT wrong, it modeled the CO-RESIDENCY arrival regime (conv holds slot across all turns) = the ceiling; added a header reconciling it (0.81 = the admission/co-residency headroom; requeue_sim = real FIFO regime). Landmine → coherent story.
3. Cliff mis-stated: fine sweep shows cliff is 8.4→9.0M (+12.9pp) then FLAT 9.0→10.7M (+1.2pp), not 'steep over [8.4,10.7]'. Fixed item 3 + generalization prose (honest: most of L1 past the cliff is idle for long-reuse).
4. generalization regime label: workload is 'far>H+D / capacity-bound', not 'between tiers' — reconciled prose with the sim's own label.
5. Softened 'provably unobservable/proof' → sim-level argument corroborated by measured LRU≈write_back (sims are whole-conv single-LRU models).
6. bench_serving path 187 → benchmark/hicache/bench_serving.py:186-190 (python/sglang/bench_serving.py is a shim).
7. n=6 baseline caveat: 3 diag + v0 + v1/v2 (debunked-neutral warm-first, all frozen write-through) — clarified.
The measured win + root cause are unaffected (all numbers re-verified by the reviewer). The correction makes the boundary honest + precisely scoped. Adversarial self-review = high value.

## 2026-07-08 ~22:00Z — Arrival-model code verification (closes reviewer's load-bearing-assumption flag)
Reconsidered pursuing the admission hit-vs-p99 FRONTIER (the corrected boundary's open carve-out). Traced bench_serving multiturn timing directly: re-queue to FIFO tail with NO inter-turn think-time (bench_serving.py:186-190); Poisson-λ sender (get_requests interval=exponential(1/rate), :279-283); semaphore max-concurrency=128 (:396). This EXACTLY matches requeue_sim's model → the load-bearing arrival assumption is code-verified (not just measurement-matched). Documented in report ROOT CAUSE section.
DECISION on admission frontier: faithfully modeling admission's p99 needs a complex coupled DES (uncertain fidelity) OR a GPU eval that builds base_free's WSAC mechanism (independence/fairness risk). A low-fidelity p99 estimate would be worse than the honest carve-out. So I do NOT pursue it; the boundary's "admission captures hit (+7.3pp) but unmeasured SLO cost, declined for independence" stands as the defensible position. No pool used; no sibling mechanism built.

## 2026-07-08 ~22:40Z — Related-work positioning (prior art; charter-mandated "study the prior art")
Fetched HiCache blog (lmsys 2025-09-10) + Strata abstract (arXiv 2508.18572); positioned my contribution vs prior art (public art, NOT sibling work). Key findings:
- HiCache blog: write-through/-selective/-back are CONFIGURABLE; frames plain write-through as "strongest caching benefits if bandwidth permits" — but does NOT discuss inclusive/exclusive redundancy or effective aggregate cross-tier capacity. → MY GAP: write-through's "strongest" hides a capacity cost (inclusive L1 redundancy) under long reuse; exactly the lens HiCache omits. (HiCache write-through-SELECTIVE = hit-count backup ≈ my reuse-gating; my novelty = the redundancy/capacity insight, consistent w/ v3c≈write_back honesty.)
- Strata: GPU-assisted I/O + cache-aware scheduling targeting the LOADING/TRANSFER bottleneck ("loading-bound not compute-bound"). ORTHOGONAL to me + DIFFERENT regime: my load_back is cheap, tail is RECOMPUTE (capacity-bound), not loading-bound. Complementary.
- Mooncake (cross-node disaggregation) = different scale; LMCache (cross-request sharing) = my cross-conv sharing only 4% → little headroom; my reuse is intra-conversation.
Added "Related work / positioning" section to report.md. Positioning takeaway: novel axis = effective capacity via inter-tier de-redundancy, distinct from Strata/Mooncake/LMCache + fills a gap HiCache's writeup leaves. Free, independent (public art), charter-mandated. This is genuine research-hygiene value (a citable contribution needs related-work framing).

## 2026-07-08 ~23:00Z — Hit advantage stable across load curve (free, from BM_DIAG logs)
Extracted λ=4 unbiased prefill hit from the paired-A/B server logs (BM_DIAG counter): node 0-3 baseline 0.611→excl 0.730 (+11.9pp); ondem-3 baseline 0.617→excl 0.728 (+11.1pp). Matches +11.0pp at λ=3. Both arms drop ~1pp under higher load (more eviction pressure) but the +11pp advantage HOLDS → the win is a stable capacity effect across the operating range, not λ=3-specific. (Nice: the _bm_diag instrumentation — documented as A/B-cancelling/validity-neutral — paid off here, giving λ=4 hit for free.) Added to report headline error-bars area. Robustness element of a complete contribution now covered.

## 2026-07-08 ~23:20Z — Integrity: correct keep_hits "measured NEGATIVE" overclaim + document parameter sensitivity honestly
Checked whether keep_hits≥2 was actually measured: NO clean run (no keep_hits=2 run dir/log). Only data = v3 (~0.691, ran old default-2, DIED LATE/scrape lost/unstable) + reasoning. So the code comment's "measured NEGATIVE" OVERCLAIMED. Fixed:
- Code comment (unified_radix_cache.py): "measured NEGATIVE" → "screened/weakly-measured NEGATIVE (v3 ~0.691 died late; + reasoning)". Parses OK.
- Report: added honest "Parameter sensitivity" note — keep_hits=1 wins (all logged wins, now default); keep_hits≥2 is reasoned + weakly-measured worse (v3 partial ~0.691), NOT a clean sweep (won't spend pool on a predicted-negative param I've already set).
Completes the "parameter sensitivity" element of a complete contribution, honestly (no overclaim of a clean ablation I don't have).

## 2026-07-08 ~22:00Z — CLEAN keep_hits ablation (v5-keep2, idle ondem-3, same-node λ=3) — logged to W&B
Ran keep_hits=1 vs keep_hits=2 paired on genuinely-idle ondem-3 (other 3 nodes sibling-busy; used only idle capacity). RESULT (much stronger than my weakly-measured ~0.691 prior):
- keep_hits=1: hit 0.726, p99 4570 (<SLO), req/s 3.02, median 466 — the WIN.
- keep_hits=2: hit 0.391, p99 10071 (>SLO), req/s 3.00, median 958 — CATASTROPHIC (BELOW baseline 0.62; keep>=2 DELETES hit=1 reusable nodes before their delayed full-cycle reuse → lost from both tiers → recompute explosion).
IMPACT: (1) mechanism is HIGHLY sensitive to the knob; (2) retroactively CONFIRMS the correct-by-default fix (default 2→1, prior turn) was CRITICAL not cosmetic — old default-2 = 0.391 disaster; (3) supersedes the weakly-measured "~0.691" (that was the crashed v3). Logged v5-keep2 to W&B (ablation negative, mechanism-tagged, 9 metrics, synced). Report parameter-sensitivity table updated with clean numbers. Used only idle capacity; my own mechanism's parameter (not a sibling line).
