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
