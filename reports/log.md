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
