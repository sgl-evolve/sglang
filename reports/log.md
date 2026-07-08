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
