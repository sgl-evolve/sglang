## 2026-07-08 — start (sgl_mech, v0.25_ablations 2-tier mechanism-only)
- Setup done: own clone evolve/sgl_mech + cu129 venv (sglang dev0, torch 2.11.0+cu129). check_env queued (a3 saturated).
- Active path verified = UnifiedRadixCache(FULL+MAMBA)+HybridCacheController (hi_mamba dormant). 2-tier, no L3.
- Baseline v0_official logged to W&B run sgl_mech (offline+synced): p99 TTFT 6326ms, hit 0.62, host_util 1.0, evict 582M ≫ load_back 298M.
- Metric = goodput under p99 TTFT ≤ 8s SLO. Plan: diagnostic baseline-repro (env control + live bottleneck evidence) → pick mechanism.
- Eval capacity is binding: shared 4-node certified pool (flock-coordinated) held by base_free; contend for it.

## 2026-07-08 — v1 NEW BEST (cost-aware eviction), clean same-node A/B
Mechanism: CostAwareStrategy (recompute-cost-weighted eviction, threshold 4096 tok) — commit f142702d3.
Clean same-node serial warm A/B on ondem-3 (toggle confirmed via server.log):
- v0-ctl (stock LRU): hit 0.627, tput 2.87, p50 736, p90 2369, p99 5189, out_tok/s 367.
- v1 (cost-aware):     hit 0.681 (+8.6%), tput 3.02 (+5.2%, sustains λ=3), p90 2054 (-13%), p99 4820 (-7.1%), out_tok/s 387 (+5.2%); p50 939 (+27.5% regression).
Takeaway: cost-aware eviction shifts recompute work from the expensive tail to the cheap median →
higher token-hit-rate + throughput + lower p90/p99 tail (favorable for goodput@p99-SLO), price = higher p50.
Lossless by construction. Logged to W&B run sgl_mech as v1 (mechanism). Next: threshold sweep + mamba-pool extension.

## 2026-07-08 — v3 NEUTRAL (mamba cost-aware extension), clean same-node A/B
v3 (cost-aware FULL+MAMBA) vs v1-repro (cost-aware FULL only), same node ondem-3, serial, back-to-back.
hit 0.671 vs 0.674 (flat), tput 3.02 vs 3.02 (flat), p50 619 vs 797 (-22%), p99 5167 vs 4601 (+12% WORSE).
Negative: mamba cost-aware eviction does NOT compound v1 (mamba pool isn't the binding recompute constraint;
full-KV host eviction, already cost-aware in v1, is). Mamba code kept gated OFF. v1-repro reproduces v1 (stable).
Next: threshold sweep (t2048/t8192) to probe p99; v4 reuse-gated cost (protect only reused-expensive) ready.
