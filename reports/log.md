## 2026-07-08 — start (sgl_mech, v0.25_ablations 2-tier mechanism-only)
- Setup done: own clone evolve/sgl_mech + cu129 venv (sglang dev0, torch 2.11.0+cu129). check_env queued (a3 saturated).
- Active path verified = UnifiedRadixCache(FULL+MAMBA)+HybridCacheController (hi_mamba dormant). 2-tier, no L3.
- Baseline v0_official logged to W&B run sgl_mech (offline+synced): p99 TTFT 6326ms, hit 0.62, host_util 1.0, evict 582M ≫ load_back 298M.
- Metric = goodput under p99 TTFT ≤ 8s SLO. Plan: diagnostic baseline-repro (env control + live bottleneck evidence) → pick mechanism.
- Eval capacity is binding: shared 4-node certified pool (flock-coordinated) held by base_free; contend for it.
