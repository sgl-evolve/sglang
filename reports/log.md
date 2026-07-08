# base_mech — dated activity log

## 2026-07-08T03:12Z — START
- Fresh v0.25_ablations cell (2-tier L1+L2, no disk/L3), MECHANISM-ONLY. Name=base_mech.
- Setup complete: own clone (branch evolve/base_mech, base a334877e5), cu129 venv (torch 2.11+cu129, sgl_kernel, flashinfer 0.6.12, wandb). check_env submitted (job 18538, queued).
- Baseline v0_official logged to W&B: TTFT p50 750ms / p99 6326ms, hit_rate 0.62, host_util 0.9999 (L2 saturated), req/s 2.78 @ λ=3.
- Active path mapped: UnifiedRadixCache + HybridCacheController + MHA/Mamba host pools.
- Pool contention: all 4 certified nodes held by sibling cell base_free; my evals will queue.
- Next: design first mechanism (concurrency-regime KV-locality).
