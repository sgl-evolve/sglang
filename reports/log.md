# sgl_free — dated research log (v0.25_ablations, 2-tier)

## 2026-07-08 (start)
- Fresh cell v0.25_ablations (2-tier L1+L2, NO disk). Setup done: own clone + cu129 venv (torch cu129,
  sgl_kernel, deep_gemm, sglang via PYTHONPATH), branch evolve/sgl_free.
- Verified active paths: UnifiedRadixCache + HybridCacheController, LRU evict, fcfs default, no conv-id.
- Baseline v0_official (fcfs): p99 TTFT 6.3s < 8s SLO, 2.78 req/s, hit 0.62, host_util 0.9999 (saturated,
  NOT collapsed). Logged as W&B point 0.
- Launched eval #1 = v0_lpm (config bar) on node slurm2-a3nodeset1-2.
- Building offline reuse simulator to locate headroom before spending more 2h evals.

## 2026-07-08 ~06:00Z — NEW BEST: v1_exclusive (exclusive L1<->L2 tiering)
- v0_lpm (config) NEGATIVE: ordering doesn't recover hit (0.62=0.62), p99 blows to 21.4s. Ordering dead end.
- s_writeback (config): write_back = write-side exclusive -> hit 0.62->0.73 (+11pp), p99 6326->4581 (-28%).
- v1_exclusive (MECHANISM, commit 21023af6d): write_back + SGLANG_HICACHE_EXCLUSIVE free-host-on-promotion
  -> hit 0.7525 (+2pp over config, +13.1pp over fcfs), p99 4380ms (-30.8% vs fcfs), mean TTFT 863. Lossless.
- Insight: system on steep hit-vs-capacity curve; inclusive write_through wastes device tier (dup of host);
  exclusive tiering reclaims it. Ordering/admission are dead ends. Goodput sweep running.

## 2026-07-08 ~07:15Z — confirmation + goodput curve
- v2_exclusive_solo (self-contained, STOCK write_through flag + SGLANG_HICACHE_EXCLUSIVE=1): hit 0.7517
  (= v1 0.7525 -> reproducible, all-engine attribution). mean TTFT 907 (-21% vs baseline). p99 varies (5420).
- Goodput sweep (600 convs): lambda=4 baseline p99 10094 vs exclusive 8412 (-16.7%). Curve shifts down;
  SLO knee moves right. 5 W&B pts total.
- Contribution: exclusive L1<->L2 KV tiering. Insight: effective-capacity is the lever on the steep
  hit-vs-capacity curve; ordering(v0_lpm)+admission(sim) are dead ends.
