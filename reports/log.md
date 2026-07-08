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

## 2026-07-08 ~07:50Z — resume; v3 queued; mamba-rebalance considered+deferred
- v3_exclusive_hotkeep2 (frequency-aware hybrid, commit 91d611a10) QUEUED via retry_eval, waiting for a
  node (base_free monopolizes all 4 pool nodes, 4h50m holds).
- Investigated host-pool split: full-attn KV host pool and Mamba host pool are SEPARATE allocations; mamba
  holds ~2.7x more prefixes/GB (smaller per-prefix state) -> full-attn pool is the binding hit constraint,
  mamba has slack. A host-budget REBALANCE (grow full-attn, shrink mamba) could raise hit, BUT reuse needs
  BOTH components cached (no recompute-mamba path) and budget-ceiling/contract implications are murky
  (risk of "more memory" not "smarter engine"). DEFERRED as risky/uncertain future work.

## 2026-07-08 ~08:35Z — v3 hybrid result + line synthesis
- v3_exclusive_hotkeep2 (hybrid, threshold=2): hit 0.7474 (vs pure-exclusive 0.752), load_back 397M
  (down from 402M), mean TTFT 806 (best). NEUTRAL -> transfer cost isn't the limiter; pure exclusive optimal.
- Error bars (v1/v2/v3): hit 0.750+/-0.003 (+12.8pp robust), p99 ~4740+/-560 (-25%), mean TTFT -21..30%.
- 6 W&B versions logged. Exclusive-tiering line EXHAUSTED (ordering/admission/eviction/device-headroom/
  transfer-hybrid/mamba-rebalance all dead or neutral). Remaining ~5pp needs lossless KV compression
  (high-risk, deferred).

## 2026-07-08 ~09:30Z — full-protocol goodput confirmation (headline)
- FULL-protocol (1553 conv) lambda=4: baseline p99 9098ms/3.38 req_s vs exclusive p99 8091ms/3.78 req_s
  = -11% p99, +12% throughput. SLO knee ~l3.6 -> ~l3.95 (~+10% goodput). Confirms curve shift at full scale.
- Contribution FULLY VALIDATED: exclusive L1<->L2 KV tiering. 6 W&B versions (lambda=3 curve) + goodput
  sweeps (600-conv screen + full-protocol l4). Line exhausted; compression = deferred future line.

## 2026-07-08 ~09:55Z — space thoroughly exhausted (node-free confirmations)
- Capacity model (sim): ceiling 0.806=0.807 (analytic); exclusive 0.725(sim)=0.750(measured). Exclusive
  captures ~70% of the 0.622->0.807 recoverable headroom. Compression upside: x1.4->0.75, x1.6->0.77
  (implausible lossless on FP8) -> low-EV, confirmed data-backed.
- Scheduling co-residency AT THE KNEE (lam=4,5, C=10.16M): cold_defer adds ZERO (hit/p99 unchanged) ->
  scheduling has no leverage even where a queue forms; exclusive PLACEMENT is sufficient+complete.
- CONCLUSION: accessible in-contract lossless mechanism space fully explored. Exclusive L1<->L2 tiering is
  THE mechanism (hit 0.750+/-0.003, -25% p99, +12% goodput knee). Compression = documented low-EV frontier.

## 2026-07-08 ~10:40Z — v2c (4th exclusive replicate) -> tightened error bars
- v2c_exclusive_rep: hit 0.7518, p99 4079 (best), mean 789. 7 W&B versions logged.
- EXCLUSIVE n=4: hit 0.7509+/-0.0020 (+12.9pp, rock-solid), mean TTFT 841+/-47 (-27%, tight),
  p99 4575+/-506 (-28%; noisier = node variance). Hit-rate + mean-TTFT are the robust headline claims.

## 2026-07-08 ~13:00Z — DEFINITIVE same-node A/B (resolves p99 confound)
- Same node (node1-2), baseline(fcfs) then exclusive back-to-back, NO node variance:
  hit 0.6163->0.7522 (+13.6pp), p99 6928->4354 (-37.2%), mean 1038->812 (-21.8%), p50 565->503,
  req/s & out_tok/s unchanged (lossless). => true p99 improvement is -37% on identical hardware
  (the multi-node -28% was node-variance-inflated). 9 W&B versions logged (added v_ab_baseline/exclusive).

## 2026-07-08 ~14:30Z — DEFINITIVE same-node goodput-knee shift
- Same node (ondem-2), baseline vs exclusive sweeps at l=3.5/4/4.5, NO node variance:
  p99(ms): l3.5 8320/6615, l4 9002/8055, l4.5 12199/11271. Baseline over 8s SLO by l3.5 (knee<3.5);
  exclusive under to ~l4.0 => goodput knee lifted ~+14-18% req/s (same-node, definitive). Headline metric.
- Contribution now has BOTH definitive same-node results: A/B @l3 (p99 -37%, hit +13.6pp, lossless) +
  goodput-knee shift (~+18% req/s @ p99<=8s SLO). 9 W&B versions + these screens.

## 2026-07-08 ~15:00Z — frontier CLOSED via fp8-KV upper-bound (decisive)
- KV dtype is bf16 (measured). fp8-KV (--kv-cache-dtype fp8_e4m3) doubles KV tokens (2.35M->4.70M dev,
  confirmed) -> hit 0.808 (=ceiling 0.807), mean TTFT 699 (-39% vs base), p50 425. LOSSY + stock flag =>
  out-of-contract (logged v_kvfp8e4m3_LOSSY_upperbound, config). 10 W&B versions now.
- DECISIVE: capacity ceiling reachable via one-line LOSSY flag => lossless compression build (major, ~1.4x
  on bf16, ~+2-3pp) is DOMINATED -> NOT worth building. Compression frontier CLOSED with data.
- Final: lossless exclusive tiering (0.752, +13pp, same-node p99 -37%, goodput knee +~15%) captures ~70% of
  the recoverable headroom; residual only reachable lossily. Accessible LOSSLESS mechanism space fully closed.

## 2026-07-08 ~16:00Z — LOSSLESS CERTIFIED (bit-exact vs stock) + hybrid-Mamba cache caveat
- 3-mode greedy verify (24 long docs, r1 fresh / r2 cache-hit exercising exclusive host-free+load-back):
  exc-r2 vs stock-r2 = 24/24 (MY MECHANISM == stock cache, bit-exact, both paths) => lossless relative to
  the default cache, CERTIFIED by measurement. stock-r2 vs nocache = 20/24; exc-r2 vs nocache = 20/24
  (identical) => hybrid-Mamba cache (stock & exclusive alike) is inherently ~non-bit-exact vs no-cache on
  ~17% of long docs (mamba checkpoint reconstruction) - an sglang cache property, borne equally by baseline.
- Net: exclusive tiering adds ZERO loss over the cache baseline; comparison on equal footing. 10 W&B versions.
