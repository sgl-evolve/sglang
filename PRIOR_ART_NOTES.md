# Prior-art mechanism candidates (for kv-flint-2c), ranked for THIS regime

Regime: saturated, prefill-heavy, host 100% full, ~24% SSD hits, heavy load-back churn (444M),
schedule_policy=fcfs, eviction=lru, mixed_chunk=off. Model hybrid-Mamba (io=direct, layout=page_first_direct forced).

## Strongest candidates (lossless, tractable, novel vs stock sglang)
1. **Balanced / loading-bound-aware prefill batching (Strata "balanced batches")** — 1.8x in Strata ablation.
   Multiturn later-turns are loading-bound: big cached prefix loaded H->D (load_stream) + tiny new suffix
   computed -> load time >> compute time -> GPU stalls. Stock sglang forms prefill batches FCFS by token
   budget, ignoring load:compute ratio. Mix loading-bound reqs with compute-heavy (miss-heavy) reqs so the
   H->D load hides behind compute. Plug-in: scheduler prefill admission loop (scheduler.py ~2861) +
   per-req load vs compute estimate from the radix match (host_hit_length -> load cost; new/miss tokens -> compute).
   Lossless (reorder/co-schedule). Medium difficulty.

2. **Bubble-filling with decode (Strata)** — +3-8%. When a prefill batch is still loading-bound, run a
   decode batch concurrently (decode=HBM-bound, load=PCIe-bound -> low contention). sglang has enable_mixed_chunk
   (off) as a related stock lever; a smarter "defer loading-bound prefill, issue decode" is the mechanism.

3. **Scheduler-aware / look-ahead eviction (CachedAttention)** — 27-31% hit-rate improvement vs LRU.
   Use the waiting queue (future accesses) to avoid evicting soon-to-be-used host/device KV. In my regime
   this shifts SSD hits -> host hits (cheaper), cutting the 24% disk reads. Lossless. Low-medium difficulty.
   Caveat: my total hit rate is near ceiling; this changes hit *tier distribution*, not total.

## Blocked / risky
- GPU-assisted I/O (kernel io-backend + page_first, Strata-IO): blocked by MambaPoolHost (direct/page_first_direct
  only; kernel crashes it). Making kernel/page_first work for Mamba host pool = legit but HARD (rebuild sgl-kernel,
  fix staging-kernel bug). Defer.
- Intermediate-activation restore (HCache), 3D restoration (CacheFlow): large, invasive. Defer.
- Speculative/approx (InfiniGen, ScoutAttention): LOSSY — out of scope (lossless gate).

## Stock config levers to map bottleneck (config-tag, not novelty): schedule_policy=lpm, enable_mixed_chunk,
##   write_policy=write_through_selective, radix_eviction_policy variants, prefetch timeout.

Sources: Strata arXiv 2508.18572; SGLang HiCache blog 2025-09-10; Mooncake FAST'25; CachedAttention ATC'24;
HCache EuroSys'25; CacheFlow; KVFlow.
