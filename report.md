# kv-lynx-4d2 — sglang HiCache KV-cache evolution report

Independent researcher. Branch `evolve/kv-lynx-4d2`. W&B run `kv-lynx-4d2` in `sgl-evolve`.
Model Qwen3.5-122B-A10B-FP8 (hybrid-Mamba), TP=8, fixed 3-tier protocol (GPU + 768 GB host +
1.8 TB disk L3), real-text Mooncake 1:1:1 mix, λ=3.5, mc=128, 1553 prompts. Headline = mean TTFT.

## Baselines (provided; logged, not re-run)
| version | mean TTFT | median TTFT | p90 | p99 | hit_rate | L3 hit frac | host_util | out tok/s |
|---|---|---|---|---|---|---|---|---|
| v0_official | 87615 ms | 1224 ms | 243466 ms | 270799 ms | 0.816 | 0.254 | 1.00 | 146.9 |
| v0_tuned    | 108824 ms | 1438 ms | 295413 ms | 322801 ms | 0.821 | 0.259 | 1.00 | 119.3 |

Both use identical resolved args (ctx 262144, mem-frac 0.85, hicache_size 96, page 64, io=direct,
layout=page_first_direct, write_through, prefetch=wait_complete). v0_tuned is *worse* on this run
(run-to-run variance / a config that didn't pan out); I target the harder bar — **below 87.6 s mean
TTFT** — which beats both.

## The bottleneck (from baseline metrics + code study)
- **Median TTFT 1.2 s but mean 87.6 s** — a catastrophic tail (p99 271 s). Most requests are fast;
  a large minority stall for minutes.
- **Host RAM 100% full** (host_util ≈ 1.0); **~25% of cache hits come from disk L3** (20.7 M tokens
  read from disk). Working set (~19 M tok) >> GPU (2.35 M) + host (~8.4 M), so ~8 M tok live only on
  disk — genuine 3-tier spill.
- **Prefetch policy = `wait_complete`**: a request whose matched prefix has disk-resident segments
  cannot enter the running batch until its *entire* storage prefetch finishes
  (`scheduler.py:2882` gates on `check_prefetch_progress`; `hi_mamba_radix_cache.py:1672`
  `can_terminate_prefetch` requires `completed_tokens == full`). So disk-read time + queue wait land
  directly in TTFT.
- **Disk IO path is single-threaded & serial per rank**: one `prefetch_io_aux` thread per rank
  (`cache_controller.py:967`) pulls one operation at a time; `HiCacheFile.batch_get`
  (`hicache_storage.py:401`) is a serial list-comprehension of blocking `open()+readinto()`.

## Screens (free; never on the curve)
### S1 — Parallel disk IO (cold-read microbench on /mnt/localssd md NVMe) — NEGATIVE as headline
Mechanism built (commit 4ce727b6): fan per-page reads/writes across a bounded thread pool in
`HiCacheFile`. **Lossless** (round-trip test: byte-identical, serial==parallel, order preserved).
- single process: **0.95 GB/s (serial) → 6.4 GB/s (W=16)** = 6.7× — big per-process headroom.
- **8 processes (= 8 TP ranks) concurrent: W=1 6.05 GB/s vs W=8 6.54 GB/s = +9% only.**
- ⇒ The array **saturates ~6.6 GB/s** and the baseline's 8 serial ranks already ~saturate it.
  Per-rank parallelism is marginal in steady state. Kept (lossless, free, tunable via
  `extra_config.hicache_io_workers`) but **not a headline win**.
- **Key implication:** disk read bandwidth is a *hardware ceiling the baseline already hits*
  (~6.6 GB/s ≈ 72k tok/s aggregate). Average disk demand over the run ≈ 3.3k tok/s (20.7 M tok /
  ~105 min) ≈ **4.5% utilization** → the tail is a **bursty queueing** phenomenon, not raw bandwidth.
  Real levers: (a) **don't wait on disk** (recompute — best_effort/timeout, and smarter adaptive
  versions), (b) **reduce disk-read/-write demand** (host admission/eviction, write policy),
  (c) **overlap disk load with prefill**.

## Versions (full evals — every one logged, kept or reverted)
### v1-timeout — `config` — RUNNING
Change: `--hicache-storage-prefetch-policy timeout` (default cap min(30 s, 2 s + 0.1 s/1k tok)).
Hypothesis: capping the per-request disk wait collapses the p99 tail (lossless — recompute of the
not-yet-loaded tail yields identical KV). Tests the "don't wait on disk" lever. Result pending.

## Next
Pick v2 mechanism from v1 result + live disk iostat: leading candidate is a **congestion/deadline-
aware adaptive prefetch** (decide wait-vs-recompute per request from live disk-queue depth & GPU
headroom) that dominates the static timeout config — genuine novelty beyond tuning.
