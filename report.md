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
### S2 — Whole-dir `os.scandir` per prefetch hit-query (the smoking gun) — STRONG POSITIVE
The hybrid-Mamba prefetch path always takes `batch_exists_v2`
(`hybrid_cache_controller.py:600`, `pool_transfers` present), whose
`_collect_existing_component_keys` (`hicache_storage.py:487`) did an **`os.scandir` over the
entire L3 storage dir**, filtered to a ~256-name target set — on **every** prefetch hit-query, on
**every** rank. The L3 dir holds ~millions of page files at steady state (1.8 TB of ~768 KB pages),
so the scan is **O(total files on disk)** and **degrades as L3 fills** — a growing tax directly on
the prefetch/TTFT critical path. Microbench (cold dir, 256-key target):
| files on disk | scandir+filter | direct `isfile` | speedup |
|---|---|---|---|
| 200k | 64.3 ms | 0.44 ms | 147× |
| 500k | 161.8 ms | 0.43 ms | 373× |
| 1M | 324.7 ms | 0.44 ms | 742× |
Linear in file count ⇒ **~650 ms/hit-query at steady-state ~2M files**. This explains the extreme,
*growing* tail (median 1.2 s but p99 271 s): late in the run, every prefetch pays ~0.65 s of
directory scan before it can even read, and they serialize on the single per-rank prefetch thread.
**Fix (commit bbf51108):** probe the specific target files with `os.path.isfile`, O(len(target_files)).
Identical result set ⇒ **lossless** (correctness test asserts set-equality vs the scandir impl).
Helps under *any* prefetch policy. **This is my v2 headline mechanism.**

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
### v1-timeout — `config` — RUNNING (on node 1-2; very slow first-time Triton compile at init)
Change: `--hicache-storage-prefetch-policy timeout` (default cap min(30 s, 2 s + 0.1 s/1k tok)).
Hypothesis: capping the per-request disk wait collapses the p99 tail (lossless — recompute of the
not-yet-loaded tail yields identical KV). Tests the "don't wait on disk" lever. Result pending.

### v2-scandirfix — `mechanism` — QUEUED (commit bbf51108, includes parallel IO 4ce727b6)
Change: O(keys) `os.path.isfile` L3 hit-query (drop the whole-dir scandir) + parallel disk IO.
Hypothesis: removing the ~0.65 s/hit-query directory scan that grows with L3 fill collapses the
growing TTFT tail, independent of prefetch policy. Lossless (byte-identical set + round-trip tests).
Highest-confidence win. Runs after a good pool node frees.

## Next
After v2, re-measure the bottleneck from the new metric profile. Candidate v3+: congestion/deadline-
aware adaptive prefetch (wait-vs-recompute), SJF prefetch ordering, or write_through_selective to cut
disk write contention — chosen from evidence, prizing novelty over tuning.
