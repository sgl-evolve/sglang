# kv-heron-e29 — sglang KV-cache evolution report

Independent researcher. Branch `evolve/kv-heron-e29`. W&B run `kv-heron-e29` in `sgl-evolve`.
Bar to beat: **v0_tuned**. Headline metric: **mean TTFT** (lower better), lossless required.

## Baselines (reference points, not re-run)

| version | TTFT mean (ms) | TTFT p99 (ms) | out tok/s | hit_rate | L3 hit frac | host_util |
|---|---|---|---|---|---|---|
| v0_official | 87615.4 | 270798.5 | 146.89 | 0.816 | 0.254 | 1.000 |
| v0_tuned    | 108824.3 | 322801.1 | 119.30 | 0.821 | 0.259 | 1.000 |

Both share identical `resolved_args` (ctx 262144, mem-frac 0.85, hicache_size 96, tp 8,
io_backend direct, layout page_first_direct, write_through, prefetch **wait_complete**, page_size 64).
v0_tuned is *slower* than v0_official → run-to-run variance is large and the regime is **saturated**:
at λ=3.5 / max-concurrency 128 the ~19M-token working set overflows all tiers, TTFT is
**queueing-dominated** (mean ~87s but p50 ~1.2s → heavy bimodal tail). Throughput is the dominant
lever on the headline metric.

## Bottleneck analysis (from code + data + prior art)

- Host tier is **full and thrashing**: `host_util≈1.0`, `evict_tokens 574M`, `disk_read 20.7M`,
  **L3/disk = 25.4% of cache hits**. The disk (L3) tier is the slow path.
- Prefetch is gated with **wait_complete** (`scheduler.py:2881`): a request is *skipped* every
  scheduler step until its full storage prefetch completes → long-prefix/L3 requests pile up in the
  queue = the TTFT tail.
- The file backend's storage reads are **serial + single-threaded**: `HiCacheFile.batch_get`
  (`hicache_storage.py:401`) is a serial list-comprehension of synchronous `readinto()`
  (`:391-393`), all executed in **one** `prefetch_io_aux_thread` (`cache_controller.py:967`). On NVMe
  this uses queue-depth ≈1, wasting most SSD bandwidth.
- Prior art agrees: HiCache blog names "latency of moving data from slower to faster tiers" as THE
  bottleneck and storage as "higher, less predictable latency"; Strata (2508.18572) names "fragmented
  I/O from paged layouts preventing full bandwidth use" and schedulers that leave systems
  "loading-bound rather than compute-bound."

## Screened-out ideas (never full-evaled — negative screens)

### S1 — parallel storage (L3) page reads  [screened out]
- **Idea:** `HiCacheFile.batch_get` reads pages serially in one `prefetch_io_aux_thread`
  (`hicache_storage.py:401`, `cache_controller.py:967`). Parallelize with a ThreadPoolExecutor to keep
  NVMe queue-depth high. Implemented + unit-tested lossless (byte-identical, order-preserved); saved to
  `screens/parallel_reads.patch` (git stash).
- **NVMe microbench (job 18121):** cold-read serial→parallel speedup peaks at **8 threads** — 64KB
  3.8×, 256KB 4.6×, 1MB 4.1× — then flat/worse (device-bandwidth-bound).
- **Why screened out:** TP=8 ⇒ **8 rank processes each already read their shard with 1 serial thread =
  ~8-way node concurrency**, which already saturates the NVMe knee (≈8). Within-rank parallelism (→64)
  is past the knee = no gain. Independent check: baseline `prefetched_tokens 166M`/105min ≈ 410
  pages/s/rank ≈ **~15% of single-thread device capacity** → disk bandwidth is NOT the sustained
  bottleneck. Screen saved a wasted 2h eval.

## Key verified facts (for choosing mechanisms)

- **Throughput-limited, not latency-limited:** server completes **1.15 req/s** vs 3.5 offered → mean
  TTFT 87s is the resulting backlog. The lever is **steady-state req/s** (decode + admission), not
  per-op speed.
- **load_back (host→device) is NOT a stall:** `cache_controller.load()` enqueues the copy on a
  `load_stream` with per-layer completion events (layer-wise overlap); `load_back_mean_ms=1.1ms`.
  (Refutes an agent hypothesis that load_back blocks prefill admission.)
- **Retraction is the device-pressure signal to watch:** `scheduler.py:3054` logs
  `"KV cache pool is full. Retract requests."` + `num_retracted_reqs` metric. Frequent retraction ⇒
  decode requests get their KV dropped and recomputed = wasted throughput.

## Versions

### v1 — prefetch policy = `timeout` (vs baseline `wait_complete`)  [config, diagnostic]
- **Hypothesis:** `wait_complete` gates each L3-hit request in the queue until its full prefetch
  completes (`scheduler.py:2881`). `timeout` (the code default) bounds the wait to
  `min(30s, 2s+0.1s/Ktok)` then proceeds, recomputing the un-fetched suffix — lossless
  (`check_prefetch_progress` inserts the partial prefix; prefill recomputes the rest). Diagnostic:
  does *not blocking indefinitely on L3* raise throughput / cut the TTFT tail?
- **Change:** none (extra arg `--hicache-storage-prefetch-policy timeout`); code = baseline (commit 953d8e197, sglang unchanged).
- **Run notes:** cold-workspace first load was ~29 min (DeepGEMM/kernel JIT compile via cicc/nvcc/ptxas
  → `~/.cache/deep_gemm`, NFS-persistent → future loads ~11 min). flashinfer allreduce-fusion crashes
  and auto-disables at runtime (recoverable). Ran on self-held certified node 0-2 (escaped pool contention).
- **Result vs baseline:** [bench running].
- **LIVE BOTTLENECK DIAGNOSIS (mid-run, the real value of v1):**
  - Running decode batch = **~110-128** (full offered concurrency 128) — NOT the ~35 I mis-inferred from
    baseline out_tok_s/TPOT. The server fully uses the offered concurrency.
  - **Device KV pool only ~30-45% used** (`full token usage`) — LARGE headroom; **NOT device-KV-bound.**
    Mamba state pool ~30-35%. So a lossless "fit more active KV on device" mechanism is NOT the lever.
  - **0 retract events** — no device-pressure thrashing (refutes the retract-mitigation hypothesis).
  - gen throughput fluctuates ~145-940 tok/s; prefill input ~22-36K tok/s; small queue early.
  - ⇒ Throughput/TTFT is limited by **pipeline efficiency** (compute for batch~128 + prefill steps
    interrupting decode with `enable_mixed_chunk=False` + KV-movement stalls as tiers fill), NOT memory,
    NOT L3 read bandwidth (S1). The TTFT tail = periodic pipeline stalls (long-doc chunked prefills
    monopolizing forwards; KV load/prefetch waits).
- **v2 candidates (data-grounded):** (a) `enable_mixed_chunk` [config] — overlap prefill+decode to kill
  the decode-pause bubble (verify hybrid-Mamba support); (b) scheduling to reduce long-prefill
  monopolization / bubble-fill; (c) reduce KV-movement stalls on the load/offload path.
- **Takeaway:** [pending bench completion].
