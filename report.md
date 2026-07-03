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
- **CORRECTION (via /metrics kv_* gauges):** the "full token usage 0.15-0.47" in decode logs is only the
  ACTIVE/protected fraction. The real device pool is ~FULL: `kv_used 489K (21% active) + kv_evictable
  1.85M (79% cached radix) + 8.8K free ≈ 2.347M (100%)`. So **there is NO wasted device headroom** —
  the "promote hot cache to device" lever is DEAD. All 3 tiers are full (working set 19M ≫ device 2.35M +
  host 7.8M = 10.2M ⇒ ~half spills to disk). cached_tokens_total: device 3.6M vs host 7.95M (host serves
  ~2× the device cache-hits). System is compute + KV-movement bound; `timeout` removes the wait_complete
  pipeline stall.
- **RESULT (vs baselines):**

  | version | TTFT mean | TTFT med | TTFT p99 | out tok/s | req/s | hit_rate | L3 frac | TPOT |
  |---|---|---|---|---|---|---|---|---|
  | v0_official | 87615 | 1225 | 270799 | 146.9 | 1.15 | 0.816 | 0.254 | 241 |
  | v0_tuned | 108824 | 1438 | 322801 | 119.3 | 0.93 | 0.821 | 0.259 | 294 |
  | **v1-timeout** | **3526** | 2221 | **32915** | **295.3** | **2.31** | 0.580 | **0.000** | 575 |

  ⇒ **~25× lower mean TTFT, ~8× lower p99, ~2.0–2.5× throughput.** Clears the tuned bar by a wide margin.
- **Lossless:** `timeout` stops blocking on the slow serial L3(disk) path and RECOMPUTES the un-loaded
  prefix at prefill (identical KV ⇒ identical outputs). L3 hits → 0 (disk skipped); device+host still
  serve 58%. "Spend less budget at better TTFT" (charter-sanctioned). Self-audit passed (on-contract, no
  SILENT FALLBACK, exit 0, 0 retracts).
- **Takeaway:** the frozen baseline's `wait_complete` + serialized disk prefetch was catastrophic (87s
  TTFT from pipeline stalls). Not waiting on disk is a huge lossless win. TPOT rose (241→575) because
  recompute-prefill now competes with decode and concurrency is higher — a hint that **prefill/decode
  overlap** and **reducing the 42% recompute** are the next levers.

### v2 — prefetch policy = `best_effort` (vs v1 `timeout`)  [config]
- **Hypothesis:** v1's TTFT median (2221ms) ≈ the `timeout` base (2s) ⇒ L3-needing requests still wait
  ~2s before giving up. `best_effort` waits 0 ⇒ may cut TTFT median/mean further. Lossless (same recompute).
- **Change:** extra arg `--hicache-storage-prefetch-policy best_effort`; code = baseline (commit a52e9a891).
- **Result: NEW BEST.** TTFT mean **2467 ms** (v1 3526), median **1528** (v1 2221), p99 **16641** (v1 32915,
  half the tail!), out 297 t/s, req 2.32/s, hit 0.543, L3 0.0, TPOT 514. vs tuned bar: **~44× lower mean
  TTFT, ~19× lower p99, ~2.5× throughput.** Confirms the hypothesis: zero-wait removes v1's ~2s timeout
  stall → lower median + tail. Lossless (recompute). Logged [config]. Emailed.
- **Takeaway:** best_effort is the winning prefetch policy (dominates wait_complete and timeout). It is the
  base for all subsequent versions.

### INFRA: `--enforce-disable-flashinfer-allreduce-fusion` (all evals v2+)
- The flashinfer allreduce-fusion attempt **intermittently HANGS** server init (v1 attempt-1 and v2
  attempt-1 both hung ~indefinitely post-GDN-init, then crash+disable). Passing
  `--enforce-disable-flashinfer-allreduce-fusion` skips the fusion path entirely → **reliable ~4-min
  loads** (was 10-30 min with hang risk). **Result-equivalent** (fusion always crashes+disables at
  runtime anyway ⇒ serving uses standard allreduce with or without the flag ⇒ TTFT/throughput
  unaffected, only load time). Used for v2 onward; does NOT affect comparability of serving metrics.


### v3 — skip L3 storage write-backups under best_effort  [MECHANISM]  ** NEW BEST **
- **Change:** `UnifiedRadixCache._finish_write_through_ack` — skip `write_backup_storage` when
  `prefetch_stop_policy == best_effort` (commit 5bfd525a1). (Note: the ACTIVE cache class for this
  hybrid-GDN model is UnifiedRadixCache, not HiRadixCache — the first edit was dormant; caught via
  backuped_tokens>0 and fixed.)
- **Why lossless:** under best_effort the disk tier is never READ (storage hits=0), so host->disk
  backups are pure waste; skipping them means a miss recomputes (identical KV) — exactly what
  best_effort already does. Verified: 0 L3 files written (v2 wrote ~770GB/776K files).
- **Result vs v2 (isolates mechanism):** TTFT mean 1331 (v2 2467, -46%), median 801 (-48%), p99 10048
  (-40%), out 387.6 t/s (+30%), req/s 3.03 (+31%, near offered 3.5), hit 0.614 (up from 0.543),
  TPOT 268 (down from 514). vs tuned bar: **~82x lower TTFT, ~32x lower p99, ~3.25x throughput.**
- **Why it works:** eliminating the host_ref pin during async disk-backup unblocks host eviction (host
  99.7% full → faster admission), and freeing the backup thread's CPU/GIL + disk BW speeds
  scheduling+decode (TPOT halved) and improves host cache quality (hit_rate up). Logged [mechanism]. Emailed.

### v3 direction — the disk (L3) tier is WASTED under timeout/best_effort
- Under both policies, L3 storage hits = 0 (best_effort cancels prefetch immediately; timeout's serial
  disk path can't finish in time). So ~42% of prefill is recomputed and the 1.8TB disk tier is idle
  (except wasted write-backups). Candidate mechanisms to reclaim it or cut recompute:
  (a) **async fire-and-forget L3 prefetch** that completes in the background to warm HOST for the next
      multiturn turn (novel; risk: host is full ⇒ eviction churn); (b) `schedule_policy lpm` cache-aware
      batching [config, lossless]; (c) `radix_eviction_policy` lfu/slru [config, lossless]; (d)
      `enable_mixed_chunk` overlap prefill+decode.

### mixed_chunk is code-proven LOSSLESS for this hybrid-GDN model
- `hybrid_linear_attn_backend.init_forward_metadata` ALWAYS calls `Mamba2Metadata.prepare_mixed`
  (mamba2_metadata.py:199), which explicitly splits a batch into `num_prefills` + `num_decodes`
  (`num_decodes = batch_size - num_prefills`, :234), builds `MixedMetadata` for the prefill (chunked
  scan) portion, and mamba.py:556-618 runs prefill+decode in one forward. Mixed prefill+decode is a
  first-class, tested path ⇒ `enable_mixed_chunk` produces identical KV/outputs (lossless by construction;
  will still sanity-check the run). This is the planned **v3** (best prefetch policy + mixed_chunk + the
  flashinfer-disable load flag).

## Strata prior-art analysis (for novel mechanisms)
Strata's scheduler pieces: delay-hit deferral (transient nodes), balanced/bundled batches, bubble-filling,
GPU-assisted I/O, storage prefetch. Assessment for THIS regime:
- **GPU-assisted I/O + storage prefetch**: already in sglang; and disk is skipped anyway → N/A.
- **I/O-aware bubble-filling**: SKEPTICAL — `load_back` (host→device) is already layer-wise overlapped
  (1.1ms, not a stall), and mixed_chunk (v3) covers prefill/decode overlap → marginal.
- **Delay-hit deferral**: low benefit — my 42% recompute is disk-skip, not redundant in-queue prefills.
- **Bundled/prefix-aware batching (`schedule_policy=lpm`)**: PROMISING for this mix — LEval/LooGLE have
  many questions on the SAME long doc; under eviction pressure (skip-disk) a shared doc prefix stays hot
  only if its requests are co-scheduled. lpm sorts the queue by prefix-match ⇒ keeps shared prefixes hot
  ⇒ more device/host hits ⇒ less recompute. Lossless (order doesn't change outputs). Default is fcfs.
  → candidate **v5** (config).

## Execution plan (serial on held node 0-2; all loads use the flashinfer-disable flag)
- **v2** = best_effort (running) — vs v1 timeout, brackets prefetch optimum.
- **v3** = `timeout + --enable-mixed-chunk` — isolates mixed_chunk vs v1 (skip-writes stays dormant
  under timeout). Attacks TPOT 575 / prefill-decode competition.
- **v4** = `best_effort` on the skip-storage-backup commit — isolates skip-writes vs v2 (mechanism).
- **v5** = best combo of the above.
- Note: decode is the throughput limiter (batch ~120, ~300 tok/s of a 122B-A10B MoE); recompute's harm
  is via *competition* with decode (fixed by mixed_chunk), not its volume — so eviction tuning is low value.

## Plan (post-v1)
Base policy = don't-block-on-L3 (timeout/best_effort). The disk tier is skipped; ~42% of prefill is
recomputed (hit_rate 0.58). Next levers, in priority:
1. **v2 best_effort** — zero-wait prefetch (config).
2. **Reduce recompute** — better device+host eviction/retention to raise hit_rate>0.58 (lossless; config
   `radix_eviction_policy` lfu/slru first, then a novel value-aware policy).
3. **Prefill/decode overlap** — `enable_mixed_chunk` to hide recompute-prefill behind decode (TPOT 575→);
   must verify hybrid-Mamba losslessness before logging.
4. **Cache-aware scheduling** — `schedule_policy` lpm vs default fcfs, to batch shared-prefix work.

### v4 — + `enable_mixed_chunk` on best stack  [VOID — reverted]
- **Change:** best_effort + skip-writes + `--enable-mixed-chunk` (commit cfb80ca41).
- **Result: VOID (crashed).** Server hit `ValueError: pool memory leak detected! [full] total=2347200,
  available=5504, evictable=2341888, protected=0` on ALL TP ranks → scheduler crash → only 3847/7037
  requests succeeded (v1/v2/v3 all completed 7037). The reported metrics (out 442 t/s, TTFT 5789) are
  artifacts of the ~45% dropped requests — NOT a valid measurement, NOT logged to the curve.
- **Diagnosis:** `enable_mixed_chunk` breaks the device KV-pool accounting for this hybrid-GDN model
  (v3 = same stack minus mixed_chunk ran clean; skip-writes only touches host->disk backup, not device
  pool). Despite `prepare_mixed` existing, mixed prefill+decode batches corrupt the full-attn pool
  accounting here. **mixed_chunk is unusable for this model — reverted.** Best remains v3-skipwrites.

### v5 — + `schedule_policy=lpm` on best stack  [config]  ** NEW BEST **
- **Stack:** best_effort + skip-writes (mechanism, v3) + lpm. Commit e2f2d6023.
- **Result vs v3 (isolates lpm):** TTFT mean 1255 (v3 1331, -5.7%), median 694 (-13.4%), p99 10207 (~=),
  out 403.1 t/s (+4.0%), req/s 3.15 (+4%), hit 0.618, TPOT 257. Valid (7037/7037, no fallback, lossless).
  vs tuned bar: **~87x lower TTFT, ~32x lower p99, ~3.4x throughput.** req/s 3.15 near offered 3.5.
- **Why:** lpm (longest-prefix-match) co-schedules requests sharing a long prefix (LEval/LooGLE: many
  questions per doc) → shared prefixes stay hot → small recompute reduction. Logged [config]. Emailed.

## Status summary (best = v5)
Curve: v0_official 87615 → v0_tuned 108824 (bar) → v1 3526 → v2 2467 → v3 **1331 (mechanism)** →
v5 **1255** ms mean TTFT. The system now serves req/s 3.15 vs offered 3.5 (near saturation-free).
Big wins: (1) don't block on the slow L3 disk tier [best_effort, config]; (2) don't WRITE a tier you
never read [skip-writes, MECHANISM — unblocks host eviction]. Remaining cost: ~38% prefill recompute
(disk skipped; capacity-bound) competing with decode. mixed_chunk (the natural overlap fix) crashes here.

### v6 — + skip storage prefetch-ISSUE under best_effort  [MECHANISM]  ** NEW BEST **
- **Change:** `UnifiedRadixCache.prefetch_from_storage` returns immediately when best_effort (commit
  bb0800d5f). Read-side mirror of v3's write-side skip. Full stack: best_effort + skip-writes +
  skip-prefetch-issue + lpm.
- **Result vs v5 (isolates this mechanism):** TTFT mean 1205 (v5 1255, -4%), median 649 (-6.5%),
  p99 8702 (v5 10207, **-15% tail**), out 412.3 t/s (+2.3%), req/s 3.22 (+2.2%), hit 0.624. Valid
  (7037/7037, lossless). Max queue 27 (v5 32, v3 37). vs tuned bar: **~90x lower TTFT, ~37x lower p99,
  ~3.5x throughput** (req/s 3.22 vs offered 3.5).
- **Why lossless / why it works:** under best_effort the prefetch is cancelled next step (storage-hit
  frac = 0.0), so issuing it loads ~0 tokens but pins a host node + allocs host pages + may evict_host
  per request — churn contending with the full host tier. Skipping it (miss=recompute, unchanged
  outputs) removes that churn. Together with skip-writes: "fully bypass the provably-dead L3 tier under
  best_effort." Logged [mechanism]. Emailed.

### v7 — + `radix_eviction_policy=slru`  [config]  ** NEGATIVE — reverted **
- **Result vs v6 (isolates eviction policy):** WORSE. TTFT mean 1707 (v6 1205, +42%), median 1093 (+68%),
  out 366 t/s (-11%), **hit_rate 0.347 (v6 0.624 — crashed)**. Valid run (7037/7037, no fallback).
- **Why:** SLRU's protected segment (hit≥2) over-protects older multi-hit nodes and evicts recent
  single-hit nodes too aggressively; this workload is recency-heavy (multiturn), so plain LRU is the
  right policy. **Eviction-policy tuning does NOT help — LRU is well-suited.** Confirms the ~38%
  recompute is capacity-bound (working set 19M ≫ device+host 10.2M), not an eviction-policy problem.
  Reverted to LRU. Best remains v6.
