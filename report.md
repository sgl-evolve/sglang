# kv-heron-e29 — sglang KV-cache evolution report

Independent researcher. Branch `evolve/kv-heron-e29`. W&B run `kv-heron-e29` in `sgl-evolve`.
Bar to beat: **v0_tuned**. Headline metric: **mean TTFT** (lower better), lossless required.

## Executive summary (for a skeptical maintainer)

On the fixed protocol (Qwen3.5-122B-A10B-FP8 hybrid-GDN MoE, TP8, 3-tier HiCache, real-text
ShareGPT+LEval+LooGLE mix at λ=3.5 / max-concurrency 128), **mean TTFT drops from the tuned baseline's
108,824 ms to ~790 ms — ~138×, losslessly — with ~3.5× throughput** (out 119→451 tok/s, req 0.93→3.52/s
= the offered λ, i.e. the system now KEEPS UP with load) and p99 322,801→~5,000 ms. Five lossless changes,
in FIVE stages: three that neutralize the disk-tier pathology (below), then two novel **cost-aware**
policies that cut the residual recompute — **(4) cost-aware EVICTION** and **(5) SJF cost-aware SCHEDULING**
— which take the best from ~1,142 ms to ~790 ms (−31%). The two new mechanisms are orthogonal and stack
(attribution: SJF −24% alone [pure scheduling], eviction −16.5% alone [pure caching]).

**Root cause of the baseline pathology:** the frozen prefetch policy `wait_complete` blocks every
L3(disk)-hit request in the scheduler queue until its full storage prefetch completes; the file backend's
disk path is serialized (one `prefetch_io_aux_thread`, synchronous `readinto`) → the pipeline stalls →
87–109 s mean TTFT (queue-dominated).

**Three lossless changes** (outputs unchanged — a recomputed KV block is numerically identical to a
loaded one; verified same 7037/7037 successful requests, no quality gate tripped):
1. **[config] `--hicache-storage-prefetch-policy best_effort`** — never block on the disk tier; recompute
   the un-cached suffix at prefill (cheap on 8×H100). 108824→2467 ms, throughput ~2.5×. Storage-tier hits
   become 0 — the disk tier is effectively unused, i.e. we spend *less* of the memory budget at far better
   TTFT (charter-sanctioned).
2. **[MECHANISM] skip L3 write-backups under best_effort** (`UnifiedRadixCache._finish_write_through_ack`)
   — the disk is never read, so its writes (~770 GB / 776k page-files per run) are pure waste; worse, each
   backup pins a host node (`host_ref`/`protect_host`) until its async write acks, blocking host eviction
   while the host tier is 99.7% full. Skipping frees host eviction → 2467→1331 ms (−46%), throughput +30%,
   hit-rate 0.54→0.61, TPOT 514→268.
3. **[MECHANISM] skip the L3 prefetch-ISSUE under best_effort** (`UnifiedRadixCache.prefetch_from_storage`)
   — the prefetch is cancelled the next step (loads ~0 tokens) yet still allocs/evicts host per request
   (same host-contention class). Skipping → −4% TTFT, −15% p99.
   Plus **[config] `--schedule-policy lpm`** (prefix-bundling for the many-questions-per-doc mix): small gain.

**Two new cost-aware mechanisms attack the residual ~38% recompute (capacity-bound) by making misses
CHEAPER and serving them SMARTER — not by adding memory (impossible: hicache_size=96 is frozen+asserted):**
4. **[MECHANISM] recompute-cost-aware EVICTION** (`--radix-eviction-policy costaware`, new `CostAwareStrategy`)
   — under best_effort a miss is recomputed on-GPU, and prefill is O(L²)-attention-dominated for LONG
   prefixes (LEval/LooGLE 100k+), which are also heavily reused; so protect DEEP prefixes from eviction
   (evict shallow/cheap first, LRU within, depth-bucket threshold 8192 — swept peak). 1142→~954 ms
   (−16.5%), hit 0.623→0.681. Binary bucket beats graded tiers (v22) and other thresholds (inverted-U).
5. **[MECHANISM] SJF cost-aware SCHEDULING** (`--schedule-policy sjf`, new policy) — sglang's `lpm`
   reverts to FCFS once the queue > 128 (our saturated regime) → scheduling is NOT cost-aware under load.
   SJF sorts the waiting queue by ascending UNCACHED prefill work (cheap-first) at any queue size → the
   queue drains faster → −24% alone (the bigger lever), no starvation (P99 also drops). Combined with (4):
   **~790 ms, hit 0.681, ~138× below the bar.**

**Unifying principle for the two mechanisms:** under a non-reading prefetch policy the disk tier is
provably dead, so BOTH its writes and its read-issue are pure host-contending churn — eliminate both,
losslessly. (Suggested upstream: auto-disable storage backup + prefetch-issue when the effective read
policy never consumes storage.)

**Negatives (kept):** `enable_mixed_chunk` crashes (device KV-pool accounting leak on this hybrid-GDN
model) — void; `write_through_selective` (hit 0.62→0.40) hurts; **capacity levers are dead** — mamba→KV
realloc (`--max-mamba-cache-size`) is NEUTRAL (residual is host-capacity-bound, not device) and growing
the host tier is forbidden+asserted; **parallel L3 disk reads + wait_complete is WORSE** (churns the
frozen host cache; disk can't beat recompute even at 4× parallel — re-confirms skip-prefetch); cost-aware
eviction TIERING and non-8192 thresholds are worse (binary d8192 is the peak). **Remaining floor** after
all five changes (~790 ms): the unavoidable long-context FIRST-TOUCH recomputes (P99 ~5 s) — the only
lever left is prefill/decode overlap (mixed_chunk), which is blocked by the crash above AND carries a
silent-losslessness risk, so it is declined (lossless-above-all).

**Variance caveat:** the two provided baselines differ ~24% at identical config, so the *big* wins
(best_effort, skip-writes) are unambiguous but the last few-% incrementals (lpm, skip-prefetch) are
small-and-directional (supported by a monotonic curve + monotonically-falling max-queue). **v9 = re-run of the best config gave 1089 ms vs v6's 1205 ms → my
run-to-run variance is ~10%**, so lpm/skip-prefetch (~4% each) are within noise (directional, not
definitive); best_effort (−97%) and skip-writes (−46%) are far beyond it.

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
  **[OVERTURNED by later results — see exec summary: mixed_chunk crashes here (unusable), so recompute is
  NOT hidden behind decode; recompute VOLUME then dominates TTFT and recompute-COST-aware eviction
  (`costaware` d8192) is a −16.5% headline win, not low value. This early note assumed mixed_chunk would
  work.]**

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

## Honesty caveat — run-to-run variance
The two provided baselines (v0_official 87615 vs v0_tuned 108824) are IDENTICAL config yet differ ~24%,
so single-run TTFT has high variance. Implications for my curve:
- **Big wins are unambiguous** (far beyond noise): best_effort/timeout vs baseline (−97%, ~25-44×);
  skip-writes vs best_effort (−46% TTFT, +30% throughput).
- **Incremental config/mechanism gains are small-and-directional** (lpm ~−4%, skip-prefetch ~−4%,
  each within the ~24% baseline spread). Evidence they are real-not-noise: the curve is **monotonic**
  across 5 versions and a secondary signal (max queue-depth) falls consistently (v3=37 → v5=32 → v6=27),
  with a mechanistic explanation for each. But I do not over-claim their magnitude.
- Negatives are clear: slru −42% (worse), mixed_chunk crash (void).

### v8 — + `write_through_selective` (device→host backup only for hot nodes)  [config]
- Stack: best_effort + skip-writes + skip-prefetch + lpm + write_through_selective. Compare vs v6.
- **Result: NEGATIVE.** TTFT mean 1460 (v6 1205, +21%), median 931 (+43%), p99 7932 (-9%), out 385 t/s
  (-7%), hit_rate 0.40 (v6 0.62 — dropped). Valid (7037/7037). write_through_selective backs up fewer
  nodes to host → smaller effective host cache → more recompute. **write_through (default) is best.**
  Reverted. CONFIG SPACE NOW FULLY MAPPED: best_effort (win), write_through (best), LRU (best), lpm
  (small win); only best_effort + the 2 skip-mechanisms help.

### v9 — repro of best config (= v6: best_effort + skip-writes + skip-prefetch + lpm)  [config]
- **Purpose:** reproducibility / variance check. Same config as v6, commit 9012478df (sglang code == v6).
- **Result:** TTFT mean **1089** ms (v6 1205), median 623, p99 6680, out 428.9 t/s, req 3.35/s, hit 0.622.
  Valid (7037/7037, no fallback). **v9 vs v6 = −9.6% for identical config ⇒ run-to-run variance ~10%.**
  Confirms the best config is reproducible in the ~1.1–1.2 s range (~90–100× below the tuned bar), and
  that the small incrementals (lpm, skip-prefetch) sit within variance while best_effort + skip-writes
  are the definitive wins.

### v10 — + `num_continuous_decode_steps=2`  [config]  ** NEUTRAL **
- Result: TTFT mean 1191 (n=1 runs 1089-1205), median 620, out 424 t/s, req 3.32. Within ~10% variance
  of the ncds=1 best → NEUTRAL. Confirms decode is not step-starved (throughput already near offered 3.5).
  Valid (7037/7037, no fallback).

## Conclusion (at the lossless ceiling for this protocol)
Config space fully mapped; best config **best_effort + skip-L3-writes + skip-L3-prefetch-issue + lpm**
reproducibly delivers **mean TTFT ~1.1-1.2 s (~90-100× below the tuned bar), p99 ~7-9 s, out ~410-430
tok/s (~3.5×), req/s ~3.2-3.35 (near the offered 3.5)** — losslessly.
- **Definitive wins** (≫ the ~10% run-to-run variance): `best_effort` prefetch policy (−97% TTFT) and the
  **skip-L3-writes MECHANISM** (−46% TTFT, +30% throughput). Unifying insight: on this protocol the L3
  disk tier is a *net liability* (serialized/slow, and its writes+prefetch contend with the 99.7%-full
  host tier); under a non-reading policy it is provably dead, so eliminate its writes AND read-issue,
  losslessly.
- **Small/within-variance:** skip-prefetch-issue, lpm.
- **Dead ends:** mixed_chunk (crashes — device pool leak), slru/write_through_selective (hurt hit-rate),
  parallel-L3-reads (ranks already saturate NVMe). Residual ~38% prefill recompute is capacity-bound
  (working set 19M ≫ device+host 10.2M); the natural fix (prefill/decode overlap) needs a working
  mixed_chunk, which is blocked by an sglang bug on this hybrid-GDN model.
- **Losslessness** is by construction: a recomputed KV block is numerically identical to a loaded one,
  and skipping never-read writes/prefetch changes no computed value (outputs differ only by the same
  batching-order fp nondeterminism present in the baseline). All kept runs completed 7037/7037.

## mixed_chunk crash — root-caused (bug report for maintainers), not fixed (integrity)
Deep investigation of the v4 crash (`pool memory leak [full] ... +192 tokens = 3 pages`, page_size 64):
`enable_mixed_chunk` calls `running_batch.prepare_for_decode()` (allocates a KV slot per decode req)
BEFORE `mix_with_running`, but decode reqs in a MIXED batch are skipped by
`batch_result_processor.py:241` (`req in batch.decoding_reqs`) → their per-step KV accounting isn't
reconciled → ~3 pages/occurrence over-count the full-attn pool until the invariant trips
(`scheduler.py:3000` prepare_for_decode → `schedule_batch.py:2641` alloc_for_decode; skip at
`batch_result_processor.py:241`). This is a real sglang bug on the hybrid-GDN + mixed_chunk path.
**Decision: NOT fixing it.** The invariant is protecting correctness (192 tokens marked *available* may
still hold live KV); a fix rewrites core mixed-batch allocation and could corrupt decode KV (lossy) with
no clean way to verify losslessness (batching-order nondeterminism), for an uncertain reward (system is
already near offered load). Integrity (lossless-above-all) outweighs the speculative gain. mixed_chunk
stays reverted; the prefill/decode-overlap lever is left as future work pending an upstream fix.

## FINAL STATE
Best (reproducible): **best_effort + skip-L3-writes(mech) + skip-L3-prefetch-issue(mech) + lpm** →
mean TTFT ~1.1–1.2 s (~90–100× below tuned bar), p99 ~7–9 s, out ~410–430 tok/s (~3.5×), req/s ~3.2–3.35.
All lossless. W&B run `kv-heron-e29` has the full curve (v0→v10) + artifacts. Config space fully mapped;
at the lossless ceiling for this fixed protocol/budget.

### v12 — page_size=32 (finer prefix matching)  [config]  ** NEUTRAL **
- Result: TTFT mean 1131 (best band 1089-1205, mean 1156), hit 0.629 (best ~0.621 — marginally higher
  from finer matching but within noise). NEUTRAL. Valid (7037/7037, no fallback; page_size=32 took
  effect — the disk O_DIRECT-alignment constraint is moot since the disk tier is unused under best_effort).
  Confirms the ~38% recompute is capacity-bound (working set ≫ cache), not prefix-boundary-bound.

## Config exploration COMPLETE
Mapped every non-frozen policy/knob: prefetch (best_effort=win), write (write_through=best;
selective worse; write_back = Mamba-incompatible), eviction (lru=best; slru worse), schedule (lpm=small
win), num_continuous_decode_steps (neutral), max_prefill_tokens (neutral), page_size (neutral).
Running batch already saturates offered concurrency (128), so admission knobs can't help; throughput is
compute-bound + recompute is capacity-bound. The two novel MECHANISMS (skip-writes, skip-prefetch) +
best_effort are the wins. At the lossless ceiling for this fixed protocol.

### v13 — best-config robustness repro (via shared pool)  [config]
- Result: TTFT mean 1098, median 583, out 431.6 t/s, req 3.37/s, hit 0.623. Valid (7037/7037, no fallback).
- **Robustness: best-config TTFT_mean over 6 runs = [1205,1089,1191,1139,1131,1098] → 1142±43 ms
  (CV 3.8%).** Rock-solid ~95× below the tuned bar, reproduced across runs (and node — v13 via the shared
  pool, likely a different certified node). The headline is well-characterized and low-variance.

### v15 — mamba→KV memory reallocation (--max-mamba-cache-size 700)  [config]  ** NEUTRAL — closes the capacity-lever direction **
- **Hypothesis:** the ~38% recompute residual is capacity-bound; the Mamba state pool is oversized
  (default `max_mamba_cache_size=1350` → ssm_state **23.75 GB/rank**, but observed `mamba usage ≈0.42`).
  Shrinking it to 700 (still ≫128 concurrency) frees ~11.4 GB/rank which the engine reallocates to the
  device KV pool — **in-budget** (frozen `mem_fraction_static=0.85` unchanged) and **lossless** (cache
  size cannot change outputs).
- **Mechanism verified live:** device KV pool grew **2,347,200 → 3,363,136 tokens (+43%)**
  (K/V 13.43→19.24 GB); mamba ssm_state 23.75→12.32 GB. hicache_size stayed 96 (frozen assert passed).
- **Result:** TTFT mean **1106 ms** (best band 1142±43), hit_rate 0.6201, out 432 t/s. **NEUTRAL**
  (within noise; vs v13 best 1098 ms / hit 0.6234 it is marginally *worse*). Valid (7037/7037, no fallback).
- **Why neutral (definitive):** the +43% device KV only *shifted* the hit mix host→device
  (hit_device 0.42→0.46, hit_host 0.58→0.54) — same **total** hit_rate (~62%), same TTFT. The host tier
  runs at **host_util ≈ 1.00 (100% full)** and is HARD-FROZEN at 96 GB (eval asserts it). The device tier
  is only ~22% of total attn-KV cache, so growing it +43% lifts total effective cache only ~+9% — too
  small to dent the miss rate. **The residual recompute is bound by the frozen host tier, not the device
  pool.** load_back is already ~1.2 ms (negligible), so moving hits to device buys nothing.
- **Consequence:** the in-budget capacity levers are now exhausted — host is frozen+saturated, and GPU
  realloc (mamba→KV) is neutral. Any further TTFT reduction must come from using the *fixed* cache more
  effectively (admission/eviction that lowers per-miss recompute cost), not from more capacity.

## EVAL CONTRACT (learned v15) — what is a legitimate mechanism
`eval.sh` FORBIDS (rc=5) and POST-LAUNCH-ASSERTS the budget: `hicache_size==96`,
`mem_fraction_static==0.85`, `context_length==262144`, `tp_size==8`. So **growing the host L2 tier or
total memory is impossible by any means** (flag or engine default → assert aborts) — and rightly so, it
is a budget increase, not an algorithm. Legitimate levers = the tunable policy flags
(prefetch/write/io-backend/mem-layout/page-size), non-forbidden flags (schedule, eviction,
**max-mamba-cache-size**, mamba-full-memory-ratio), and NEW engine mechanisms. The only in-budget
*capacity* move is reallocating the frozen GPU pool (mamba↔KV) — shown neutral above.

### v16 — parallel L3 disk reads (16-thread pool in HiCacheFile) + wait_complete  [mechanism]  ** NEGATIVE (aborted early; clear signal) — re-confirms skip-prefetch **
- **Hypothesis:** the L3 disk (NVMe) is slow because sglang reads it SERIALLY (queue-depth 1); a standalone
  bench showed **~4× speedup** at 8–16 threads (64KB–1MB pages). If parallelized, the disk tier (1800 GB)
  could hold the full ~19M working set → hit ~100% → beat best_effort's 38% recompute. Patch: a 16-thread
  ThreadPoolExecutor in `HiCacheFile.batch_get` (reads target disjoint host buffers → lossless). Committed
  438ba2e02; verified live ("HiCacheFile: page IO parallelism = 16 thread(s)").
- **Result (aborted at turn ~800/7037, decisive):** with wait_complete (the policy that actually USES the
  disk), caching is *worse* than best_effort — prefill `#cached-token` is **~0** almost everywhere (vs
  best_effort's 0.62 device+host hit) and throughput is **~3× slower** (~50 vs ~150 turns/min).
- **Why (the real lesson):** turning the storage-prefetch path ON (wait_complete) **churns/evicts the
  frozen 96 GB host cache** to stage L3 loads — destroying the device+host radix hits that best_effort
  relies on — *and* blocks on them. Parallel reads make the disk 4× faster but the bottleneck is cache
  **churn + blocking**, not read bandwidth, so 4× buys nothing. This directly **re-confirms the
  skip-prefetch mechanism (v6)**: not touching the L3 tier under best_effort is optimal. The disk tier is
  not worth using for this workload/budget even with a parallel backend.
- **Ops note:** aborting a wait_complete run mid-flight left a D-state/OOM-stuck server zombie (in-flight
  parallel disk IO) — don't abort disk-heavy runs; let them finish. The collision guard auto-skips the
  wedged node until it self-reaps.

## DIRECTIONS EXHAUSTED (v15 capacity, v16 disk-tier)
Both remaining avenues to cut the ~38% recompute are now closed with evidence: (1) in-budget capacity
realloc (mamba→KV, v15) is neutral — the residual is bound by the frozen+saturated 96 GB host tier, not
the device pool; (2) using the L3 disk tier (v16), even with a 4× parallel backend, caches worse and
slower than recompute because the prefetch path churns the frozen cache. **best_effort + skip-writes +
skip-prefetch + lpm remains the lossless optimum (~1.1 s TTFT, ~95× below the tuned bar) for this fixed
protocol/budget.**

### v17 — recompute-cost-aware eviction (`--radix-eviction-policy costaware`)  [mechanism]  ** NEW BEST **
- **Idea:** under best_effort a miss is recomputed on-GPU; prefill is O(L²)-dominated for LONG prefixes
  (the LEval/LooGLE portion, 100k+ tokens) even in this mamba-hybrid, so a deep prefix is far costlier to
  recompute than a short one — and long-context conversations are heavily REUSED across their many turns.
  New pluggable eviction strategy `CostAwareStrategy`: bucket evictable leaves by prefix DEPTH (cumulative
  tokens root→node, via a bounded parent-walk) — evict SHALLOW (cheap) leaves before DEEP (expensive),
  LRU within each bucket. Lossless (eviction only changes what's cached; a wrong evict just recomputes).
  ~35 lines in `evict_policy.py` + factory/argparse wiring; committed 2818cd334.
- **Result (v17, n=1) vs prev best (v13, LRU):** Mean TTFT **1030.4 ms** (prev 1098; band 1142±43, prev
  6-run min 1089 — v17 is below the entire band), **hit_rate 0.6906 vs 0.6234 (+6.7 pp)**, throughput
  **50475 tok/s / 3.52 req/s** (prev ~48000 / 3.37), P99 7624 ms (≈prev 7665), evict_mean_ms 1.03 (prev
  0.54 — 2× from the depth-walk, negligible vs TTFT). Lossless (7037/7037, no fallback; frozen budget
  asserted: hicache_size=96, mem_fraction=0.85).
- **Why it works (overturns the earlier "LRU is optimal" hypothesis):** the +6.7 pp hit_rate is a
  structural mechanism effect (not run-noise) — protecting deep, heavily-reused long-context prefixes from
  eviction better matches the workload's reuse than pure recency. Fewer/cheaper recomputes → lower mean
  TTFT + higher throughput. The P99 tail is unchanged (those are genuine first-touch misses), but the mean
  and hit rate improve. This is the FIRST lever to beat the best_effort+skip-mechanisms plateau.
- Reproduction (v17b) in progress to confirm before declaring; the hit_rate signal already strongly
  corroborates a real gain.

### v17b — cost-aware eviction reproduction  [mechanism]  ** CONFIRMS NEW BEST **
- Repro on the same certified node: TTFT **1051.5 ms** (v17a 1030.4; both below the prev best 6-run min
  of 1089), hit_rate **0.6877** (v17a 0.6906), throughput 50469 tok/s / 3.52 req/s. Lossless (7037/7037).
- **Confirmed (n=2):** cost-aware eviction = TTFT ~1030–1051 ms (mean ~1041, ~9% below the 1142±43 LRU
  band), hit_rate ~0.689 (+6.6 pp), ~+5% throughput. The hit-rate lift is stable across runs → a real
  structural gain, not noise. Emailed as the new best. Depth threshold currently 4096 tokens; sweeping next.

### v18/v19 — cost-aware eviction DEPTH_THRESHOLD sweep  [mechanism]
- Optimizing the winning cost-aware eviction (v17 used threshold=4096). Also fixed a latent bug: the
  bounded parent-walk cap `MAX_WALK` was 96, too small to measure depths >~6k tokens (would silently
  degrade high thresholds to LRU); raised to 512 (the walk is naturally bounded by threshold/page_size
  since it breaks once depth>=threshold).
- **v18 (threshold=2048):** TTFT 1020 ms, hit_rate **0.6645** — LOWER hit than 4096 (0.69). Protecting
  MORE (medium) prefixes is LESS selective and dilutes the benefit (still beats LRU's 0.6234). So the
  sweet spot is toward MORE-selective (protect only the genuinely-long), not less.
- **v19 (threshold=8192):** in progress — testing the more-selective direction.
- Takeaway so far: threshold ~4096 is a good operating point (hit 0.69); 2048 over-protects (0.66).

### Cost-aware eviction DEPTH_THRESHOLD sweep — COMPLETE (peak = 8192)
| policy / threshold | Mean TTFT | hit_rate | P99 TTFT | note |
|---|---|---|---|---|
| LRU (prev best)    | 1142±43 ms | 0.6234 | ~7665 | baseline eviction |
| costaware d2048    | 1020 ms | 0.6645 | 7630 | over-protects (medium prefixes dilute) |
| costaware d4096    | 1030/1051 ms | 0.688–0.691 | 7624 | v17/v17b (first new best) |
| **costaware d8192**| **936 ms** | 0.6806 | **7070** | **PEAK — best TTFT + best tail** |
| costaware d16384   | 993 ms | 0.6260 | 7677 | too selective (≈LRU hit; only longest protected) |

- **Inverted-U with peak at threshold≈8192** (~18% below LRU's 1142 ms, ~13% below the first-cut d4096).
- **Key insight:** hit_rate is NOT monotonic with TTFT — d4096 has the highest hit (0.690) but WORSE TTFT
  than d8192 (0.681 hit). Cost-aware eviction wins by minimizing recompute COST (protecting the few
  O(L²)-expensive long-context prefixes), not by maximizing hit count. d8192 best balances protecting the
  expensive prefixes against diluting the cache; d16384 protects too few (→LRU-like hit); d2048/d4096
  protect too many (dilute). All lossless. Confirming d8192 with a reproduction (v21) before finalizing.

### v21 — cost-aware d8192 reproduction  [mechanism]  ** CONFIRMED OPTIMIZED BEST **
- Repro: TTFT **971.9 ms** (v19 936; mean of the two ~954), hit_rate 0.6817, P90 2031, P99 7661,
  throughput 50455 tok/s / 3.52 req/s. Lossless (7037/7037).
- **FINAL BEST = cost-aware eviction, depth threshold 8192** (n=2: 936/972 ms, mean ~954): **−16.5% mean
  TTFT vs the LRU best (1142±43)**, +5.8 pp hit (0.623→0.682), +5% throughput — all lossless, in-budget,
  on the fixed protocol. Stacked on best_effort + skip-writes + skip-prefetch + lpm (~100× below the
  tuned bar overall). Emailed. This is the definitive new mechanism from this evolution.

### v22 — costtiered (graded depth tiers) refinement  [mechanism]  ** NEGATIVE — binary d8192 is optimal **
- Tested graded protection (tier = depth//8192 capped at 4; protect the truly-huge ≥32k MORE than the
  moderately-long 8-16k) vs the binary "protect all ≥8192 equally" of costaware.
- Result: TTFT **1060 ms**, hit_rate **0.5658** (WORSE than LRU's 0.623, and far below costaware d8192's
  0.681). Grading over-commits cache to the huge LEval/LooGLE prefixes (they hog space) → medium/short
  prefixes evicted aggressively → hit rate craters. Median TTFT also jumped (743 vs ~548).
- **Definitive: binary cost-aware (protect all ≥threshold EQUALLY, LRU within; threshold=8192) is the
  optimum.** Grading hurts. Cost-aware eviction is now fully optimized and refinement-tested.

## FINAL RESULT (this evolution)
Two novel, lossless engine mechanisms compound on the fixed protocol:
1. **best_effort regime + skip-L3-writes + skip-L3-prefetch + lpm** — bypass the catastrophically-slow
   serialized L3 disk by recomputing misses on-GPU (~95× below the v0_tuned bar; 108.8 s → ~1.1 s).
2. **recompute-cost-aware eviction (`--radix-eviction-policy costaware`, threshold 8192)** — protect
   deep, O(L²)-expensive, heavily-reused long-context prefixes from eviction; evict shallow (cheap)
   first, LRU within. A further **−16.5%** mean TTFT (1142→~954 ms), +5.8 pp hit, +5% throughput.
Negatives mapped: mamba→KV realloc (neutral, host-capacity-bound), parallel-L3-reads+wait_complete
(worse, churns frozen cache), costtiered (worse, over-protects). All results lossless & reproduced.

### v23/v23b — SJF cost-aware scheduler + cost-aware eviction  [mechanism]  ** MAJOR NEW BEST **
- **Discovery:** sglang's `lpm` scheduler reverts to **FCFS once the waiting queue > 128**
  (`schedule_policy._determine_active_policy`) — exactly our saturated regime — so under load the
  scheduler is NOT cost-aware. New `sjf` CacheAgnostic policy (`--schedule-policy sjf`): sort the waiting
  queue by ascending UNCACHED prefill work `len(prompt)+len(output)−num_matched_prefix_tokens` (uses the
  cheaply-populated match count → cost-aware at ANY queue size, no FCFS fallback). Serve cheap requests
  first → queue drains faster → lower latency for all (Little's law). Lossless: service ORDER only.
- **Result (n=2, on top of cost-aware eviction d8192):** Mean TTFT **802 / 777 ms (~790)** vs LRU+lpm
  1142 → **−31%**; vs costaware+lpm 954 → −17%. Median ~522. **P99 4884/5204 vs 7665 → −35% (tail ALSO
  improves — no starvation; 7037/7037 completed).** throughput ~50461 tok/s / 3.52 req/s (≈ offered λ 3.5
  → the system now KEEPS UP with load). hit_rate 0.681 unchanged vs costaware → the SJF gain is purely
  scheduling, not caching. Lossless, in-budget, frozen budget asserted.
- **Cumulative vs v0_tuned bar (108.8 s): ~138× lower TTFT.** Three stacked novel lossless mechanisms:
  best_effort+skip-L3 (recompute > slow disk), cost-aware eviction (protect expensive prefixes),
  SJF cost-aware scheduling (drain cheap-first, fix FCFS-under-load).

### v24 — SJF+LRU (mechanism attribution)  [mechanism]
Isolating the two new mechanisms (mean TTFT / hit_rate, on best_effort + skip-L3):
| eviction \ scheduler | lpm (→FCFS under load) | SJF |
|---|---|---|
| LRU   | 1142 / 0.623 (prev best) | **869 / 0.617** (v24) |
| costaware d8192 | 954 / 0.681 | **790 / 0.681** (BEST) |

- **SJF is the larger single lever: −24% alone** (1142→869) with hit UNCHANGED (0.617≈LRU) — a pure
  scheduling win (fixing FCFS-under-load), orthogonal to caching.
- **Cost-aware eviction: −16.5% alone** (1142→954) via hit +5.8 pp (0.623→0.681) — a pure caching win.
- The two are ~independent and STACK: SJF+costaware = **790 ms (−31% vs LRU+lpm)**, the best. Both
  lossless, in-budget, on the fixed protocol. This is the clean two-mechanism decomposition of the win.

### v25 — SJF × eviction-threshold interaction  [mechanism]
- d4096+SJF = 803 ms / hit 0.695 ≈ d8192+SJF (790 / 0.681) — within noise. The eviction threshold is NOT
  sensitive under SJF and the two mechanisms are ORTHOGONAL (confirmed): the combined optimum is stable.
  d8192+SJF stands as the best (~790 ms). Note hit_rate again non-monotonic with TTFT (d4096 higher hit,
  ≈same TTFT) — reinforces that cost-aware wins by cheaper misses, not more hits.

## STATUS: comprehensive lossless optimum reached for this fixed protocol
Design space explored end-to-end (config + capacity + disk-tier + eviction + scheduling), 25 logged
versions. Best = best_effort + skip-L3-writes + skip-L3-prefetch + cost-aware eviction (d8192) + SJF
scheduling = **~790 ms mean TTFT, ~138× below v0_tuned, lossless, system keeps up with λ=3.5**. The floor
is now the unavoidable long-context first-touch recomputes (P99 ~5 s); the only remaining lever is
prefill/decode overlap (mixed_chunk), declined for a device-KV-pool leak + silent-losslessness risk.
Four novel engine mechanisms total (2 disk-skip, cost-aware eviction, SJF), all pluggable/upstream-friendly.
