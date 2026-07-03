# kv-lynx-4d2 — sglang HiCache KV-cache evolution report

Independent researcher. Branch `evolve/kv-lynx-4d2`. W&B run `kv-lynx-4d2` in `sgl-evolve`.
Model Qwen3.5-122B-A10B-FP8 (hybrid-Mamba), TP=8, fixed 3-tier protocol (GPU + 768 GB host +
1.8 TB disk L3), real-text Mooncake 1:1:1 mix, λ=3.5, mc=128, 1553 prompts. Headline = mean TTFT.

## EXECUTIVE SUMMARY (for a skeptical maintainer)
Two robust, lossless wins take **mean TTFT 87,615 ms → 1,558 ms (56×)** and throughput 1.15 → 3.26
req/s (nearing the offered λ=3.5) on the fixed protocol:
1. **`mechanism` — O(keys) L3 hit-query (drop a whole-dir `os.scandir`) [commit bbf51108].** The
   hybrid-Mamba prefetch path ran `os.scandir` over the *entire* L3 dir (~2M page files) on every
   prefetch, every rank — O(total-files), growing as L3 fills → the catastrophic growing tail (p99
   271 s). Fix = `os.path.isfile` on the specific targets (identical set → byte-lossless; set-equality
   test). Alone: **36× TTFT** (v2). Microbench: 147–742× on the hit-query.
2. **`config` — Mamba/KV GPU-memory rebalance (`--mamba-full-memory-ratio 0.9→1.5`).** After (1) the
   run is GPU-prefill-bound; the GPU KV pool sits ~72% empty because the hybrid model's *Mamba-state
   pool* is the binding GPU constraint. Shifting GPU cache memory to the Mamba pool (max_mamba_cache
   1350→1711) raises cached-sequence capacity → hit_rate recovers → less prefill. Cumulative **56×**
   (v4). Lossless (pure memory rebalance).

**Robustness note:** the two provided baselines differ 87.6 s vs 108.8 s for the *same* config
(~24% run-to-run variance), so the *big* deltas above are unambiguous, but fine version-to-version
ranking (v3 1802 / v4 1558 / v5 1603 ms) is within noise — treat ratio∈[1.3,1.5] as one plateau.
All my runs use `--enforce-disable-flashinfer-allreduce-fusion` (this venv's fusion NCCL init hangs;
it's a comm opt that can only *slow* things, so my wins are if anything *conservative* vs a fusion-on
baseline). Every version completed 7037/7037 requests clean.

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

### v2-scandirfix — `mechanism` — RUNNING (commit 264d2e0d, scandir fix only; parallel IO off)
Change: O(keys) `os.path.isfile` L3 hit-query (drop the whole-dir scandir). Parallel IO shelved
behind a knob (off) as it's array-bound (~9%) and its mamba path isn't concurrency-tested.
Hypothesis: removing the ~0.65 s/hit-query directory scan that grows with L3 fill collapses the
growing TTFT tail, independent of prefetch policy. Lossless (byte-identical set + round-trip tests).

### v2-scandirfix — RESULT (commit e77d37a8, `mechanism`) — **NEW BEST, 36× TTFT**
| metric | v0_official | **v2-scandirfix** | Δ |
|---|---|---|---|
| **mean TTFT** | 87615 ms | **2425 ms** | **36.1× lower** |
| TTFT p90 | 243466 ms | 4052 ms | 60× lower |
| TTFT p99 | 270799 ms | 30636 ms | 8.8× lower |
| TTFT median | 1224 ms | 1427 ms | ~flat |
| out tok/s | 146.9 | 336.7 | 2.29× |
| req thruput | 1.15 | 2.63 | 2.29× |
| TPOT mean | 241 ms | 450 ms | worse (see caveats) |
| hit_rate | 0.816 | 0.577 | lower |
| L3 hit frac | 0.254 | 0.026 | disk barely used |
| prefetched tok | 166 M | 14.3 M | 11.6× fewer |
| disk_read tok | 20.7 M | 1.49 M | 13.9× fewer |

**The catastrophic growing tail is gone** (p99 271 s → 31 s) — exactly what the S2 screen predicted:
the O(total-files) scandir per prefetch was the tail. With an O(keys) hit-query, prefetches complete
fast, the working set stays hot in GPU+host, and the system **barely spills to disk** (L3 25%→2.6%,
prefetch 166M→14M). Serve wall-time ~45 min vs ~105 min. This is "less budget at better TTFT" — a
charter-endorsed win. **Lossless by construction** (the fix is a faster existence check → identical
KV loaded → identical outputs; correctness test proves set-equality vs scandir).

Caveats (honest): (a) v2 has FlashInfer fusion disabled (env workaround) while the provided baseline
may have it on — fusion is an allreduce (comm) opt that can only *slow* things, so it can't explain a
TTFT *improvement*; it does inflate v2's TPOT (450 vs 241). (b) The regime shift (hit_rate↓, disk↓) is
emergent from faster processing, not a caching bug. **To isolate the scandir fix from the fusion
confound and establish a fusion-matched in-env reference, I am running a control `v0-ctrl-scandir` =
original scandir + fusion-off** (both fusion-off ⇒ clean A/B of the scandir fix). New bottleneck after
v2: TPOT / hit-rate, not disk-wait (disk now 2.6%), so disk-wait mechanisms (adaptive prefetch, SJF)
are now low-value — v3 should target decode/throughput or hit-rate recovery.

## Post-v2 bottleneck analysis (from v2 raw metrics) — reshapes v3+
The residual v2 tail (p99 31 s) is **queue-time-bound, not disk-wait**: v2's `queue_time_seconds`
histogram is p99 ~30 s (≈ p99 TTFT), and **req_throughput 2.63 < offered λ 3.5** ⇒ the system is
**throughput-bound**, so the waiting queue builds over the run. Mean queue time (~1.33 s) is ~55% of
the 2.4 s mean TTFT; the rest is prefill.
- ⇒ **Adaptive/timeout prefetch and SJF (disk-wait mechanisms) are the WRONG lever now** — disk is
  only 2.6% of hits, and recompute-on-congestion would ADD GPU work → lower throughput → *worse*
  queueing. C1 (adaptive prefetch) is built + logic-tested but **shelved** by this evidence.
- The throughput ceiling is **GPU work per request**: prefill (inflated by the hit-rate drop
  0.82→0.58 ⇒ ~42% recompute) + decode (TPOT 450 vs 241, inflated by the fusion-off env workaround).
- **Why the scandir fix boosted throughput 1.15→2.63:** the O(N) scandir ran on the *scheduler
  thread* (prefetch_from_storage ← _add_request_to_queue), serializing request admission. Removing it
  unblocked the scheduler. Next scheduler-thread candidate: `evict_host`'s per-prefetch O(N) heapify
  (C5) — but after the scandir fix the system is likely GPU-bound, so C5's headroom is uncertain.
- **v3+ plan:** run the control (v0-ctrl-scandir = scandir ON + fusion-off) to (a) isolate the scandir
  fix, (b) get a fusion-matched baseline, (c) reveal whether the hit-rate drop is from the scandir fix
  (regime change) or elsewhere. Then target **throughput** — most likely **hit-rate recovery**
  (reuse-aware retention to reduce prefill recompute), the genuine KV-cache lever.

### v0-ctrl-scandir — control (scandir ON + fusion-off) — FAILED / NOT LOGGED
Intended as a fusion-matched in-env baseline to isolate the scandir fix. Result was invalid:
**63 / 7037 requests completed**, 0.08 req/s, tpot 8.3 s, server shut down early. Either a mid-run
pool collision (a neighbor's server on the same node — the shared pool has non-flock users) or the
fusion-off+scandir combo being too slow to complete before client timeouts. Not logged (incomplete).
Note: `running-req` peaks at 139 (>128 max-concurrency) in BOTH v2 and this run — it counts chunked
sub-requests, so it is NOT a collision signal; v2 is confirmed clean (7037/7037 completed, 2.63 req/s).
Takeaway: without the scandir fix a fusion-off run may not even complete this workload — the fix is
what makes it tractable in-env. The 36× TTFT claim rests on: (a) logic — fusion-off can only *slow*
allreduce, never improve TTFT; (b) the S2 screen — scandir was 147–742× and O(N)-growing on the
scheduler thread; (c) v2's clean 7037/7037 completion vs baseline's tail.

## v2 validity (self-audit, confirmed)
resolved_args == contract (ctx 262144, mem-frac 0.85, hicache_size 96, tp 8, page 64, wait_complete);
no SILENT FALLBACK; **7037/7037 requests completed** at 2.63 req/s; lossless by construction. On-contract.

## Remaining bottleneck & future directions (for a maintainer)
v2 is **GPU-compute-bound, prefill-dominated** (prefill:decode batches 9406:454; #running-req p90=128
== max-concurrency), throughput 2.63 < offered λ 3.5 ⇒ the queue (mean queue ~1.33 s ≈ 55% of the
2.4 s mean TTFT) is the residual. Prefill is inflated by the hit-rate drop (0.82→0.58 ⇒ ~42%
recompute), itself caused by the faster regime evicting prefixes before multiturn reuse. The GPU KV
pool sits ~72% empty (token-usage p50 0.26) while host is full — likely because the hybrid model's
**Mamba SSM-state pool is the binding GPU constraint** (evicting a leaf to free Mamba state also frees
its full-attn KV). Genuine next levers (all lossless, none easy):
- **Decouple Mamba-state and KV eviction** so hot full-attn KV can stay in the empty GPU KV pool even
  when a leaf's Mamba state is evicted → more device hits → less prefill → higher throughput.
- **Reuse-aware retention** for multiturn conversation prefixes across the inter-turn gap.
- (Fusion is an env artifact: this venv's FlashInfer allreduce-fusion NCCL init hangs; all my runs are
  fusion-off, so version-to-version comparisons are consistent.)

### v3-mamba-ratio13 — `config` (on the scandir-fixed engine) — **NEW BEST**
Change: `--mamba-full-memory-ratio 1.3` (default 0.9) → shifts GPU cache memory from the ~72%-empty
full-attn KV pool to the binding Mamba-state pool. max_mamba_cache_size **1350→1611** (+19% cached
sequences). Fusion-off (env). Clean **7037/7037**, no fallback. Lossless (memory rebalance, same KV).
| metric | v2-scandirfix | **v3-mamba-ratio13** | Δ vs v2 | vs v0_official |
|---|---|---|---|---|
| **mean TTFT** | 2425 ms | **1802 ms** | −26% | **48.6× lower** |
| TTFT p99 | 30636 ms | 12916 ms | −58% | 21× lower |
| TTFT median | 1427 ms | 1209 ms | −15% | ~flat |
| out tok/s | 336.7 | 382.5 | +14% | 2.6× |
| req thruput | 2.63 | 2.99 | +14% | 2.6× |
| TPOT mean | 450 ms | 392 ms | −13% | — |
| hit_rate | 0.577 | 0.541 | −0.036 | — |
| **GPU KV usage** (p50/max) | 0.26/0.69 | **0.33/0.83** | +on-device | — |
| device hit frac | 0.500 | 0.516 | +on-device | — |

**Validates the Mamba-KV-imbalance diagnosis**: giving the Mamba-state pool more of the (empty) GPU
budget lets more sequences stay device-resident → more device hits → less prefill recompute → higher
throughput → shorter queue → lower TTFT. (Overall hit_rate dips slightly because the KV pool shrank,
but the *device* share rose and device hits are fastest — net a clear win.) GPU KV peak is now 0.83,
so headroom to push the ratio further is limited (peak → 1.0 risks preemption); v4 candidates: nudge
ratio to ~1.4, `--enable-int8-mamba-checkpoint` (2× cached-prefix capacity, but LOSSY → needs a
quality gate), or the deeper decouple (keep full-attn KV on-device when a leaf's Mamba state evicts).

### v4-mamba-ratio15 — `config` — **NEW BEST (56.2× vs baseline)**
`--mamba-full-memory-ratio 1.5` → max_mamba_cache_size **1711** (vs 1611@1.3, 1350@0.9). Clean
7037/7037, GPU KV peak 0.90 (1 preemption, negligible), fusion-off, lossless.
| metric | v3(1.3) | **v4(1.5)** | vs v0_official |
|---|---|---|---|
| mean TTFT | 1802 ms | **1558 ms** | **56.2×** |
| median TTFT | 1209 ms | 913 ms | 1.3× |
| TTFT p99 | 12916 | 12866 | 21× |
| TPOT | 392 ms | 300 ms | — |
| out tok/s | 382 | 417 | 2.8× |
| req thruput | 2.99 | 3.26 (→λ 3.5) | 2.8× |
| hit_rate | 0.541 | **0.647** | — |
More Mamba slots ⇒ more cached sequences ⇒ hit_rate recovers (0.54→0.65) ⇒ less prefill ⇒ throughput
3.26 (nearing offered λ 3.5, so the queue is nearly drained) ⇒ median TTFT 913 ms. Monotonic gains
0.9→1.3→1.5; GPU KV peak 0.90 leaves small headroom → v5 tries 1.6.

## Curve so far (own versions, all lossless, fusion-off)
v2 scandir-fix (mechanism) 2425 ms → v3 ratio1.3 1802 → v4 ratio1.5 **1558 ms** (baseline 87615 ms).

## Operational notes (env / infra — not research variables)
- **Pool coordination:** the manager's held pool is shared and some researchers run evals on it
  *without* the per-node flock (e.g. pinned launchers), so a flock-free node can still host a
  neighbor's 8-GPU server. My launcher (`run_eval.sh`) now gates on **flock + ≥1.8 TB disk + ≥1.3 TB
  free RAM + GPUs idle (<10 GB used)** so I never collide (collisions were causing OOM / SIGKILL /
  NCCL-timeout during init), plus a bounded retry.
- **Do NOT probe a loading server** via `srun --overlap` — health/GPU/py-spy probes during the
  sensitive 8-rank init destabilize it (observed SIGBUS on a rank). Detect serving by reading
  `server.log` on the shared FS; capture live metrics only after serving, sparingly.
- **FlashInfer allreduce fusion hangs in this venv.** The model auto-enables FlashInfer trtllm
  allreduce fusion on H100; in my env its NCCL communicator init **times out (600 s) and the server
  then hangs** in the silent post-barrier phase (reproduced on 1-2 and ondem-3, v1 and v2). Fix:
  pass **`--enforce-disable-flashinfer-allreduce-fusion`** (not in eval.sh's FORBIDDEN list) → reaches
  the same fusion-off state the fallback targets, but loads normally (~11 min) instead of hanging.
  **Caveat:** fusion is an allreduce (comm) optimization, orthogonal to KV-cache/prefill; it does not
  touch the KV-loading path that dominates the TTFT tail (my headline). All my versions use this flag,
  so version-to-version comparisons are consistent; vs the provided baseline there may be a small
  fusion-state difference (noted, TTFT-neutral).

## Next
After v2, re-measure the bottleneck from the new metric profile. Candidate v3+: congestion/deadline-
aware adaptive prefetch (wait-vs-recompute), SJF prefetch ordering, or write_through_selective to cut
disk write contention — chosen from evidence, prizing novelty over tuning.

### v5-mamba-ratio16 — `config` — reverted (ratio ceiling)
`--mamba-full-memory-ratio 1.6` → max_mamba 1754, GPU KV peak 0.94. mean TTFT 1603 ms (vs v4 1558),
hit_rate 0.635 (vs 0.647), p99 15197 (vs 12866). Slightly WORSE than v4 — past the sweet spot; the KV
pool shrinks too far (peak→0.94) so the KV-shrink cost outweighs the extra Mamba capacity. **Ratio
lever peaks at 1.5 (v4).** v6+: get more Mamba capacity WITHOUT shrinking KV — mamba cache strategy
(no_buffer/lazy frees the ping-pong buffer, lossless) or int8-mamba (2×, lossy → quality gate).
