# v2 design — Exclusive/pipelined KV tiering (capacity lever, latency-free)

## Why (from v1 + diagnostics)
- Binding constraint = **distinct cache capacity** (~8.4M inclusive) ≪ working set 19M → thrash → 21pp
  hit headroom (ceiling 0.80 vs baseline 0.60).
- v1 admission (reduce working set) FAILED the p99≤8s SLO (deferral latency, decode-bound turns).
- Scheduler loop is CPU-sensitive (observe-instrumentation inflated p99 4×) → the mechanism must live in
  the **cache/controller layer**, not the scheduler admission loop.
- write_through is INCLUSIVE: device (2.35M) mirrors host's hot subset → wastes device as distinct cap.
  Making tiers EXCLUSIVE → distinct 10.75M → sim hit **0.59→0.73 (+14pp)**, extra hits are cheap load_back.

## The problem with the stock write_back proxy
`_evict_device_leaf` write_back branch (unified_radix_cache.py:1494-1503): on device eviction of an
un-backed node it does `write_backup(write_back=True)` then **synchronous `writing_check(write_back=True)`**
(blocks until D→H copy done) then `_evict_to_host`. Eviction runs in the scheduler thread (via
`evict_from_tree_cache` on allocation pressure), so the sync write STALLS scheduling. Under heavy eviction
(180M evicted) this likely negates the capacity gain. → screen `cfg-writeback` to measure net.

## Novel mechanism: asynchronous / pipelined exclusive eviction ("write-behind exclusive tier")
Gain exclusivity's capacity WITHOUT the sync-evict stall.
- Keep content **device-exclusive** (don't back up on hit — like write_back).
- **Write-behind watermark:** when device evictable/used crosses a high-watermark, a background pass
  (cache_controller aux thread OR piggybacked on writing_check polling) **asynchronously enqueues D→H
  writes for the LRU device leaves ahead of need** (via `cache_controller.write`, non-blocking), marking
  them "backup-in-flight".
- At real eviction time, the LRU leaf's backup is already in-flight/done → just poll ack (fast) → free
  device. **No synchronous write on the critical path.**
- Lossless: content preserved in host before device free; identical to write_back semantics, only the
  timing of the D→H copy changes (async vs sync). Outputs unchanged.
- Env-gated `SGLANG_XTIER_*` (default off = stock write_through). Cache-layer only (no scheduler-loop cost).

## Implementation sketch (targets)
- `cache_controller.py`: async write queue already exists (`write`/`start_writing`/`ack_write_queue`).
- `unified_radix_cache.py`: add a proactive-backup pass (iterate `evictable_device_leaves` LRU, enqueue
  write_backup for the coldest N that aren't backed/in-flight) triggered from `check_hicache_events` or a
  cheap periodic hook; make `_evict_device_leaf` fast-path free when backup already acked.
- Watermarks via `mem_pool_host.available_size()` / device evictable size.

## REFINED mechanism (safer): "Lazy-backup exclusive tiering"
Instead of surgery on write_back's sync eviction, keep write_through's stall-free async eviction but
change WHEN backup happens — gaining exclusivity/capacity with LOSSLESS-BY-CONSTRUCTION safety:
- **Skip eager backup** (`_inc_hit_count`, ura.py:1810 — gate off write_backup on hit when XTIER on) →
  content stays DEVICE-EXCLUSIVE (not in host) → distinct capacity gain.
- **Proactive async backup pass** in `check_hicache_events` (ura.py:2463, per-step, after writing_check):
  when device free < watermark, async `write_backup` the coldest-K UNBACKED device leaves (write_through
  path = non-blocking; LRU order deterministic across ranks → distributed-consistent).
- **Eviction unchanged (write_through path):** backed leaf → `_evict_to_host` (stall-free, host copy ready);
  unbacked leaf reached by eviction → DROPPED → recompute. **Both lossless** (recompute = identical KV).
  So correctness holds even if the proactive pass lags (only perf, never corruption).
- Net: hot device content exclusive (+capacity → +hit), cold margin pre-backed async (stall-free evict).
- Env: `SGLANG_XTIER_LAZY=1` (default 0=stock write_through), `SGLANG_XTIER_WATERMARK` (device-free frac
  trigger, e.g. 0.15), `SGLANG_XTIER_BATCH` (K coldest to back per pass). Cache-layer only (no sched-loop cost).
- Accessors: device free = allocator.available_size(); coldest device leaves = evictable_device_leaves in
  UnifiedLRUList order; write_backup(node) already async for write_through.
- RISK: distributed consistency of the proactive selection + ack; Mamba multi-component backup. Mitigation:
  reuse existing write_backup/writing_check machinery; deterministic LRU selection; env-gated + lossless fallback.

## Screen plan
1. cfg-writeback (write_back config) → does exclusive tiering raise hit rate? net goodput vs v0_official?
2. If capacity gain confirmed → implement async version → eval v2 (mechanism). Compare hit, p99 TTFT,
   req/s, load_back vs v0_official. Sweep watermark. Knee sweep (λ 3-4) for the goodput curve.
