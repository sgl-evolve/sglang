# kv-onyx-mf76 — research notes (prior art + sglang HiCache code map)

Source: independent prior-art + code study (Strata arXiv 2508.18572, HiCache blog 2025-09-10,
LMCache/Mooncake) cross-read against my clone's code (base commit 72812db13). All line numbers
refer to `python/sglang/srt/**` at that commit.

## Architecture map (how sglang HiCache KV works today)

**Three tiers.** L1 = GPU KV pool (`memory_pool.py`, MHA/MLA/DSA `TokenToKVPool`, always
layer_first). L2 = host pinned pool (`memory_pool_host.py`: `MHATokenToKVPoolHost` /
`MLATokenToKVPoolHost`, size = `hicache_ratio*device` or `hicache_size` GiB, default layout
`page_first`). L3 = optional storage backend. Unified page table = `HiRadixCache` radix tree
(`hiradix_cache.py`); each `TreeNode` has `value` (device idx) + `host_value` (host idx) + flags
evicted/backuped.

**Controller** (`cache_controller.py` `HiCacheController`). Two CUDA streams: `write_stream`
(L1→L2, `start_writing:674`), `load_stream` (L2→L1, `start_loading:758`). In-proc op lists
load/write queues merged via `CacheOperation.merge_ops:126` (one merged op/round, priority=min).
`LayerDoneCounter:74` / `LayerLoadingEvent:56`, num_counters=3 rotating → layer-wise overlap:
`start_loading:771` loops layers on load_stream recording `producer_event.complete(i)` per layer;
model reads via `wait_until` in memory_pool. Storage threads: `prefetch_thread`
(`prefetch_thread_func:1024`) + `prefetch_io_aux_thread` (`prefetch_io_aux_func:967`) +
`backup_thread` (1193). `move_indices:736` uploads host idx to GPU for kernel backend.

**Write policy** (`_inc_hit_count:898`, `write_through_threshold` hiradix:183): write_through→1,
selective→2, write_back→disabled. Backed-up nodes must be contiguous prefix from root.

**Prefetch (L3→L2).** Issued in `scheduler._prefetch_kvcache:2280` at `_add_request_to_queue:2311`
— i.e. at ARRIVAL time, BEFORE `policy.calc_priority` sort (2805). `prefetch_from_storage:1483`
page-aligns key, gates on `prefetch_threshold=max(256,page_size)`, allocs host, enqueues
`PrefetchOperation`. `prefetch_thread_func:1024` runs `_storage_hit_query:994` (serial batch
`batch_exists`, STORAGE_BATCH_SIZE=128 pages), all-reduce MIN per request (1042), revokes if
hit<threshold else pushes to `prefetch_buffer`; `prefetch_io_aux_func` does `_page_transfer:937`.
Stop policy (`can_terminate_prefetch:1336`): best_effort / wait_complete / timeout.
`check_prefetch_progress:1377` polled in prefill loop (scheduler 2882); requests still prefetching
are SKIPPED (continue, not reordered). Rate limit `prefetch_rate_limited:984` caps at
0.8*(host-device).

**Load-back (L2→L1).** `schedule_policy.add_one_req:942` → `init_load_back:1215` → `load_back:1143`
(all-or-nothing torch.cat; skip if <load_back_threshold=10 or >mem_quota). Load triggered after
batch built: scheduler 2969 `hicache_consumer_index = ready_to_load_host_cache()` → start_loading
(one merged op for whole batch).

**Scheduler** (`_get_new_batch_prefill_raw:2758`). `calc_priority` sorts queue (LPM/FCFS, LPM
disabled when queue>128). `PrefillAdder.add_one_req` admits by token budget + longest-prefix.
NO loading-bound classification, NO bubble-filling decode-during-load, NO transient/in-queue dedup
nodes (only I/O-in-flight bookkeeping).

**File L3 backend** (`hicache_storage.py` `HiCacheFile`). `_batch_io_v2:563` is a SERIAL python
loop over pages (`_read_page`/`_write_page` = open/readinto/tofile), no thread pool, no O_DIRECT.
Contrast hf3fs (`storage/hf3fs/storage_hf3fs.py`) which uses a ThreadPoolExecutor.

**Layouts** (`memory_pool_host.py init_kv_buffer:148`): layer_first / page_first(default) /
page_first_direct / page_head; `load_to_device_per_layer:223` & `backup_from_device_all_layer:340`
dispatch JIT kernels.

## Ranked lossless hypotheses (effect / effort, lossless-gated)

1. **Parallelize file-L3 `_batch_io_v2` with a ThreadPoolExecutor** (effort low, risk low).
   Serial per-page IO → bounded pool (config `file_io_threads`, default 4–8), reassemble bool vec
   in order. Lossless: disjoint host idx / disjoint files, same bytes. → faster L3→L2 so more
   prefetches land before dispatch. **Best first move (lowest risk).**
2. **Prefetch in priority/LPM order, not arrival order** (low, medium). After `calc_priority`
   sorts the waiting queue, issue/raise prefetch for the top-K soon-to-dispatch reqs. Lossless:
   prefetch is best-effort/partial-safe (untransferred tokens recomputed). → fewer dispatch-time
   prefetch misses → lower TTFT under deep queues.
3. **Pipeline `_storage_hit_query` probe with page transfer** (medium, medium). Probe chunk N,
   transfer it while probing N+1 (per-chunk all-reduce MIN keeps TP lockstep). Hides probe latency.
4. **Bubble-filling: run a decode batch when the prepared prefill batch is load-bound** (medium,
   medium; Strata mech 2). Gate determinism by NOT changing batch composition — defer the whole
   prefill one iteration, run the already-formed decode. Throughput +3–8% (Strata).
5. **Adaptive prefetch rate limit** instead of fixed 0.8*(host-device) (low, medium).
6. **Auto-tune `load_back_threshold`/`prefetch_threshold` to the FP8-397B break-even** (low, low).
   Compute load-vs-recompute break-even from pool sizes + measured prefill tput.
7. **Start L2→L1 load earlier** (issue during admission, not after batch close) (medium, medium).
8. **Transient in-queue dedup nodes** to avoid redundant prefill of in-flight shared prefixes
   (high, high; Strata mech 4). Big under ShareGPT 60-client locality.
9. **Balanced batch formation**: cap aggregate load-vs-compute per prefill batch, backfill with
   cache-hit reqs (high, high; Strata mech 3). Throughput +~11–12% (Strata).
10. **O_DIRECT file backend** to avoid page-cache double-buffering (medium, medium). Frees host RAM
    for L2.
11. **Concurrent `_page_transfer` (multi-thread L3→L2)** to remove head-of-line blocking (medium,
    medium).

Lossless gate for all: any KV byte not delivered exactly is **recomputed** by normal prefill,
never approximated. Scheduling/ordering changes must keep each *executed batch's composition*
deterministic so FP reduction order (hence outputs) is bit-identical.

## Planned search order
v1 = #1 (file-L3 thread pool, safest). Then #6 (threshold tuning) + #2 (prefetch ordering).
Then the bigger Strata mechanisms (#4 bubble-filling, #9 balanced batches, #8 transient nodes).
Each: implement → fast-screen (micro/short) → full protocol → log W&B (kept or reverted) → report.
