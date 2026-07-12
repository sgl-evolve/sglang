# SLOP implementation design — Speculative Load-back Overlap under Pressure
(kleinrock; gated on baseline showing load-back is a real critical-path cost. This doc de-risks the
eventual implementation by resolving the non-obvious correctness pitfalls BEFORE coding, so I don't waste
a scarce GPU eval on a lossy/crashy bug.)

## Goal
Overlap the L2→L1 (host→device) load-back of a cache-hit request's prefix with the running batch's
compute, by STARTING it while the request waits in `waiting_queue`, instead of synchronously at admission.
Net: remove the load-back stall + admission-path alloc/evict from the critical path → lower TTFT for
warm turns. LOSSLESS (only moves WHEN KV is transferred, never changes KV contents).

## Baseline path (stock, confirmed by reading)
- `_add_request_to_queue` (scheduler.py:2305) → `_prefetch_kvcache` (2280) [NO-OP in 2-tier] → append to
  waiting_queue.
- admission `_get_new_batch_prefill_raw` (2775) loop (2861): `req.init_next_round_input(tree_cache)` →
  `adder.add_one_req` → if `req.needs_host_load_back()` → `tree_cache.init_load_back(...)` →
  `hiradix.load_back(node)`: `cache_controller.load()` (alloc device idx + enqueue) [+ possible evict].
- after admission: `scheduler.py:2970 ready_to_load_host_cache()` → `cache_controller.start_loading()`
  merges load_queue → layer-pipelined DMA on load_stream; forward waits per-layer (`LayerDoneCounter`).

## SLOP change (minimal, hook the EXISTING prefetch seam)
1. In `_prefetch_kvcache` (or a new `_prefetch_loadback`): when a req enters the queue AND storage is
   OFF (2-tier), run `req.init_next_round_input(self.tree_cache)` to populate match (host_hit_length,
   last_node). If `req.needs_host_load_back()` and `host_hit_length >= SLOP_MIN` and device-prefetch
   budget available → call a NEW `tree_cache.init_load_back(...)` early, recording the returned
   device_indices + node in a per-req field `req._slop_loaded` and in `tree_cache.ongoing_load_back`.
   Then `cache_controller.start_loading()` immediately (kick DMA now, overlapping running-batch compute).
2. Admission loop: BEFORE `add_one_req`, if `req._slop_loaded` is set → the KV is already resident (or
   in-flight). Mirror the storage-prefetch skip pattern (scheduler.py:2882): if the load event isn't done
   (`is_load_back_event_done`), `continue` (leave in queue, try next iteration); else admit — and CRUCIALLY
   make `add_one_req`/`init_next_round_input` see the already-loaded device indices so it does NOT
   re-issue `load_back` (double-load).

## Correctness pitfalls & resolutions (the whole point of this doc)
- **P1 Double-load.** If SLOP already loaded the prefix, admission must NOT call `load_back` again.
  Resolution: SLOP sets `node.value` device indices + registers in `ongoing_load_back` exactly as stock
  `load_back` does (hiradix:1196-1205). Then at admission, `node.evicted` is False (value set) →
  `needs_host_load_back()` returns False for that node → `init_load_back` is skipped naturally. VERIFY:
  `needs_host_load_back` keys on `host_hit_length>0` set during match; re-running match after SLOP load
  must recompute host_hit_length=0 for the now-resident prefix. So re-run `init_next_round_input` at
  admission (stock already does) → it re-matches and sees the prefix as device-resident → host_hit_length
  drops → no reload. KEY: SLOP must PROMOTE the loaded nodes in the radix tree (set value, emit
  store(GPU)) so the re-match classifies them device-resident. Stock `load_back` already does this.
- **P2 Lossless.** SLOP never alters KV bytes; it only pre-issues the same H2D copy stock would do. The
  loaded KV is identical. Output equivalence holds by construction (same prefix indices feed the forward).
  GATE: still run the lossless check (outputs vs no-cache) as required.
- **P3 Device memory pressure / eviction cascade.** Early load-back allocates device KV slots for reqs not
  yet admitted → could evict running-batch KV (BAD). Resolution: a HARD budget — SLOP only prefetches if
  `cache_controller.mem_pool_device_allocator` free ≥ (running headroom + this load). Track
  `slop_inflight_tokens`; cap at `SLOP_BUDGET` (e.g. a small fraction of device KV). If over budget, DON'T
  prefetch (fall back to stock synchronous load at admission). NEVER evict running/active KV for a
  speculative load (only evict already-evictable radix nodes, as stock load_back does via its evict path —
  but for SLOP, prefer to SKIP rather than evict, to avoid churn).
- **P4 Revocation.** A req may be aborted/finished before admission, or the prefetched prefix may be
  needed elsewhere. On req removal from queue, free its `_slop_loaded` device indices
  (`cache_controller.evict_device`) and clear `ongoing_load_back[node.id]`, dec_lock_ref. Add cleanup in
  the req-abort path.
- **P5 LayerDoneCounter producer/consumer.** `start_loading` uses 3 rotating producer counters
  (cache_controller.py:78). Calling it at queue-entry (SLOP) AND at admission (stock 2970) must not
  overflow/misalign the 3-counter ring. Resolution: SLOP calls start_loading for its own load; the forward
  consumes via the counter index returned by `ready_to_load_host_cache`. Ensure SLOP's load completes
  (event done) BEFORE the admitting forward, so the forward's consumer index isn't waiting on a SLOP
  producer. Since SLOP loads are kicked earlier and checked done at admission, they finish first. VERIFY
  the counter ring doesn't wrap with >3 concurrent SLOP loads: cap concurrent SLOP loads ≤ 2 (leave 1 for
  the admitting batch). This is the SUBTLEST pitfall — test carefully on GPU.
- **P6 lock_ref accounting.** stock `load_back` does inc_lock_ref(last_hit_node) to protect loaded KV
  until use. SLOP must hold the lock from prefetch until admission consumes it, then transfer to the batch
  (or release on revocation). Mirror stock's inc/dec exactly.

## Toggle & metrics
- Env `SLOP=0/1` (default 0 = stock) OR a policy arg. Add Prometheus counters: `slop_prefetch_tokens`,
  `slop_hit` (prefetch done before admission), `slop_late` (admission waited), `slop_revoked`,
  `slop_skipped_budget`. These decompose whether SLOP actually overlapped (slop_hit) vs added latency.
- Per-request diag (crash-safe, try/except): log prefix_dev/host tokens, extend tokens, slop state, wait,
  ttft — for the §5 tail decomposition.

## Kill criteria (when to abandon SLOP → impossibility)
- Baseline `load_back_duration` p99 tiny (< ~10 ms) or load-back tokens small → nothing to overlap → SKIP
  SLOP, report as impossibility evidence (load-back not on the critical path).
- SLOP implemented but goodput@SLO unchanged (n≥2) AND slop_hit high → confirms tail is cold-prefill-bound,
  not load-back-bound → strong impossibility negative control.

## Files touched (est.)
- `python/sglang/srt/managers/scheduler.py` (~30 LOC: _prefetch_loadback + admission skip + revoke).
- `python/sglang/srt/mem_cache/hiradix_cache.py` (~15 LOC: reuse init_load_back; budget check).
- `python/sglang/srt/managers/cache_controller.py` (~10 LOC: metrics; maybe a budget query).
- Total ~55 LOC. Develop in a git worktree so the queued stock baseline tree stays pristine; merge post-baseline.
