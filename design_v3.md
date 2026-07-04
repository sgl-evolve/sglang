# v3+ mechanism candidates (pick from v2 evidence)

Context: v2 makes the L3 hit-query O(keys) (was O(total-files) scandir). Strata's thesis:
prior tiered-KV systems are *loading-bound, not compute-bound* — schedulers ignore cache-load
delays. After v2 the hit-query is cheap; the residual TTFT tail (if any) comes from (a) actual disk
read time under bursts (bandwidth ~6.6 GB/s, already ~saturated by 8 ranks) and (b) requests blocking
on those reads (wait_complete) while the GPU may idle. Decide v3 from v2's iostat + GPU-util +
per-tier metrics.

## POST-v2 DATA (from v2 server.log) — reframes everything
- #running-req p50=125/p90=128 (==max-concurrency) ⇒ batch SATURATED ⇒ **GPU-compute-bound**, NOT
  scheduler-bound ⇒ C5 (evict_host) and other scheduler-thread fixes WON'T raise throughput.
- prefill:decode batches = 9406:454 ⇒ GPU work is **prefill-dominated** (long-ctx mix + chunked
  prefill), and prefill is inflated by the hit-rate drop (0.58 ⇒ ~42% recompute).
- GPU KV token-usage p50=0.26 / max=0.69 ⇒ **GPU KV pool ~72% EMPTY** while host is full (0.999).
  HiCache offloads cached prefixes to the (full) host, leaving the fast GPU tier underused.

### C6 — GPU KV retention (use the empty 72% of GPU) [NOVEL, top pick for v4]
Keep more hot radix prefixes resident in the GPU KV pool (which sits ~72% empty) instead of
offloading everything to the saturated host tier ⇒ more device hits ⇒ fewer misses ⇒ less prefill
recompute ⇒ higher throughput ⇒ shorter queue ⇒ lower TTFT. This is the confirmed lever (GPU-bound on
prefill + GPU underutilized). Need to find WHY GPU is underfilled (HiCache GPU->host eviction/write
policy) and retain hot prefixes on-device up to capacity. Lossless (retention only changes tier, not
values). Gate on control data (does the slow baseline also underfill GPU, or is this the fast regime?).

## C1 — Congestion-aware adaptive prefetch (wait ↔ recompute)  [SHELVED: tail is queue-bound, not disk]
`hi_mamba_radix_cache.py:can_terminate_prefetch` (L1659). Today: static wait_complete/timeout/
best_effort. Idea: terminate a prefetch early (admit with loaded prefix, recompute the tail on GPU)
when the disk is congested, but wait when it's idle — best of both. Signals in
`self.cache_controller`: `prefetch_buffer.qsize()` (read backlog), `prefetch_queue.qsize()`,
`prefetch_tokens_occupied` vs `prefetch_capacity_limit`, `operation.completed_tokens`/elapsed.
Rule: effective wait shrinks as backlog grows → under bursts, recompute (drains disk, hides latency
on GPU which has headroom during loading stalls); when quiet, full wait (keep cache benefit, offload
GPU). Lossless (recompute = identical KV; already-on-disk pages aren't re-written — `set()` L424 skips
existing keys). Risk: recompute needs GPU headroom; if GPU is the bottleneck this hurts TPOT/throughput.
Gate on v2's GPU-util during stalls.

## C2 — SJF / size-aware prefetch ordering  [NOVEL, robust, policy-independent]
`cache_controller.py`: `prefetch_queue`/`prefetch_buffer` are plain FIFO `Queue()` (L358, L1028);
`prefetch_thread_func` puts to buffer at L1066. Make `prefetch_buffer` a priority queue keyed by
op read size (pages = `len(operation.hash_value)`), smallest-first. Classic SJF minimizes mean wait
on the single per-rank IO server → many short prefetches finish fast instead of queueing behind a few
huge ones → mean+median TTFT drop. Lossless (only reorders independent reads). Risk: starves the
largest reads (few, already-slow tail); bound with aging if p99 regresses. Helps under any policy,
needs no GPU headroom — good complement to C1.

## C3 — Cache-aware scheduling / overlap (Strata pillar)  [bigger, riskier]
`scheduler.py:2861-2890`: waiting_queue is FIFO; requests mid-prefetch are `continue`-skipped. Make
admission prefer requests whose KV is already resident (device/host) and defer disk-waiters, so the
GPU runs ready work while disk loads (fill bubbles). Partially already happens (skip → admit others).
Enhance: prioritize by resident-prefix fraction; cap concurrent disk-waiters. Lossless. Risk: fairness
/ starvation; more invasive in the scheduler (must-read large-class-init-style if touching __init__).

## C4 — write_through_selective to cut disk write contention  [CONFIG, low-risk fallback]
Baseline write_through offloads every new page to disk (offload 145M tok/run); those writes compete
with prefetch reads on the ~6.6 GB/s array. `write_through_selective` (hit_count≥2) writes only reused
("hot") pages → less write traffic → faster reads. Blog: write-through only best "if bandwidth
permits" — it doesn't here. Config flag `--hicache-write-policy write_through_selective`. Maps whether
write contention matters; a mechanism could make the write threshold congestion-adaptive.

## C5 — Incremental host-eviction LRU (kill per-prefetch O(N) heapify)  [NOVEL, gate on evidence]
`hi_mamba_radix_cache.py:evict_host` (L737) rebuilds a heap over the ENTIRE
`evictable_full_host_leaves` set on every call: `heap=[(n.last_access_time,n) for n in
evictable_full_host_leaves]; heapq.heapify(heap)`, then pops only ~num_tokens worth. Host is 100%
full (host_util≈1.0) so ~every prefetch triggers eviction, and this runs in the SCHEDULER main loop
(`_add_request_to_queue`→`_prefetch_kvcache`→`prefetch_from_storage`→`_alloc_with_evict(...,evict_host)`)
— O(N_evictable_leaves) on the critical path, growing with cache occupancy (scandir-like). Fix: keep a
persistent LRU (like the existing `mamba_host_lru_list`) and pop the tail, instead of rebuilding the
heap each call. Lossless (same LRU victim order). Risk: must keep the LRU in sync on insert/access/
evict — invasive. MEASURE N first (instrument or infer from v2 residual scheduler stalls) before building.

Order of attack after v2: read iostat (read vs write GB/s, %util) + GPU-util during stalls.
- disk %util high & writes large → C4 then C2.
- disk %util moderate but GPU idles during stalls → C1 (recompute) and/or C3.
- reads dominate & serialized → C2 (SJF) first (safe, policy-independent).

### C6 refinement — GPU underfill is likely MAMBA-state-constrained (hybrid model)
GPU holds full-attn KV + Mamba SSM states. If the Mamba-state pool is the binding constraint,
`evict()` frees leaves to relieve Mamba pressure, which ALSO frees their full-attn KV → KV pool
underfills (28%). So "retain more KV in GPU" is bounded by Mamba capacity, not KV. A real fix would
decouple: keep full-attn KV on-device even when a leaf's Mamba state is evicted (partial-tier node).
Deep/risky. CONFIRM via control server.log: if the slow baseline ALSO shows GPU~28%, it's a
structural Mamba-KV imbalance (worth a bold v4); if it fills GPU, v2's fast regime is the cause.
Cheap alt to probe hit-rate lever: --page-size 32 (finer prefix match; config; uncertain).

## v8+ plan — frequency-aware eviction (built v8; hold-session batch)
Context (from v6-page32 metrics): GPU KV pool ~72% empty (NOT binding), Mamba pool IS binding,
host tier full (util 1.0, evict_tokens 612M), host hits = 50% of all hits, disk irrelevant (1.14%).

- **v8 freq-evict (α=50) [BUILT, committed 97d6058ec]**: aged-LFU key `last_access + α·hit_count` on the
  full-KV device+host eviction heaps (evict/evict_host). Mainly affects the ACTIVE host full-KV eviction
  (device rarely evicts since KV pool 72% empty). Tests: does protecting hot host prefixes raise host
  hit-rate → fewer misses → lower TTFT? Lossless (victim order only). Run FIRST in the hold.
- **Decision tree after v8:**
  - v8 beats v4 (1558ms) above ~24% noise → α sweep (α∈{20,100,200}) to find sweet spot; log best.
  - v8 within noise of v4 → host-eviction policy isn't the lever. Then test the BINDING tier:
    **v9 = freq-aware MAMBA eviction** (evict_mamba LRU walk, line 818): CLOCK-style bounded
    second-chance — skip (reset_node_mru) an LRU candidate with hit_count≥thr up to a skip budget,
    evict a colder node instead. Higher-leverage (Mamba is binding) but riskier (LRU-list mutation on
    critical path) → implement only if v8 shows the mechanism class has ANY signal, and verify lossless.
  - both neutral → eviction policy is not a TTFT lever at fixed capacity; v4 stands as near-optimal;
    pivot to a different axis (scheduler cache-aware admission) or conclude.
- Also worth 1 clean exclusive-node re-run of v4 (best) to tighten the headline vs the fusion/noise caveat.

## v10 (planned, HIGH-EV, SAFE config) — cache-aware scheduling `--schedule-policy lpm`
Baseline uses `schedule_policy=fcfs` (eval.sh doesn't set it -> sglang default fcfs). sglang supports
`lpm` (longest-prefix-match): admit requests whose prefix is already cached FIRST, so they skip prefill.
In this GPU-prefill-bound, batch-saturated(128) regime, serving cache-hit (cheap-prefill) requests first
-> they finish faster + free batch slots faster -> higher throughput -> shorter queue -> lower mean TTFT.
This is the Strata/HiCache "cache-aware scheduling" pillar, as a NON-forbidden CONFIG flag (no code, no
risk). Lossless: scheduling ORDER doesn't change greedy(temp0) outputs. Risk: may starve cache-miss
(long-prefill) reqs -> watch p99/tail. Different axis than eviction (v8/v9) -> diversifies the bets.
Run (on v4's best config): `bash <EVAL/hold_run> v10-sched-lpm --schedule-policy lpm
--mamba-full-memory-ratio 1.5 --enforce-disable-flashinfer-allreduce-fusion`. Also worth `lof`.
Priority when a slot opens: this (v10, safe+high-EV) is a strong candidate to run FIRST alongside/ before
the novel eviction mechanisms, given eval scarcity.
