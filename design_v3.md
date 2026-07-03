# v3+ mechanism candidates (pick from v2 evidence)

Context: v2 makes the L3 hit-query O(keys) (was O(total-files) scandir). Strata's thesis:
prior tiered-KV systems are *loading-bound, not compute-bound* — schedulers ignore cache-load
delays. After v2 the hit-query is cheap; the residual TTFT tail (if any) comes from (a) actual disk
read time under bursts (bandwidth ~6.6 GB/s, already ~saturated by 8 ranks) and (b) requests blocking
on those reads (wait_complete) while the GPU may idle. Decide v3 from v2's iostat + GPU-util +
per-tier metrics.

## C1 — Congestion-aware adaptive prefetch (wait ↔ recompute)  [NOVEL, top pick]
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

Order of attack after v2: read iostat (read vs write GB/s, %util) + GPU-util during stalls.
- disk %util high & writes large → C4 then C2.
- disk %util moderate but GPU idles during stalls → C1 (recompute) and/or C3.
- reads dominate & serialized → C2 (SJF) first (safe, policy-independent).
