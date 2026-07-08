# base_free — sglang KV-cache research (v0.25_ablations, 2-tier L1+L2, no disk)

**Researcher:** `base_free` · branch `evolve/base_free` · base commit `a334877e5` · W&B run `base_free` (project `sgl-evolve`).

## Regime (this cell, v0.25 — distinct from v0.2)
2-tier HiCache: **L1** GPU HBM (~2.35 M tok) + **L2** host DRAM 768 GB (~8.4 M tok) = **~10.7 M** capacity.
**No L3/disk** — overflow beyond L1+L2 is *recompute*. Model Qwen3.5-122B-A10B-FP8 (hybrid Mamba), TP8,
ctx 262144. Load: real-text Mooncake 1:1:1 mix (1553 convs / 7163 turns / ~19 M tok working set),
**λ=3** Poisson, max-concurrency 128, multiturn. Active cache path = **UnifiedRadixCache** +
`managers/cache_controller.py` (hiradix/hi_mamba dormant).

## Baseline `v0_official` (given, not re-run)
- TTFT p50 **750 ms** / p99 **6326 ms** (within the 8 s SLO) / mean 1146 ms
- req throughput **2.78** req/s (λ=3), out 355 tok/s
- hit_rate **0.6217**; of hits: device 40.2% / host 59.8%
- host_util **0.9999** (L2 saturated → forced eviction — the pressure signal)
- churn: load_back **298 M** tok, evict **582 M** tok (≫ 99.9 M prompt tok) = heavy thrashing
- resolved: io=direct, layout=page_first_direct, write=write_through, page_size 64

## Workload structure (measured from mooncake_mix_v1.jsonl)
- 1553 convs, **7163 turns**, mean **4.6 turns/conv** (median 3, max 61).
- Each conv = a big **document** (median ~29.5K chars ≈ 7K tok; mean ~49K; max 763K ≈ 190K tok, heavy tail)
  + growing Q/A. Turn N prompt = doc + QA[1..N-1] + Q[N].
- Dispatch (bench_serving): each conv re-enqueued to **back of a FIFO queue** after each turn, interleaved
  across ~128 concurrent convs at λ=3. ⇒ turn N+1 reuses `doc+QA[1..N]` after a **full queue-cycle
  eviction window**. Cross-conv prefix sharing ≈ 0 (distinct docs); within-conv prefixes nested/growing.

## Target
Concurrency-regime KV-locality, **lossless**: keep a conversation's about-to-be-reused prefix (esp. its
large document) co-resident in L1+L2 under load. NOT eviction-policy tuning (LRU≈Belady per charter).
Levers: reuse-/prefix-aware routing · admission · prefetch (=proactive load_back here) · L1↔L2 placement.

---

## Offline simulator (sim/) — validated proxy, screens DELTAS
Faithful 2-tier LRU sim over the REAL tokenized workload + exact bench_serving FIFO re-enqueue
(round-robin waves). Validation vs baseline.json: prompt_tokens 99.66M vs 99.9M (exact access model);
device/host hit split 0.36/0.64 vs 0.40/0.60; host_util 1.0; hit 0.594 vs 0.6217 (sim slightly
over-evicts — fine for deltas). **Infinite-cache ceiling hit = 0.805.**

### Key diagnosis
- Headroom = **ceiling 0.805 − baseline ~0.60 = ~21 pp lost to capacity eviction** (recoverable),
  on top of ~19.5% unavoidable cold (turn-0 docs + each turn's new Q).
- Per-turn hit: **turn 0/1/2 ≈ 0.00** (doc evicted by wide early waves before its reuse; 34% of prompt
  volume), turn 3 ≈ 0.50, turn 4+ ≈ 0.996. The loss is concentrated in the FIRST reuses.
- Root cause: bench_serving re-enqueues a conv to the BACK of a long queue → ~300 s inter-turn gap →
  ~900 concurrently-ACTIVE convs → working set ~11 M > cap 8.4 M → docs thrash out before reuse.
- **GANG admission sweep (bound active conv set to G):** G=1553→hit 0.594; G=700→0.801; **G≤500→0.805
  (=ceiling), capacity_recompute→0.** ⇒ bounding concurrent conversations to fit the working set
  recovers the ENTIRE headroom. Eviction *order* can't (LRU≈Belady); admission *can*.

### Mechanism direction (v1)
**Cache-pressure-aware conversation admission**: in the scheduler prefill-admission path, when the host
tier is under pressure, DEFER starting new-conversation prefills (low resident-prefix reuse) in favor of
requests that reuse resident prefixes — bounding the active conversation working set to L1+L2 capacity.
Lossless (reorders scheduling only). Env-gated, default off = exact baseline. Backpressure via the client
concurrency limit naturally caps active convs. Latency trade-off measured on the full eval.

## Prior-art positioning (novelty)
- **Strata (2508.18572):** insight = serving is *loading-bound* not compute-bound; fixes = GPU-assisted I/O
  (decouple GPU/host layouts) + cache-aware *scheduling* (order requests to balance compute vs I/O) +
  overlap stalls. **No working-set / conversation admission control.**
- **HiCache blog:** L1/L2/L3 tiering, write-through/selective/back, storage→host prefetch (best-effort vs
  stage), layer-wise H→D overlap. **No locality-preserving admission / concurrency limiting.**
- Both optimize the *transfer/layout/order* but still THRASH under multi-turn concurrency when the active
  working set exceeds L1+L2. **WSAC is orthogonal & complementary:** it bounds *what must be resident* so
  eviction never touches about-to-be-reused documents. This is the missing admission piece.

## Hardware diagnostics (diag-baseline, mechanism OFF, live /metrics)
Steady state (host_util=1.0, from live server): run_bs≈128 (=max-conc), wait≈0-8, **cold_active≈124
(~ALL reqs COLD → reuses MISS, prefill `#cached-token:0`)** — the concurrency-eviction thrash, CONFIRMED
on hardware. `evicted_tokens`≈180M at 30% through (heavy thrash). **Device pool `/metrics`:
kv_available=1472 (~0 free), kv_evictable=2.08M (FULL of cached KV), kv_used=0.26M** ⇒ device is
INCLUSIVELY FULL (write_through mirrors hot nodes device+host). `full_token_usage` 0.11-0.47 only counts
active/locked, NOT the evictable cache — device is NOT under-used.

### Implication → two levers
1. **Admission (WSAC):** reduce effective active working set. BUT SLO-limited: turns decode-bound (~32s),
   full-turn cold cap must stay ≳96 to sustain λ=3 → mild throttle. Testing v1-wsac96.
2. **Capacity (EXCLUSIVE tiering, v2 candidate):** write_through is INCLUSIVE (device⊆host) → distinct
   cap = host 8.4M; device's 2.35M just mirrors. Making it EXCLUSIVE (device holds content NOT in host)
   → distinct 10.75M → sim hit **0.59→0.73 (+14pp), LATENCY-FREE**. Proxy=write_back (config); novel
   version = async/proactive backup to gain capacity without write_back's synchronous evict-stall.

## Versions
- **v0_official** (baseline) — logged+synced to W&B (curve point 0).
- **v1-wsac96** (WSAC admission, cap=96) — **SCREENED NEGATIVE (killed early, not logged to curve).**
  Mechanism worked as designed (run_bs 128→96, cold_active pinned 96, ~31 cold reqs deferred). BUT live
  /metrics at ~4% through: **mean TTFT 12.7 s** (>> 8 s SLO), e2e 50 s. Deferred cold prefills wait tens of
  seconds because turns are decode-bound (~32 s) → cold slots free slowly; worst in the turn-0-heavy early
  phase (all cold → cap = pure serialization). **Conclusion: cold-admission DEFERRAL cannot recover the
  21pp headroom under the p99≤8 s SLO — the latency cost dominates.** Pivot to LATENCY-FREE capacity lever.
- (v2 → EXCLUSIVE tiering: screen write_back next)
