# base_mech — sglang KV-cache mechanism research (v0.25, 2-tier L1+L2, MECHANISM-ONLY)

**Researcher:** base_mech · branch `evolve/base_mech` · base commit `a334877e5` · W&B run `base_mech` (project `sgl-evolve`)
**Setup:** v0.25_ablations, 2-tier (L1 GPU HBM + L2 host DRAM 768 GB, **no disk/L3**). Config/flag tuning is OFF-CONTRACT — every logged version must be an **engine-code mechanism** change.

## The contract (never change)
- Model `Qwen/Qwen3.5-122B-A10B-FP8` (hybrid Mamba+attn MoE), TP=8, ctx 262144, mem-frac 0.85, hicache-size 96 (768 GB L2), page-size 64, io-backend `direct`, mem-layout `page_first_direct`, write-through.
- Workload: real-text Mooncake 1:1:1 mix (`mooncake_mix_v1.jsonl`, 1553 convs / ~19 M tok), loogle loader, multiturn, **λ=3**, max-concurrency 128, num-prompts 1553.
- **Headline metric:** goodput under p99 TTFT ≤ 8 s SLO — a mechanism must shift the whole load curve up. Per-version: TTFT p50/p99, hit-rate, L2 host-util, req/s tracks λ.
- **Lossless gate:** outputs must match no-cache run.

## Active code paths (verified from clone)
- Cache class: **UnifiedRadixCache** (`mem_cache/registry.py:101-104` → `_create_unified_radix_cache` for hybrid SSM + hierarchical).
- Host pools (L2): `MHATokenToKVPoolHost` (attn KV) + `MambaPoolHost` (SSM state), `page_first_direct`, wrapped in `HostPoolGroup`.
- Controller: **HybridCacheController** (`hybrid_cache_controller.py`): `write()`/`start_writing()` (D→H), `load()`/`start_loading()` (H→D per-layer), async `write_stream`/`load_stream`.
- Scheduler: `scheduler.py` `_get_new_batch_prefill_raw` → `check_hicache_events()`, `policy.calc_priority()` (lpm/fcfs), `PrefillAdder`. Prefix match in `schedule_policy.py match_prefix_for_req()`.
- Write-through: `unified_radix_cache.py:1810-1825 _inc_hit_count` → `write_backup` (hit_count≥1). L1 evict: `_evict_device_leaf` demotes to L2 if backed up else cascade-deletes. L2 evict: `evict_host` frees host indices → **dropped** (no L3).

## Baseline — v0_official (2-tier stock, λ=3)
| metric | value |
|---|---|
| TTFT p50 | 750.4 ms |
| TTFT p99 | 6326 ms (under 8 s SLO) |
| TTFT mean | 1146 ms |
| hit_rate | 0.6217 |
| host_util (L2) | **0.9999 (saturated)** |
| req throughput | 2.78 req/s (λ=3) |
| out_tok/s | 355.4 |
| load_back mean | 1.65 ms |
| hit split | device 0.40 / host 0.60 |

**Read of the baseline:** L2 host is 100% full (forced eviction). hit_rate 0.62 with 19 M working set vs ~10.7 M capacity. load_back is cheap (1.65 ms, ~300 GB/s host→device). The lever (per charter) is **concurrency-regime KV-locality**: keep a conversation's about-to-be-reused turns co-resident in L1+L2 under interleaved load — via reuse-/prefix-aware routing + admission + prefetch + placement. NOT eviction-policy tuning (LRU≈Belady here).

---

## Analysis (offline, sim/)
- Trace: 1553 convs / 7037 turns; each conv = 1 large doc (mean 12.3K tok, median 7K, max 192K) + ~4.5 QA turns; turn 0 prefills the doc, turns 1..n reuse it (verbatim re-send).
- Total presented prompt tok = 99.5M (≈ baseline 99.9M ✓). **Ceiling hit = 0.806** (full-history reuse), 0.780 (doc-only, retokenization-pessimistic). Baseline actual = 0.622 → **~16-18M tok/run of reusable history is recomputed.**
- Plain-LRU DES → hit ≈ 0.81 (active-set reuse distance fits easily): the 0.62 is a **HiCache mechanism inefficiency, not capacity.**
- Only truly-LOSSY path in engine: `_evict_device_leaf` DELETE of an unbacked device leaf under write-through (fires when `write_backup`→0 = host full & `evict_host` can't free). Everything else demotes to host (recoverable via load_back). Baseline: 582M dev-evict + 298M load-back for 62M hits = ~8-16× L1↔L2 thrash.
- Structural note: an ACTIVE multi-turn conv's doc node is an INTERNAL node (has next-turn child) → not a device/host leaf → protected from eviction while the conv extends. So loss should concentrate on (a) cold single-turn docs, (b) the arrival ramp where host fills faster than completed-conv leaves free.
- Prior art (HiCache blog, Strata): multi-turn session KV co-residency under concurrency is UNSOLVED; eviction not session/turn-aware; Strata is I/O-latency-scheduling, orthogonal. → my angle is novel.

## s0-diag findings (partial run, live server.log; killed at 22%)
- **dev_delete ≈ 0 (44K tok), wb_fail = 0** → the write-through DELETE path is NOT the leak; nothing lost that way. All device eviction = lossless demote-to-host (14.9M).
- **host_evict large & growing (8.5M @ 22%)** → host (L2) is the capacity bottleneck; reused content is dropped from host.
- Pool sizes (per rank): KV device **2.35M tok** (26.9GB); Mamba device **1351 slots** (ssm_state 23.8GB, ~17.6MB/state); host = **96GB KV + 96GB Mamba** (SEPARATE pools). Mamba is a distinct, tightly-bounded tier.
- `full token usage` p90=0.51 / `mamba usage` p90=0.34 = **LOCKED (running-batch) fraction only** (evictable cached counts as "available", pool_stats_observer.py:264). So device is effectively full; NOT idle.
- match_prefix consensus (`best_match_node`) requires ALL components valid incl. Mamba; mamba validator = state on device OR host (mamba_component.py:67). Mamba evicts INDEPENDENTLY of KV (separate pools/LRU) → a node's KV can be on host while its Mamba state is dropped → **consensus truncates → KV reuse lost even though KV resident.** Leading hypothesis.
- s1-diag (enhanced) adds kv_only-vs-consensus match-depth counters to confirm mamba-truncation vs KV-host-capacity.

## Decision tree (after diagnostic s0-diag counters)
- **dev_delete_tok large (~tens of M):** delete-path is the leak → Mechanism = *lossless write-through eviction* (back up or defer before deleting; never recompute what we can demote).
- **dev_delete_tok small, host_evict_tok large & reused:** host-pressure leak → Mechanism = *reduce host write-through pressure / conversation-aware admission to bound the resident working set*.
- **both small:** loss is partial-path / match / load_back timing → deeper instrumentation.

## Candidate mechanisms (choose after s1-diag discriminator)
**M-mamba (if kv_only >> consensus): Mamba/KV co-residency for hybrid models.** Mamba SSM-state evicts independently of its KV (separate pools+LRU; device tombstone mamba_component.py:220, host tombstone :548), truncating the consensus prefix match even when KV is host-resident. Fix: couple Mamba eviction to KV — never drop a node's Mamba state (device+host) while its KV prefix is retained & reusable (and prefer evicting Mamba of nodes whose KV is also being evicted). Novel: no prior serving cache co-manages SSM-state + KV residency for hybrid Mamba/attention models. Generalizable to all Qwen3.5-MoE / hybrid models. Must respect the small Mamba budget (1351 device slots) — evaluate for OOM.
**M-host (if kv_only ≈ consensus ≈ 0.62): conversation-coherent host retention.** Reused KV dropped from host under pressure/ramp. Fix: conversation-/reuse-aware host admission + retention that keeps an active conversation's prefix co-resident across its turns (charter thesis). 
**M-thrash (secondary): reduce 298M load-back / 582M evict churn** by keeping hot prefixes device-resident (fewer H→D reloads) — improves TTFT even if reuse is preserved.

## Versions
| ver | commit | tag | hit | TTFT p50/p99 | host_util | req/s | note |
|---|---|---|---|---|---|---|---|
| v0_official | stock | baseline | 0.6217 | 750/6326 | 0.9999 | 2.78 | given baseline |
| s0-diag | ed175dc05 | (screening) | _running_ | | | | baseline behavior + loss counters |
