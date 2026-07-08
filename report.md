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

## Versions

_(none logged yet — designing first mechanism)_
