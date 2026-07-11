# Research log — researcher **lamport** (v0.3, RESEARCH-TIER)

**Name:** lamport (Leslie Lamport). Branch `evolve/lamport`; W&B run `lamport` in `sgl-evolve`; workspace `researchers/lamport`.
**Bar:** top-venue only (SOSP/OSDI/MLSys/ASPLOS/FAST). One deep, novel mechanism that shifts the goodput@SLO curve, lossless.
**Protocol (fixed):** full-decode Poisson rate sweep λ∈{3,5,7,10}, NUMP=500 multiturn, MAXC=256, headline = max req/s with p99 TTFT ≤ 8 s. 2-tier (L1 GPU + L2 768 GB host, NO L3). Model Qwen3.5-122B-A10B-FP8 (hybrid GDN/Mamba + full-attn), TP8.

## System map (LIVE code, my clone @ a334877e5)
- Cache class = **`UnifiedRadixCache`** with components `[FULL, MAMBA]` (registry.py:99-104). `hi_mamba_radix_cache.py` / `mamba_radix_cache.py` are **dormant** for this config.
- Component architecture: `unified_cache_components/` (README is the design doc). Per-component LRUs, lock_refs, eviction drivers. FULL = path-structured KV (device+host leaf sets). MAMBA = **leaf-only** constant-size SSM state (copy-on-write on match), separate device+host LRUs, tiered to host (`MambaPoolHost`, page_first_direct/direct only).
- **Reuse boundary** (`_match_prefix_helper`, unified_radix_cache.py:885) = deepest node where **ALL** components validate (Full device|host AND Mamba device|host). Mamba checkpoints created at chunk boundaries (chunked prefill, `cache_unfinished_req`) + request ends. **Branch-point mamba checkpointing is DISABLED in HiCache mode** (mamba_component.py:86 "can add a HiCache-aware branching policy later") — a documented gap.
- Load-back (H→D) already **overlapped** with compute (load_stream + LayerDoneCounter). No recompute-vs-load arbitration (always load if cached; init_load_back @2410). Cascade eviction priority Full(2)>SWA(1)>Mamba(0): evicting Full cascades Mamba, not vice-versa.

## Baseline (stock 2-tier), single-λ reference (baseline.json, from supervisor, λ=3)
ttft p50 750 ms, p99 6326 ms, hit 0.622, host_util 1.0, req/s 2.78, evict 582 M tok, load_back 298 M tok, hit_device 0.40 / hit_host 0.60.
→ **Capacity-bound** (working set ~19 M ≫ cache ~10.7 M). Capacity/exclusive-tiering & eviction-policy are BANNED/dead-end. Goodput headroom must come from the **latency side under load** (shift curve left/down), not hit-rate.

## Direction (being pinned by data)
Candidate leads, hybrid-specific & non-banned:
- (A) Reuse-frontier-aligned dense/branch-point Mamba checkpointing + host tiering (raise effective reuse depth → less recompute). Headroom = cross-request document sharing; bounded by chunk_size granularity.
- (B) Latency-side: reduce load-back / eviction churn on the critical path under load; better transfer/compute co-scheduling.
- Deciding via baseline curve + instrumented characterization (trace-driven motivation). Will NOT commit until headroom is measured.

---
## Versions
| ver | tag | hypothesis | curve result vs baseline | lossless | takeaway |
|-----|-----|-----------|--------------------------|----------|----------|
| v0-baseline | config | stock rate-sweep = my comparison curve | (running) | — | — |

## Formal submissions
(none yet)
