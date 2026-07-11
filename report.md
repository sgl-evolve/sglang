# Research log — researcher **lamport** (v0.3, RESEARCH-TIER)

**Name:** lamport (Leslie Lamport). Branch `evolve/lamport`; W&B run `lamport` in `sgl-evolve`; workspace `researchers/lamport`.
**Bar:** top-venue only (SOSP/OSDI/MLSys/ASPLOS/FAST). One deep, novel mechanism that shifts the goodput@SLO curve, lossless.
**Protocol (fixed):** full-decode Poisson rate sweep λ∈{3,5,7,10}, NUMP=500 multiturn, MAXC=256, headline = max req/s with p99 TTFT ≤ 8 s. 2-tier (L1 GPU + L2 768 GB host, NO L3). Model Qwen3.5-122B-A10B-FP8 (hybrid GDN/Mamba + full-attn), TP8.

## System map (LIVE code, my clone @ a334877e5)
- Cache class = **`UnifiedRadixCache`** with components `[FULL, MAMBA]` (registry.py:99-104). `hi_mamba_radix_cache.py` / `mamba_radix_cache.py` are **dormant** for this config.
- Component architecture: `unified_cache_components/` (README is the design doc). Per-component LRUs, lock_refs, eviction drivers. FULL = path-structured KV (device+host leaf sets). MAMBA = **leaf-only** constant-size SSM state (copy-on-write on match), separate device+host LRUs, tiered to host (`MambaPoolHost`, page_first_direct/direct only).
- **Reuse boundary** (`_match_prefix_helper`, unified_radix_cache.py:885) = deepest node where **ALL** components validate (Full device|host AND Mamba device|host). Mamba checkpoints created at chunk boundaries (chunked prefill, `cache_unfinished_req`) + request ends. **Branch-point mamba checkpointing is DISABLED in HiCache mode** (mamba_component.py:86 "can add a HiCache-aware branching policy later") — a documented gap.
- Load-back (H→D) already **overlapped** with compute (load_stream + LayerDoneCounter). No recompute-vs-load arbitration (always load if cached; init_load_back @2410). Cascade eviction priority Full(2)>SWA(1)>Mamba(0): evicting Full cascades Mamba, not vice-versa.

## Hardware / pool sizes (per TP rank ×8, from server.log)
- **Mamba pool: 1351 slots, ssm 23.77 GB → ~18 MB per sequence-state** (LARGE, scarce). max_running_requests=270 → ~1081 free cache slots. **Mamba HOST pool = 96 GB/rank (~5360 slots).**
- KV pool: 2.35 M device tokens (27 GB); **KV HOST = 96 GB (~8.4 M tokens).** So Mamba host pool (768 GB total) is co-equal to KV host pool.
- Model 48 layers = 36 GDN linear-attn + 12 full-attn (interval 4); GQA 2 KV heads / head_dim 256. `mamba_cache_chunk_size` = max(FLA 64, page 64); `enable_int8_mamba_checkpoint`=False (off, lossy).

## Baseline v0-baseline (my rate-sweep, stock), λ=3 point [running]
req/s 2.80, hit **0.842**, input **42953 tok/s** vs output **310 tok/s** → **PREFILL-BOUND**. TTFT: median **1.64 s**, P90 **27.4 s**, P99 **53.5 s**; E2E median 7.2 s. TPOT median 200 ms, P99 15.5 s.
→ **Tail-latency regime**: median fine, but P99 TTFT 53 s (goodput@8s-SLO ≈ 0). Tail = **queue-driven head-of-line blocking by large prefill recomputes**. Reducing aggregate recompute drains the queue → cuts the tail → raises goodput. Capacity/exclusive-tiering/eviction-policy remain BANNED/dead-end.

## Central hypothesis (gated by v1-instr instrumentation)
The **large, scarce Mamba state (18 MB; 1351 dev / 5360 host slots)** is the binding reuse resource. Independent LRU eviction of Mamba vs Full → a Mamba checkpoint is evicted while its Full-KV prefix stays resident (KV host pool has room) → **Full KV stranded (present but un-continuable) → catastrophic full-history recompute → the P99 TTFT tail.** Mechanism = **hybrid dual-cache reuse-frontier co-management** to eliminate stranding → kill the tail → goodput. Lossless (exact states, exact reuse).

### Decision tree (from `tools/parse_instr.sh` on v1-instr)
- stranded ≥5% **and** mamba_host_evict>0 → EVICTION-driven stranding → **frontier-coupled retention** (protect load-bearing Mamba, evict orphaned first).
- stranded ≥5%, mamba_host_evict≈0 → SPARSE-checkpoint stranding → **denser/branch-point Mamba checkpointing** (HiCache-aware, currently disabled).
- stranded <1% → PIVOT (elastic host-pool sharing between Mamba/KV host; or latency-hiding/scheduling of the recompute tail).

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
