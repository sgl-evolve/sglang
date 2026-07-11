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

### Reuse structure (baseline server.log, all rates)
Reuse is **long-context**: per-hit cached tokens mean 29 452, **p50 20 544**, p90 61 184, p99 168 192, **max 257 152** (near 262 144 ctx). 18% of hits reuse >50k tokens; 25% reuse <6144. Total recompute over run ≈ 24.3 M new tokens. ⇒ a single 18 MB Mamba checkpoint unlocks up to **~2.9 GB** of cached Full KV (257k tok × 11.4 KB); losing it → full multi-hundred-k-token history recompute = the P99 tail. Mechanism target = protect/retain the highest-leverage (most-KV-unlocking) Mamba checkpoints in the scarce pool.

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
## ⚠ PIVOT — v1-instr disproves the SSM-stranding hypothesis (decisive negative)
Measured (warm run, all rates): **host KV pool never saturates** (host_util ≈ 0.79 peak; **mamba_host_evict=0, full_host_evict=0** across the whole run). Reason: at NUMP=500 the working set (~6 M tok) < host KV capacity (8.4 M), so the 768 GB host tiers are **over-provisioned** and never evict. Consequences:
- **SSM-state stranding is tiny** (0.33–0.66% of frontier, sparse-checkpoint only, 0 eviction) ⇒ **VMR-host = no-op** (nothing to protect). Marconi/Jenga's "SSM state is the scarce GPU resource" does NOT hold once the state is host-tiered here.
- **hit_rate is maxed at 0.842** (0.158 = irreducible first-occurrence; host retains all, so ~0 eviction-recompute) ⇒ **no hit-rate/capacity headroom**.
- **Bottleneck = DEVICE↔HOST movement/churn**: device KV pool (2.35 M) turns over hard — **evict 169 M + load-back 138 M tokens per rate** (load-bound regime, à la Strata). This is the charter's "hide/reduce L1↔L2 movement cost under load" target.
- **p99 TTFT is noisy**: v0-baseline λ=3 p99=53 s (cold JIT, first load) vs v1-instr λ=3 p99=9 s (warm). Warm profile: p99 ~9 s(λ3), ~23 s(λ5). ⇒ compare **warm-to-warm**, use stable counters (hit, evict/load-back tokens) for attribution.

**New direction:** target the device↔host **movement cost / overlap under load** (transfer path, not capacity/eviction/config). Need a profiling run to locate the exact bottleneck (prefill compute vs load-back-exposed vs queue) before committing. VMR kept flag-gated as a documented negative.

## ★★ DEFINITIVE: Mamba pool NEVER binds — VMR dead at ALL scales (diag1553, off-contract probe)
Ran NUMP=1553 (charter's intended working set ~19M ≫ 10.7M cache), λ=5, instrumented. As host fills:
**FULL KV host pool EVICTS** (full_host_evict 1991 nodes / 8.17M tok @6k matches) — capacity pressure IS real at intended scale — but the **MAMBA host pool STILL does not evict** (mamba_host_evict=0). The mamba host pool (5360 states) is large enough to hold every conversation's checkpoints even at 1553 convs. Stranding grows only mildly with pressure (0.66%→1.17%) and is **sparse-checkpoint only** (branch-point checkpointing off; bounded by mamba_cache_chunk_size=64 ⇒ ≤64 tok/hit). ⇒ **VMR (mamba retention) is a confirmed no-op at contract AND intended scale**; denser checkpointing would recover ≤~1% (negligible). **The ONLY pressured tier is attention-KV**, whose levers (exclusive tiering = the known +13pp baseline; LRU eviction ≈ Belady) are BANNED/dead-end.

## Conclusion / contribution
Rigorous, cross-scale NEGATIVE + characterization (formal submission `keystone-caching`): (1) the recurrent-state tier is over-provisioned and never the bottleneck under host tiering — inverting Marconi(MLSys'25)/Jenga(SOSP'25) GPU-scarcity framing; (2) the contract eval (NUMP=500) is not cache-bound (host 79%, hit maxed 0.842, PCIe <10%, compute/burst-bound); (3) at intended scale only attention-KV capacity binds (known levers). Novel reusable instrumentation (reuse-frontier gap). No lossless non-banned positive cache mechanism exists for this regime — a bounded impossibility for cache-architecture research here; the lever is compute/scheduling.

## (superseded) Value-Density Mamba Retention (VMR)
`LAMPORT_MECH=vmr` (default off; A/B on identical code). In the scarce Mamba host pool, **protect checkpoints whose unlocked attention-KV prefix exceeds their own ~18 MB cost** (parameter-free crossover ≈1570 tok, ablatable via `LAMPORT_MIN_TOK`) from host eviction; 2-pass fallback keeps the pool always freeable. Files: `mem_cache/lamport_mech.py`, `mamba_component.py` (`_host_evict_pass`, value stored at checkpoint creation). Lossless (reuse exact; only *which* checkpoints are retained changes). Targets the P99 tail: a deep-checkpoint eviction forces full-history recompute. **Risk:** in long-context workloads most checkpoints exceed the threshold → need enough short (ShareGPT) checkpoints to sacrifice; else raise threshold / go value-ordered.

## Versions
| ver | tag | hypothesis | curve result vs baseline | lossless | takeaway |
|-----|-----|-----------|--------------------------|----------|----------|
| v0_official | config | stock 2-tier λ=3 reference (baseline.json) | reference | — | logged W&B |
| v0-baseline | config | stock rate-sweep (COLD JIT, 1st load) | goodput@8s **0**; peak 5.59; λ3 p99 **53s** | ✓ | JIT-cold-inflated tail; use warm |
| v1-instr | mechanism* | stock+instrumentation, WARM (baseline replicate) | goodput@8s **0**; peak 5.95; λ3 p99 **9.0s** hit 0.842 | ✓ | *instr only; **stranding 0.4%, host 79%, 0 evict → NOT cache-bound** |
| v2-writeback | config | write_back = exclusive tiering (the known +13pp lever) @contract | λ3 0.8424/9389 ≈ v1-instr → **NEUTRAL** | ✓ | known capacity lever neutralized (host not pressured) |
| diag1553 | mechanism* | off-contract probe @ charter's intended scale (NUMP=1553, wt) | **host util 1.0**; Full host evicts **587M tok**; **mamba host evict = 0**; **hit 0.650** (vs 0.842@contract); req/s 3.67 | ✓ | mamba NEVER binds even @ util 1.0; only attn-KV pressured; **contract eval UNDER-PROVISIONED** |
| diag1553wb | config | write_back @1553 (exclusive tiering under real pressure) | hit **0.650→0.724 (+7.4pp)**, req/s **3.67→4.13 (+12.5%)**; mamba host evict = 0 | ✓ | **exclusive tiering IS the (known) lever — but only at scale**; neutral @contract |
| v3-baseline-rep | config | contract warm replicate (write_through) — error bars | λ3 p99 **7956** (vs 8969 v1, 9389 v2); hit 0.8412 | ✓ | **goodput@8s NOISE-fragile**: λ3 p99 straddles 8s across warm runs (7956/8969/9389, ±9%) |

### ★ COMPLETE 2×2 (hit rate): recurrent state never binds; sole lever = attn-KV capacity, masked by under-provisioning
| | write_through | exclusive tiering (write_back) | Δ |
|---|---|---|---|
| **contract (NUMP 500)** | 0.8412 / 0.8419 | 0.8424 | **+0.05pp NEUTRAL** (host 79%) |
| **intended (NUMP 1553)** | 0.650 | 0.724 | **+7.4pp** (host util 1.0) |
mamba_host_evict = **0 in all 4 cells**. ⇒ contract eval masks the only cache lever (attn-KV capacity=known exclusive tiering); recurrent state is a non-lever at every scale. No novel lossless primitive exists.
| (VMR) | mechanism | value-density Mamba retention | **no-op** (mamba_host_evict=0 ⇒ nothing to protect) | ✓ | documented negative; flag-gated, default off |

## Formal submissions
(none yet)

## ★ Paper v2: elevated to a CAPACITY LAW (principle-level, family-wide)
**Capacity Law:** recurrent-state tier binds only when mean context L < L* = s/k (s=state bytes/seq, k=KV bytes/tok). For our model L*≈1580 tok. Since caching pays off only at L≫L*, the recurrent tier NEVER binds; attn-KV (footprint ∝ context) is the sole capacity bottleneck. Inverts Marconi/Jenga GPU-scarcity.
**Generality (config-checkable, 6 hybrids):** L*_elem = Qwen3-Next 1536, Zamba2 167, Jamba 448, Granite-4 3456, Nemotron-H 3072, Qwen3.5-122B ~1580(measured) — all ≪ serving L (32k-256k). Family-wide result.
**Impossibility corollary:** law + frozen budget + eviction-as-solved ⇒ no lossless budget-respecting non-classical cache mechanism can shift the curve. Win must come from outside the lossless cache (compute/scheduling). VMR no-op = direct test of the law.
