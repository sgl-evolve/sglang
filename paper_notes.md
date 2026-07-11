# Paper notes — stable framing (fill results after v1-instr/v2-vmr)

**Working title:** *Keystone Caching: Value-Density Retention of Recurrent State in Tiered Hybrid-Attention LLM Serving*

**One-line claim:** In hybrid (linear-attention + full-attention) LLM serving with tiered KV, the tiny recurrent SSM checkpoint is a *keystone* that gates reuse of a much larger, still-resident attention-KV prefix; managing the scarce SSM-state pool by the KV it unlocks (not by recency) eliminates catastrophic full-history recomputes and shifts the throughput–latency curve, losslessly.

## The gap (novelty anchor) — verified via prior-art sweep
- **Marconi (MLSys'25)** — hybrid prefix caching, FLOP-efficiency eviction `S=recency+α·FLOPs/mem`. **GPU-HBM only: on eviction it DISCARDS SSM state, never tiers it.**
- **Jenga (SOSP'25)** — heterogeneous on-GPU allocator (page = LCM of embedding sizes) so big Mamba states coexist with small KV blocks. **No host offloading.**
- **SGLang HiCache (this system)** — DOES tier the SSM state to host (MambaPoolHost) + overlapped load-back — but manages the two pools (KV, Mamba) by **independent LRU**, blind to the cross-component coupling. Branch-point checkpointing disabled in HiCache mode (documented TODO).
- **vLLM** — RFC #17140: "SSM state not managed by block manager → incompatible with prefix caching / KV offloading / PD-disagg." Hybrid prefix caching still experimental.
- **Cake (ICML'25) / KVPR (ACL'25)** — recompute-vs-load arbitration, **for transformer KV only**.
- **AttentionStore/CacheGen/Strata/HCache/InfiniGen** — transfer/compute overlap, layout, importance-prefetch — **all for transformer KV; none manages the constant-size recurrent state under a host tier.**

**⇒ Unclaimed:** cross-component, value-aware co-management of the *tiered* recurrent-state pool, where the SSM checkpoint's eviction cost = the attention-KV it unlocks. This is the "keystone" coupling; it is specific to hybrid models and does not exist in transformer-KV tiering.

## Why it's not the banned levers
- NOT capacity/exclusive-tiering (+13pp): total memory unchanged; pools not resized (hicache_size frozen).
- NOT a classic policy on plumbing: the value is a *cross-component* quantity (SSM checkpoint ↦ unlocked attention-KV), a coupling absent in single-cache LRU/LFU/GDSF. Insight generalizes to any hybrid (Jamba/Zamba/Qwen3-Next/MiniMax/Falcon-Mamba).
- Lossless: exact states, exact reuse; only *which* checkpoints are retained changes.

## Motivation data (measured, this system, Qwen3.5-122B, 8×H100)
- Model 48 layers = 36 GDN linear + 12 full-attn. **Mamba state ≈ 18 MB/seq**, pool 1351 dev / ~5360 host slots. KV 2.35M dev tok, 8.4M host tok.
- Baseline rate-sweep: goodput@8s-SLO **0**; peak 5.59 req/s; p50 TTFT 1.2–1.6 s but **p99 28–53 s**; hit 0.841; PREFILL-BOUND (input 43k tok/s vs output 310 tok/s).
- Reuse is long-context: per-hit cached **p50 20 544 tok, max 257 152**. A single 18 MB checkpoint unlocks up to ~2.9 GB of cached KV.
- [PENDING v1-instr] stranded_tok = Full KV present but un-continuable because its Mamba checkpoint was evicted (= avoidable recompute); mamba_host_evict_nodes.

## Evaluation plan
- Curve: goodput@SLO + p50/p90/p99 TTFT and req/s across λ∈{3,5,7,10}, VMR vs stock. n≥2 replicates.
- Attribution: instrumentation shows stranded_tok ↓ and protect_skips; recompute-token delta.
- Ablation: threshold sweep (LAMPORT_MIN_TOK); VMR-off = stock (identical code).
- Lossless check: outputs vs baseline.
