# Recompute-Cost-Aware KV Eviction for Tiered Caches under a Tail-Latency SLO

**Researcher:** `sgl_mech` (v0.25_ablations, mechanism-only). **Branch:** `evolve/sgl_mech`.
**One line:** eviction from a tiered LLM KV cache should minimize the *recompute cost* of misses, not their
*count* — a distinct objective from LRU/Belady that shifts the goodput curve up losslessly.

This is the reviewable summary; `report.md` is the full dated log and `reports/log.md` the milestone trail.
Every number below is reproducible from the commit noted; the mechanism lives entirely in `python/sglang/srt/**`.

## 1. Problem & setting
SGLang serves a hybrid attention/SSM MoE (Qwen3.5-122B-A10B-FP8, TP8, ctx 262144) under KV-cache memory
pressure with **HiCache** tiering KV across L1 (GPU HBM) and L2 (768 GB host DRAM) — **2 tiers, no disk**.
Fixed protocol: the real-text 1:1:1 mix (ShareGPT+LEval+LooGLE, `mooncake_mix_v1.jsonl`, 1553 convs → 7037
turns / ~19 M tok), multiturn, healthy rate λ=3, max-concurrency 128. Headline metric: **max sustainable
req/s under p99 TTFT ≤ 8 s**. The working set (~19 M tok) far exceeds L1+L2 (~10.7 M) → genuine pressure.

## 2. Key insight
Under in-order reuse, LRU ≈ Belady on hit *count* (<0.1 pp headroom) — so count-optimal eviction is a dead
end. But **per-miss recompute cost is not uniform**: it scales with the lost prefix length, which spans orders
of magnitude (a 1 k chat turn vs a 128 k document prefix). Under a **tail SLO**, and for compute efficiency,
what matters is the *cost* of the misses, not their number. The few catastrophic long-prefix recomputes
dominate both the p99 tail and total prefill work. **Eviction should therefore be recompute-cost-aware:** evict
cheap-to-recompute prefixes before expensive ones, LRU within a cost tier. LRU is count-optimal; this is
cost-optimal — a different, and under an SLO more relevant, objective.

## 3. Mechanism (engine code)
`CostAwareStrategy` (`python/sglang/srt/mem_cache/evict_policy.py`) plugs into the radix-cache eviction victim
heap (device `drive_eviction` + host `drive_host_eviction` in the FULL component). `get_priority(node)` returns
`(tier, last_access_time)`: `tier = 0` (evict first) if the node's recompute cost `< threshold`, else `1`
(protected); the min-heap pops cheap-then-old. Cost = the node's own segment length (`len(node.key)`). Selected
via `--radix-eviction-policy cost_aware` (registered first-class in `_EVICTION_POLICY_FACTORIES`, `utils.py`);
an env override of the default `lru` (`SGLANG_ENABLE_COST_AWARE_EVICTION`, `SGLANG_COST_AWARE_EVICT_THRESHOLD`)
exists only for the mechanism-only ablation. **Lossless by construction:** eviction *order* only decides hit vs
miss; a miss recomputes bit-identical KV. Best operating point: **threshold ≈ 2048 tokens** (a U-optimum —
t1024 over-protects and worsens p99, t8192 under-protects and loses hit-rate). Hardened with 22 CI-registered
unit tests (`test/registered/unit/mem_cache/test_evict_policy.py`).

## 4. Results (fixed protocol; same-node paired A/B; error bars over repeats)
| metric | stock LRU | cost-aware @t2048 | delta | robustness |
|---|---|---|---|---|
| **max throughput** (λ=6 saturation) | 3.98 req/s | 4.445 req/s | **+11.7%** | n=2 sweeps, reproducible (cost 4.44/4.45) |
| total prefill new-tokens (λ=3–6 sweep) | 151.9 M | 129.4 M | **−14.8%** | n=2, NON-overlapping, <1% within-cond spread |
| token hit-rate | 0.6185±.006 | 0.6786±.005 | **+6.0 pp** | non-overlapping (n=4 stock / n=8 cost) |
| p50 TTFT (λ=3) | 591±97 ms | 496±15 ms | **−16%** | non-overlapping |
| p99 TTFT (λ=3) | — | — | **−10%** | paired same-node (3 paired deltas, all negative) |
| lossless gate | — | — | Successful = 7037/7037 | verified across all sweep points |

**Deterministic core:** the −14.8% prefill-compute reduction is a cache-behavior *count* (immune to the
cluster's cross-run latency noise), measured at the SAME memory budget (device-KV-util 0.33 vs 0.33). In the
compute-bound overload regime this is exactly why max throughput rises +11.7%: same memory, less compute, more
goodput. Commits: mechanism `f142702d3`; error-bar correction `a9665d9f7`; upstream policy `b250344dc`; tests
`5997f8f37`.

## 5. Why it works — two regimes, one lever
- **Aggregate hit-rate is capacity-bound** (19 M ≫ 10.7 M, ceiling ~0.67); cost-aware ordering wins the
  accessible part by keeping the expensive long prefixes resident.
- **The p99-SLO knee is prefill-compute-bound** (from server counters at peak queue: device KV only 33%/70%
  used, Mamba 18%, but ~450 k pending prefill tokens, 95% of queued prefills cold). The lossless lever there is
  cutting the *warm-recompute* fraction — which cost-aware does. Per-λ (contamination-immune scheduler
  counters), at the knee (λ=4) cost-aware cuts prefill backlog **−17.5%** and queue depth **−13.6%** vs stock,
  at every λ → the goodput curve shifts right.

## 6. Ablations & design-space bounding (all neutral-or-worse / infeasible / off-limits)
- Threshold sweep t1024/t2048/t8192 → **t2048 U-optimum**.
- 3-tier cost segmentation, depth-cost (both ends: over-protect→~LRU / under-protect), cost-aware **Mamba**
  eviction, reuse-gating → **all NEUTRAL** (empirically or by the cost-correctness argument).
- Partial hybrid reuse (cache 12 attn-KV, recompute 36 GDN states) → **infeasible**: layer interleaving couples
  GDN recompute to O(L²) attention recompute.
- Scheduling / admission → cannot reduce the cold-prefill compute that sets the knee (compute-bound, with data).
- Concurrent prefill coalescing → **<1%** headroom (888/1553 docs unique; dedup dominated by an empty
  ShareGPT doc ×538 + one 4.2 k-tok doc ×100).
- Compression / footprint → FP8 already; further lossless ~1.1×, marginal.
- Cost-aware write-back admission → redundant with eviction (same steady-state L2 composition).
- Non-redundant/exclusive tiering → the one larger remaining lever, but a sibling cell's finding in shared
  memory; per the independence charter I do **not** adopt it.

## 7. Honest limitations
Gains are modest — the workload is capacity-bound. The **precise p99-knee Δλ is noise-limited**: a fine paired
sweep (λ=3.0/3.6/3.8/4.0, clean, anchor-validated) gave stock knee λ≈3.84 vs cost λ≈3.78 with per-λ p99 deltas
that **alternate sign** (−11.5/−1.1/+5.9/−1.7%) — i.e. no robust *direct* p99 knee shift in either direction.
The knee *direction* rests on the deterministic scheduler-state proxy (§5), not on p99 latency. Two earlier
overclaims (a single-run "Pareto +5%/−10%" and an n=1 "+8.3% knee") were **retracted** after powering/repeats.

## 8. Novelty & reproducibility
Strata (cache-aware scheduling + GPU-IO) and HiCache (hit-count-selective write-through, layer overlap) both
optimize count-hit-rate / loading latency; **neither makes eviction recompute-cost-aware for a tail SLO** —
eviction is explicitly an open area in the HiCache blog. The insight (recompute *cost*, not miss *count*, under
an SLO) generalizes to any tiered KV cache. Reproduce: `--radix-eviction-policy cost_aware` (or the
`SGLANG_COST_AWARE_*` env for the ablation); W&B run `sgl_mech` in project `sgl-evolve` holds the full curve.
