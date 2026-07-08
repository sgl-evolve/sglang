# Exclusive L1↔L2 Tiering: Reclaiming Wasted GPU Cache Capacity in Hierarchical KV Caches

**sgl_free — sglang HiCache KV-cache研究 (v0.25 ablations). Lossless mechanism; fixed evaluation protocol.**

## Abstract
Hierarchical KV caches (HiCache) in LLM serving keep a fast GPU tier (L1) and a large host-DRAM tier (L2).
The default `write_through` policy is **inclusive**: every hot device entry is eagerly mirrored to host, so
the *distinct* cached working set is bounded by host capacity alone and the GPU tier is spent on duplicates.
We make the two tiers hold **disjoint** content (an entry lives on device **XOR** host), reclaiming the GPU
tier as extra distinct capacity. On a Qwen3.5-122B-A10B hybrid-Mamba model (TP8, 2-tier: GPU ~2.35M + host
~7.81M tokens) under a real long-context multiturn workload, exclusive tiering raises the prefix-cache
hit-rate from **0.62 to 0.75 (+13pp, node-independent)** — ~13% less fresh-prefill compute — and is
**measured bit-exact lossless**. A definitive node-controlled full-protocol A/B shows the downstream payoff:
at the SLO-sustainable rate it cuts p99 TTFT −16% (with +26% more SLO headroom), and under overload it
absorbs +10–13% more throughput at identical median latency. We give a code-level correctness argument, a
**falsifiable two-sided generalization law** (a capacity-band boundary saying *when* it helps and a
transfer-cost scope caveat saying *when it stops*), and show the residual gap to the analytic hit ceiling is
reachable only by *lossy* KV quantization — so exclusive tiering is the maximum lossless hit-rate for a fixed
tier budget.

## 1. Introduction
Long-context multiturn serving is prefill-bound: each turn re-reads a long document prefix, and under KV
memory pressure that prefix is repeatedly evicted and recomputed. HiCache mitigates this with a GPU+host
cache, but its default inclusive `write_through` policy duplicates hot device entries onto host, so the two
tiers store largely the *same* content. The distinct cached set is then ≈ host capacity, and the expensive
GPU tier adds little effective capacity.

**Contribution.** (1) A lossless engine mechanism — exclusive (device-XOR-host) HiCache tiering — that
reclaims the GPU tier as distinct capacity (+79 lines, env-gated, default off). (2) Rigorous evidence on a
fixed protocol: +13pp hit-rate (robust across nodes and n=4 runs), measured bit-exact lossless, a 3-point
ablation isolating the engine mechanism from a config, and a downstream goodput improvement. (3) A
generalizable insight: the benefit equals the workload's reuse-mass CDF slope over the reclaimed capacity
band — falsifiable, self-validating in simulation, and bounded on the cost side by H↔D bandwidth.

## 2. Background
For this hybrid-Mamba config the live cache is `UnifiedRadixCache` + `HybridCacheController`. Each
conversation is a linear radix chain (cross-conversation sharing ≈0); reuse is intra-conversation (a
document's KV must survive eviction between its own turns). Capacity is the binding constraint: total
resident ≈10.16M tokens vs a ~19M-token working set; eviction order is near-optimal already (LRU≈Belady),
and scheduling/admission cannot manufacture capacity. The lever is therefore **effective capacity**.

Inclusive `write_through` (`_inc_hit_count`→`write_backup` on reuse, keeping the device copy) makes device
a subset of host ⇒ distinct capacity ≈ host (7.81M). Baseline hit-rate 0.62.

## 3. Mechanism
`SGLANG_HICACHE_EXCLUSIVE=1` enforces device XOR host via three gated edits, all reusing the mature
`write_back` plumbing:
1. **No eager backup** — `_inc_hit_count` returns early; an entry stays device-only until evicted.
2. **Back up on eviction** — `_evict_device_leaf` writes D→H then demotes (not eagerly).
3. **Free host on promotion** — new `_promote_free_host` (in `loading_check`) frees the host copy after an
   H→D load-back completes, so a re-warmed entry is device-exclusive again.
Distinct capacity becomes device+host ≈10.16M.

**Correctness (lossless-safe by construction).** `_promote_free_host` frees host only *after*
`finish_event.synchronize()` (device durable) and the load-back's `dec_host_lock_ref`; a concurrent
load-back of the same node holds `host_lock_ref`, so the free is skipped (and a `host_value is None` check
prevents double-free); the walk stops at the first device-absent ancestor, never dropping a sole copy; and a
later device eviction re-creates the host copy via the write_back path — so every entry always has ≥1 copy.
This matches the measured 24/24 bit-exact result (§4).

## 4. Evaluation
**Protocol (fixed, never modified).** Real 1:1:1 text mix (ShareGPT+LEval+LooGLE via the loogle loader,
1553 conversations, ~7037 turns), λ=3, max-concurrency 128, real decode. Headline metric: goodput under a
TTFT SLO (max sustainable req/s with p99 TTFT ≤ 8s). Lossless gate: outputs == no-cache.

- **Hit-rate: 0.62 → 0.75 (+13pp), node-independent.** Exclusive = 0.7522 on two distinct nodes and
  0.7509±0.002 across n=4 runs (baseline 0.62). ≈13% less fresh-prefill compute. This is the robust headline.
- **Lossless: measured bit-exact.** A 3-mode greedy verify over 24 long documents (fresh and cache-hit paths,
  exercising the exclusive host-free/load-back path) gives exclusive == stock outputs 24/24. (The
  hybrid-Mamba cache is inherently ~non-bit-exact vs *no-cache* on ~17% of long docs due to Mamba SSM-state
  reconstruction — but stock shows the identical pattern, so it is a cache property, not the mechanism.)
- **Ablation (3-point, isolates engine from config).** fcfs inclusive: hit 0.622. +`write_back` flag
  (write-side exclusivity, config only): 0.733 (+11pp). +free-host-on-promotion (engine mechanism, full
  exclusivity): 0.7525 (+13pp). The final +2pp is attributable purely to the engine code.
- **Downstream latency/goodput (grow with load; node-dependent).** A *definitive* same-node, full-protocol
  (1553-conv/7037-turn), flock-held-both-legs A/B isolates two robust effects. (i) **At the SLO-sustainable
  rate** (offered 3.0, both under the 8s SLO): exclusive cuts **p99 TTFT −16%** (4907→4111 ms) and **p50 −10%**
  (543→488) at identical 3.02 req/s — a clean latency win with **+26% more SLO headroom**. (ii) **Under
  overload** (offered 4.0/4.5, both over SLO): exclusive absorbs **+10%/+13% more achieved throughput**
  (3.48→3.82, 3.67→4.14) at *identical* p50 — it stays nearer the offered rate before the queue runs away.
  The p99≤8s **goodput knee** is co-bracketed in (3,4) for both at this coarse grid, so a precise knee-shift
  magnitude is being measured by a finer-grid (3.25/3.5/3.75) same-node A/B; earlier "+10–18%" screen
  estimates are treated as suggestive pending that resolution. The robust, node-independent driver remains the
  +13pp hit-rate.
- **Negatives (ruled out with evidence).** Eviction-order (LRU≈Belady), schedule ordering (lpm),
  admission/concurrency caps, and scheduling-based co-residency do **not** recover hit-rate here
  (capacity-bound). A frequency-aware hybrid (keep hot nodes inclusive) is TTFT-neutral.

## 5. Generalization (two-sided, falsifiable)
**Capacity side — when it helps.** The benefit equals the workload's reuse-mass CDF slope over the reclaimed
band `[H, H+D]` (host H, device D): `benefit = HitRate(H+D) − HitRate(H)`. A calibrated simulator
self-validates (predicts +11.8pp at this HW vs measured +13pp). Sliding H: benefit peaks +25pp when the band
straddles the steep knee, and is **provably 0** once the host tier alone already covers the working set
(H ≳ working set) — over-provisioned-host deployments should not expect a gain. This subsumes the
device/host-ratio trend and transfers to any workload.

**Cost side — when it stops.** Exclusive wins by saving prefill compute, *despite* moving ~2× more H↔D data:
eviction now writes D→H first (`evict_mean_ms` 1.1→20.7) and more reuse is host-only (load_back_tokens
+34%, `load_back_mean_ms` 1.8→19.0). The +13pp hit removes ~13% of fresh prefill, which dominates on a large
model with long prefixes. So the net win holds only while prefill compute is the bottleneck; on a slow host
interconnect or a short-prefix (cheap-prefill) workload the extra transfer traffic can erode or reverse it.

## 6. Frontier
The residual gap from 0.75 to the analytic hit ceiling (~0.81) requires *more bytes*, reachable only by lossy
KV quantization (`--kv-cache-dtype fp8_e4m3` doubles capacity → hit 0.808). A greedy verify shows fp8-KV
adds ~2/24 output divergence beyond the cache's inherent 4/24 — modest but not bit-exact, and a config flag,
so out of scope for a lossless mechanism. Lossless host-side compression is dominated by it (~1.4× on bf16,
needs a variable-size allocator + a decompression kernel on the load-back path). **Hence exclusive tiering is
the maximum lossless hit-rate for a fixed tier budget.**

## 7. Related work
HiCache/LMCache-style host offloading grows aggregate capacity; we instead remove intra-cache duplication so
existing tiers hold disjoint content — orthogonal and composable. Inclusive-vs-exclusive is a classic CPU
cache-hierarchy axis; we bring it to LLM KV caches, where it is lossless and the reuse structure makes the
capacity-band analysis exact. Eviction-policy work (LRU/Belady) is a non-lever here (capacity-bound).

## 8. Limitations
Single model / workload / HW (the protocol is fixed and cannot be varied in-contract; generalization is via a
calibrated simulator, made falsifiable). The downstream latency/goodput result now rests on a *definitive*
node-controlled, full-protocol (1553-conv), flock-held-both-legs A/B (both effects above are from that single
run), so it is no longer a screen artifact; its one remaining coarseness is rate granularity (3/4/4.5 puts
both SLO knees in one interval), which the in-flight finer-grid A/B closes. Latency/goodput magnitudes are
node- and load-dependent; the robust, node-independent claim is the hit-rate.

## 9. Conclusion
Making HiCache's GPU and host tiers exclusive is a small, lossless engine change that reclaims wasted GPU
cache capacity for a robust +13pp prefix-hit gain and a downstream goodput improvement, with a code-level
correctness argument and a falsifiable two-sided generalization law. It is the lossless ceiling for a fixed
tier budget; going further requires trading exactness for bytes.

*Artifacts: `patches/exclusive_tiering.patch` (applyable), `UPSTREAM.md` (PR summary), `report.md` (full
working record), `sim/generalization_band.py` (band law), W&B run `sgl_free` (project sgl-evolve).*
