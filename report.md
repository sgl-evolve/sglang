# base — sglang KV-cache research (v0.31, full-decode rate sweep, goodput@SLO)

Researcher: **base** (independent replicate). Branch `evolve/base`. Clone base commit `a334877e5`.
W&B: project `sgl-evolve`, run `base` (group v0.31).

## Protocol (v0.31, recalibrated vs v0.3)
- 2-tier HiCache: L1 GPU HBM + L2 host DRAM (768 GB, `--hicache-size 96`), **no L3/disk**.
- Model Qwen3.5-122B-A10B-FP8 (hybrid-Mamba MoE), tp8, ctx 262144, full real decode.
- **Rate sweep** λ∈{3,5,7,10}, Poisson open-loop, `--max-concurrency 256`, `num-prompts 1553`
  (~19M-tok working set ≫ L1+L2 ~10.7M → genuine pressure), **warmup burst (300 convs) + NO per-rate
  flush** → warm steady-state (fixes v0.3's bimodal cold-start metric).
- **Headline: goodput@SLO = max achieved req/s among rates with p99 TTFT ≤ 8 s.** Honest controls
  (decode-bound, expected ~flat): peak out tok/s, peak req/s. Lossless gate: outputs == no-cache.

## Active code path (verified from registry.py, not assumed)
- registry.py `default_radix_cache_factory`: hybrid-SSM + `--enable-hierarchical-cache`
  ⇒ **`UnifiedRadixCache`** via `_create_unified_radix_cache` (comment: "HybridModel launches HiCache
  via UnifiedRadixCache by default"). `HiMambaRadixCache`/`HiRadixCache` are **DORMANT** for this model.
  (Runtime confirmation pending via the `Tree cache initialized: ... impl=UnifiedRadixCache` server.log line.)
- Live 2-tier levers (no L3, so `prefetch_from_storage` is dormant):
  - `_inc_hit_count` (unified_radix_cache.py:1810) → `write_backup` on hit (write_through eager backup, D→H).
  - `load_back` (:1660) H→D on prefix match when device-evicted; `init_load_back` (:2410) from scheduler.
  - eviction: `evict`→`FullComponent.drive_eviction` (LRU via `eviction_strategy`, default `LRUStrategy`);
    `_evict_device_leaf` (:1482) demotes backuped leaves to host (`_evict_to_host`) or deletes if unbacked;
    `evict_host`/`_evict_host_leaf` for L2.
  - `full_component.commit_hicache_transfer` LOAD_BACK (full_component.py:316) sets `cd.value` but
    **never frees `cd.host_value`** ⇒ device-resident KV stays duplicated in host (inclusive tiering).
- Invariant: `node.backuped` ⟺ FULL host_value present; `node.evicted` ⟺ FULL device value None.
  Mamba host state is O(#seq) — tiny, never the binding tier (host_util pressure is the FULL-KV host pool).

## Key regime fact (baseline.json, single-point λ=3, old format)
hit 0.62, **host_util 0.9999** (L2 saturated, forced eviction), ttft_p99 6326 ms, req/s 2.78,
load_back_mean 1.65 ms, evict_mean 1.02 ms, load_back_tokens 298M, evict_tokens 582M. TTFT is
prefill-compute-bound at this point; load_back is cheap. ⇒ the lever on TTFT/goodput is **hit-rate**,
which is **capacity-bound** at host_util=1.0. Raising hit without more memory ⇒ use L1+L2 more
efficiently ⇒ eliminate inclusive duplication ⇒ **exclusive (device-XOR-host) L1↔L2 placement**
(explicitly in-charter scope: "L1↔L2 placement/layout"). NOT eviction-order tuning (LRU≈Belady dead end).

## Version log

### v0 — stock baseline sweep (control)  [RUNNING/queued: job 19435]
- Hypothesis: n/a (control). Establishes the goodput@SLO curve + where p99 crosses 8 s (prefill- vs
  decode-bound), the linchpin for choosing the mechanism.
- Change: none (stock `UnifiedRadixCache`, write_through, inclusive tiering).
- Result: PENDING (queued behind v0.3 holds + sibling v0 baselines; certified pool saturated).
- Lossless: n/a (reference).

### v1 — exclusive L1↔L2 tiering (mechanism, PLANNED)  [implemented, env-gated SGLANG_XTIER, not yet run]
- Hypothesis: host_util=1.0 means the host tier is the binding capacity; write_through keeps a host copy
  of every device-resident hit node (inclusive) ⇒ the host wastes ~L1-worth of capacity on duplicates.
  An **exclusive** hierarchy (KV in device XOR host, never both) frees that capacity for unique evicted
  KV ⇒ higher hit ⇒ fewer prefill tokens ⇒ lower p99 TTFT ⇒ higher goodput@SLO. Pure code; no flag
  achieves free-host-on-loadback (write_back still keeps the post-loadback host copy).
- Change (gated `SGLANG_XTIER=1`, default off ⇒ resolved_args frozen at write_through):
  (1) suppress eager write_through backup-on-hit; (2) backup-on-eviction (write_back-on-evict) so a
  device-only node is demoted not deleted; (3) free `cd.host_value` on load-back commit (the exclusive part).
- Certification plan: SAME-node A/B (env toggle 0 vs 1), resolved_args identical (write_through), code
  diff = the mechanism; replicate to beat node variance (±14% req/s, ±30% p99 per prior campaigns).
- Result: PENDING.

## Ops notes
- eval.sh has a path bug (computes `workspace/sgl/v0.3_ablations/base`); fixed by symlink
  `v0.3_ablations/base → v0.31/base` (frozen eval.sh untouched — fairness-clean).
- Contended pool: 4-way (base + 3 research) + v0.3 holds. Evals queue; research/code in parallel.
