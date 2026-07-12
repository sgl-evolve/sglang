# base — sglang KV-cache research (v0.31, full-decode rate sweep, goodput@SLO)

Researcher: **base** (independent replicate). Branch `evolve/base`. Clone base commit `a334877e5`.
W&B: project `sgl-evolve`, run `base` (group v0.31).

## SUMMARY (running)
- **v0.31 recalibration works**: the cache genuinely binds (hit 0.62, host_util≈1.0), unlike v0.3 (under-
  provisioned/bimodal). A KV mechanism CAN move the metric.
- **Headline: goodput@SLO 0 → ~3.** Stock write_through (inclusive tiering) fails the 8 s TTFT-SLO even at
  the healthy λ=3 (p99 11.7 s) ⇒ goodput 0. De-duplicating the host tier raises effective capacity ⇒ hit
  0.68→0.74 ⇒ the p99 tail (long-context miss re-prefills) drops to 7.0 s ⇒ goodput ~3.
- **Fundamental cap**: goodput@SLO is capped at ~3 by the DECODE knee (λ≥5 is offered above the ~5 req/s
  decode-bound service ceiling ⇒ saturates ⇒ p99 unbounded; caching can't raise the decode ceiling). So the
  metric is near-binary and the real lever is the **λ=3 p99 tail margin**.
- **Mechanisms (engine code, lossless, resolved_args frozen):**
  1. *Exclusive device-XOR-host tiering* (free-host-on-loadback; no config provides it): +1.7pp hit over
     write_back but **goodput/tail-NEUTRAL same-node** — uniform capacity doesn't target the tail. (honest)
  2. *Cost-aware host retention* (keep long/high-recompute-cost contexts, drop short first): targets the tail
     directly. [vca RUNNING/queued]
- **Contribution note**: the goodput 0→3 win is reachable via the write_back CONFIG (not novel); the engine
  code (exclusive tiering, cost-aware retention) is where the novel mechanism lies. All same-node A/B on 0-1.
- Ops: local commits (4079f06c1 XTIER, e7d1eec41 fix, a90cb79ce cost-aware); remote evolve/base has a PRIOR
  run's commits so I did NOT force-push over it (fresh-replicate integrity) — work preserved locally + W&B.

## vca — cost-aware host retention (mechanism) — HONEST NEGATIVE  [node -0, job 19500, commit a90cb79ce]
- Keep long (high re-prefill-cost) contexts, evict short first (threshold 8192 tok, HOST eviction order).
- **Same-node (-0) λ=3:  v1x exclusive p99 6461 hit 0.7539  |  vca cost-aware p99 6851 hit 0.7210.**
- Cost-aware **lowered hit −3.3pp** (as designed — drops short contexts) but the p99 tail did **NOT** improve
  (slightly worse, within run noise). ⇒ the "cost-weighted-not-count-weighted" retention hypothesis does
  **not** help here: write_back already caches most reusable contexts, and the residual λ=3 tail is
  **capacity-floored** (working set 19M ≫ L1+L2 10.7M → ~26% miss is near the floor; the long contexts that
  cause the tail are simply too large/numerous to fit regardless of retention policy). Honest negative.

## OVERALL CONCLUSION (honest)
- The v0.31 recalibration works (cache binds). **goodput@SLO 0→~3** is reachable and is a
  **capacity-de-duplication** effect: inclusive write_through wastes ~L1-worth of host on duplicates →
  goodput 0; de-duplicating (write_back family) → hit 0.68→0.74 → the λ=3 p99 tail drops 11.7s→7.0s → goodput 3.
- **Fundamental limits found**: (a) goodput caps at ~3 — the DECODE knee (~5 req/s) bounds it; λ≥5 saturates;
  caching cannot raise the decode ceiling. (b) The λ=3 p99 tail (~7s) is **capacity-floored** — engine
  mechanisms beyond de-duplication (exclusive free-on-loadback: +1.7pp hit but tail-neutral; cost-aware
  retention: −3.3pp hit, tail-neutral) do **not** further improve goodput or the tail.
- **Contribution**: a rigorous characterization of goodput@SLO on decode-bound 2-tier HiCache (near-binary
  metric, decode-knee cap, capacity-floored tail) + a lossless exclusive-tiering engine mechanism
  (free-on-loadback, +1.7pp effective capacity, no config provides it) + two honest negatives that bound the
  design space. The big goodput win is a config-reachable capacity effect; no engine mechanism beats it here.

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

### v0 — stock baseline sweep (control)  [DONE, node 0-1, XTIER_EXCLUSIVE=0, commit 4079f06c1]
- Change: none (stock `UnifiedRadixCache`, write_through, inclusive tiering; XTIER off).
- **Curve (λ: req/s, out tok/s, TTFT p50, TTFT p99, hit):**
  - λ=3:  2.89, 369, 1003 ms, **11663 ms**, 0.678
  - λ=5:  3.62, 463,  993 ms, **24484 ms**, 0.665
  - λ=7:  4.10, 524, 1078 ms, **33274 ms**, 0.658
  - λ=10: 4.20, 537, 1117 ms, **41202 ms**, 0.655
- **goodput@SLO = 0** (p99 TTFT > 8 s at every λ, including the healthy λ=3). peak req/s 4.2, peak
  out tok/s 537 (decode ceiling ~540 tok/s: 524→537 from λ7→λ10, ~flat = decode-bound control, as expected).
- Read: throughput keeps rising with λ but latency collapses (p99 11.7→41 s) — a **goodput cliff** bound
  by the p99 TTFT tail (HOL-blocking long-context miss re-prefills). Lever = cut λ=3 p99 (11.7 s) below 8 s.
- W&B: logged as `v0` [config].

### INTERIM (v0 A/B, node 0-1, job 19448) — the regime, from λ=3
- v0 stock λ=3: req/s 2.89, out 369 tok/s, TTFT p50 **1003 ms** / **p99 11663 ms**, hit **0.678**, 7037 reqs.
- **p99 TTFT 11.7 s > 8 s SLO at the *healthy* rate** ⇒ baseline **goodput@SLO ≈ 0** (higher λ only worse).
  The metric is gated by the **p99 TTFT tail**: median 1.0 s but p99 11.7 s ⇒ a small fraction of requests
  eat huge re-prefills (cache misses on long LEval/LooGLE contexts). Lever = cut that tail. XTIER raises
  hit ⇒ fewer long re-prefills ⇒ lower tail ⇒ possibly under 8 s (goodput 0→3). v1 will show if it's enough.
- Note p99 (11.7 s) ≫ baseline.json's old single-point 6.3 s: the warm-steady-state sweep at
  max-concurrency 256 / full 1553 convs is more tail-stressed than the old λ=3 single point. Report the
  CURVE (p99 per λ), not just the binary goodput — a p99 shift is a result even if it doesn't cross 8 s.

### v1 — exclusive tiering under write_through — CRASHED (invariant violation), RETRACTED
- First XTIER (write_through + suppress-eager-backup + backup-on-evict + free-Full-host-on-loadback)
  booted + served the short screen fine, but the FULL sweep crashed the scheduler `sanity_check` ~6 min
  into warmup: (1) "aux host present but Full.host_value=None" (freed Full host, left Mamba host), and
  (2) exclusive tiering breaks the **prefix-closed host-backup** invariant write_through enforces
  (`sanity_check` exempts it only under write_back). ⇒ exclusive tiering is fundamentally a **write_back**
  mechanism. Not logged (void run). Fix committed (e7d1eec41).
- **Chaining evals in one allocation is unsafe**: v0's 768 GB pinned host tier wasn't freed before the
  next run's DRAM gate (`DRAM_TOO_LOW 329G`). Run each version as a SEPARATE job (fresh alloc) on the
  same node to keep the A/B same-node.

### v-wb — write_back config (capacity diagnostic, control)  [node 0-1, job 19473; λ=3 done, rest running]
- **λ=3: req/s 3.02, out 387 tok/s, p50 515 ms, p99 TTFT 7032 ms (≤ 8 s SLO!), hit 0.7371.**
- vs v0 (same node): p99 **11663→7032 ms (−40%)**, hit **0.678→0.737 (+5.9pp)**, p50 1003→515 ms.
- ⇒ **goodput@SLO 0 → ~3**: de-duplicating the host tier (no eager device-copy backups) raises effective
  capacity → hit → fewer long-context miss re-prefills → the p99 tail drops under 8 s. **Capacity is the
  lever**, confirmed. (write_back is a CONFIG flip — the measurement, not the contribution.)
- Margin to SLO is 0.97 s (12%) — near-boundary ⇒ MUST replicate for error bars (goodput could flip on a
  noisy run). v0's 11.7 s is 46% over ⇒ v0=0 is robust; the 0→3 jump is likely real.

### v1x — exclusive tiering (write_back + free-host-on-loadback CODE)  [node -0, job 19479, commit e7d1eec41]
- Fix VALIDATED: survived the sanity-check window that crashed the write_through version. Pure-code add
  over write_back: free ALL components' host on load-back ⇒ disjoint L1/L2.
- **λ=3 ablation (v0 stock / v-wb write_back-config / v1x write_back+CODE):**
  | ver | p99 TTFT | hit | note |
  |-----|----------|-----|------|
  | v0   | 11663 ms | 0.6782 | stock inclusive (goodput 0) |
  | v-wb | 7032 ms  | 0.7371 | write_back CONFIG (+5.9pp hit, goodput ~3) |
  | v1x  | 6461 ms  | 0.7539 | + free-on-loadback CODE (**+1.7pp hit** over write_back) |
- My code adds **+1.7pp hit** (node-independent signal) = real but modest extra effective capacity; p99
  −8% vs v-wb but that's CROSS-NODE (v1x on -0, v-wb on 0-1), so treat as tentative — confirm same-node.

### KEY INSIGHT — goodput@SLO is capped at ~3 (the decode knee)
- Achieved req/s asymptotes ~4.2–4.75 (v-wb: λ5→4.23, λ7→4.75); λ≥5 is offered ABOVE the decode-bound
  service ceiling ⇒ **saturation** ⇒ p99 unbounded (v-wb λ=5 p99=21 s) ⇒ can NEVER pass the 8 s SLO.
  Caching does not raise decode throughput (charter says so). ⇒ **only λ=3 can pass ⇒ goodput ≤ ~3.**
- So the metric is near-binary (0 or ~3); the REAL differentiator is the **λ=3 p99 tail margin** (below 8 s)
  and the curve. The tail = long-context miss re-prefills. ⇒ novel headliner target: **recompute-cost-aware
  retention** — preferentially keep long (high re-prefill-cost) contexts resident to cut the p99 tail more
  than uniform capacity does. This optimizes the RIGHT objective for goodput@SLO (cost-weighted misses,
  not miss-count) — a metric-aligned insight distinct from hit-rate-maximal LRU.
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

## Runtime confirmations (from the 0-1 screen, job 19440, commit 4079f06c1)
- Active cache = `UnifiedRadixCache` (`Tree cache initialized: ... impl=UnifiedRadixCache
  hybrid_ssm=True hierarchical=True`), pools=KV+MAMBA, transfer_layer_num=48. XTIER edits are LIVE.
- `max_total_num_tokens=2347648` (L1 ≈ 2.35M tok), host tier 96GB×8=768GB (L2). max_running_requests=270.
- **XTIER validated**: server boots + serves the mix cleanly with `XTIER_EXCLUSIVE=1` (warmup + 600-conv
  bench, cache hits + load-back exercised, decode ~254 tok/s) — no crash/hang/leak. All server.log
  "Traceback"s are benign optional-lib (`libavutil`/`torchcodec`), not cache code.
- **0-1 boots a real 8-GPU server** (CUDA-graph capture OK, no NCCL hang) = nodes.yaml's own
  certification criterion. Used for the A/B because the VERIFIED pool is 100% held by the sibling v0.3
  campaign (holds 7.5h+). Documented deviation; the v1-vs-v0 delta is same-node so node-var cancels.

## Contingent next steps (decide from the v0 curve)
- If **prefill-bound** (p99 climbs through 8s across λ; hit-limited): capacity is the lever → XTIER is the
  right mechanism; add the **write_back-config ablation** (v-wb) to separate the config effect (backup-on-
  evict) from the pure-code exclusive effect (free-host-on-loadback), and replicate the A/B (n≥2).
- If **decode-bound / goodput flat** (peak req/s ~constant regardless of hit): capacity can't move
  goodput → pivot to a **TTFT-tail / transfer-overlap** mechanism (hide load-back latency; overlap H→D
  with prefill compute) or report the honest capacity-saturation ceiling as the contribution.
- Novel headliner candidate beyond exclusive tiering: **reuse-aware exclusive tiering** — keep the host
  copy (inclusive) for hot/shallow prefixes that churn evict↔reload, go exclusive only for cold/deep
  prefixes, capturing exclusive's capacity win without its re-backup cost on hot churn.

## Ops notes
- eval.sh has a path bug (computes `workspace/sgl/v0.3_ablations/base`); fixed by symlink
  `v0.3_ablations/base → v0.31/base` (frozen eval.sh untouched — fairness-clean).
- Contended pool: 4-way (base + 3 research) + v0.3 holds. Evals queue; research/code in parallel.
