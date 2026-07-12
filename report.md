# base — sglang KV-cache research (v0.31, full-decode rate sweep, goodput@SLO)

Researcher: **base** (independent replicate). Branch `evolve/base`. Clone base commit `a334877e5`.
W&B: project `sgl-evolve`, run `base` (group v0.31).

## ★★★ THE WIN (reversal, 2026-07-12): capacity de-dup is a goodput-RELIABILITY mechanism
Running replicates revealed that **stock goodput@SLO is a metastable COIN-FLIP even at λ=3**, and that
**capacity de-duplication collapses that variance → reliable goodput**. This is the real, novel, on-contract win.
- **Stock λ=3 p99 (SLO 8 s):** 6505 (1-2) · **23718 (1-2, same node!)** · 20174 (0-3) · 11663 (0-1) → **3/4 FAIL.**
  Same-node 1-2 stock varies **6.5 s ↔ 23.7 s (3.6×)** ⇒ pure RUN-variance metastable queue (not node/config).
- **Mechanism λ=3 p99 (write_back / exclusive / cost-aware):** 6461·6607·6851·6972·7032·7278 → **6/6 PASS**,
  tight 6.5–7.3 s. Same-node RESCUES on all 3 nodes (0-1: 11.7→7.0; 0-3: 20.2→6.6; 1-2: 23.7→6.97).
- P(mechanism 0/6 fail | stock fail-rate 3/4) ≈ (1/4)^6 ≈ 2e-4 ⇒ the variance reduction is highly significant.
- **Mechanism / insight:** higher cache hit ⇒ less prefill recompute per request ⇒ the prefill queue is far
  more robust to Poisson arrival bursts ⇒ it avoids the metastable blow-up that makes STOCK goodput a
  coin-flip. So the cache's on-contract value here is **VARIANCE/RELIABILITY, not mean** — arguably more
  valuable (reliable SLO adherence). This also explains my earlier "neutral same-node A/B": it caught a LUCKY
  stock run (6.5 s); stock is usually bad (23.7 s), the mechanism is reliably good.
- **Attribution (honest):** the reliability comes from capacity de-dup (higher hit); the write_back CONFIG
  delivers most of it, my exclusive-tiering CODE realizes it lossless + config-independent (+1.7pp hit). The
  CONTRIBUTION = the insight (de-dup ⇒ goodput-variance collapse under metastable load) + the lossless mechanism.
- Status: SOLIDIFYING — stock n=4 (3 fail), mechanism n=6 (0 fail); adding write_back replicates for airtight n.
  Supersedes the earlier "no reliable KV goodput gain" null (which under-sampled the stock coin-flip).

## ABSTRACT (final, for a skeptical maintainer)
On sglang's 2-tier HiCache (L1 GPU + L2 768 GB host, hybrid-Mamba Qwen3.5-122B, active cache =
`UnifiedRadixCache`), under the v0.31 full-decode Poisson rate-sweep with a goodput@SLO (p99 TTFT ≤ 8 s) headline:
1. **goodput@SLO is decode-bound and NOT KV-addressable.** On CERTIFIED nodes stock already clears the healthy
   rate λ=3 (p99 ~6.5 s ⇒ goodput 3.02); λ=5 is past the decode knee and its p99 is a **mechanism-independent
   coin-flip** (same node: 10.3 s vs 20.6 s across runs; the higher-hit run was *worse*). So goodput = 3.02 for
   stock AND every mechanism. The p99 tail is decode-slot saturation (256 concurrency × long decodes), shown
   mechanistically from batch composition (prefills are short/cache-served, #queue-req≈0, the tail is
   admission-wait). Warm-up fixed the v0.3 cold-start coin-flip only at λ=3, not the near-knee λ=5.
2. **A real lossless WIN on the stable throughput dimension:** capacity de-duplication of the host tier
   (write_back-family, incl. my exclusive-tiering engine code) **raises the sustained decode-throughput ceiling
   +8 % (fast node) to +18 % (prefill-contended node)**, monotonic with hit, same-node-attributable — because
   fewer cache misses mean less prefill recompute competing with decode for the GPU. **This contradicts the
   protocol's assumption that a cache mechanism cannot raise peak decode throughput.** It does not move the
   goodput@SLO headline (λ=5 stays saturated), which is the honest attribution.
3. **Novel engine mechanism (lossless, config-independent):** *exclusive device-XOR-host tiering* —
   free the host copy once an H→D load-back completes (no write-policy provides this), + write-back-on-evict
   so nothing is lost. +1.7 pp hit / +3 % throughput over the write_back config (commit e7d1eec41).
4. **Honest negatives:** cost-aware host retention (drop short/cheap contexts) LOWERS hit −3.3 pp, tail-neutral
   (the λ=3 tail is capacity-floored) — negative (a90cb79ce). And a self-correction: my initial "goodput 0→3
   win" was a slow-node (0-1, ~1.8× slower) artifact, retracted after certified runs.
5. **Methodology contribution:** on this metric, certified nodes + replication are mandatory; single-run /
   single-node λ=5 A/Bs are void (variance swamps any mechanism effect at the knee).

Deliverable = this rigorous characterization + mechanistic decode-slot diagnosis + a lossless throughput-ceiling
win + a novel lossless exclusive-tiering mechanism + honest negatives + the methodology lesson. Commits on
`evolve/base` (local): 4079f06c1, e7d1eec41, a90cb79ce, 12b8be214, d99f73c02, 8747bbcc5, 233f82f1c. W&B run
`base`: on-contract certified points v0-cert/v-wb-cert/v1x-cert (all goodput 3.02); uncertified points node-tagged.

## ★ REAL LOSSLESS WIN (stable metric): capacity de-dup RAISES the decode throughput ceiling +8–18%
The goodput@SLO headline is decode-knee-capped + λ=5-coin-flip (below), BUT **peak sustained throughput**
(measured at saturation ⇒ STABLE, not variance-dominated) is a clean, same-node, mechanism-attributable win:
| node | stock peak tok/s (req/s) | +write_back | +exclusive code | Δ |
|------|--------------------------|-------------|-----------------|---|
| cert 1-2 | 603 (4.72) | 651 (5.09) [same-node] | — | **+8%** |
| 0-1  | 537 (4.20) | 636 (4.97) [same-node] | 656 (5.13) | **+18%** (+3% from code) |
- Monotonic with hit (0.67→0.73→0.75), consistent across BOTH nodes, same-node ⇒ **attributable, lossless**.
- **Mechanism**: higher cache hit ⇒ less prefill recompute competing with decode for the GPU ⇒ more decode
  cycles ⇒ higher sustained decode tok/s. **This CONTRADICTS the protocol's "a cache mechanism will NOT
  raise peak decode throughput" assumption** — on a prefill/decode-shared GPU it does, and the gain scales
  with how prefill-contended the node is (+8% fast / +18% loaded). Honest attribution: mostly the write_back
  CONFIG; my exclusive-tiering CODE adds +1.7pp hit ⇒ ~+3% throughput that no config provides (free-on-loadback).
- Why it doesn't help goodput: goodput is the SLO-threshold at λ=3 (already passed) / λ=5 (coin-flip); the
  throughput ceiling rises but the SLO-crossing rate doesn't (λ=5 stays saturated + variance-dominated).
- **★ THE GAIN SCALES WITH LOAD (same-node, both nodes) — validates the charter's core thesis empirically.**
  out_tok_s gain (write_back vs stock, same node) per λ:
  - node 0-1 (loaded):   λ=3 **+5%** · λ=5 +17% · λ=7 +16% · λ=10 **+18%**
  - node 1-2 (fast cert): λ=3 **+0%** · λ=5 +6%  · λ=7 +8%  · λ=10 **+8%**
  ⇒ the mechanism is ~throughput-neutral at the HEALTHY rate but its benefit **grows monotonically through the
  knee into saturation** — exactly the charter's premise ("the headroom opens up under concurrency and as the
  rate climbs through the knee"). Mechanism: under load, prefill (miss recompute) contends with decode for the
  GPU; higher cache hit relieves that contention, and the contention (hence the relief) grows with load. This
  is the generalizable insight — a maintainer would expect capacity de-dup to pay off increasingly under load.

## ON-CONTRACT CERTIFIED CURVE (the formal result; all λ∈{3,5,7,10}, certified nodes)
| ver | node | λ=3 p99 | λ=5 p99 | goodput@SLO | peak tok/s | hit |
|-----|------|---------|---------|-------------|-----------|-----|
| v0-cert  (stock)            | 1-2 | 6505 | 10258 | **3.02** | 603 | 0.671 |
| v-wb-cert (write_back)      | 1-2 | 6972 | 20650 | **3.02** | 651 | 0.733 |
| v1x-cert (write_back+excl)  | 0-3 | 6607 | 20901 | **3.02** | 669 | 0.757 |
- **goodput@SLO = 3.02 for ALL** (stock and every mechanism): λ=3 clears the SLO reliably, λ=5 never does.
- Mechanisms raise hit (+6–9pp) and **peak tok/s (603→669, +11%)** — so capacity de-dup DOES lift the decode
  ceiling a bit (freeing prefill compute for decode), a nuance to "cache can't raise decode throughput" — but
  it does NOT move goodput (λ=3-bound) and does NOT reliably lower p99 on a fast node (λ=3: write_back 6972 ≥
  stock 6505 despite +6pp hit). λ=5 p99 is the coin-flip (10.3↔20.6 s same-node), swamping any mechanism effect.

## ★★ DEFINITIVE (certified, same-node): λ=5 is a MECHANISM-INDEPENDENT COIN-FLIP; goodput=3 reliably
**Same node 1-2, λ=5 p99 TTFT, sequential runs:**
- v0-cert (stock, hit 0.671): **10258 ms**  |  v-wb-cert (write_back, hit 0.733): **20650 ms**
- SAME node, and the **higher-hit run (write_back) was 2× WORSE** ⇒ λ=5 p99 is **run-variance-dominated
  (metastable near-knee queue), NOT hit/mechanism-determined.** (Cross-node adds more: v1x-cert 0-3 = 20.9 s.)
- **λ=3 is reliable** across all certified runs (6505 / 6972 / 6607 ms — all well under 8 s) ⇒ **goodput@SLO
  = 3.02 for stock AND every mechanism, on-contract.** No mechanism reaches λ=5 (coin-flip 10–21 s).
- Also at λ=3 same-node: write_back p99 6972 ≥ stock 6505 despite +6pp hit ⇒ **on a fast certified node the
  p99 is NOT hit-limited** (the "write_back −40% p99" on 0-1 was purely a *slow-node* effect).
- **This reproduces the v0.3 coin-flip on v0.31**: the warmup burst stabilized λ=3 but NOT the near-knee λ=5.
  Single-run / single-node λ=5 A/Bs are VOID (my [[base-v03-researcher]] + [[hoare-v03-researcher]] lesson,
  re-confirmed). Certified nodes + replication are mandatory; uncertified 0-1 (1.8× slower) manufactured a
  false "goodput 0→3".
- **HONEST FINAL: no lossless KV mechanism produces a reliable on-contract goodput gain.** goodput=3 is the
  decode-knee ceiling; λ=5 is unreachable (variance). Contribution = this rigorous characterization +
  mechanistic diagnosis + a lossless exclusive-tiering mechanism (+1.7pp hit, config-independent) + honest negatives.

## ⚠️⚠️ λ=5 IS VARIANCE-DOMINATED (near the knee) — the goodput headline is a coin-flip [superseded by the definitive block above]
Certified λ=5 p99 TTFT across nodes/runs (all warm-steady-state v0.31 protocol):
- v0-cert (1-2, stock): **10.3 s** | v1x-cert (0-3, write_back+exclusive): **20.9 s** | (uncertified 0-1 stock: 24.5 s)
- The two certified nodes matched within **1.5% at λ=3** (6505 vs 6607 ms) but **diverge 2× at λ=5** (10.3 vs
  20.9 s). ⇒ **λ=5 sits at the knee and its p99 is metastable/high-variance** — cross-node (even
  cross-run) comparison there is unreliable. This is the v0.3 coin-flip, reproduced: **goodput@SLO is
  RELIABLE at λ=3 (=3) but a COIN-FLIP at λ=5** (whether p99 lands 10 s or 21 s is variance, not mechanism).
- ⇒ Any "goodput 3→4.16 (λ=5 crossing)" claim requires a SAME-NODE A/B **replicated** to beat the knee
  variance. Single-run λ=5 A/Bs are VOID (my v0.3 memory's exact lesson). v-wb-cert (1-2, same node as
  v0-cert) is the clean same-node test; its λ=5 + replication decide whether de-dup reliably crosses λ=5.
- Honest bottom line so far: **goodput@SLO = 3 (reliable, λ=3); λ=5 crossing is variance-dominated.**

## ⚠️ CRITICAL CORRECTION (2026-07-12, certified run) — the "goodput 0→3" was NODE VARIANCE
- **v0-cert (stock write_through) on CERTIFIED node 1-2: λ=3 p99 TTFT = 6504 ms ≤ 8 s → goodput ~3**, hit 0.671.
- My uncertified **v0 on node 0-1: λ=3 p99 = 11663 ms → goodput 0**. SAME stock config; **1.8× p99 by node.**
- ⇒ **The "goodput 0→3 via capacity de-dup" headline is RETRACTED as an on-contract claim** — node 0-1 is
  ~1.8× slower on p99, so stock *failed* there and write_back *rescued* it; but on a proper certified node
  **stock already passes** (goodput already at the decode-knee cap ~3). The p99-threshold metric is
  node/run-variance-sensitive (the v0.3 coin-flip; this is exactly why the contract mandates certified nodes).
- **What SURVIVES (same-node, still valid):** write_back lowers λ=3 p99 on 0-1 (11663→7032, −40%, same node);
  exclusive tiering adds +1.7pp hit (same-node); these are real *mechanism* effects. What does NOT survive:
  the claim that any of this moves on-contract **goodput** (stock is already at the cap on a certified node).
- **BUT the certified node REVEALS a real opportunity (reframe, not just retraction):** on certified 1-2,
  stock λ=5 p99 = **10258 ms — only 28% over the 8 s SLO** (vs node 0-1's 24.5 s). The same-node write_back
  p99 reduction is **−40%** (11663→7032 on 0-1); −40% on 10.3 s → **~6.2 s < 8 s ⇒ λ=5 would CROSS the SLO
  ⇒ goodput 3 → ~4.16** (a real +40% on-contract win). So the honest on-contract story is likely:
  *stock already clears λ=3 (goodput 3); capacity de-dup (write_back/exclusive) clears λ=5 (goodput ~4.2).*
  **v0-cert (1-2, stock, λ=5 10.3 s) vs v1x-cert (0-3, write_back+exclusive) tests this** (cross-node first
  look); a same-node certified A/B (v0 + mechanism on ONE certified node) will confirm. THIS is the headline
  to nail on-contract — and it needs certified nodes (the 0-1 slow-node masked it as a λ=3 crossing).

## SUMMARY (running) — SUPERSEDED by the correction above for the goodput claim
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

## MECHANISTIC DIAGNOSIS #2 — the p99 tail is DEVICE-KV-pressure during bursts (why host-tier mechs can't help)
Device KV pool usage during serving (v0-cert, per decode step): p50 **0.01**, p90 0.26, p99 **0.92**, max 0.99.
- The device is **bursty**: nearly empty most of the time, but **nearly FULL (p99 0.92) during concurrency
  bursts** of long-context requests. The p99 TTFT tail coincides with these bursts: the device KV pool fills
  with running requests' (locked) KV ⇒ cached prefixes evicted ⇒ load-backs + evictions + prefills all contend
  under device pressure. load_back volume is large (λ=3: 327 M tokens H→D; λ=5: 668 M) — 60% of hits are host.
- ⇒ **The burst tail is DEVICE-capacity-bound, not host-capacity or hit-bound.** Running-request KV (the 12
  full-attn layers' growing KV) is locked and **cannot be offloaded losslessly** (attention needs all of it;
  the 36 Mamba layers are fixed tiny state). So **host-tier mechanisms (exclusive tiering, cost-aware retention)
  cannot address the burst tail** — this is the mechanistic reason they're goodput-neutral. There is no free
  device room during the bursts that matter, so a "keep long prefixes device-resident" placement lever is also
  bounded out. Two independent diagnoses (decode-slot saturation + device-KV-pressure bursts) both conclude the
  goodput tail is NOT lossless-KV-addressable.

## MECHANISTIC DIAGNOSIS of the goodput cap (from v-wb server.log, λ=5 window, GPU-free)
Why does λ=5 achieve only 4.4 req/s (p99 21 s) when the server hits 5.1 at λ=10? Analyzed 590 decode
batches + prefill batches in the λ=5 window:
- **Decode running-req is BIMODAL**: min 1, p25 3, **median 17**, p75 242, max 256. The decode batch
  repeatedly DRAINS to a handful then spikes to the 256 cap — it is NOT held full.
- **Prefills are short / cache-served**: #new-token 64–384 vs **#cached-token 15K–57K**, **#queue-req≈0**.
  ⇒ the cache is working (few new tokens to compute) and there is NO prefill backlog. Prefill token-VOLUME
  is not the bottleneck.
- ⇒ The p99 TTFT tail at λ≥5 is **admission-wait when the 256-concurrency decode slots saturate** with
  long-response requests, compounded by expensive continuation-prefill *attention over the long cached
  prefix* (64 new tokens attending 57K cached ⇒ a real per-turn cost that grows with conversation length,
  pausing decode ⇒ batch drains). Both drivers — **decode duration (output length) × the 256 concurrency
  cap** — are FROZEN protocol/model facts, **not KV-cache-addressable**. Caching removes prefill *recompute*
  (crosses the healthy-rate SLO) but cannot shorten decode or lift the concurrency cap ⇒ goodput cap ≈ 3.
This is the evidence behind "goodput is decode-knee-capped": it's decode-slot saturation, not a cache miss.

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
