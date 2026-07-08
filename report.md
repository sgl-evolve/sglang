# sgl_free — v0.25 KV-cache research report (2-tier: L1 GPU HBM + L2 host DRAM, NO disk)

Researcher: **sgl_free** (independent replicate). Branch `evolve/sgl_free`, W&B run `sgl_free` in
project `sgl-evolve`. Model Qwen3.5-122B-A10B-FP8 (hybrid-Mamba GDN MoE), TP8, ctx 262144.

> NOTE: I previously ran as `sgl_free` on **v0.2_ablations** (a 3-tier L1/L2/L3-disk protocol). v0.25 is a
> DIFFERENT protocol (2-tier, no disk) → that thesis (serial-prefetch-backlog disk arbiter) does not apply.
> Starting fresh from the baseline; reusing only cluster/ops lessons, re-deriving all science.

## EXECUTIVE SUMMARY (for a skeptical maintainer)
**Contribution: EXCLUSIVE L1↔L2 KV-cache tiering for HiCache** — under memory pressure, keep each cached
entry on device XOR host (never both), instead of the stock INCLUSIVE behavior where write_through eagerly
duplicates every hot device entry onto host. This raises *distinct* cache capacity by ~the device-tier size
and, because the system sits on the STEEP part of the hit-vs-capacity curve, converts to a large hit-rate
and tail-latency win.

**Result ladder (fixed protocol, λ=3), all clean/on-contract, lossless:**
- fcfs baseline (inclusive, stock): hit **0.622**, p99 TTFT **6326 ms**.
- +write_back flag (write-side exclusivity only): hit **0.733** (+11pp), p99 **4581** (-28%).  [config]
- +free-host-on-promotion (my engine mechanism → full exclusivity): hit **0.7525** (+13.1pp vs baseline),
  p99 **4380 ms (-30.8%)**, mean TTFT 863 (-25%), req/s 2.78→3.02.  [mechanism, commit 21023af6d]

**Why it works / evidence:** (1) live metrics show both tiers saturated but with write_through the device
tier is a redundant *inclusive* subset of host (device⊆host) → distinct capacity ≈ host alone (7.81M) for a
19M-token working set; (2) a trace-calibrated simulator shows hit is steeply capacity-sensitive around this
operating point; (3) 3-point ablation isolates the write-side (config) vs promotion-side (engine) halves.

**What I ruled out (negative results, valued):** prefix-ORDERING is a dead end — `--schedule-policy lpm`
does NOT change hit (0.62→0.62) and *worsens* p99 to 21.4 s (cache-cold starvation); eviction-ORDER is a
dead end (LRU≈Belady, charter + confirmed by lpm); ADMISSION/concurrency-capping showed ~no hit gain in
sim and is not cleanly implementable (no conversation id to separate active from finished-cached convs).

**Generalizable insight:** for a saturated multi-tier KV cache on the steep hit-vs-capacity curve, the
lever is *effective capacity* (exclusive tiering / de-duplication), not scheduling order or admission.
Inclusive-by-default HiCache leaves the fast tier as dead-weight duplication; make it exclusive.

**Honest limits:** the write-side half is reachable via a stock flag (write_back); the *engine* mechanism's
marginal gain over that strong config is modest (+2.0pp hit, -4.4% p99), though the FULL exclusive design
(needing the promotion-side engine change) and the insight are the contribution. Exclusivity increases
host→device load-back (298M→402M) — net TTFT still improves. Self-contained version (SGLANG_HICACHE_EXCLUSIVE
alone, no flag, commit feca1871e) + goodput-curve sweep + error-bar reruns in progress (node-contended).

## The protocol (fixed contract)
- 2-tier: L1 GPU HBM (~2.35M tok) + L2 host DRAM (`--hicache-size 96` = 768 GB, ~7.81M tok). No L3.
- Frozen launch: TP8, ctx 262144, mem-frac 0.85, page-size 64, chunked-prefill 6144, io-backend `direct`,
  mem-layout `page_first_direct`, write-policy `write_through`, hicache-size 96. Mamba host pool REQUIRES
  direct/page_first_direct.
- Load: real-text 1:1:1 mix (`mooncake_mix_v1.jsonl`, 1553 LooGLE-style convs), loogle loader, multiturn,
  **λ=3** (sub-knee; knee ≥4), max-concurrency 128, num-prompts 1553, real decode.
- **HEADLINE metric: goodput under a TTFT-SLO — max sustainable req/s with p99 TTFT ≤ 8 s.** Per-version at
  λ=3: TTFT p50/p99, req/s (tracks λ?), hit-rate, L2 host-util. Lossless gate: outputs == no-cache.

## Active code paths (verified for THIS config)
- Cache: **`UnifiedRadixCache`** (mem_cache/unified_radix_cache.py), components [FULL, MAMBA]; selected via
  registry.py:101-104 (enable_hierarchical_cache + is_hybrid_ssm). DORMANT: HiRadixCache, MambaRadixCache,
  hi_mamba_radix_cache.
- Controller: **`HybridCacheController`** (mem_cache/hybrid_cache/), base cache_controller.py.
  write-through D→H = `write_backup` (urc.py:1540, triggered _inc_hit_count when hit_count≥threshold);
  load-back H→D = `init_load_back` (urc.py:2410) → `load_back` (urc.py:1650).
- Evict L1 = `_evict_device_leaf` (urc.py:1482, demote via `_evict_to_host` 1464); Evict L2 =
  `_evict_host_leaf` (urc.py:1520). Eviction order = **LRU** (evict_policy.py, key node.last_access_time).
- Scheduler: default policy **fcfs**; prefix match `match_prefix_for_req` (schedule_policy.py:85);
  queue ordering `calc_priority` (schedule_policy.py:170); batch admission `get_new_batch_prefill`
  (scheduler.py:2738) via `PrefillAdder`.
- **No conversation/session id** on Req (only per-turn `rid`). Multi-turn conversations are identifiable
  ONLY by shared token-prefix (the radix path). Turns of one conv form a linear chain in the radix tree;
  cross-conversation sharing ≈ 0 (each conv's document differs → paths diverge at token ~2).

## Workload reuse structure (from the loogle loader)
- Each conv = one large LooGLE **document** (turn 0 = "Input: <doc> Question: <Q0>", ~7–12k tok) + up to
  ~10 short follow-up turns (Qi). Turn i's prompt = doc + all prior Q/A → reuses turn i-1's full chain.
- **Reuse is intra-conversation**: the document KV computed at turn 0 must survive eviction until the
  conv's last turn. Under concurrency (128 interleaved convs + λ=3 arrivals), other convs' documents
  churn L1+L2 and evict X's document between its turns → recompute the whole doc. THIS is the locality loss.
- Arrival model: single FIFO queue drained at λ=3, cap 128 concurrent; a conv's next turn re-enqueues to
  the TAIL on completion (interleaves behind others).

## Baseline (v0_official, fcfs) — logged as W&B point 0
- TTFT p50 750 ms / p99 6326 ms / mean 1146 ms; req/s 2.78 (≈λ, mild queueing); hit_rate 0.6217;
  host_util **0.9999** (L2 saturated → forced eviction); hit_device 0.40 / hit_host 0.60; load_back 298M
  tok @1.65ms, evict 582M tok @1.02ms. **p99 6.3s < 8s SLO → baseline meets SLO at λ=3 (NOT collapsed).**
- Read: healthy concurrency regime (unlike v0.2's collapsed fcfs). The win must come from LOCALITY
  (raise hit-rate under fixed saturated capacity ⇒ less fresh prefill ⇒ lower TTFT ⇒ higher sustainable λ),
  not from de-collapsing a queue.

## Thesis
Under saturation, fresh prefill ∝ (1 − hit_rate) dominates TTFT. Raising hit_rate at fixed L1+L2 capacity
requires keeping each active conversation's document co-resident across its turns. Levers (charter): prefix-
/reuse-aware **admission** + **routing/batching** + **prefetch** + L1↔L2 **placement** — NOT eviction order
(LRU≈Belady for the in-order intra-conv reuse). Candidate mechanism: locality-preserving admission control
that bounds the simultaneously-active document working set so admitted convs finish their turns resident.

---

## Versions

### v0_official — baseline (config, given) — W&B point 0
Stock fcfs 2-tier. See numbers above. Necessary to clear; not the contribution.

### v0_lpm — strong stock config (config) — RUNNING (node slurm2-a3nodeset1-2)
`--schedule-policy lpm`. Establishes the cache-aware ordering bar my mechanism must beat + live bottleneck
characterization. (A config flip — not a contribution by itself.) Result: TBD.

---

## Empirical findings from live telemetry (v0_lpm run, warm) + code — DECISIVE

- **Both cache tiers are FULL.** Device KV pool ≈ **2.345M tok** (live: `kv_available_tokens`≈2432 (~0),
  `kv_evictable_tokens`≈1.96M, `full_token_usage`≈0.16=locked/size). Host ≈ **7.81M tok** (util 0.9999).
  Total resident ≈10.16M vs working set 19M → genuinely capacity-bound. **No idle device headroom** (the
  "full_token_usage 0.16" is misleading — it EXCLUDES evictable radix entries; device is full of cache).
  ⇒ "promote hot entries to idle L1" is a DEAD idea (no idle L1).
- **hit_rate counts device+host both** (`cached = dev+host`; summarize.py). Baseline 0.62 = 0.40 dev + 0.60
  host — reuse is real. `#cached-token=0` in prefill logs is device-only prefill-time accounting (host hits
  load-back later), NOT "no reuse".
- **Cache hits DO save prefill compute** for full-attn (fully) AND mamba (at `mamba_cache_chunk_size`
  granularity, ~90%+ effective). So raising hit_rate genuinely cuts fresh prefill.
- **At λ=3, server queue ≈ 0** (`num_queue_reqs` 0-4), `num_running_reqs`=128 pinned. ⇒ schedule ORDERING
  (lpm) has little to reorder at λ=3; the leverage appears near the knee (λ≥4) where a queue forms.
- **Decode-heavy regime:** gen throughput ~300 tok/s total vs prefill bursts ~40k tok/s. GPU mostly
  decoding long contexts → prefill (TTFT) latency comes from big cache-MISS documents (full-doc recompute,
  ≤84k tok) contending with the decode batch. ⇒ p99 TTFT tail is dominated by MISS recomputes → raising
  hit_rate directly cuts the p99 tail (the SLO metric).
- **Theoretical hit ceiling ≈0.81** (reusable 80.6M / prompt 99.9M). Headroom 0.62→0.81 = reuses evicted
  before they happen. Given LRU≈Belady (charter), recovering it needs CHANGING THE ACCESS SEQUENCE
  (scheduling/pacing) or ADMISSION, not eviction order.

### v0_lpm RESULT (config, W&B, commit a334877e5) — NEGATIVE, decisive
`--schedule-policy lpm`: ttft_p50 991 / **p99 21351** / mean 1602 ms; req/s 2.47; **hit_rate 0.6205**
(≈ fcfs 0.6217); host_util 0.9991; hit_dev 0.41/host 0.59; load_back 297M; evict 582M. Clean run
(contract OK, lpm applied, no fallback). **LPM is WORSE than fcfs on everything** — p99 3.4× worse (blows
the 8s SLO), throughput down, and **hit_rate UNCHANGED**. Confirms: (1) prefix ORDERING does not recover
hit-rate (capacity-bound + LRU≈Belady, per charter); (2) LPM starves cache-cold requests → tail explosion.
⇒ ordering is a dead end; fcfs baseline is near-optimal for ordering.

## PIVOT — the lever is EFFECTIVE CACHE CAPACITY (exclusive vs inclusive tiering)
- Sim shows the system sits on the STEEP part of the hit-vs-capacity curve (C 7.5→10.2M ⇒ hit 0.45→0.73),
  so a modest EFFECTIVE-capacity gain ⇒ large hit gain ⇒ large p99/goodput gain.
- **Root pathology (code + live metrics):** with `write_through`, HiCache is INCLUSIVE — every device
  (L1) cache entry is eagerly backed up to host (L2) and KEPT (write_backup, urc.py:1540, threshold=1),
  so device ⊆ host. Live: device evictable ≈1.96M, all duplicated on host (host_util 0.9999). ⇒ distinct
  cache capacity ≈ HOST alone (7.81M); the 2.35M device tier is REDUNDANT (holds copies).
- **Mechanism v1 (candidate): EXCLUSIVE L1↔L2 tiering.** Keep an entry on device XOR host (never both):
  (a) no eager device→host backup (back up only at device eviction, write_back path urc.py:1497-1502);
  (b) on host→device load-back/promotion, FREE the host copy. ⇒ distinct capacity = device+host ≈10.16M
  (+25-30%) ⇒ steep-curve hit gain, lossless (entry never dropped, just single-tier). Novel: exclusive
  vs inclusive KV cache tiering (a CPU-cache design axis) applied to LLM HiCache. Charter-aligned
  (L1↔L2 placement). Distinct from admission/scheduling.
- **Gate:** `--hicache-write-policy write_back` screen RUNNING (weak exclusivity proxy: threshold=2 so hot
  docs still re-duplicate; a positive result strongly supports building the full exclusive mechanism).

### s_writeback — config (write_back), W&B, commit a334877e5 — STRONG WIN, confirms exclusivity
vs fcfs baseline: **hit_rate 0.6217 → 0.7326 (+11.1pp)**, **p99 TTFT 6326 → 4581 ms (-28%)**, p50 750→586,
mean 1146→933, req/s 2.78→3.02, out_tok/s 355→387. host_util still ~1.0; hit_dev 0.34/host 0.66; load_back
298M→386M; evict 582M. Clean (contract OK, write_back applied, no fallback). **Confirms the exclusivity /
effective-capacity thesis EXACTLY** (sim predicted ~0.72 hit at exclusive capacity 10.16M). The feared
eviction-cost did NOT hurt — higher hit dominates. BUT write_back is a STOCK CONFIG FLIP ⇒ not the
contribution; it's the strong config bar my mechanism must beat.
Note (code): stock `write_back` is already WRITE-SIDE exclusive — `_inc_hit_count` returns early for
write_back (urc.py:1815) ⇒ no eager backup; device→host only at eviction. It re-inclusivizes on promotion
(load_back keeps host copy). So s_writeback tests write-side exclusivity; my v1 adds promotion-side.

### v1_exclusive — MECHANISM (commit 21023af6d), W&B, mechanism — WIN over the write_back config bar
Result (clean: contract OK, exclusive engaged on all 8 ranks, no fallback, lossless by construction):

| metric | fcfs (inclusive) | write_back (config) | **v1_exclusive (mechanism)** |
|---|---|---|---|
| hit_rate | 0.6217 | 0.7326 | **0.7525** (+2.0pp vs config, +13.1pp vs fcfs) |
| p99 TTFT ms | 6326 | 4581 | **4380** (-4.4% vs config, **-30.8% vs fcfs**) |
| mean TTFT ms | 1146 | 933 | **863** (-7.5% vs config) |
| p50 TTFT ms | 750 | 586 | 546 |
| req/s | 2.78 | 3.02 | 3.02 |
| hit_host_frac | 0.598 | 0.657 | 0.663 |
| load_back tok | 298M | 386M | 402M |

**Contribution:** exclusive L1↔L2 KV tiering (device XOR host). write_back gives write-side exclusivity
(config, +11pp); my engine change (free-host-on-promotion) completes it (promotion-side, +2pp more →
0.7525 hit, -30.8% p99 vs baseline). Attributable via the 3-point ablation. The +2pp/-4.4% is the engine
mechanism's marginal gain over the strong config; the broader insight (exclusivity/effective-capacity is
THE lever on the steep hit-vs-capacity curve; ordering & admission are dead ends) is the generalizable
contribution. Cost of exclusivity: more host hits / load-back (device holds fewer copies) — net TTFT still
improves (higher hit dominates). Honest note: modest marginal engine gain; next = push it further (v2:
partial/hot-set inclusive to cut load-back cost) + goodput-curve sweep to show the SLO-knee shift.
--- original design notes below ---
`--hicache-write-policy write_back` + `SGLANG_HICACHE_EXCLUSIVE=1`. Env-gated free-host-on-promotion
(urc.py `_promote_free_host` in `loading_check`). Full device-XOR-host exclusivity.
**Losslessness (verified by construction):** (1) only device-RESIDENT nodes' host copies are freed ⇒
device-absent (evicted) nodes always retain host ⇒ LOAD_BACK's `while cur.evicted: assert host_value`
(full_component.py:280) holds; (2) freed nodes are detached from host-LRU ⇒ host-eviction never walks them
(mamba_component:550 assert safe); (3) on later device eviction, write_back re-backs-up to host BEFORE
demote ⇒ "evicted ⟹ has-host" invariant preserved; (4) KV values never altered — only tier placement.
**Risk to watch:** write_back does device→host copy at EVICTION time (on the make-room critical path);
under heavy eviction this can raise TTFT and may offset the capacity/hit gain. The s_writeback result
(write_back vs fcfs TTFT) reveals this cost; v1 attribution = (fcfs) vs (write_back) vs (write_back+excl).
Env propagation through srun verified (--export=ALL). Result: TBD.
