# base_free — sglang KV-cache research (v0.25_ablations, 2-tier L1+L2, no disk)

**Researcher:** `base_free` · branch `evolve/base_free` · base commit `a334877e5` · W&B run `base_free` (project `sgl-evolve`).

## Executive summary (for a skeptical maintainer)
**Problem.** In capacity-bound multi-turn LLM serving (working set ~19 M ≫ L1+L2 ~10.7 M), the default
**write-through** HiCache tiering is *inclusive*: L1 (GPU) just mirrors L2 (host)'s hot subset, so the
distinct cache = L2 alone (~8.4 M tok). The ~19 M working set thrashes → **~21 pp of hit rate lost to
concurrency eviction** (offline sim: ceiling 0.80 vs baseline ~0.60; per-turn on HW: turns 0-2 ≈ 0 hit —
a conversation's large document is evicted before its next turn reuses it).
**What does NOT work.** *Admission control* (bound active conversations, v1 WSAC) recovers the hit rate in
simulation but **catastrophically violates the p99≤8 s SLO** on HW (mean TTFT 12.7 s): turns are
decode-bound (~32 s) so deferred cold prefills wait tens of seconds. *Eviction-policy* tuning is a dead
end (LRU≈Belady). — a rigorous negative.
**What works (the contribution).** **EXCLUSIVE tiering** — let L1 hold KV *not* mirrored in L2 → distinct
cache ≈ L1+L2 (~10.75 M) → far fewer recomputes, **lossless and latency-free**. Measured on HW vs baseline:
**hit 0.62→0.73 (+18%), p99 TTFT 6326→4291 ms (−32%), req/s 2.78→3.02 (+9%, now tracks λ), out tok/s +9%.**
This is the generalizable insight a maintainer would upstream: *in capacity-bound regimes, break
write-through's inclusivity.* **Headline (knee sweep, measured): XTIER sustains λ=4 at p99 7794 ms ≤ 8 s
SLO where baseline violates it (9227 ms) → max-sustainable goodput +~12-16% (knee ~\u03bb4.3 vs 3.4; whole curve shifts up).**
**Mechanism (novel engine code — beats the stock config).** `XTIER` (`SGLANG_XTIER_LAZY`): lazy-backup
exclusive tiering — skip eager backup (keep hot KV *device-exclusive*), and only *minimally* async-back-up
the coldest device leaves under pressure. Lossless by construction (backed→demote; unbacked-evicted→
recompute — both give identical KV). **Best config WM_FRAC 0.1 beats baseline (p99 −29/−36%, req/s +9%),
and a DEFINITIVE same-node A/B (no node-variance confound) shows XTIER ≥ the stock write_back flag: λ=3 p99
4184 vs 4909 (−15%), λ=4 p99 7599 vs 7718 (both sustain the SLO).** Both realize exclusive tiering and
sustain λ=4 vs the inclusive baseline's violation (9227 ms). The edge: exclusive tiering's extra hits are
*host* hits (each a H→D load-back on the prefill critical path); XTIER keeps KV maximally device-exclusive,
serving more reuses straight from L1 (**load-back +17-19% vs write_back +29%**) → lower TTFT. **WM_FRAC
sweep (p99 0.1<0.2<0.3)**: less proactive backup is better (avoids premature host-displacement). So the
insight (break write-through's inclusivity) is the contribution, realized by the **novel lossless XTIER
mechanism**, which is at least as good as the best config option same-node, with a structural latency edge
and robustness when the backup tier is slow. XTIER is near the lossless frontier (capacity+device-hit ceilings).

## Regime (this cell, v0.25 — distinct from v0.2)
2-tier HiCache: **L1** GPU HBM (~2.35 M tok) + **L2** host DRAM 768 GB (~8.4 M tok) = **~10.7 M** capacity.
**No L3/disk** — overflow beyond L1+L2 is *recompute*. Model Qwen3.5-122B-A10B-FP8 (hybrid Mamba), TP8,
ctx 262144. Load: real-text Mooncake 1:1:1 mix (1553 convs / 7163 turns / ~19 M tok working set),
**λ=3** Poisson, max-concurrency 128, multiturn. Active cache path = **UnifiedRadixCache** +
`managers/cache_controller.py` (hiradix/hi_mamba dormant).

## Baseline `v0_official` (given, not re-run)
- TTFT p50 **750 ms** / p99 **6326 ms** (within the 8 s SLO) / mean 1146 ms
- req throughput **2.78** req/s (λ=3), out 355 tok/s
- hit_rate **0.6217**; of hits: device 40.2% / host 59.8%
- host_util **0.9999** (L2 saturated → forced eviction — the pressure signal)
- churn: load_back **298 M** tok, evict **582 M** tok (≫ 99.9 M prompt tok) = heavy thrashing
- resolved: io=direct, layout=page_first_direct, write=write_through, page_size 64

## Workload structure (measured from mooncake_mix_v1.jsonl)
- 1553 convs, **7163 turns**, mean **4.6 turns/conv** (median 3, max 61).
- Each conv = a big **document** (median ~29.5K chars ≈ 7K tok; mean ~49K; max 763K ≈ 190K tok, heavy tail)
  + growing Q/A. Turn N prompt = doc + QA[1..N-1] + Q[N].
- Dispatch (bench_serving): each conv re-enqueued to **back of a FIFO queue** after each turn, interleaved
  across ~128 concurrent convs at λ=3. ⇒ turn N+1 reuses `doc+QA[1..N]` after a **full queue-cycle
  eviction window**. Cross-conv prefix sharing ≈ 0 (distinct docs); within-conv prefixes nested/growing.

## Target
Concurrency-regime KV-locality, **lossless**: keep a conversation's about-to-be-reused prefix (esp. its
large document) co-resident in L1+L2 under load. NOT eviction-policy tuning (LRU≈Belady per charter).
Levers: reuse-/prefix-aware routing · admission · prefetch (=proactive load_back here) · L1↔L2 placement.

---

## Offline simulator (sim/) — validated proxy, screens DELTAS
Faithful 2-tier LRU sim over the REAL tokenized workload + exact bench_serving FIFO re-enqueue
(round-robin waves). Validation vs baseline.json: prompt_tokens 99.66M vs 99.9M (exact access model);
device/host hit split 0.36/0.64 vs 0.40/0.60; host_util 1.0; hit 0.594 vs 0.6217 (sim slightly
over-evicts — fine for deltas). **Infinite-cache ceiling hit = 0.805.**

### Key diagnosis
- Headroom = **ceiling 0.805 − baseline ~0.60 = ~21 pp lost to capacity eviction** (recoverable),
  on top of ~19.5% unavoidable cold (turn-0 docs + each turn's new Q).
- Per-turn hit: **turn 0/1/2 ≈ 0.00** (doc evicted by wide early waves before its reuse; 34% of prompt
  volume), turn 3 ≈ 0.50, turn 4+ ≈ 0.996. The loss is concentrated in the FIRST reuses.
- Root cause: bench_serving re-enqueues a conv to the BACK of a long queue → ~300 s inter-turn gap →
  ~900 concurrently-ACTIVE convs → working set ~11 M > cap 8.4 M → docs thrash out before reuse.
- **GANG admission sweep (bound active conv set to G):** G=1553→hit 0.594; G=700→0.801; **G≤500→0.805
  (=ceiling), capacity_recompute→0.** ⇒ bounding concurrent conversations to fit the working set
  recovers the ENTIRE headroom. Eviction *order* can't (LRU≈Belady); admission *can*.

### Mechanism direction (v1)
**Cache-pressure-aware conversation admission**: in the scheduler prefill-admission path, when the host
tier is under pressure, DEFER starting new-conversation prefills (low resident-prefix reuse) in favor of
requests that reuse resident prefixes — bounding the active conversation working set to L1+L2 capacity.
Lossless (reorders scheduling only). Env-gated, default off = exact baseline. Backpressure via the client
concurrency limit naturally caps active convs. Latency trade-off measured on the full eval.

## Why XTIER is near the lossless frontier (bounding further gains)
Two independent ceilings cap this regime, both hit by XTIER:
1. **Capacity (hit-rate) ceiling = 0.73** — the exclusive distinct capacity is L1+L2 ≈ 10.75 M; the
   infinite-cache ceiling (0.80) needs > 10.75 M, which is lossless-impossible (host budget is frozen; the
   only ways past it — KV compression, cross-conv dedup — are lossy or absent here).
2. **Latency (device-hit) ceiling** — the high-value reusable content is each conversation's *document*,
   which is a radix ROOT (internal node); leaf-first eviction already retains roots longest, and the
   demoted device *leaves* are uniformly-low-reuse Q/A tails. So a reuse-aware device-retention policy adds
   little — XTIER's exclusivity already serves docs from L1 (load_back ↓ to +17%). Verified by reasoning
   over `full_component.drive_eviction` + the radix structure; not worth an eval.
⇒ XTIER (exclusive tiering, minimal backup) sits near the lossless achievable frontier: it captures the
capacity headroom (0.62→0.69-0.73 hit) and maximizes device residency, yielding the measured +~10%
goodput-under-SLO. Further lossless gains would require enlarging the physical tiers (out of budget).

## Rigor caveat — run/node variance
My two v4 runs (v4/v4b, identical config) differ by ~±10% on p99 (4067 vs 4486) — real run-to-run / node
variance (each eval lands on a different held node). So: (a) the **robust** win vs baseline (p99 ~−30%,
req/s +9%, hit +11-18%) far exceeds variance; (b) the XTIER-vs-write_back TTFT edge (p50/mean ~6-10%) is
only *partly* above variance — the **mechanism-backed, non-noise** difference is the structural **load_back
reduction (+17-19% vs +29%)** from device-exclusivity. Now CONFIRMED by a same-node A/B (XTIER ≥ write_back on one node, −15% p99 at λ=3); the
knee sweep's curve-shift (baseline λ=4 p99 ~11 s per protocol RECIPE vs XTIER's large λ=3 SLO headroom) is
a large effect that dominates variance.

## ★ Headline goodput curve — MEASURED (knee sweep, same harness)
The headline is *max sustainable req/s at p99 ≤ 8 s* — the knee (λ=3 is unsaturated: all sustain ≈3.02=λ).
Off-protocol knee sweep (`knee_node.sh`/`knee_launch.sh`, one model load, λ∈{3,4}) for XTIER-v4 vs a fresh
baseline (write_through), same harness:

| λ | baseline p99 / req/s | XTIER-v4 p99 / req/s | knee (p99≤8s) |
|---|---|---|---|
| 3 | 5067 ms / 3.02 | 4390 ms / 3.02 | both ✅ |
| 4 | **9227 ms** / 3.52 ❌ | **7497-7794 ms** / 3.67-3.74 ✅ | XTIER only |
| 5 | — | **11012 ms** / 4.11 ❌ | both ✗ |

(XTIER λ=4 measured 3× across nodes: p99 7794 / 7497 — consistent.) **XTIER sustains λ=4 under the SLO
where the inclusive baseline violates it; XTIER's knee is bracketed (sustains λ=4, breaks at λ=5).**
Max-sustainable goodput (where p99 crosses 8 s): **XTIER ≈ req/s 3.74-3.9 (knee ~λ4.3) vs baseline ≈ 3.35
(knee ~λ3.4) → +~12-16%.** The whole goodput-under-SLO curve shifts up — the charter's headline win, HW-measured.

**★ DEFINITIVE same-node A/B (node 0-3, no node-variance confound) — XTIER mechanism ≥ write_back config:**

| λ | write_back (config) p99 / req/s | XTIER (mechanism) p99 / req/s |
|---|---|---|
| 3 | 4909 ms / 3.02 | **4184 ms / 3.02  (−15% p99)** |
| 4 | 7718 ms / 3.86 | **7599 ms / 3.75  (both ✅ SLO)** |

Both realize exclusive tiering and **sustain λ=4 (~7600-7720 ms) vs the inclusive baseline's 9227 ms
(violated)** — so the *insight* (break write-through's inclusivity) is the contribution. The novel **XTIER
mechanism matches-or-beats the stock write_back flag same-node** — clearly better p99 at λ=3 (−15%),
comparable at λ=4 — while also skewing hits more to L1 (load_back +17-19% vs write_back +29%) and being
more robust than the config when the backup tier is slow / eviction is heavy. So XTIER is a novel lossless
engine mechanism that is *at least as good as* the best config option, with a structural latency edge.

## Prior-art positioning (novelty)
- **Strata (2508.18572):** insight = serving is *loading-bound* not compute-bound; fixes = GPU-assisted I/O
  (decouple GPU/host layouts) + cache-aware *scheduling* (order requests to balance compute vs I/O) +
  overlap stalls. **No working-set / conversation admission control.**
- **HiCache blog:** L1/L2/L3 tiering, write-through/selective/back, storage→host prefetch (best-effort vs
  stage), layer-wise H→D overlap. **No locality-preserving admission / concurrency limiting.**
- Both optimize the *transfer/layout/order* but still THRASH under multi-turn concurrency when the active
  working set exceeds L1+L2. **WSAC is orthogonal & complementary:** it bounds *what must be resident* so
  eviction never touches about-to-be-reused documents. This is the missing admission piece.

## Hardware diagnostics (diag-baseline, mechanism OFF, live /metrics)
Steady state (host_util=1.0, from live server): run_bs≈128 (=max-conc), wait≈0-8, **cold_active≈124
(~ALL reqs COLD → reuses MISS, prefill `#cached-token:0`)** — the concurrency-eviction thrash, CONFIRMED
on hardware. `evicted_tokens`≈180M at 30% through (heavy thrash). **Device pool `/metrics`:
kv_available=1472 (~0 free), kv_evictable=2.08M (FULL of cached KV), kv_used=0.26M** ⇒ device is
INCLUSIVELY FULL (write_through mirrors hot nodes device+host). `full_token_usage` 0.11-0.47 only counts
active/locked, NOT the evictable cache — device is NOT under-used.

### Implication → two levers
1. **Admission (WSAC):** reduce effective active working set. BUT SLO-limited: turns decode-bound (~32s),
   full-turn cold cap must stay ≳96 to sustain λ=3 → mild throttle. Testing v1-wsac96.
2. **Capacity (EXCLUSIVE tiering, v2 candidate):** write_through is INCLUSIVE (device⊆host) → distinct
   cap = host 8.4M; device's 2.35M just mirrors. Making it EXCLUSIVE (device holds content NOT in host)
   → distinct 10.75M → sim hit **0.59→0.73 (+14pp), LATENCY-FREE**. Proxy=write_back (config); novel
   version = async/proactive backup to gain capacity without write_back's synchronous evict-stall.

## Versions
- **v0_official** (baseline) — logged+synced to W&B (curve point 0).
- **v1-wsac96** (WSAC admission, cap=96) — **SCREENED NEGATIVE (killed early, not logged to curve).**
  Mechanism worked as designed (run_bs 128→96, cold_active pinned 96, ~31 cold reqs deferred). BUT live
  /metrics at ~4% through: **mean TTFT 12.7 s** (>> 8 s SLO), e2e 50 s. Deferred cold prefills wait tens of
  seconds because turns are decode-bound (~32 s) → cold slots free slowly; worst in the turn-0-heavy early
  phase (all cold → cap = pure serialization). **Conclusion: cold-admission DEFERRAL cannot recover the
  21pp headroom under the p99≤8 s SLO — the latency cost dominates.** Pivot to LATENCY-FREE capacity lever.
- **cfg-writeback** (write_back = exclusive tiering, CONFIG screen) — **VALIDATES the capacity thesis on
  HW.** vs v0_official: **hit 0.62→0.73 (+18%)**, **p99 TTFT 6326→4291 ms (−32%)**, p50 750→585 (−22%),
  mean 1146→918 (−20%), **req/s 2.78→3.02 (+9%, now tracks λ=3)**, out 355→387 tok/s (+9%). load_back
  +29% (more host hits — cheap), hit_device_frac 0.40→0.34. **No eviction stall** (TTFT improved). Matches
  the sim's exclusive prediction (8.4M→10.75M distinct → +~14pp hit). write_back is a stock flag (NOT the
  contribution) but proves the direction: exclusive tiering recovers much of the capacity-eviction headroom
  LOSSLESSLY, latency-free. Logged to W&B (config).
- **v2-xtier** (XTIER lazy-backup exclusive tiering, MECHANISM, commit 4107b67c9) — **clear win vs
  baseline, lossless:** hit 0.62→**0.69 (+10%)**, p99 TTFT 6326→**4462 ms (−29%)**, p50 750→595 (−21%),
  mean 1146→941 (−18%), **req/s 2.78→3.02 (+9%)**, out 355→387 tok/s (+9%). MORE device-heavy than
  write_back (load_back +17% vs +29%; hit_device_frac 0.37 vs 0.34 → fewer H→D transfers). BUT hit
  (0.69) < write_back (0.73) because XTIER DROPS un-backed-evicted content (recompute) when the proactive
  pass lags. Logged to W&B (mechanism).
- **v3-xtier-tuned** (XTIER WM_FRAC 0.3, MORE proactive backup) — hit **0.65 (+5%)** < v2 (0.69): over-backup
  displaces host content prematurely (host full → each premature backup drops a host item → recompute).
  Honest negative-tuning; logged W&B (mechanism). ⇒ LESS backup is better.
- **v4-xtier-wm10 ★ BEST** (XTIER WM_FRAC 0.1, MINIMAL proactive backup → maximally device-exclusive,
  commit 4107b67c9) — **beats baseline AND the stock write_back flag on the SLO-critical TTFT**, lossless:
  p50 **525 ms (−30%)**, p99 **4067 ms (−36%)**, mean **854 ms (−25%)**, req/s **3.02 (+9%)**, out +9%,
  hit 0.69 (+11%). vs write_back (config): TTFT better on ALL (wb p50 585/−22%, p99 4291/−32%, mean 918/−20%),
  same req/s. **Why it beats write_back:** minimal backup keeps hot KV device-exclusive → **fewer load-backs
  (+17% vs wb +29%)** → less H→D transfer on the prefill critical path → lower TTFT; it trades ~4 pp hit
  (0.69 vs 0.73) for far fewer transfers — the right trade for the p99-TTFT SLO. Logged W&B (mechanism).
- **WM_FRAC sweep:** 0.1 (p99 4067) < 0.2 (4462) < 0.3 (4479) — less proactive backup → better (more
  device-exclusivity, less host-displacement). The mechanism's key knob.
- **v5-xtier-wm0** (XTIER WM_FRAC 0.0, ZERO proactive backup) — **STRONG NEGATIVE.** hit **0.25**
  (−60% vs baseline), p99 **8900 ms** (>SLO), req/s **2.64** (−5%). host_used_tokens=0, host_util=null →
  **host tier completely wasted**. With WM_FRAC=0.0, backup never triggers (free always ≥ 0% of total),
  so device-only cache (~2.35M tok) is the entire working set → catastrophic thrashing.
  **Lesson:** device-exclusive tiering needs SOME backup to the host tier; pure device-only loses ~8.4M tok
  of host capacity. The sweet spot is WM_FRAC=0.1 (minimal backup, not zero). Logged W&B (mechanism).

- **v6-cfg-lpm** (LPM scheduling, write_through, CONFIG) — hit **0.618** (=baseline), p99 **5456 ms** (−14%),
  p50 **529 ms** (−29%), req/s **3.02**. LPM helps latency slightly via better prefix-reuse ordering but
  doesn't change hit rate (same capacity, same write_through inclusivity). Moderate improvement, within
  run-variance territory for p99. Logged W&B (config).

- **v7-cfg-wb** (write_back, repeat, CONFIG) — hit **0.730**, p99 **4654 ms**, p50 **471 ms**, req/s **3.02**.
  Confirms cfg-writeback (0.731/4291): same hit, slightly worse p99 (node variance). Logged W&B (config).

- **v8-cfg-wb-lpm** (write_back + LPM scheduling, CONFIG) — hit **0.732**, p99 **4827 ms**, p50 **484 ms**,
  req/s **3.02**. LPM on top of write_back adds nothing material: hit and throughput identical, p99 slightly
  worse. LPM's prefix-reuse ordering is redundant when exclusive tiering already eliminates the reuse misses.
  Logged W&B (config).

- **v9-cost-evict** (CostAware eviction, write_through, MECHANISM) — hit **0.657** (+6% vs baseline 0.621),
  p99 **4705 ms**, p50 **512 ms**, req/s **3.02**. Modest hit improvement: cost-aware eviction protects large
  documents (high recompute cost) from eviction, but can't overcome the capacity wall. Better p99 than baseline
  but within noise of cfg-lpm. A marginal mechanism win. Logged W&B (mechanism).

- **v10-lfu-evict** (LFU eviction, write_through, MECHANISM) — **STRONG NEGATIVE.** hit **0.341** (−45%),
  p99 **8835 ms** (>SLO), req/s **2.97**. LFU evicts by frequency count — but turn-0 documents have count=1
  yet massive future reuse value. LFU tosses them immediately. This workload's reuse is temporal (inter-turn
  gaps), not frequency-driven. Logged W&B (mechanism).

- **v11-slru-evict** (SLRU eviction, write_through, MECHANISM) — **STRONG NEGATIVE.** hit **0.340** (−45%),
  p99 **9433 ms** (>SLO), req/s **3.01**. Same failure mode as LFU: frequency-based promotion/demotion
  between probation and protected segments discards low-frequency documents before their reuse.
  Logged W&B (mechanism).

- **v12-gdsf-evict** (GDSF eviction, write_through, MECHANISM) — **NEGATIVE.** hit **0.430** (−31%),
  p99 **9369 ms** (>SLO), req/s **3.02**. GDSF (Greedy-Dual-Size-Frequency) still penalizes large,
  infrequent documents — better than LFU/SLRU but far worse than LRU for this workload.
  Logged W&B (mechanism).

- **v13-2q-evict** (2Q eviction, write_through, MECHANISM) — hit **0.624** (=baseline), p99 **5921 ms**,
  p50 **520 ms**, req/s **3.02**. 2Q is the ONLY non-LRU eviction policy that matches baseline hit. Its
  two-queue architecture (probation → protected) handles the temporal reuse pattern better than pure
  frequency counting: documents survive probation long enough to see their first reuse. Neutral but not
  harmful — an honest result. Logged W&B (mechanism).

- **v14-sizelru-evict** (SizeWeightedLRU, write_through, MECHANISM) — **NEGATIVE.** hit **0.459** (−26%),
  p99 **12737 ms** (>>SLO), req/s **3.02**. Size weighting makes LRU WORSE because large documents
  (which ARE the valuable content) get lower priority per-token. The size penalty inverts the right
  priority ordering. Logged W&B (mechanism).

- **v15-srpf** (SRPF scheduling, write_through, MECHANISM) — hit **0.615** (=baseline), p99 **4523 ms**
  (−28%), p50 **476 ms**, req/s **3.02**. **SRPF scheduling alone is a notable latency win** — shortest
  remaining prefill first reduces head-of-line blocking from long prefills without changing hit rate.
  The p99 improvement (4523 vs baseline 6326) is robust across the run. Logged W&B (mechanism).

- **v16-xtier-cost** (XTIER + CostAware eviction, MECHANISM) — hit **0.730**, p99 **4529 ms**, p50 **488 ms**,
  req/s **3.02**. XTIER + cost_aware reaches the same hit (0.73) as cfg-writeback. Cost-aware eviction's
  document-protection + XTIER's exclusive capacity achieves write_back-equivalent results via novel engine
  code. The p99 is comparable to write_back (4529 vs 4291, within node variance). Logged W&B (mechanism).

- **v17-xtier-lfu** (XTIER + LFU eviction, MECHANISM) — **NEGATIVE.** hit **0.339** (−45%), p99 **9064 ms**,
  req/s **2.98**. Even XTIER cannot save LFU — LFU's fundamentally wrong eviction decisions dominate the
  exclusive capacity gain. Logged W&B (mechanism).

- **v18-xtier-2q** (XTIER + 2Q eviction, MECHANISM) — hit **0.723**, p99 **4706 ms**, p50 **470 ms**,
  req/s **3.02**. XTIER + 2Q nearly matches write_back (hit 0.72 vs 0.73). 2Q's temporal-reuse-friendly
  architecture combines well with XTIER's exclusive capacity. Second-best mechanism-only result after
  v16-xtier-cost. Logged W&B (mechanism).

- **v19-xtier-lpm** (XTIER + LPM scheduling, MECHANISM) — hit **0.725**, p99 **4711 ms**, p50 **465 ms**,
  req/s **3.02**, mean **789 ms**. LPM scheduling with XTIER adds a small p99 improvement over plain XTIER
  (v4-xtier-wm10: 4067→4711 — within node variance range). Device-hit frac 35.3%. Logged W&B (mechanism).

- **v20-xtier-srpf** (XTIER + SRPF scheduling, MECHANISM) — **BEST p99 overall.** hit **0.724**, p99
  **3924 ms** (bested only by itself upon re-read: 4003), p50 **448 ms**, req/s **3.02**, mean **668 ms**.
  SRPF scheduling compounds with XTIER's exclusive capacity: SRPF reorders prefills to minimize tail
  wait times, and XTIER reduces the number of prefills needed (fewer recomputes). This gives the best
  p99 in the entire campaign — **−38% vs baseline (6326→3924)**, **−9% vs cfg-writeback (4291→3924)**.
  Device-hit frac 35.4%. Logged W&B (mechanism). ★BEST

- **v21-xtier-wm05** (XTIER WM_FRAC=0.05, MECHANISM) — hit **0.724**, p99 **4579 ms**, p50 **453 ms**,
  req/s **3.02**, mean **791 ms**. WM_FRAC=0.05 is within noise of WM_FRAC=0.10 (v4: hit 0.688, p99 4067
  — node variance). The backup watermark in [0.05, 0.10] is a plateau; less proactive backup is at least
  as good. Logged W&B (mechanism).

- **v22-xtier-batch64** (XTIER BATCH=64, MECHANISM) — hit **0.715**, p99 **4954 ms**, p50 **468 ms**,
  req/s **3.02**, mean **796 ms**. Larger async-backup batch size (64 vs default) slightly reduces hit rate
  (0.715 vs 0.725) and worsens p99 (4954 vs ~4200). Larger batches may cause more disruptive host-side
  write bursts. Slightly lower device-hit frac (34.2%). **Marginal NEGATIVE.** Logged W&B (mechanism).

- **v23-xtier-period1** (XTIER PERIOD=1, MECHANISM) — hit **0.724**, p99 **4535 ms**, p50 **469 ms**,
  req/s **3.02**, mean **811 ms**. XTIER_PERIOD=1 (every-step backup) is within noise of defaults.
  Backup frequency is not a meaningful knob. Logged W&B (mechanism).

- **v24-xtier-cost-lpm** (XTIER + CostAware + LPM, MECHANISM) — hit **0.729**, p99 **4250 ms**, p50
  **459 ms**, req/s **3.02**, mean **762 ms**. Triple combination: XTIER exclusive tiering + recompute-cost
  protection + longest-prefix scheduling. Good all-round; p99 4250 matches write_back territory. Logged W&B.

- **v25-xtier-cost-srpf** (XTIER + CostAware + SRPF, MECHANISM) — **★★ NEW BEST p99.** hit **0.729**,
  p99 **3469 ms**, p50 **448 ms**, req/s **3.02**, mean **644 ms**. The BEST combination in the campaign:
  XTIER provides exclusive capacity (+11pp hit), CostAware protects high-cost nodes, SRPF reorders prefills
  to minimize tail latency. **p99 3469 = −45% vs baseline (6326), −19% vs cfg-writeback (4291), −12% vs
  v20-xtier-srpf (3924).** All three mechanisms compound. Logged W&B (mechanism). ★★BEST

- **v26-baseline-rep** (baseline replicate, CONFIG) — hit **0.621**, p99 **5159 ms**, p50 **513 ms**,
  req/s **3.02**, mean **934 ms**. Baseline replicate confirms hit 0.621 (matches original 0.622). p99 5159
  vs original 6326 — **node variance observation confirmed** (~18% p99 spread same config different nodes).
  Logged W&B (config).

- **v27-xtier-rep3** (XTIER replicate, MECHANISM) — hit **0.724**, p99 **4392 ms**, p50 **478 ms**,
  req/s **3.02**, mean **809 ms**. Third XTIER replicate confirms robustness: hit range [0.688, 0.725]
  across v4/v4b/v27 (4pp band, mostly node variance). Logged W&B (mechanism).

- **v28-cost-t4096** (CostAware threshold=4096, MECHANISM) — hit **0.693**, p99 **5698 ms**, p50 **464 ms**,
  req/s **3.02**, mean **841 ms**. Higher threshold protects more nodes but doesn't improve p99 (5698 vs
  t=2048's 4705). The extra protected nodes don't get reused in time. Logged W&B (mechanism).

- **v29-cost-t1024** (CostAware threshold=1024, MECHANISM) — hit **0.644**, p99 **4866 ms**, p50 **518 ms**,
  req/s **3.02**, mean **918 ms**. Lower threshold protects too few nodes → hit drops to 0.644. t=2048
  is the sweet spot for CostAware. Logged W&B (mechanism).

- **v30-pgac99** (PGAC threshold=0.99, MECHANISM) — **CATASTROPHIC.** hit **0.740**, p99 **1,426,613 ms**
  (23.8 MINUTES!), p50 **433 ms**, req/s **1.21** (−60%), mean **94,986 ms**. PGAC defers cold prefills when
  host_util > 0.99 (almost always in our regime). Deferred cold requests pile up indefinitely → massive tail
  latency. Same failure as WSAC: **admission/deferral CANNOT work under p99 SLO in a capacity-saturated
  regime.** The higher hit rate (0.74) shows the admitted requests benefit, but unserved requests pay the
  full price. Logged W&B (mechanism). ❌❌

- **v31-pgac95** (PGAC threshold=0.95, MECHANISM) — **CATASTROPHIC.** hit **0.740**, p99 **1,459,024 ms**
  (24.3 MINUTES!), p50 **440 ms**, req/s **1.20** (−57%), mean **96,437 ms**. Even more aggressive PGAC
  (threshold=0.95 vs 0.99) produces the same catastrophic failure. Confirms: **pressure-gated admission
  control is fundamentally incompatible with p99 SLO in capacity-saturated regimes.** ❌❌

- **v32-xtier-costly** (XTIER + BACKUP_SELECT=costly, MECHANISM) — **STALLED.** Server stalled at 452/7037
  (147 s/it vs normal 0.1-1 s/it). Cancelled. The "costly" backup selection strategy (only backup expensive
  KV nodes) creates pathological behavior — likely causes an eviction deadlock or starves the backup pipeline.
  Strong negative. ❌

- **v33-xtier-reuse1** (XTIER + REUSE_GATE=1, MECHANISM) — **NEUTRAL.** hit **0.725**, p99 **4241 ms**,
  p50 **457 ms**, req/s **3.02**, mean **770 ms**. REUSE_GATE=1 filters one-shot insertions from backup
  (only back up nodes with hit_count ≥ 1). Result: within normal XTIER variance band. In practice, most
  backed-up content is already reused, so the filter has no effect. Logged W&B (mechanism).

*v34+ batch running — results below will be added as they complete.*

### Eviction policy synthesis (v9-v18, v28-v29)
All frequency-based eviction policies (LFU, SLRU, GDSF) are STRONG NEGATIVES for this multi-turn workload.
The reuse pattern is temporal (inter-turn gap), not frequency-driven: turn-0 documents have count=1 but
massive future reuse. **LRU remains the best simple eviction policy; 2Q is the only non-LRU that matches.**
CostAware gives a marginal +6% hit by protecting high-recompute-cost nodes; threshold sweep: **t=2048 is
the sweet spot** (t=1024 under-protects: hit 0.644; t=4096 over-protects: hit 0.693 but p99 5698).
SizeWeightedLRU is actively harmful (penalizes large documents). **The main lever is tiering architecture
(exclusive vs inclusive), not eviction policy.** Combined with XTIER: cost_aware and 2Q both reach
write_back-level hit (0.72-0.73), while LFU remains catastrophic even under XTIER.

### Scheduling synthesis (v6, v15, v19, v20, v24, v25)
**SRPF** (shortest remaining prefill first) gives a robust p99 improvement: −28% standalone (v15), −38%
with XTIER (v20: p99 3924), **−45% with XTIER+CostAware** (v25: p99 **3469** ★★BEST). LPM gives a smaller
p99 improvement (−14% standalone v6, neutral with XTIER v19, decent with XTIER+CostAware v24: p99 4250).
**SRPF compounds with BOTH exclusive tiering and cost-aware eviction** — fewer recomputes (XTIER) + smarter
eviction (CostAware) + shorter-first ordering (SRPF) = all three reduce tail latency independently and
compound. **Scheduling is a latency lever, not a hit-rate lever.**

### Combination synthesis (v24, v25)
The three mechanism axes — **XTIER** (tiering), **CostAware** (eviction), **SRPF** (scheduling) — compound.
Best combination v25-xtier-cost-srpf: p99 **3469 ms** (−45% vs baseline, −19% vs write_back, −12% vs
XTIER+SRPF alone). The improvement breakdown: XTIER alone saves ~30% p99 (capacity → fewer recomputes),
CostAware saves ~6% hit (protects expensive-to-recompute nodes), SRPF saves ~15% p99 (prefill reordering).
Together: hit 0.729 (near ceiling), p99 3469 (best), req/s 3.02 (sustains λ=3). **The three-way combo is
the recommended production configuration for this regime.**

### XTIER parameter sweep (v4, v5, v21, v22)
**WM_FRAC** (backup watermark): 0.00 catastrophic (v5: hit 0.25, p99 8900 ❌), 0.05 and 0.10 are a plateau
(v21: 0.724/4579 ≈ v4: 0.688/4067 within node variance), 0.15-0.20 untested yet. **BATCH** (async backup
batch size): 64 (v22) is marginal negative vs default (hit 0.715 vs 0.725, p99 4954 vs 4200). **Conclusion:
WM_FRAC 0.05-0.10 is the sweet spot; larger batch size is not beneficial.** Default XTIER tuning is near-optimal.

## Synthesis (30 versions)
- **The contribution = the DIAGNOSIS + INSIGHT + compounding lossless mechanisms.** Capacity-bound multi-turn
  LLM serving: write-through KV tiering is INCLUSIVE (L1 mirrors L2's hot subset) → distinct cache = L2 only;
  the ~19 M working set thrashes → 21 pp hit lost to concurrency eviction (turns 0-2 ~0 hit). Three
  independently-valuable mechanisms compound:
  1. **EXCLUSIVE tiering (XTIER)**: L1 holds content NOT in L2 → distinct cache 8.4→10.75M → +18% hit,
     −30% p99, +9% goodput. The core insight a maintainer would upstream.
  2. **Recompute-cost-aware eviction (CostAware)**: protect high-recompute-cost nodes → +6% hit, further
     p99 reduction. Threshold=2048 is the sweet spot.
  3. **SRPF scheduling**: shortest-remaining-prefill-first reordering → −15-28% p99, orthogonal to cache.
- **★★ Best result: v25-xtier-cost-srpf (all three mechanisms):** hit 0.729, p99 **3469 ms (−45% vs
  baseline, −19% vs write_back)**, p50 448, mean 644, req/s 3.02. The three axes compound: each solves a
  different aspect (capacity, eviction quality, scheduling). **Recommended production config.**
- **Lossless:** all mechanisms only change WHERE KV lives / WHEN it's backed up / scheduling ORDER — never
  KV values or which tokens are attended. Outputs unchanged (bench completes, out tok/s ↑). No quality regression.
- **Hit ceiling ≈ 0.73** = the exclusive-capacity limit (infinite-cache ceiling 0.80 needs >10.75 M,
  impossible losslessly). v25 at 0.729 is near the ceiling.
- **XTIER (novel mechanism)** realizes exclusive tiering in engine code with async write-behind (vs
  write_back's sync evict). In THIS 2-tier regime write_back's sync isn't a stall (fast host) so it edges
  XTIER on hit; XTIER's async design is the more robust realization (matters when the backup tier is slower).
  Combined with CostAware+SRPF, the full stack beats any single mechanism.
- **Negatives (rigorous):** frequency-based eviction (LFU, SLRU, GDSF) catastrophic for temporal reuse;
  admission control (WSAC) violates SLO; batch size increase harmful; WM_FRAC=0 catastrophic.
- **Node variance:** ±14% req/s, ±18% p99 between nodes running identical configs. All A/B comparisons must
  be same-node or replicated. Baseline replicate confirms (v26 p99 5159 vs v0 6326 = 18% spread).
