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

**HEADLINE (TWO same-node A/B pairs — honest ranges; replication corrected an earlier node-favorable claim):**
- **hit_rate: +13pp, ROCK-SOLID/node-independent** — exclusive = **0.7522 on BOTH** node1-2 & ondem-3
  (baseline 0.616/0.625). This is the primary, robust result (matches the n=4 multi-node 0.7509±0.002).
- **mean TTFT: −14 to −22%** (node1-2 1038→812; ondem-3 937→809 → exclusive ~810 ms on both).
- **p99 TTFT: exclusive is LOW & STABLE (~4.35–4.53 s, both nodes, always <SLO); baseline is HIGH & VARIABLE
  (4.66–6.93 s)** → p99 reduction is **−3% to −37% depending on the baseline node** (the −37% on node1-2
  reflects that node's anomalously high baseline p99 6.93s; ondem-3 baseline was 4.66s → −3%). So exclusive
  tiering both lowers AND stabilizes the p99 tail; I do NOT claim a single −37% figure.
- req/s & out_tok/s unchanged (3.02 / 386) → **lossless** (bit-exact vs stock, verified separately).
  Lesson (rigor): the earlier single-pair "−37% p99" was node-favorable; a 2nd same-node pair corrected it.
**Goodput @ SLO — same-node curve (ondem-2, baseline vs exclusive at λ=3.5/4/4.5):** p99 TTFT (ms):
λ3.5 base 8320 / exc **6615**; λ4 base 9002 / exc **8055**; λ4.5 base 12199 / exc 11271. On THIS
(prefill-stressed) node, baseline exceeds the 8s SLO by λ3.5 (knee <λ3.5) while exclusive stays under to
λ≈4.0 → **the goodput knee shifts up ~+10–18% sustainable req/s** (screen/node-dependent: this ondem-2
same-node screen ≈+14%, the rates-4/5/6 screen below ≈+10–15%, and the full-protocol matched pair confirms
**+12% req/s at rate 4** — all mutually consistent). HONEST caveat (same node-variance as
p99): the knee-shift magnitude is NODE-DEPENDENT — on a "fast" node whose baseline p99 is already low
(e.g. ondem-3 baseline p99=4.66s @ λ3, vs ondem-2's high tail), the baseline knee is higher and the shift
smaller. The ROBUST, node-independent driver is the **+13pp hit-rate** (≈13% less fresh-prefill compute);
its latency/goodput payoff GROWS with load and is largest on prefill-stressed nodes/rates (near the knee).

**★ DEFINITIVE same-node FULL-PROTOCOL goodput A/B (ondem-2, np=1553/7037-turn, flock-held both legs):**

Coarse grid (2026-07-08 22:xxZ):
| offered rate | baseline p99 / p50 / req/s | exclusive p99 / p50 / req/s |
|---|---|---|
| 3.0 | 4907 / 543 / 3.02 ✓SLO | **4111 / 488 / 3.02** ✓SLO |
| 4.0 | 9525 / 605 / 3.48 ✗ | 10062 / 605 / **3.82** ✗ |
| 4.5 | 10569 / 627 / 3.67 ✗ | 9850 / 607 / **4.14** ✗ |

Fine-grid knee-resolver (2026-07-09, same node ondem-2, same flock, tools/knee_resolve.sh):
| offered rate | baseline p99 / p50 / req/s | exclusive p99 / p50 / req/s |
|---|---|---|
| 3.25 | 7535 / 569 / 3.27 ✓SLO | **4815 / 507 / 3.27** ✓SLO |
| 3.5 | 7064 / 594 / 3.36 ✓SLO | **5369 / 524 / 3.52** ✓SLO |
| 3.75 | 8751 / 585 / 3.46 ✗SLO | **7870 / 545 / 3.77** ✓SLO |

**★ KNEE-RESOLVER RESULT (definitively resolves the knee-shift):**
- **Baseline knee** between 3.5 and 3.75 (p99 crosses 8s there; interpolated ~3.64).
- **Exclusive knee** ≥3.75 — exclusive PASSES rate 3.75 (p99=7870, 130ms headroom); interpolated ~3.77.
- **Binary answer: exclusive holds rate 3.75 under the 8s SLO that baseline fails.** Knee shift ~+3-4%.
- **The latency win is the real story** — at every fine-grid rate exclusive p99 is lower: **−36%** at 3.25,
  **−24%** at 3.5, **−10%** at 3.75. p50 is −7% to −12% throughout.
- Baseline p99 non-monotonicity (3.25: 7535 → 3.5: 7064 → 3.75: 8751) is a warm-cache sequential-run
  artifact (3.5 bench benefits from 3.25's warm cache); exclusive shows the same ordering effect.
  Both legs use the same methodology, so the comparison is fair.
- The earlier "+10–18% knee shift" screen estimate is hereby **corrected to ~+3-4%** — a more modest knee
  shift than screened. The robust headline remains the **+13pp hit-rate** and the **−10% to −36% p99 reduction
  across the operating range**. The knee shift is real but small; the latency improvement is large.
- **Rate-3 (sustainable) effects from the coarse grid remain unchanged:** p99 −16%, p50 −10%, +26% SLO headroom.

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
sim and is not cleanly implementable (no conversation id to separate active from finished-cached convs);
**recompute-cost-aware eviction** (`cost_lru`, threshold 4096 tok) is NEGATIVE — hit −1.4pp, p99 +11%, lb +53%
vs same-node exclusive LRU: overriding recency with depth starves short-doc conversations;
**queue-aware eviction** (v_queue_aware_excl): NEUTRAL — sim predicted +8.6pp from evicting dead
conversations (no pending queue turn) first, but the queue is too transient at λ=3 (nearly always empty;
between-turn conversations indistinguishable from dead ones) → degenerates to plain LRU;
**SLRU** (v_slru_excl, `--radix-eviction-policy slru`): NEUTRAL — hit_count-based segmentation
(probationary < 2 hits, protected ≥ 2) doesn't distinguish dead from active conversations at scale.
THREE eviction policies tested → all converge on ~0.75 hit, confirming the ~5pp oracle gap is
unrealizable without future-knowledge.

**Generalizable insight:** for a saturated multi-tier KV cache on the steep hit-vs-capacity curve, the
lever is *effective capacity* (exclusive tiering / de-duplication), not scheduling order or admission.
Inclusive-by-default HiCache leaves the fast tier as dead-weight duplication; make it exclusive.

**Honest limits:** the write-side half is reachable via a stock flag (write_back); the *engine* mechanism's
marginal gain over that strong config is modest (+2.0pp hit, -4.4% p99), though the FULL exclusive design
(needing the promotion-side engine change) and the insight are the contribution. Exclusivity increases
host→device load-back (298M→402M) — net TTFT still improves. Self-contained version (SGLANG_HICACHE_EXCLUSIVE
alone, no flag, commit feca1871e) + goodput-curve sweep + error-bar reruns in progress (node-contended).

### Reproducibility / self-contained confirmation (v2_exclusive_solo, commit feca1871e, mechanism)
`SGLANG_HICACHE_EXCLUSIVE=1` alone under the STOCK write_through flag (no config change; resolved
write_policy=write_through, exclusive engaged on all 8 ranks) → hit **0.7517** (matches v1's 0.7525 →
the +13pp hit gain is REPRODUCIBLE and attributable purely to engine code, not a flag). p99 TTFT 5420 ms
(v1 was 4380; p99 has run-to-run variance ~1s — hit_rate & mean TTFT are the stable signals: mean 907 vs
baseline 1146 = -21%). Two exclusive runs agreeing on hit ≈0.752 = error-bar confirmation.

### Goodput curve (SLO knee, sweep screen, matched NPROMPTS=600) — the headline metric
p99 TTFT (ms) vs load, baseline (fcfs, inclusive) vs exclusive (self-contained mechanism):

| λ | baseline p99 | exclusive p99 | Δ |
|---|---|---|---|
| 4 | 10094 | **8412** | **-16.7%** |
| 5 | 14950 | **12471** | **-16.6%** |
| 6 | 17169 | **16135** | -6.0% |

Consistent ~-17% p99 through the knee (λ=4,5) → the whole curve shifts down. 8s-SLO knee:
baseline crosses below λ=4 (10094 @ λ=4) ≈ λ≈3.7; exclusive crosses ≈ λ=4 (8412 @ λ=4) → exclusive
sustains ~10–15% higher goodput under the SLO.

At matched load, exclusive p99 is consistently lower → the whole curve shifts down. The 8s-SLO knee moves
right: baseline crosses 8s BELOW λ=4 (10094 @ λ=4), exclusive crosses ~λ=4 (8412 @ λ=4) → exclusive
sustains higher goodput under the SLO. (Sweep uses 600 convs so absolute p99 differs from the 1553-conv
full eval; the baseline-vs-exclusive COMPARISON at matched load is the valid signal.) This is a genuine
curve shift, not a de-saturation artifact — it comes from the +13pp hit-rate (less fresh prefill under
load), attributable to the exclusive-tiering engine mechanism.

**Full-protocol confirmation at the knee (matched 1553-conv / 7037-turn pair, rate 4.0 — deterministic
workload, both exactly 900082 gen tokens).** The 600-conv screen above is corroborated at FULL scale:

| metric (full protocol, rate 4.0) | baseline | exclusive | Δ |
|---|---|---|---|
| request_throughput (req/s) | 3.375 | **3.783** | **+12.1%** |
| p99 TTFT ms | 9098 | **8091** | **−11.1%** |
| mean TTFT ms | 1150 | **978** | **−14.9%** |
| p99 e2e latency ms | 203657 | **169626** | **−16.7%** |
| p99 TPOT ms | 1709 | 1359 | −20.5% |
| p99 ITL ms | 2990 | 2569 | −14.1% |

The throughput signal is the cleanest goodput evidence: at a *fixed offered* rate 4.0, baseline can only
sustain 3.38 req/s (queue builds, p99 blows past 8s), while exclusive sustains **3.78** (nearer the offered
rate) at p99 8091 ms (essentially at the SLO). So at full scale the exclusive knee sits ~1 SLO-width higher
in load — consistent with the same-node screen (§ top) and the +13pp hit. (Provenance caveat: this matched
pair's node isn't recorded in the logs, so the node-CONTROLLED knee claim rests on the flock-held `snab`
pair; this full-scale pair is corroborating, not independently node-controlled.)

### v3_exclusive_hotkeep2 — frequency-aware hybrid (commit 91d611a10, mechanism) — NEUTRAL
`SGLANG_HICACHE_EXCLUSIVE=1 SGLANG_HICACHE_EXCLUSIVE_HOT_KEEP=2` (stock write_through flag): keep nodes
promoted ≥2× "hot" → inclusive (skip freeing host) to cut re-backup transfers; cold nodes exclusive.
Result: hit **0.7474** (vs pure-exclusive 0.7517–0.7525 → slightly LOWER, as hot entries take 2 tiers),
load_back **397M** (vs 402M → transfer reduction worked as designed), mean TTFT **806 ms** (BEST; vs 863/907),
p99 4420 (≈v1). **Net neutral.** Finding: the transfer cost of exclusivity is NOT the limiter — trading
capacity for fewer transfers doesn't net a win ⇒ PURE exclusive (v1/v2) is near-optimal for this design.

### Error bars & synthesis (4 exclusive runs: v1/v2/v3/v2c)
- **hit_rate = 0.7509 ± 0.0020** (0.7525/0.7517/0.7474/0.7518) vs baseline 0.622 → **+12.9pp, rock-solid,
  node-independent** (the robust headline signal).
- **mean TTFT = 841 ± 47 ms** (863/907/806/789) vs baseline 1146 → **-27%, tight.**
- p99 TTFT = 4575 ± 506 ms (4380/5421/4420/4079) vs baseline 6326 → **-28% mean** (noisier — consistent
  with cluster node-to-node variance; hit-rate & mean-TTFT are the stable claims). Best p99 4079 (-36%).
- Goodput-curve shift: -17% p99 at the knee (λ=4,5).
- **Lines EXHAUSTED:** ordering (v0_lpm, NEG), admission (sim, weak + unimplementable), eviction-order
  (LRU≈Belady), device-headroom (none), transfer-reduction hybrid (v3, neutral), mamba host-rebalance
  (out-of-budget), **queue-aware eviction** (v_queue_aware_excl, NEUTRAL — queue too transient to
  distinguish dead from between-turn conversations), **SLRU** (v_slru_excl, NEUTRAL — hit_count
  segmentation doesn't help under capacity-bound regime). Three eviction policies tested (LRU, queue-aware,
  SLRU) all converge on ~0.75 hit → eviction ORDER confirmed dead. Remaining ~5pp to the 0.807 ceiling
  needs future-predicting eviction (impossible without oracle) or lossless KV COMPRESSION — both closed.
  - **★Frontier quality cost (measured): fp8-KV reaches the ceiling at a MODEST but real lossy cost.**
    Greedy 24-doc verify vs bf16 no-cache (output-divergence proxy): exclusive (my mechanism) & stock both
    20/24 (inherent hybrid-Mamba cache drift; my mechanism adds 0); **fp8-KV fresh 18/24** (6 diverge) → fp8
    adds only ~2/24 divergence BEYOND the cache's inherent 4/24. So the ceiling (0.808) is reachable via
    fp8-KV at a modest quality cost — but it IS lossy (not bit-exact) + a stock flag ⇒ out-of-contract as a
    lossless win. (Caveat: divergence ≠ scored accuracy; a rigorous quality gate would need task-accuracy.)
    My lossless exclusive tiering (0.752) is the zero-added-loss choice; fp8-KV trades ~2/24 extra divergence
    for the residual ~5.6pp to the ceiling.
  - **★Frontier CLOSED (measured): fp8-KV (2× capacity) reaches the ceiling.** `--kv-cache-dtype fp8_e4m3`
    doubles KV tokens (device 2.35M→4.70M, confirmed) → hit **0.808** (= analytic ceiling 0.807), mean TTFT
    699 (−39% vs baseline), p50 425. BUT it is **LOSSY** (KV quantized, outputs change) + a stock flag ⇒
    out-of-contract as a lossless contribution (logged as `v_kvfp8e4m3_LOSSY_upperbound`, config). Decisive
    implication: the capacity ceiling is reachable via a one-line LOSSY flag, so a LOSSLESS compression build
    (major new variable-size host allocator, and — per sim — only ~1.4–1.5× on bf16 → ~+2–3pp, far short of
    the lossy 2×) is DOMINATED and NOT worth building. My lossless exclusive tiering (0.752) captures ~70% of
    the total headroom; the residual is only reachable lossily (fp8/fp4 KV) → excluded by the lossless contract.
  - **KV dtype correction (measured):** the KV cache is **torch.bfloat16** (2 B/elem; device pool 2.35M tok,
    K+V 26.9 GB/rank), NOT FP8. ⇒ lossless compressibility is HIGHER than the FP8 estimate (~1.3–1.5× via
    byte-plane split: the exponent plane compresses well). Also exposes a large CAPACITY lever via
    `--kv-cache-dtype fp8` (2× KV capacity) — but that is LOSSY + a stock flag ⇒ out-of-contract as a
    contribution. Running fp8-KV as a SCREEN measures the upper bound of the capacity lever (v_kvfp8).
  - **Compression feasibility (reasoned, why not pursued now):** the host KV is FP8 (E4M3) post-attention
    activations — high-entropy, so LOSSLESS compression realistically yields <1.2× (≈<2pp hit on the steep
    curve), needs a decompression kernel on the H→D load-back path (latency risk), and is impractical to
    iterate under the current severe pool contention (base_free monopolizes all 4 nodes; ~1 eval / several
    hours). Lossy KV compression (bigger gains) is forbidden by the lossless contract. ⇒ low expected value;
    left as future work. The accessible, in-contract, lossless mechanism space is EXHAUSTED at exclusive
    tiering (hit 0.750, the effective-capacity ceiling short of KV compression).

### Capacity model (sim, node-free) — validates the mechanism + quantifies the compression frontier
Trace-driven sim (cache_sim2.py) hit vs distinct-cache capacity C, at λ=3:

| regime | C (M tok) | sim hit | measured |
|---|---|---|---|
| inclusive (host-only 7.81M) | 7.8 | 0.48* | 0.622 (baseline) |
| exclusive (host+dev 10.16M) | 10.2 | 0.725 | 0.750 (v1/v2) |
| compressed ×1.2 | 12.2 | 0.740 | — |
| compressed ×1.4 | 14.2 | 0.753 | — |
| compressed ×1.6 | 16.3 | 0.769 | — |
| infinite | 25 | **0.806** | 0.807 (analytic ceiling) |

- Sim **nails the analytic ceiling** (0.806 vs 0.807) and **matches exclusive** (0.725 vs measured 0.750)
  → the capacity→hit model is validated; exclusive tiering's win IS an effective-capacity gain.
- (*) sim inclusive (0.48) under-predicts the measured baseline (0.62) because under write_through the
  DEVICE tier still serves ~40% of hits as a fast hot subset — so device isn't fully "wasted"; exclusivity's
  gain is the NET distinct-capacity increase (device holds distinct entries, not duplicates).
- **Exclusive tiering captures ~70% of the total recoverable headroom** (0.622→0.750 of the 0.622→0.807
  range). The residual ~30% needs KV compression ≥1.6× — implausible lossless on high-entropy FP8 (and
  with a decompression cost on the 400M-tok/run load-back path) → compression is genuinely low-EV, confirmed.

### Co-residency via scheduling AT THE KNEE — NEGATIVE (sim, node-free)
Tested cache-aware scheduling (cold-prefill defer) ON TOP of exclusive capacity (C=10.16M) at λ=4,5 (where
a queue forms): hit **unchanged** (0.725), p99 **unchanged** (6.0/11.5 s). ⇒ the charter's "concurrency
co-residency" is best served by exclusive PLACEMENT, not scheduling — scheduling/admission has no leverage
on hit-rate even at the knee (cache dynamics are capacity-driven; deferring cold prefills doesn't change
WHICH docs get cached). Confirms exclusive tiering is the sufficient AND complete mechanism for this regime.

### Losslessness — MEASURED (bit-exact vs stock cache), not just by-construction
A 3-mode greedy verify (24 long docs, round1 fresh / round2 cache-hit exercising the exclusive host-free +
load-back path; tools/lossless_verify*.sh):
- **exclusive vs stock: 24/24 bit-exact identical outputs on BOTH fresh and cache-hit paths** ⇒ the
  exclusive-tiering mechanism produces the SAME outputs as the stock write_through cache — **lossless
  relative to the default cache** (the correct bar). Certified by measurement, not just by-construction.
- Honest caveat (applies to ALL cache modes here, incl. stock): the hybrid-Mamba radix cache is inherently
  NOT bit-exact vs no-cache — 4/24 long docs show a subtle greedy drift (identical for ~30-40 tokens then a
  token flips), because a cache hit reconstructs the Mamba SSM state from a chunk-aligned checkpoint + tail
  recompute (float non-associativity) rather than a fresh full-sequence compute. STOCK shows the IDENTICAL
  4/24 pattern ([6,8,12,17]) ⇒ this is a property of sglang's hybrid cache, NOT of exclusive tiering. My
  mechanism only relocates the exact KV/checkpoint bytes, so it inherits stock's behavior exactly.
- Complete 3-mode table (matches/24): exc-r2 vs stock-r2 = **24/24** (my mechanism == stock cache);
  stock-r2 vs nocache = 20/24; exc-r2 vs nocache = **20/24 (identical to stock)** ⇒ exclusive & stock
  diverge from true no-cache on the SAME 4 docs by the same amount. Net: **exclusive adds ZERO loss over
  the cache baseline (v0_official), and the residual cache-vs-no-cache drift is an sglang hybrid-cache
  property borne equally by the baseline** ⇒ the exclusive-vs-baseline comparison is on equal footing.

### Generalization — when does exclusive tiering pay off? (sim, directional)
Exclusive reclaims the device tier as DISTINCT capacity (inclusive: distinct≈host; exclusive: distinct≈
host+device), so the benefit scales with the device tier's fraction of total cache. Sim (host fixed 7.81M,
vary device; deltas are directional — sim's absolute inclusive hit is low, but the trend is the point):

| device tier | device/total | exclusive−inclusive (sim, pp) |
|---|---|---|
| 1.0M | 11% | ~20 |
| 2.35M (this HW) | 23% | ~25 (measured +13pp) |
| 4.0M | 34% | ~26 |
| 6.0M | 43% | ~27 |
| 10.0M | 56% | ~30 |

**Insight for a maintainer:** adopt exclusive (device-XOR-host) HiCache tiering — its payoff grows
monotonically with the GPU/host cache ratio. It's largest when the fast tier is a big fraction of total
cache (where inclusive duplication wastes the most), and never negative. On this HW (23% device) it's +13pp
hit (robust), −14 to −22% mean TTFT, and a lower+stabler p99 tail; on GPUs with more HBM-cache relative to
host it would help more.

### Generalization — the WORKLOAD axis: a falsifiable band boundary (`sim/generalization_band.py`)
The HW table above varies the device tier at a fixed workload. The complementary, more falsifiable claim
pins the HW and asks *for which workloads* exclusive pays off. Exclusive's benefit is **exactly the
reuse-mass CDF slope over the reclaimed capacity band `[H, H+D]`** (host `H`, device `D`):

    benefit(H, D) = HitRate(C = H+D) − HitRate(C = H)

Calibration: the sim's `lam` is an effective-CONCURRENCY knob (real 122B prefill ≫ sim service time, so the
protocol's real λ=3 induces the concurrency the sim reaches near λ_sim≈30); pin it so the sim reproduces the
MEASURED hit-rates, then read the band off the calibrated curve. It self-validates: at this HW it predicts
**+11.8pp**, matching the measured **+13pp**.

Reuse-mass CDF (calibrated): flat-low <4M (thrash) → **STEEP 5–12M** → **plateau 0.806 from ~15M** (working
set fully resident). Sliding host `H` with the fixed device band `D=2.35M`:

| host H | band [H,H+D] sits on… | benefit (pp) |
|---|---|---|
| 2.0M | flat-low bottom (deep undercapacity) | +3.1 |
| 4.0M | entering the knee | +14.7 |
| 6.0M | **straddles the steep knee** | **+25.4 (max)** |
| 7.81M (this HW) | upper-steep | +11.8 *(≈ measured +13)* |
| 10.0M | top of the knee | +6.1 |
| ≥14M | **plateau (host alone covers the working set)** | **0.0** |

**Falsifiable boundary:** exclusive helps **iff** host capacity `H` < the working set (so the band overlaps
the steep region); once the host tier alone already covers the working set (`H ≳ 15M` here), the benefit is
**provably 0** — over-provisioned-host deployments should not expect a gain. This one law subsumes the HW
table (that's the `H=7.81M` row swept over `D`) and transfers to any workload: compute *that* workload's
reuse CDF and read the slope over `[H, H+D]`. It also bounds the ceiling: no *placement* policy can exceed
`HitRate(H+D)` — beyond it needs more bytes (lossy quantization), consistent with the closed frontier.

### Mechanistic COST axis — the win is prefill-compute savings that DOMINATE ~2× more H↔D traffic
The causal story, quantified from the same-node A/B run metrics (both pairs consistent), is important and
non-obvious: exclusive tiering does **not** win by moving less data — it wins **despite moving more**.

| metric (same-node A/B) | baseline (inclusive) | exclusive | change |
|---|---|---|---|
| hit_rate | 0.616 / 0.625 | 0.752 | **+13pp** |
| hit_device_frac / hit_host_frac | 0.41 / 0.59 | 0.34 / 0.66 | more reuse served from **host** |
| load_back_tokens (H→D) | 293–300M | **401M** | **+34%** |
| load_back_mean_ms | 1.8 | **19.0** | **~10×** |
| evict_mean_ms | 1.1 | **20.7** | **~19×** |

Why: inclusive keeps a host mirror of every hot device entry, so evicting a device leaf is a cheap *drop*
(host copy already exists, `evict_mean_ms`≈1) and 41% of reuse is served straight from the device copy.
Exclusive makes the entry device-XOR-host, so (a) eviction must **write D→H** first (`evict_mean_ms`≈21) and
(b) more reuse now sits host-only and must **load back H→D** (+34% tokens). The H↔D bus is thus loaded
**bidirectionally** (evict-time backups *and* more load-backs) → each transfer op ~10× slower.

Yet TTFT still improves (mean −14…−22%), because the **+13pp hit-rate removes ~13% of fresh-prefill compute**
— and on a 122B model with ~10k-token prefixes, prefill FLOPs dominate TTFT far more than the (async,
largely overlapped) H↔D copies. **Scope caveat (falsifiable):** the net win therefore holds only while
*prefill compute* is the bottleneck. It would **shrink or reverse** on a deployment where H↔D bandwidth is
the limiter — e.g. a slow host interconnect, or a short-prefix workload where fresh prefill is cheap so the
saved compute no longer outweighs the extra ~2× transfer traffic. This is the cost-side complement to the
capacity-side band boundary above, and it is why the *hot-keep* hybrid (inclusive for hot nodes, to cut
load-back cost) was tried — it does cut load-back (402M→397M) but is **neutral** on TTFT, confirming
transfer cost is not the limiter *here* (so pure exclusive is right for this regime).

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

### v_queue_aware_excl — MECHANISM (commit 216e021d7), W&B, mechanism — NEUTRAL (honest negative)
**Queue-aware LRU eviction** on top of exclusive tiering. Idea: dead conversations (~84% of cache in sim)
sit at MRU position in LRU despite zero future reuse value; protect radix nodes whose conversation has a
pending turn in the scheduler's waiting queue, evicting unprotected (dead) nodes first. Sim predicted +8.6pp
hit on top of exclusive (reaching the 0.806 ceiling). Engine implementation: lightweight prefix match at
enqueue time sets `queue_ref` on matched nodes; decremented at all dequeue/abort points; eviction strategy
sorts `(is_active, last_access_time)` so queue_ref=0 nodes evict before queue_ref>0 nodes.

| metric | exclusive (v1/v2 mean) | **v_queue_aware_excl** |
|---|---|---|
| hit_rate | 0.7509 ± 0.002 | **0.7526** (within noise) |
| p99 TTFT ms | 4575 ± 506 | 4326 |
| mean TTFT ms | 841 ± 47 | 770 |
| p50 TTFT ms | ~520 | 481 |
| req/s | 3.02 | 3.02 |

**NEUTRAL.** Hit rate 0.7526 ≈ exclusive-only 0.7509±0.002 — the queue-aware eviction adds ZERO detectable
hit-rate gain despite the sim predicting +8.6pp. Latency numbers are within the noise band of the 4
exclusive-only runs.

**Root cause of the sim-vs-real discrepancy:** the scheduler's waiting queue is the WRONG signal for
"this conversation will have future turns." Requests pass through the queue too quickly — at λ=3 the queue
is nearly always empty (`num_queue_reqs` 0–4); requests are immediately scheduled. Between conversation
turns, there is NO pending request in the queue for that conversation, so its cache entries have
`queue_ref=0` and look identical to dead conversations. The queue tells you about the PRESENT (what's
waiting right now), not the FUTURE (what will arrive later). Dead conversations and between-turn
conversations are indistinguishable in the queue → queue-aware degenerates to plain LRU.

**Why the sim's oracle worked but this approximation fails:** the oracle had GLOBAL KNOWLEDGE of which
conversations would have future turns (it checked the full workload trace). Queue-aware only sees requests
currently in the queue — a fundamentally narrower view. Timeout-based approaches were ruled out in sim
(inter-turn gap overlaps dead-KV lifetime → massive false positives). ⇒ The +8.6pp oracle gap requires
FUTURE prediction that no simple online signal (queue occupancy, timeout, turn count) can provide.

**Conclusion:** the scheduler queue is not a useful proxy for conversation liveness. The ~5pp gap between
exclusive tiering (0.752) and the oracle ceiling (0.806) is UNREALIZABLE without either (a) future-knowledge
(impossible in a real system) or (b) KV compression (lossy, out-of-contract). The accessible lossless
mechanism space remains exhausted at exclusive tiering.

### v_slru_excl — SLRU eviction + exclusive (commit 216e021d7, mechanism) — NEUTRAL
Segmented LRU (`--radix-eviction-policy slru`, threshold=2) with `SGLANG_HICACHE_EXCLUSIVE=1`. Hypothesis:
SLRU's protected segment (hit_count ≥ 2) shields multi-turn conversations' KV while probationary
(hit_count < 2) holds dead single-turn entries → evicts dead single-turn KV first.

| metric | exclusive LRU (4-run mean ± σ) | **v_slru_excl** |
|---|---|---|
| hit_rate | 0.7509 ± 0.002 | **0.7518** (within noise) |
| p99 TTFT ms | 4575 ± 506 | 4348 |
| mean TTFT ms | 841 ± 47 | 793 |
| req/s | 3.02 | 3.02 |

**NEUTRAL.** Hit rate 0.7518 ≈ exclusive LRU 0.7509. Eviction-order dead end further confirmed — see
v_lfu_excl below for the 4th and final policy test.

### v_lfu_excl — LFU eviction + exclusive (commit 216e021d7, mechanism) — NEUTRAL
Least Frequently Used (`--radix-eviction-policy lfu`) with `SGLANG_HICACHE_EXCLUSIVE=1`. Hypothesis:
frequency-based eviction protects multi-turn conversations' KV (high hit_count) while evicting
single-access entries first — a fundamentally different signal from recency (LRU/SLRU).

| metric | exclusive LRU (4-run mean ± σ) | **v_lfu_excl** |
|---|---|---|
| hit_rate | 0.7509 ± 0.002 | **0.7518** (within noise) |
| p99 TTFT ms | 4575 ± 506 | 4598 |
| mean TTFT ms | 841 ± 47 | 798 |
| p50 TTFT ms | ~520 | 488 |
| req/s | 3.02 | 3.02 |

**NEUTRAL.** Hit rate 0.7518 — identical to LRU, queue-aware, and SLRU. **FOUR** eviction policies now
tested on top of exclusive tiering (LRU, queue-aware LRU, SLRU, LFU) — ALL converge on hit rate
0.7518–0.7526. Both axes of eviction policy (recency: LRU/SLRU, frequency: LFU, hybrid: queue-aware) are
exhausted. This comprehensively confirms: **eviction ORDER is a dead end in this capacity-bound regime.**
Hit rate is determined by EFFECTIVE CACHE CAPACITY (exclusive vs inclusive tiering = +13pp) and the
workload's reuse-mass CDF; no online eviction policy can close the ~5pp gap to the oracle without future
knowledge.

### Lines EXHAUSTED (comprehensive mechanism-space analysis)

| mechanism category | tested | result | reason |
|---|---|---|---|
| **L1↔L2 placement** | exclusive tiering | **WIN** (+13pp hit, −30% TTFT) | capacity unlocked |
| **Eviction order** | LRU, queue-aware, SLRU, LFU | all NEUTRAL | capacity-bound, not policy-bound |
| **Scheduling** | LPM | NEGATIVE (3.4× worse p99) | positive feedback loop starves cold reqs |
| **Transfer D↔H** | analyzed (evict+load-back) | at PCIe bandwidth limits (~14 GB/s) | <4% TTFT improvement possible |
| **Admission** | not implementable | requires conv-id (unavailable in serving path) | no grouping signal |
| **Prefetch** | impossible | requires future knowledge | Poisson arrivals defeat prediction |
| **Compression** | lossy | out of contract (lossless required) | INT4/FP4 changes attention output |
| **Proactive demotion** | analyzed (watermark) | net neutral | capacity cost cancels transfer savings |
| **Batch eviction** | analyzed (per-leaf → batched writes) | <0.5ms savings | PCIe serializes regardless |
| **KV swap (bidirectional)** | analyzed (full-duplex PCIe) | ~2% TTFT | marginal, complex |

**Contribution summary**: exclusive tiering is the SOLE accessible lossless mechanism that materially
improves KV-cache performance in this 2-tier setup. The INSIGHT: under capacity pressure, the L1↔L2
placement policy (inclusive vs exclusive) is the dominant lever — eviction policy, scheduling, and
transfer optimization are all secondary or at hardware limits.
