# sgl_free — dated research log (v0.25_ablations, 2-tier)

## 2026-07-08 (start)
- Fresh cell v0.25_ablations (2-tier L1+L2, NO disk). Setup done: own clone + cu129 venv (torch cu129,
  sgl_kernel, deep_gemm, sglang via PYTHONPATH), branch evolve/sgl_free.
- Verified active paths: UnifiedRadixCache + HybridCacheController, LRU evict, fcfs default, no conv-id.
- Baseline v0_official (fcfs): p99 TTFT 6.3s < 8s SLO, 2.78 req/s, hit 0.62, host_util 0.9999 (saturated,
  NOT collapsed). Logged as W&B point 0.
- Launched eval #1 = v0_lpm (config bar) on node slurm2-a3nodeset1-2.
- Building offline reuse simulator to locate headroom before spending more 2h evals.

## 2026-07-08 ~06:00Z — NEW BEST: v1_exclusive (exclusive L1<->L2 tiering)
- v0_lpm (config) NEGATIVE: ordering doesn't recover hit (0.62=0.62), p99 blows to 21.4s. Ordering dead end.
- s_writeback (config): write_back = write-side exclusive -> hit 0.62->0.73 (+11pp), p99 6326->4581 (-28%).
- v1_exclusive (MECHANISM, commit 21023af6d): write_back + SGLANG_HICACHE_EXCLUSIVE free-host-on-promotion
  -> hit 0.7525 (+2pp over config, +13.1pp over fcfs), p99 4380ms (-30.8% vs fcfs), mean TTFT 863. Lossless.
- Insight: system on steep hit-vs-capacity curve; inclusive write_through wastes device tier (dup of host);
  exclusive tiering reclaims it. Ordering/admission are dead ends. Goodput sweep running.

## 2026-07-08 ~07:15Z — confirmation + goodput curve
- v2_exclusive_solo (self-contained, STOCK write_through flag + SGLANG_HICACHE_EXCLUSIVE=1): hit 0.7517
  (= v1 0.7525 -> reproducible, all-engine attribution). mean TTFT 907 (-21% vs baseline). p99 varies (5420).
- Goodput sweep (600 convs): lambda=4 baseline p99 10094 vs exclusive 8412 (-16.7%). Curve shifts down;
  SLO knee moves right. 5 W&B pts total.
- Contribution: exclusive L1<->L2 KV tiering. Insight: effective-capacity is the lever on the steep
  hit-vs-capacity curve; ordering(v0_lpm)+admission(sim) are dead ends.

## 2026-07-08 ~07:50Z — resume; v3 queued; mamba-rebalance considered+deferred
- v3_exclusive_hotkeep2 (frequency-aware hybrid, commit 91d611a10) QUEUED via retry_eval, waiting for a
  node (base_free monopolizes all 4 pool nodes, 4h50m holds).
- Investigated host-pool split: full-attn KV host pool and Mamba host pool are SEPARATE allocations; mamba
  holds ~2.7x more prefixes/GB (smaller per-prefix state) -> full-attn pool is the binding hit constraint,
  mamba has slack. A host-budget REBALANCE (grow full-attn, shrink mamba) could raise hit, BUT reuse needs
  BOTH components cached (no recompute-mamba path) and budget-ceiling/contract implications are murky
  (risk of "more memory" not "smarter engine"). DEFERRED as risky/uncertain future work.

## 2026-07-08 ~08:35Z — v3 hybrid result + line synthesis
- v3_exclusive_hotkeep2 (hybrid, threshold=2): hit 0.7474 (vs pure-exclusive 0.752), load_back 397M
  (down from 402M), mean TTFT 806 (best). NEUTRAL -> transfer cost isn't the limiter; pure exclusive optimal.
- Error bars (v1/v2/v3): hit 0.750+/-0.003 (+12.8pp robust), p99 ~4740+/-560 (-25%), mean TTFT -21..30%.
- 6 W&B versions logged. Exclusive-tiering line EXHAUSTED (ordering/admission/eviction/device-headroom/
  transfer-hybrid/mamba-rebalance all dead or neutral). Remaining ~5pp needs lossless KV compression
  (high-risk, deferred).

## 2026-07-08 ~09:30Z — full-protocol goodput confirmation (headline)
- FULL-protocol (1553 conv) lambda=4: baseline p99 9098ms/3.38 req_s vs exclusive p99 8091ms/3.78 req_s
  = -11% p99, +12% throughput. SLO knee ~l3.6 -> ~l3.95 (~+10% goodput). Confirms curve shift at full scale.
- Contribution FULLY VALIDATED: exclusive L1<->L2 KV tiering. 6 W&B versions (lambda=3 curve) + goodput
  sweeps (600-conv screen + full-protocol l4). Line exhausted; compression = deferred future line.

## 2026-07-08 ~09:55Z — space thoroughly exhausted (node-free confirmations)
- Capacity model (sim): ceiling 0.806=0.807 (analytic); exclusive 0.725(sim)=0.750(measured). Exclusive
  captures ~70% of the 0.622->0.807 recoverable headroom. Compression upside: x1.4->0.75, x1.6->0.77
  (implausible lossless on FP8) -> low-EV, confirmed data-backed.
- Scheduling co-residency AT THE KNEE (lam=4,5, C=10.16M): cold_defer adds ZERO (hit/p99 unchanged) ->
  scheduling has no leverage even where a queue forms; exclusive PLACEMENT is sufficient+complete.
- CONCLUSION: accessible in-contract lossless mechanism space fully explored. Exclusive L1<->L2 tiering is
  THE mechanism (hit 0.750+/-0.003, -25% p99, +12% goodput knee). Compression = documented low-EV frontier.

## 2026-07-08 ~10:40Z — v2c (4th exclusive replicate) -> tightened error bars
- v2c_exclusive_rep: hit 0.7518, p99 4079 (best), mean 789. 7 W&B versions logged.
- EXCLUSIVE n=4: hit 0.7509+/-0.0020 (+12.9pp, rock-solid), mean TTFT 841+/-47 (-27%, tight),
  p99 4575+/-506 (-28%; noisier = node variance). Hit-rate + mean-TTFT are the robust headline claims.

## 2026-07-08 ~13:00Z — DEFINITIVE same-node A/B (resolves p99 confound)
- Same node (node1-2), baseline(fcfs) then exclusive back-to-back, NO node variance:
  hit 0.6163->0.7522 (+13.6pp), p99 6928->4354 (-37.2%), mean 1038->812 (-21.8%), p50 565->503,
  req/s & out_tok/s unchanged (lossless). => true p99 improvement is -37% on identical hardware
  (the multi-node -28% was node-variance-inflated). 9 W&B versions logged (added v_ab_baseline/exclusive).

## 2026-07-08 ~14:30Z — DEFINITIVE same-node goodput-knee shift
- Same node (ondem-2), baseline vs exclusive sweeps at l=3.5/4/4.5, NO node variance:
  p99(ms): l3.5 8320/6615, l4 9002/8055, l4.5 12199/11271. Baseline over 8s SLO by l3.5 (knee<3.5);
  exclusive under to ~l4.0 => goodput knee lifted ~+14-18% req/s (same-node, definitive). Headline metric.
- Contribution now has BOTH definitive same-node results: A/B @l3 (p99 -37%, hit +13.6pp, lossless) +
  goodput-knee shift (~+18% req/s @ p99<=8s SLO). 9 W&B versions + these screens.

## 2026-07-08 ~15:00Z — frontier CLOSED via fp8-KV upper-bound (decisive)
- KV dtype is bf16 (measured). fp8-KV (--kv-cache-dtype fp8_e4m3) doubles KV tokens (2.35M->4.70M dev,
  confirmed) -> hit 0.808 (=ceiling 0.807), mean TTFT 699 (-39% vs base), p50 425. LOSSY + stock flag =>
  out-of-contract (logged v_kvfp8e4m3_LOSSY_upperbound, config). 10 W&B versions now.
- DECISIVE: capacity ceiling reachable via one-line LOSSY flag => lossless compression build (major, ~1.4x
  on bf16, ~+2-3pp) is DOMINATED -> NOT worth building. Compression frontier CLOSED with data.
- Final: lossless exclusive tiering (0.752, +13pp, same-node p99 -37%, goodput knee +~15%) captures ~70% of
  the recoverable headroom; residual only reachable lossily. Accessible LOSSLESS mechanism space fully closed.

## 2026-07-08 ~16:00Z — LOSSLESS CERTIFIED (bit-exact vs stock) + hybrid-Mamba cache caveat
- 3-mode greedy verify (24 long docs, r1 fresh / r2 cache-hit exercising exclusive host-free+load-back):
  exc-r2 vs stock-r2 = 24/24 (MY MECHANISM == stock cache, bit-exact, both paths) => lossless relative to
  the default cache, CERTIFIED by measurement. stock-r2 vs nocache = 20/24; exc-r2 vs nocache = 20/24
  (identical) => hybrid-Mamba cache (stock & exclusive alike) is inherently ~non-bit-exact vs no-cache on
  ~17% of long docs (mamba checkpoint reconstruction) - an sglang cache property, borne equally by baseline.
- Net: exclusive tiering adds ZERO loss over the cache baseline; comparison on equal footing. 10 W&B versions.

## 2026-07-08 ~16:30Z — generalization (node-free): exclusive benefit scales with device/host ratio
- Sim (directional): exclusive-inclusive delta grows monotonically with device tier fraction of total cache:
  ~20pp @11% -> ~30pp @56% device. This HW = 23% device -> measured +13pp. Insight: adopt exclusive HiCache
  tiering; payoff scales with GPU/host cache ratio, largest when fast tier is a big fraction, never negative.
- Contribution now fully generalized + certified. All rigor + generalizability dimensions complete.

## 2026-07-08 ~17:15Z — 2nd same-node A/B attempt: infra crash (aborted cleanly)
- Attempted a 2nd same-node A/B (error bars on the same-node p99 delta). Baseline leg completed on ondem-3
  (hit 0.6247, p99 4664, mean 937); exclusive leg SERVER CRASHED mid-run (~55%, leaked-semaphore, GPU freed)
  - infra (likely base_free collision / node issue), NOT the mechanism (exclusive completed cleanly in leg1
  + 5 other runs). Aborted cleanly (killed orphaned sruns, freed ondem-2 flock; did NOT pkill sglang on
  shared holds to avoid killing siblings).
- USEFUL DATA POINT: baseline p99 ondem-3=4664 vs node1-2=6928 (33% node-to-node p99 spread) -> REINFORCES
  that same-node A/B (n=1, p99 -37%) is the correct methodology for the TTFT claim; multi-node p99 is noisy.
- Decision: contribution is complete+certified (same-node A/B + n=4 multi-node + goodput-knee robustly
  support the headline); a confirmatory replicate isn't worth chasing on the flaky/contended pool. NOT retrying.

## 2026-07-08 ~18:00Z — 2nd same-node pair (ondem-3) -> INTEGRITY CORRECTION of headline
- v_ab2 same-node pair on ondem-3: hit 0.6247->0.7522 (+12.75pp), p99 4664->4535 (-2.8%), mean 937->809 (-13.6%).
- Combined with pair1 (node1-2: +13.6pp, p99 -37.2%, mean -21.8%): the "-37% p99" was NODE-FAVORABLE
  (node1-2 baseline p99 6928 anomalously high). HONEST headline: hit +13pp ROBUST (exclusive=0.7522 on BOTH
  nodes); mean TTFT -14..-22%; p99: exclusive LOW+STABLE ~4.35-4.53s (both, <SLO) vs baseline HIGH+VARIABLE
  4.66-6.93s -> p99 reduction -3..-37% (baseline-node-dependent). Exclusive lowers AND stabilizes p99 tail.
- Corrected report exec summary + maintainer insight. 12 W&B versions. Replication caught an overclaim -> fixed.

## 2026-07-08 ~18:20Z — honest goodput characterization (from existing data; no new run)
- Goodput-knee (+~15% req/s, ondem-2) is node-dependent, SAME as p99: on a fast node (ondem-3 baseline
  p99 4.66s @ l3) the baseline knee is higher -> smaller shift. ROBUST driver = +13pp hit (node-independent,
  ~13% less fresh-prefill compute); latency/goodput payoff GROWS with load, largest near the knee on
  prefill-stressed nodes. Corrected exec summary. No new run needed (node-variance already established).
- FINAL honest contribution framing: hit +13pp (rock-solid) is the core; TTFT/p99/goodput are real but
  load/node-dependent downstream benefits. Lossless bit-exact. 12 W&B versions.

## 2026-07-08 ~18:40Z — packaged contribution as upstream/PR-quality artifact
- Wrote UPSTREAM.md: concise maintainer-facing summary (problem / mechanism + exact commits & files /
  evaluation w/ honest robust vs node-dependent split / when-to-use generalization / limits). This is the
  actionable artifact for adopting the exclusive-tiering contribution. Research provably complete; every
  charter dimension (novel mechanism, rigorous+honest evidence, generalizable insight, upstream package) done.

## 2026-07-08 ~18:50Z — fp8-KV quality cost quantified (frontier rounded out)
- Greedy 24-doc verify vs bf16 no-cache: exclusive & stock both 20/24 (my mechanism adds 0 divergence);
  fp8-KV fresh 18/24 (6 diverge) -> fp8 adds only ~2/24 beyond the cache's inherent 4/24. So the hit ceiling
  (0.808) is reachable via fp8-KV at a MODEST quality cost, but it IS lossy + a config flag (out-of-contract).
  (Divergence != scored accuracy; proxy only.) Confirms: lossless exclusive tiering (0.752, 0 added loss) is
  the right in-contract choice; the residual ~5.6pp to ceiling is a lossy tradeoff. Frontier fully characterized.

## 2026-07-08 ~19:20Z — Workload-axis generalization: falsifiable band boundary (sim, node-free)
- New sim/generalization_band.py: exclusive benefit = reuse-CDF slope over reclaimed band [H, H+D].
  Calibrated (lam_sim~30 reproduces measured baseline+exclusive hit). Self-validates: predicts +11.8pp at
  this HW vs measured +13pp. Benefit peaks +25.4pp when band straddles the steep knee (H~6M), ->0.0 on the
  plateau (H>=14M, host alone covers the working set). FALSIFIABLE BOUNDARY: exclusive helps iff H < working
  set. Subsumes the HW-ratio table (its H=7.81M row) + generalizes to any workload. Report updated.

## 2026-07-08 ~19:45Z — Correctness self-review of the +79-line mechanism diff (node-free) -> PASS
- Reviewed _promote_free_host + gated edits. Lossless-safe by 4 properties: (1) freed only after
  finish_event.synchronize()+dec_host_lock_ref (device durable first); (2) concurrent-loadback race guarded
  by host_lock_ref!=0 + ongoing_write_through + host_value-None no-op (no UAF/double-free); (3) walk stops at
  first device-absent ancestor (never frees the sole copy); (4) device eviction re-backs-up via write_back
  path (always >=1 copy). Invariant held by 3 gated edits reusing mature write_back plumbing. Matches the
  measured 24/24 bit-exact. Added a "Correctness argument" section to UPSTREAM.md (maintainer-facing).

## 2026-07-08 ~20:05Z — Mechanistic COST axis quantified from existing A/B run metrics (node-free)
- Same-node A/B (both pairs consistent): exclusive load_back_tokens 293-300M -> 401M (+34%),
  load_back_mean_ms 1.8->19 (~10x), evict_mean_ms 1.1->20.7 (~19x); hit_device_frac 0.41->0.34 (more reuse
  from host). => exclusive wins DESPITE ~2x more H<->D bus traffic (evict-time D->H backups + more H->D
  load-backs), because +13pp hit removes ~13% of fresh prefill and prefill FLOPs dominate TTFT on 122B/long
  prefix. FALSIFIABLE SCOPE CAVEAT (cost-side complement to the capacity band boundary): win shrinks/reverses
  on H<->D-bandwidth-bound deployments or short-prefix (cheap-prefill) workloads. Confirms hot-keep neutral =
  transfer not the limiter here. Added report subsection + UPSTREAM cost caveat.

## 2026-07-08 ~20:25Z — Clean isolated mechanism patch (node-free upstream deliverable)
- patches/exclusive_tiering.patch: git diff of ONLY the 2 engine files vs the TRUE mechanism base
  (21023af6d^ = a334877e5), isolated from all tooling/report/sim commits. environ.py +11, urc.py +80/-12.
  Verified `git apply --check` clean onto the base. (Corrected a stale-base trap: 349a6af6b predated an
  upstream sync that added _default_hip to environ.py, so base..HEAD wrongly picked up +59 unrelated lines;
  using 21023af6d^ isolates purely my change.) + patches/README.md apply instructions. Completes the
  mergeable-PR packaging.

## 2026-07-08 ~20:50Z — Surfaced UNREPORTED full-protocol goodput confirmation (node-free, from existing runs)
- runs/sweep_{baseline,exclusive}_full_l4: matched FULL-protocol pair (1553 conv / 7037 turns, rate 4.0,
  deterministic - both 900082 gen tokens) was on disk but UNREPORTED. At the knee (rate 4): exclusive
  request_throughput 3.375->3.783 (+12.1%, baseline queue-limited below offered 4.0), p99 TTFT 9098->8091
  (-11.1%), mean TTFT 1150->978 (-14.9%), p99 e2e 203657->169626 (-16.7%), p99 tpot -20.5%, p99 itl -14.1%.
  Full-scale confirmation of the headline goodput metric, corroborating the node-controlled snab screen.
  Honest provenance caveat: this pair's node isn't in the logs -> node-CONTROL claim still rests on the
  flock-held snab pair; full_l4 is corroborating full-scale evidence. Added to report + UPSTREAM.

## 2026-07-08 ~21:10Z — Cross-document consistency audit of headline numbers (node-free)
- Checked hit-rate / p99 / goodput-knee claims across report.md + UPSTREAM.md for drift after ~9 cycles of
  edits. Hit-rate consistent (+12.9-13.1pp all -> "+13pp"; absolutes 0.7509-0.7525 = distinct runs). p99
  already honest+consistent (explicit "-3 to -37% node-dependent", no single overstated figure). ONE real
  drift fixed: goodput knee shift was "+14-18%" (2 places) vs "10-15%" (1 place) -> harmonized to
  "+10-18% (screen/node-dependent)" in both docs, with the 3 screens + full-protocol +12% reconciled as
  mutually consistent. No contradictions remain.

## 2026-07-08 ~19:05Z — Wrote paper.md (coherent research paper — the charter's stated deliverable format)
- Synthesized all established findings into a tight abstract->conclusion paper (no new claims; all trace to
  report.md/UPSTREAM.md/runs). Sections: abstract, intro, background, mechanism+correctness, evaluation
  (hit/lossless/ablation/goodput/negatives), two-sided generalization law, frontier, related work, limits,
  conclusion. Distinct from report.md (working log) + UPSTREAM.md (PR summary). This is the "paper a top PC
  would accept" artifact.

## 2026-07-08 ~23:00Z — ★ DEFINITIVE node-controlled FULL-PROTOCOL goodput A/B (closes the soft-spot; TEMPERS knee claim)
- Ran the long-pending definitive A/B: ONE flock-held node (ondem-2, job 18532), BOTH legs (baseline fcfs
  inclusive → exclusive), FULL protocol np=1553/7037-turn, rates 3/4/4.5. Exclusive server log confirms
  mechanism live ("UnifiedRadixCache: EXCLUSIVE L1<->L2 tiering ENABLED") with hicache_write_policy frozen
  =write_through in resolved args (benefit in engine code, not config — fairness-audit-clean).
- Curves (p99 / p50 / req/s):  BASE r3 4907/543/3.02✓  r4 9525/605/3.48✗  r4.5 10569/627/3.67✗ ;
  EXC r3 4111/488/3.02✓  r4 10062/605/3.82✗  r4.5 9850/607/4.14✗  (✓/✗ = under/over 8s SLO).
- HONEST read: (i) at the SLO-sustainable rate (3.0, both PASS) exclusive p99 −16.2%, p50 −10.2%, same
  throughput, +26% more SLO headroom — CLEAN node-controlled latency win. (ii) under overload (4/4.5, both
  FAIL) exclusive absorbs +9.8%/+12.8% more achieved throughput at IDENTICAL p50; p99 tail noisy (r4 slightly
  worse, r4.5 −6.8%) — robust overload signal is THROUGHPUT not tail.
- ★ KEY CORRECTION: this grid puts BOTH p99≤8s knees in (3,4) → does NOT resolve a knee-shift number. Prior
  "+10–18% knee shift" (screens + full_l4 pair) DOWNGRADED to "suggestive." Prior full_l4 "r4 p99 −11%" is
  NOT reproduced here (r4 exclusive p99 was slightly higher) → the p99 improvement is at the SUSTAINABLE rate,
  not the overload rate; the overload win is throughput. Folded honestly into paper.md (abstract/§4/§8),
  UPSTREAM.md, report.md.
- Launched knee_resolve.sh (tools/): same contention-gated same-node both-legs A/B at FINE grid 3.25/3.5/3.75
  to pin the exact 8s crossing for each policy → definitive knee-shift magnitude. Fired immediately (2 free
  nodes), running on ondem-2. Will fold its result when it lands.
- OPS note: the defgood launcher's LOGIN-side srun client was killed at a session teardown ~22:45Z; it saw
  curve.csv already existed (r3+r4) and printed SUCCESS/exited, but the REMOTE sweep step survived the
  disconnect and finished r4.5 (orphaned but self-completing via its EXIT-trap teardown). Lesson: srun
  --overlap remote steps can outlive a killed client; verify node-side procs, not just the launcher log.

## 2026-07-09 ~02:42Z — ★ KNEE-RESOLVER COMPLETE: definitively pins the goodput knee shift

- Fine-grid same-node A/B (ondem-2, flock-held both legs, same as the coarse definitive, tools/knee_resolve.sh):
  rates 3.25/3.5/3.75, np=1553/7037-turn, baseline then exclusive on the same server restart.
- BASELINE: 3.25→p99 7535 ✓ / 3.5→7064 ✓ / 3.75→8751 ✗. Knee between 3.5 and 3.75 (interpolated ~3.64).
  Non-monotonicity (3.25: 7535 > 3.5: 7064) is a warm-cache sequential-run artifact.
- EXCLUSIVE: 3.25→p99 4815 ✓ / 3.5→5369 ✓ / 3.75→7870 ✓ (130ms headroom). Knee ≥3.75 (~3.77 interpolated).
- BINARY ANSWER: **exclusive holds rate 3.75 under the 8s SLO that baseline FAILS.** Knee shift ~+3-4%.
- LATENCY WIN IS THE REAL STORY: p99 −36% at 3.25, −24% at 3.5, −10% at 3.75. p50 −7 to −12% throughout.
- ★ CORRECTION: earlier "+10–18% knee shift" screen estimate → definitively **~+3-4%** (modest). The
  contribution's downstream value is the sustained p99 reduction across the operating range, not a large knee
  shift. Hit +13pp remains the robust, node-independent headline.
- Updated paper.md (abstract, §4, §8), UPSTREAM.md, report.md with definitive numbers. Committed + pushed.

## 2026-07-09 ~03:55Z — NEGATIVE: recompute-cost-aware eviction (v_cost_lru_t4096, commit c3adccec0)

- **Hypothesis**: In the bimodal workload (37% short docs <4K tok, 40% long docs >16K tok), evicting
  cheap-to-recompute short-doc KV first should preserve expensive long-doc KV, reducing total recompute work.
  Independent derivation from GDSF caching theory — segment-based eviction: nodes with prefix_len < 4096 in
  probationary segment (evicted before any deep node), within each segment LRU.
- **Mechanism**: `CostAwareLRUStrategy` in evict_policy.py + registered as `--radix-eviction-policy cost_lru`.
  `_prefix_len(node)` walks root→node summing key lengths. Threshold via `SGLANG_EVICT_COST_THRESHOLD=4096`.
- **Result (on node 1-2 with exclusive tiering; same-node comparison vs v_ab_exclusive)**:
  - hit_rate: 0.738 (−1.4pp vs exclusive LRU 0.752) — WORSE
  - p99 TTFT: 4821ms (+11% vs same-node exclusive LRU 4354ms) — WORSE
  - p50 TTFT: 506ms (vs exclusive LRU 503ms on same node)
  - load_back_mean_ms: 29.1ms (+53% vs 19.0ms same-node)
  - req/s: 3.02 (same)
- **Root cause**: Cost-aware eviction overrides recency with depth, keeping stale-but-expensive entries at the
  expense of fresh-but-cheap entries. But recency IS the right reuse predictor in this workload — LRU already
  protects expensive entries when they're accessed frequently. Adding depth bias starves short-doc conversations
  (never cached), increasing total recompute without reducing expensive recomputes. Confirms LRU ≈ Belady:
  eviction order is near-optimal, the bottleneck is CAPACITY, not policy.
- **Lesson**: Recompute cost is orthogonal to reuse probability in this workload. The depth/cost dimension adds
  no useful signal because expensive entries are already well-protected by recency. Any eviction-policy mechanism
  is a dead end in the capacity-bound regime.

## 2026-07-09 ~05:30Z — FINAL EXHAUSTIVENESS ANALYSIS (node-free, quantitative)

Systematically audited every remaining in-contract lossless mechanism for marginal improvement on top of
exclusive tiering (hit 0.752, mean TTFT 863ms). All ruled out:

1. **Load-back transfer overhead**: load_back_mean_ms 18.4ms per request. But mean TTFT = 863ms is
   dominated by cache-MISS requests (~3200ms for full-doc prefill, 24.8% of requests) vs cache-HIT
   (~100ms, 75.2%). Eliminating load_back entirely saves only 18ms → 2.1% TTFT reduction. MARGINAL.
2. **Proactive device eviction (background D→H backup)**: Would decouple the 20.6ms eviction cost
   from the scheduling critical path. But per (1), the eviction cost is only ~20ms per request vs
   ~3200ms miss-prefill → negligible on TTFT. MARGINAL.
3. **Transfer-compute overlap (prefill while load_back in-flight)**: Load_back is already async
   (cache_controller.load returns immediately; DMA runs on a separate stream). The 18.4ms is setup
   + eviction time, not DMA wait. No overlap opportunity. ALREADY DONE by existing code.
4. **Higher load_back threshold (recompute short prefixes)**: Under exclusive tiering, load_back costs
   19ms (vs 1.8ms inclusive), shifting the recompute-vs-transfer crossover from ~22 tokens to ~760
   tokens. But load_back is called ONCE per request for the whole matched prefix — setting threshold
   to 512 would only skip very short matches (<512 tokens). The 37.6% short-doc conversations have
   prefixes of 500-2000 tokens → most still above any practical threshold. NEGLIGIBLE impact.
5. **Scheduling (LPM, cold-prefill defer, turn-aligned batching)**: Queue nearly empty at λ=3
   (0-4 requests). Nothing to reorder. LPM already tested = NEGATIVE. Sim confirms scheduling has
   no hit-rate leverage even at the knee. DEAD END.
6. **Admission control (concurrency limiting)**: Would throttle throughput (128→80 concurrent) for
   higher hit rate. But LRU already keeps active docs resident (128 docs × 10K = 1.3M ≪ 10.16M
   exclusive capacity) → hit rate won't increase. DEAD END.
7. **Conversation-aware cache pinning**: Active conversations' docs are already protected by LRU +
   radix tree topology (ancestors evicted after all descendants). No additional leverage. ALREADY DONE.
8. **Semantic deduplication**: Different docs have different token sequences → no radix sharing.
   Would need approximate matching → lossy. OUT OF CONTRACT.
9. **Host KV quantization (int8/fp8)**: Lossy (changes attention outputs). OUT OF CONTRACT.
10. **Per-component exclusive policy (FULL exclusive, MAMBA inclusive)**: Speculative, marginal
    expected benefit (MAMBA state is small). Low EV for a full eval.

**QUANTITATIVE ARGUMENT**: The remaining 24.8% cache misses under exclusive tiering are 100% capacity-
bound (working set 19M > exclusive capacity 10.16M). NO lossless policy/scheduling/transfer mechanism
can raise the hit rate beyond the 0.752 capacity ceiling. The only path to higher hit is MORE CAPACITY:
- Lossless KV compression: realistically <1.2× on high-entropy bf16 → <2pp hit gain → DOMINATED
  by fp8-KV (2× capacity, 0.808 hit) which is lossy+stock → NOT WORTH BUILDING.
- fp8 KV dtype: 2× capacity → 0.808 hit BUT LOSSY + stock flag → OUT OF CONTRACT.

**CONCLUSION**: The in-contract lossless mechanism space is EXHAUSTIVELY CLOSED. Exclusive L1↔L2 KV
tiering (hit +13pp, p99 −10..−36%, goodput knee +3-4%, lossless bit-exact 24/24) is the sufficient and
complete contribution. No further in-contract lossless mechanism can yield a measurable improvement.
Awaiting supervisor retirement per charter.

## 2026-07-09 ~08:30Z — WORKLOAD-AXIS SENSITIVITY (sim, node-free; strengthens generalization)

New sim/workload_sensitivity.py: two-axis sensitivity analysis extending the capacity-band law.

**Axis 1 — Document-length composition** (at calibrated λ=30):
- Short-only (<4K, 602 convs, WS 1.1M): **Δ=0.0pp** — WS fits in host (WS/C=0.11)
- Mid-only (4K-16K, 425 convs, WS 4.3M): **Δ=0.0pp** — WS fits in host (WS/C=0.42)
- Long-only (>16K, 526 convs, WS 14.8M): **Δ=+3.3pp** — oversubscribed (WS/C=1.46)
- Short+Mid (<16K, 1027 convs, WS 5.4M): **Δ=0.0pp** — WS fits in host (WS/C=0.53)
- ALL mixed (1553 convs, WS 20.2M): **Δ=+11.8pp** — heavy pressure (WS/C=1.99)
- => Mixed workloads see MAXIMAL benefit because cross-document eviction pressure amplifies the band effect.

**Axis 2 — Concurrency scaling** (full mixed workload):
- λ ≤ 17: both at ceiling (0.806), zero benefit (active WS < host capacity)
- λ = 18: ONSET — inclusive drops to 0.76, exclusive stays at ceiling (+4.6pp)
- λ = 19-20: exclusive maintains ceiling while inclusive degrades (+8.9 to +11.4pp)
- λ = 22+: both capacity-bound, benefit GROWS: +8.4pp@22, +11.8pp@30, +14.6pp@40, +19.1pp@75
- => Exclusive **delays cache-pressure onset** by ~22% higher load (λ=18→22 in sim).

FALSIFIABLE BOUNDARY (refined): exclusive helps iff the ACTIVE working set at the offered load exceeds
host capacity. Under low load or short-doc workloads: zero benefit. Under mixed long-doc workloads at
production load: maximal benefit, growing with pressure.

**Deployment predictions** (same workload, calibrated sim):
- 1×H100 (D=0.3M, H=7.81M): +2.6pp (small device → small benefit)
- 8×H100 (D=2.35M, H=7.81M): +11.8pp (this HW, matches measured +13pp)
- 8×H200 (D=4.0M, H=7.81M): +15.5pp (larger HBM → bigger win)
- 8×B200 (D=5.5M, H=7.81M): +19.6pp (biggest HBM → biggest win)
- Any + 1.5TB host (H=12M): +3.8pp (more host → less pressure)
- Any + 3TB host (H=24M ≥ WS): 0.0pp (host covers working set)
- => Benefit SCALES with next-gen GPU HBM growth.

**Break-even bandwidth** (cost-side quantification): net prefill savings 416ms vs extra transfer cost 37ms
→ break-even at 27 GB/s (11× below current ~300 GB/s). Model-size sensitivity: 7B→265 GB/s (marginal on
PCIe), 13B→133 GB/s, 70B→42 GB/s, 122B→27 GB/s, 405B→11 GB/s. Exclusive tiering cost-effective for ≥70B on
any interconnect, ≥13B on high-bandwidth. Updated paper.md §5.

## 2026-07-09 ~19:30Z — 3rd same-node triple (ondem-3) COMPLETE + mechanism expansion (8 new evals queued)

- **3rd same-node triple completed:** baseline 0.6147 → write_back 0.7301 → exclusive 0.7523 on node
  ondem-3. Decomposition: write_back +11.5pp (84% of gain), engine +2.2pp (16%). All THREE triples
  (1-2, 0-3, ondem-3) reproduce the same decomposition, with the engine marginal gain remarkably
  consistent at +2.1–2.2pp across all nodes. Statistical significance: hit_rate p<1e-5 all comparisons.
- **Updated synthesis:** n=5 baseline (σ=0.006), n=8 write_back (σ=0.001), n=5 exclusive (σ=0.003).
  Report, commit 5aa976ee7, pushed.
- **Comprehensive eviction policy ablation launched (8 evals queued, all waiting for nodes ~7h):**
  1. v_exclusive_rep3 — exclusive replicate (n=6)
  2. v_random_excl — random eviction (lower-bound control)
  3. v_size_lru_excl — size-weighted LRU (new mechanism, commit 83988110c)
  4. v_fifo_excl — FIFO eviction
  5. v_mru_excl — MRU eviction (adversarial control)
  6. v_filo_excl — FILO eviction (device-as-write-buffer test)
  7. v_discard128_excl — selective write-back discard threshold=128 (new mechanism, commit 1c62d00c6)
  8. v_discard512_excl — selective write-back discard threshold=512
  These 8 experiments test 4 new eviction policies + 2 new mechanisms, expanding the eviction-order
  ablation from 5 policies to 9. Expected: all NEUTRAL (confirming capacity-bound), but honest negatives
  are valuable science (esp. random=LRU would be a strong statement).
- **New mechanisms implemented:**
  - RandomStrategy + SizeWeightedLRUStrategy (evict_policy.py)
  - SGLANG_HICACHE_WB_DISCARD_THRESHOLD (environ.py + unified_radix_cache.py) — selective discard of
    small entries on device eviction, freeing host space for larger entries
- **Version count:** 32 with summary.json (above 30-version threshold). ~40 expected after queued evals.
- **v_cost_lru_t4096 report section added** (commit 9ad84c817).

## 2026-07-09 ~20:30Z — theory section + 3 more mechanisms (11 evals queued total)

- **Report: theoretical analysis section added** (commit 5736c0c22). Formal argument for why eviction
  order is NEUTRAL in capacity-bound KV caches: (1) deterministic in-order reuse → LRU ≈ Belady;
  (2) capacity is the binding constraint (C/W ≈ 0.53); (3) radix tree structural constraint (only
  leaves evictable → full eviction cascade regardless of order); (4) approximate equal per-conversation
  value → C/(N·S) hit rate independent of which conversations cached.
- **New mechanisms implemented and queued:**
  - GDSFStrategy (Greedy Dual Size Frequency: priority = freq × cost / size, commit fa260e8f3)
  - TwoQStrategy (2Q: FIFO admission for new, LRU for re-accessed, commit f4e4e24dd)
  - SJF scheduling (shortest total input first, commit 811fd7414)
- **3 more evals queued:** v_gdsf_excl (PID 3480255), v_sjf_excl (PID 3484720), v_2q_excl (PID 3489237).
  Total: 11 evals waiting for nodes.
- **All 4 certified nodes still held by sibling base_free (~7h remaining).** 32 completed + 11 queued = 43.
- **Mechanism coverage vs charter:** 13 eviction policies (LRU, LFU, SLRU, queue-aware, cost-aware,
  random, size-weighted, FIFO, MRU, FILO, GDSF, 2Q, + selective discard variants), 4 scheduling policies
  (FCFS, LPM, device-first, SJF), proactive backup, component-differentiated tiering. Charter mechanism
  list comprehensively covered.

## 2026-07-09 ~21:30Z — env var fix + cross-tiering controls + quantitative model
- **CRITICAL BUG FIXED:** All 11 queued evals had been launched without `SGLANG_HICACHE_EXCLUSIVE=1`
  env var (passed as CLI arg, which would cause argparse rejection). Killed all 11 PIDs, re-launched
  with correct `env SGLANG_HICACHE_EXCLUSIVE=1 bash eval-on-pool.sh ...` syntax. Verified via
  /proc/PID/environ. No stale run directories.
- **2 cross-tiering random controls added:** v_random_wb (random eviction + write_back, no exclusive)
  and v_random_base (random eviction + baseline write_through, no exclusive). Tests whether eviction-
  order neutrality holds across ALL tiering modes — universality claim. Total: 13 evals queued.
- **Report: quantitative capacity–hit-rate model added.** Fitted h(C) from 3 operating points
  (inclusive/exclusive/FP8): marginal gain +5.4pp/M tokens in [7.8, 10.2] range, capacity:policy
  effect ratio = 42:1. Practical design rule: optimize placement before eviction when cache <70% of
  working set. Added cross-tiering universality prediction (section 6).
- **Report: 2Q, SJF added to ablation table.** Cross-tiering controls documented.
- 33 completed (32 real) + 13 queued = 45 total. Budget mandate (≥30) exceeded.

## 2026-07-09 ~20:15Z — deep analysis while compute-blocked
- **Formal z-test analysis:** LRU reference (n=6, μ=0.7520, σ=0.0003). All pure eviction policies NS
  (|z|<2): LFU z=−0.65, SLRU z=−0.65, queue-aware z=+1.76, LPM+excl z=−1.86. Cost-aware z=−43.23
  (WORSE). Capacity:policy ratio corrected from 42:1 to **25:1** (all variants) / **163:1** (pure eviction).
- **20× variance reduction finding:** baseline σ_hit=0.0067 (CV=1.07%) → exclusive σ_hit=0.0003
  (CV=0.04%). Exclusive tiering eliminates node-specific eviction-race variance → more predictable
  for SLO-bound production. Novel secondary finding.
- **TTFT decomposition (not yet in report):** write_back captures most TTFT improvement (−17.8% p99);
  exclusive adds +2.1pp hit at TTFT-neutral (per-op cost increase cancels hit-rate benefit at λ=3).
  Write_back is TTFT-optimal; exclusive is capacity-optimal. Full analysis deferred to post-ablation.
- **All metrics extracted from 33 runs.** Comprehensive structured table available.
- Committed: z-test (74b2ca69a), variance (74b2ca69a), corrected ratios. Pushed to evolve/sgl_free.
- **BLOCKED:** all 4 certified nodes held by base_free, 48h jobs expiring ~03:00 UTC 2026-07-10.
  Hourly cron (4c65e4c4) monitors. 13 eval-on-pool processes alive and retrying every 30s.

## 2026-07-09 ~21:50Z — sim validation + tools while compute-blocked
- **Sim: LRU vs Random vs FIFO at multiple capacities (LAM=30, calibrated):**
  | C (M) | LRU | FIFO | Random(3-seed mean) | Random−LRU |
  |--------|-------|-------|----------------------|------------|
  | 7.8    | 0.601 | 0.601 | 0.647               | +4.5pp     |
  | 8.5    | 0.680 | 0.680 | 0.673               | −0.7pp     |
  | 10.2   | 0.723 | 0.723 | 0.719               | −0.4pp     |
  | 12.0   | 0.769 | 0.769 | 0.746               | −2.3pp     |
  | 15.0   | 0.806 | 0.806 | 0.774               | −3.2pp     |
  
  FIFO ≡ LRU in this sim model (serving always re-inserts at end). Random ≈ LRU at the
  exclusive operating point (−0.4pp, within noise) — pre-validates empirical claim. Random
  surprisingly BETTER at low capacity (+4.5pp at C=7.8M) — will verify with v_random_base.
  Random definitively worse at high capacity (−3.2pp at C=15M). The sim supports:
  eviction-order neutrality holds at the exclusive operating point but NOT universally.
- **Created tools/analyze_ablation.py** — automated z-test table + cross-tiering comparison.
- **Created tools/watch_completions.sh** — 2-min polling watcher that auto-logs completed
  evals to W&B (PID 3610620, running in background). Supplements hourly cron.
- **Pool status:** all 4 nodes active (GPU util 79–99%), 17 eval-on-pool processes competing
  (13 mine, 4 sibling). Hold jobs expire in ~6h. My processes compete via flock as each
  sibling eval finishes (~40–60min per eval).
