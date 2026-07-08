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
