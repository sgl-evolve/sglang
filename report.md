# EXECUTIVE SUMMARY — kv-flint-2c (independent sglang HiCache researcher)

**Protocol (fixed):** Qwen3.5-122B-A10B-FP8, TP8, ctx 262144, mem-frac 0.85, hicache 96 (768GB host) + 1.8TB
file L3, Mooncake 1:1:1 mix (1553 convs), lambda=3.5, max-conc 128. Headline: mean TTFT (lower better).
Baselines: v0_official (wait_complete stock) 87615 ms; **v0_tuned 108824 ms (the bar)**.

## HEADLINE (best config): best_effort + SKIP_L3_WRITE + SKIP_L3_PREFETCH + write_back = ~886 ms  (~123x < v0_tuned)
THREE composing wins that all attack "work for a resource you don't need": (A) fully bypass the never-read L3 disk
tier [my two novel engine mechanisms], then (B) stop churning the full host tier [write_back config]. Confirmed
n>=2 same-allocation at each step (cf. RETRACTED v24 below). Curve: best_effort ~35-45x > +skip-write ~2x >
+skip-prefetch ~7% > **+write_back ~21% = ~886 ms**.
 0) **write_back** (--hicache-write-policy) -- once the disk is bypassed, eager write_through offloads EVERY write
    device->host, churning the FULL host tier (evicts useful KV); write_back offloads only on device-eviction ->
    host hit_rate 0.63->0.73, throughput 3.46->3.52 (>=lambda -> queue drains), mean 1122->886. TWO same-alloc
    confirms both -21% (884/1122, 888/1130), write_back {884,888} non-overlapping with write_through {1122,1130}.
    REGIME-DEPENDENT: write_back was NEGATIVE without skip (v21 2819) -- only wins once host-churn is the bottleneck.
The two disk-bypass MECHANISMS (both self-guarded to best_effort where hit_storage_frac=0.0, LOSSLESS):
 1) **SKIP_L3_WRITE** -- stop offloading KV host->disk (~252M tokens of write-only waste contending with the
    essential host<->device load_back). ~2x alone: throughput 2.55->3.40 req/s, host hit 0.52->0.62, TTFT halves.
 2) **SKIP_L3_PREFETCH** -- stop even ISSUING the disk prefetch (under best_effort it completes 0 tokens yet allocates
    a host buffer and EVICTS useful host KV to make room, then discards it -> host-tier churn). Adds ~7% (hit
    0.616->0.622, p99 10244->7736).
Both are one-line self-guarded early-returns in UnifiedRadixCache (write_backup_storage / prefetch_from_storage),
env-gated, LOSSLESS. **Full-bypass is n=3 ALLOCATION-INVARIANT: {1094.5, 1094.0, 1096.4} across two allocations
(spread 2.4ms = 0.2%) even though the no-skip baseline swings 1818(fast)<->2530(slow)** -- the mechanisms remove the
disk-contention variance source, leaving a stable GPU-prefill-bound system. Same-allocation decompositions (both
mechanisms are large standalone wins that compose): 18328 no-skip 1818 > skip-write 1176 (-35%) > full 1094 (-40%);
18332 no-skip 2530 > skip-prefetch 1319 (-48%) > full 1096 (-57%). Upstream one-liner: **under a storage tier you
never read, skip BOTH its writes AND its prefetch issue.** (Contrast the RETRACTED v24 below: claims made only after
n>=2 same-allocation confirmation with non-overlapping ranges.) [lpm becomes redundant once skip is applied.]

## The dominant enabling win: prefetch_policy = best_effort  (~35-45x < v0_tuned)
The stock HiCache default `wait_complete` SYNCHRONOUSLY waits for slow L3 (disk) prefetches before admitting a
request. In this saturated regime (throughput < lambda) that wait backs up the waiting queue, so mean TTFT is
dominated by queue time (baseline ~87-109 s). Switching to `--hicache-storage-prefetch-policy best_effort` (skip the
disk read, recompute the missing suffix on GPU, keep the queue drained) collapses mean TTFT to ~2.0-2.8 s
= a ~35-45x reduction. This is a CONFIG change (no code), lossless (recompute reproduces the same KV), and by far
the dominant effect. Everything else is second-order.

## A modest, real config win on top: schedule_policy = lpm  (~8.5%, same-node n=2)
Cache-aware `lpm` ordering helps ONLY because best_effort keeps the waiting queue small (<128): stock lpm reverts to
fcfs once the queue exceeds 128 (schedule_policy.py::_determine_active_policy), which is why it is worthless under the
queue-saturating wait_complete default but useful under best_effort. Same-node (node -0) n=2: be+lpm {2540,2375}
mean 2457 vs be-alone {2586,2782} mean 2684 -> ~8.5%, non-overlapping ranges. Best config = best_effort + lpm.

## CRITICAL METHODOLOGY FINDING: eval variance is large; control for the NODE.
Even on `--exclusive` certified nodes, NODE-TO-NODE variance is ~25-30% (same config best_effort+lpm: ~2010 ms on
ondem-3 vs ~2457 ms on -0) and same-node run-to-run is ~7-8%. So any two SINGLE-RUN results within ~30% on DIFFERENT
nodes are statistically meaningless. I initially reported a "59x new best" (v24, cost-gate=4096, 1837.9 ms) and a
"lpm +38%" -- BOTH were node confounds. Controlled same-node repeats corrected them (v24 retracted in W&B + email;
lpm's true effect is ~8.5%). RULE: never claim a mechanism win from single cross-node runs; require n>=2 same-node
vs a same-node control; only effects >~30% (or same-node effects clearly >~8%) are real.

## Everything else is neutral or negative (honestly logged, kept on the curve)
NEUTRAL (my two novel engine mechanisms -- do NOT help in this regime):
 - Balanced/loading-bound prefill batching (SGLANG_ENABLE_BALANCED_PREFILL): same-node {2746,2789} ~= be-alone.
 - Length/cost-aware prefetch gate (SGLANG_PREFETCH_COST_GATE): same-node {2586,2519} ~= no-gate.
NEUTRAL config: dfs-weight scheduling; write_through_selective / write_back (default write_through is best).
NEGATIVE config: mixed_chunk (16180 ms, fragments prefill); page-size 128/256 (monotonically worse than 64 --
 coarser prefix-match lowers hit rate). timeout prefetch beats best_effort ALONE but loses once lpm is added.

## Why the mechanisms are neutral -- the bottleneck (v35 metrics, be+lpm on -0)
throughput 2.55 req/s (< lambda 3.5 -> still saturated) | hit_rate 0.52 (~48% miss = GPU recompute) | hit tiers
device 59% / host 41% / DISK 0% | host_util 0.998 (FULL) | 252M tokens offloaded to disk that are NEVER read back
(l3_hit=0) | mean TTFT tail-dominated (median 1135, p99 17791). Once best_effort drains the queue the system is
PREFILL-RECOMPUTE-bound and HOST-CAPACITY-bound. Levers that don't cut the 48% recompute (scheduling, batching,
prefetch-gate, layout, write-policy) cannot move the throughput floor -- which is exactly what the data shows.
Cutting the recompute would require better caching (eviction/admission) or cheaper prefill -- the former I hold
off-limits to stay an independent replicate, the latter is frozen by the contract (chunk size) or lossy (mixed_chunk).

## Bottom line for a maintainer
Reproducible frontier (lossless, non-eviction): **best_effort + lpm + skip_L3_write ~1167 ms (~93x < v0_tuned)**.
Three upstream-ready takeaways, in order of impact: (1) for HiCache under queue-saturating load, **best_effort
prefetch beats the wait_complete default by ~35-45x** (don't synchronously block admission on slow-disk reads);
(2) **don't offload KV to a storage tier you never read** -- under best_effort the disk is write-only, and skipping
those writes frees the transfer path for a further ~2x (my SKIP_L3_WRITE mechanism); (3) cache-aware scheduling
(lpm) should not silently revert to fcfs exactly when the queue is largest (~8.5%). NEUTRAL in this regime: my
balanced-batching and cost-gate mechanisms, dfs-weight, write-policy. NEGATIVE: mixed_chunk, larger page-size.
METHODOLOGY: node-to-node variance ~25-30% even on --exclusive nodes; require same-node n>=2 vs a same-node control
before claiming (I retracted one single-run "59x" outlier that was a node confound). Every number traces to a commit
+ W&B point; all negatives kept on the curve.

---
# kv-flint-2c — sglang HiCache KV-cache research log

Independent researcher. Branch `evolve/kv-flint-2c`. W&B run `kv-flint-2c` in `sgl-evolve`.
Bar to beat: **v0_tuned** (ttft_mean 108824 ms). Better stock reference: **v0_official** (ttft_mean 87615 ms).
All evals on the fixed protocol (122B-A10B-FP8, TP8, ctx 262144, mem-frac 0.85, hicache 96=768GB host,
file L3 1.8TB, mix 1553 convs, λ=3.5, max-conc 128). Headline: mean TTFT (lower better); lossless gate.

## Baseline reading (from baseline.json / baseline_tuned.json)
The system is **saturated**: arrival λ=3.5 req/s but req_throughput ≈ 1.15 → the queue grows over the run,
so **mean TTFT (~87.6s official) is dominated by queue wait** (median TTFT is only ~1.2s). Mean TTFT is
therefore effectively a **throughput proxy**; out_tok_s / req_throughput are cleaner, less tail-noisy signals.

Key baseline signals:
- hit_rate 0.816 (device 31.3% / host 43.4% / **storage/SSD 25.4%** of hits). host_util ≈ 1.0 (host cache full).
- Prompt tokens 99.9M, cached 81.5M → **~18.3M miss tokens must be prefilled** (workload-structural; misses
  are genuinely-new tokens, so hit rate is near its ceiling for this reuse structure).
- Decode output ≈ 0.9M tokens total → GPU work is **prefill-dominated**.
- Huge transfer churn: load_back 444M (~4.4× prompt → multiturn reload tax), evict 574M, offload 145M,
  prefetch 166M, disk_read 20.7M.
- Device KV pool = 2.35M tokens (215 GB) — tiny vs ~19M working set → heavy device eviction/reload.

Two baseline files have **identical resolved_args** yet differ ~24% (87.6 vs 108.8s) → either large
run-to-run TTFT variance or tuning in knobs not captured by resolved_args (schedule_policy / mixed_chunk /
eviction_policy are NOT in resolved_args). Implication: a win must be **large** to clear the noise floor;
I prioritize throughput deltas over tail-TTFT.

## Active code path (verified)
Hybrid-Mamba + hierarchical cache → `registry.default_radix_cache_factory` returns **`UnifiedRadixCache`**
(`mem_cache/unified_radix_cache.py`) with `HybridCacheController` + `MambaPoolHost`. NOT `hi_mamba_radix_cache.py`.
Defaults (server_args): schedule_policy=**fcfs**, radix_eviction_policy=**lru**, enable_mixed_chunk=**False**,
disable_overlap_schedule=False, write_through_threshold=1 (write_through), prefetch_threshold=256,
prefetch_capacity_limit≈0.8·(host−device)≈4.4M tok, load_back_threshold=10.

Prefetch (L3→host) is async in bg threads; `wait_complete` holds a req in the waiting queue until its full
storage prefetch completes, but the scheduler admission loop `continue`s past not-ready reqs to admit
cache-ready ones — so wait_complete HoL bubbles are **largely mitigated already**. load_back (host→device)
runs synchronously in the prefill adder (can trigger device eviction on the critical path), transfer on load_stream.

---

## Versions

### v0_official — stock default (reference, not re-run)  [config]
ttft_mean 87615 ms, out_tok_s 146.9, hit 0.816, L3 25.4%.

### v0_tuned — strongest stock config (reference, not re-run; BAR)  [config]
ttft_mean 108824 ms, out_tok_s 119.3, hit 0.821, L3 25.9%.

### v1-besteffort — prefetch policy best_effort (diagnostic)  [config]  — RUNNING
Hypothesis: if GPU idles on synchronous disk-prefetch waits, best_effort (serve partial + recompute
suffix, lossless) fills those bubbles → lower TTFT. If compute-bound (bubbles already mitigated by the
admission `continue`), best_effort adds recompute → neutral/worse. Sign is the diagnostic:
bubble-bound vs compute-bound. Result: _pending_.
Regime read (from live server.log): `Prefill batch #cached-token`/`#new-token` ratio = load:compute per
batch. If #cached-token >> #new-token, batches are LOADING-BOUND (H→D load can't hide behind compute) →
motivates v2 balanced batching. Also watch retract count, token usage, gen throughput, #queue-req growth.

---

## Planned mechanism — v2: loading-bound-aware ("balanced") prefill batching  [mechanism]
Motivation (Strata "balanced batches", 1.8× in their ablation): in this multiturn workload later turns
are loading-bound (big cached prefix loaded H→D via load_back + tiny new suffix computed). Stock sglang
forms prefill batches FCFS by token budget only (schedule_policy=fcfs), ignoring the per-batch
load:compute ratio → loading-bound batches where the GPU forward stalls waiting for each layer's H→D load.

Mechanism: in the prefill admission loop (`scheduler._get_new_batch_prefill_raw`, ~L2861), track running
`batch_load` (Σ req.host_hit_length) and `batch_compute` (Σ real_input_tokens = extend_len − host_hit_length).
Defer (skip this round) a request that would push `batch_load > R·batch_compute` while the batch already
carries compute — so the H→D load of load-heavy turns is paired with enough compute to hide it. Always
admit ≥1 req (progress) and cap consecutive defers (anti-starvation). Lossless: pure admission reorder,
outputs unchanged. Gated by `SGLANG_ENABLE_BALANCED_PREFILL` (EnvBool False; A/B vs stock).
Gate on v1 regime: implement only if v1/live shows loading-bound batches.

## Environmental block (honest record) — 2026-07-03, ~06:40–23:10+
The shared held eval-pool (4 certified nodes for 8 workers, 2:1 oversubscribed) has been
un-winnable for me for ~17h despite a correct, collision-safe, autonomous pipeline:
- One pool node (-0) was jammed ~5h by a competitor's `launch_eval` looping evals without releasing
  the flock; later competitors' `node_waiter` launchers were observed SQUATTING flocks on
  free+usable nodes (flock held, 0 servers, disk≥1.8T for minutes) — monopolizing capacity.
- No free certified node exists to self-lock (all 8 certified are pool-held / kv-heron-self-locked /
  drain), so a queued `sbatch` hedge (18160) sits PENDING indefinitely.
- I evolved the launcher to the correct strategy: infinite-blocking `flock` (compete with competitors'
  `flock -w` blockers; hold kernel FIFO wake position; no restarts). Early hours were lost to launcher
  bugs (a pkill pattern self-killing my shell; FIFO-position resets from repeated restarts).
I have NOT gamed the pool (no flock-squatting, no flock-bypass that risks OOMing a neighbor) — that is a
fairness matter for the manager's fairness-guard. The autonomous orchestrator (plan.tsv: v1-besteffort,
v2-balanced-r2 [mechanism], v3-wtselective, v4-lpm, v5-mixedchunk) keeps blocking fairly and will run +
self-audit + log to W&B the moment it wins node access. v2 mechanism (balanced/loading-bound prefill
batching) is built, unit-tested, committed, flag-gated (SGLANG_ENABLE_BALANCED_PREFILL).

## RESULTS (autonomous dedicated-node runner; all with --enforce-disable-flashinfer-allreduce-fusion for node stability)

### v1-besteffort  [config, LOSSLESS] — NEW BEST (beats v0_tuned 34x)
prefetch=best_effort. Mean TTFT **3241.8 ms** (v0_official 87615 → 27.0x; v0_tuned 108824 → 34.4x).
median 1711 / P99 29465 ms; out 248 tok/s (+69%); req 1.94/s; bench 60min; hit_rate 0.44 (device+host
only, L3 skipped). FINDING: default wait_complete prefetch blocks requests on slow SSD(L3) reads →
queue saturates (baseline ~88s is queue-wait). best_effort (serve partial + recompute suffix on GPU) →
#queue-req 0-6 → ~27-34x lower TTFT. Lossless (recompute = identical KV). This is CONFIG (bottleneck map),
not novelty; establishes best_effort as the winning regime for mechanism work.
NOTE: ~28h lost to eval-pool contention/monopolization + dirty freed nodes (GPU zombies/bad disk);
solved with an autonomous self-correcting dedicated-node pipeline (keeper+runner, 2h TTL bad-node
exclusion, retry, claim-lock, flashinfer-disable) — see eval_good_node.sh/dedicated_runner.sh/keeper.sh.

### Plan (best_effort winning regime): v2-balanced-r2 (mechanism, wait_complete+balanced — tests my
loading-bound batching in its target regime), v6-be-balanced (mechanism in winning regime),
v7-be-lpm, v8-be-mixedchunk, v9-be-wtselective, v10-timeout. Genuine-novelty goal: beat 3241ms with a mechanism.

### v2-balanced-r2 [mechanism, LOSSLESS]  ttft_mean 82853.9 ms (wait_complete + balanced, R=2)
vs v0_official 87615 (wait_complete baseline): ~5% better but WITHIN ~24% run-to-run noise -> balanced
batching is ~neutral in the slow wait_complete regime (the disk-wait dominates; low headroom to help).

### v6-be-balanced [mechanism, LOSSLESS] — NEW BEST  ttft_mean 3023.2 ms (best_effort + balanced, R=2)
vs v1-besteffort 3241.8 (best_effort alone): **-6.7% TTFT, +9.8% req tput (2.13), +9.6% out tput (271.8)**,
hit 0.452. Measured same-node back-to-back (ondem-3) => low variance; throughput corroborates. GENUINE
MECHANISM WIN: loading-bound-aware balanced prefill batching pays off in the low-queue best_effort regime
(pairs unavoidable per-layer H->D load_back with enough prefill compute to hide it). Lossless.
Next: ratio sweep (v11 R=1.0, v12 R=1.5, v13 R=3.0) to find the balancing optimum; +be-configs v7-v10.

### More results (best_effort/timeout winning regime; all --enforce-disable-flashinfer-allreduce-fusion)
- v7-be-lpm [config] **2010.3 ms — BEST (54x < v0_tuned)**. lpm cache-aware sched: hit 0.44->0.618 -> fewer recomputes.
- v8-be-mixedchunk [config] 16180 ms — NEGATIVE (mixed-chunk fragments prefill throughput in recompute-heavy regime).
- v10-timeout [config] 2588 ms — timeout prefetch BEATS best_effort-alone (3241): bounded disk wait -> hit 0.44->0.612.
- v14-be-lpm-balanced [mechanism] 2708 ms — NEGATIVE: my balanced batching CONFLICTS with lpm (both reorder the
  queue; balanced's deferral breaks lpm's prefix co-scheduling -> hit 0.618->0.469). Balanced helps best_effort-ALONE
  (v6 +7%) but not with lpm. KEY INSIGHT: two cache-aware schedulers interfere; lpm alone dominates.
Curve summary (mean TTFT, lower=better): v0_tuned 108824 > v0_official 87615 >> v1 3241 (best_effort, 34x) >
  v6 3023 (best_effort+balanced, mechanism +7%) > v10 2588 (timeout) > v7 2010 (best_effort+lpm, BEST 54x).

### v18-to-lpm [config] 2551.2 ms — best_effort+lpm (v7 2010) STILL BEST.
Finding: timeout BEATS best_effort ALONE (v10 2588 < v1 3241), but WITH lpm best_effort wins (v7 2010 < v18 2551).
lpm's high hit rate (0.618) makes indiscriminate disk-waits (timeout) a NET latency cost. => winning config = best_effort+lpm.
Next (my novelty): LENGTH/COST-AWARE prefetch — wait for disk KV only on long prefixes (recompute O(L^2)-expensive),
skip on short (cheap recompute). Beats blanket timeout's indiscriminate waiting; distinct from my balanced batching.

### v20-be-lpm-wtsel [config] 2705.6 ms — NEGATIVE. write_through_selective (thr 2) < default write_through (thr 1, v7 2010).
Delaying host writes -> host tier populates slower -> fewer host hits -> more recompute. Aggressive write-through best here.

### v21-be-lpm-wb [config] 2819.1 ms — NEGATIVE. write_back (write only on eviction) < write_through default.
Confirms host tier must populate EAGERLY (write_through thr=1) to serve host hits; deferred writes -> more recompute.
Write-policy exploration complete: write_through(default,v7 2010) > wtsel(2705) > write_back(2819). All non-default HURT.

### v23-be-lpm-cgate1k [MECHANISM: length/cost-aware prefetch gate, gate=1024tok cap=0.3s] 2212.0 ms.
vs v7 be+lpm (no wait) 2010 -> +10% WORSE, but MUCH better than blanket timeout+lpm (v18 2551): selective bounded
waiting hurts less than indiscriminate waiting, but still loses to pure best_effort. FINDING: the saturated regime is
QUEUE-bound not recompute-cost-bound -> disk is NEVER worth waiting for (even for long O(L^2) prefixes, even a 0.3s
page-cache-hot wait), because any wait backs up the queue. Re-confirms best_effort (skip all disk, recompute) optimal.
Sweep continues: v24 gate=4096 (fewer prefixes wait -> should approach but not beat v7), v25 cap=0.8s.

### Key stock-code insight (my own reading of schedule_policy.py::_determine_active_policy):
lpm REVERTS to fcfs when waiting_queue > 128 (prefix-sort too expensive for big queues). This EXPLAINS why
best_effort+lpm wins: best_effort keeps the queue SMALL (<128, no disk waits) so lpm actually engages; under
wait_complete the queue is huge -> lpm silently == fcfs. The revert is LPM-ONLY: dfs-weight (other stock cache-aware
policy) does NOT revert -> stays cache-aware even in queue bursts >128. Hypothesis: v26 be+dfs-weight may beat be+lpm.

### *** v24-be-lpm-cgate4k [MECHANISM] 1837.9 ms — NEW BEST (~59x < v0_tuned, -8.6% vs v7) *** EMAILED.
My length/cost-aware prefetch gate WINS when tuned: gate=1024 (v23) too aggressive 2212 (too many prefixes wait ->
queue saturates) BUT gate=4096 = 1837.9 (only the few genuinely-LONG prefixes wait 0.3s; their O(L^2) recompute is
expensive enough that a short page-cache-hot disk wait pays off). SWEET SPOT between 1024 and 4096. This is a genuine
MECHANISM win over the best config (v7 be+lpm 2010). Lossless. env SGLANG_PREFETCH_COST_GATE=4096 (cap 0.3s).
MUST CONFIRM (repeat, given ~24% baseline variance) + refine sweet spot (gate 2048/6144/8192, cap 0.1/0.5).

### v26-be-dfsweight [config] 2636.0 ms — NEGATIVE. dfs-weight < lpm (v7 2010), despite dfs-weight NOT reverting to
fcfs at queue>128. Implies: under best_effort the queue rarely exceeds 128 (so lpm rarely reverts anyway), AND/OR
dfs-weight's DFS-order co-scheduling is simply worse than lpm's longest-prefix-match for this workload. lpm stays best.

### *** RETRACTION + VARIANCE FINDING (v28) ***
v28-be-lpm-cgate4k-rpt = 2586.4 ms = EXACT REPEAT of v24 (same env/args, gate=4096) which was 1837.9 ms.
A ~40% swing for the SAME config => v24's 1837.9 was a FAVORABLE-VARIANCE OUTLIER, not a real win. RETRACT the
"v24 cost-gate mechanism win" (correction emailed). The eval has LARGE run-to-run / node-to-node variance (~35-40%),
so ANY two single-run results in the ~2000-2600 ms band are statistically indistinguishable. Earlier fine-grained
rankings (be+lpm 2010 vs timeout+lpm 2551 vs cost-gate vs write-policy) are largely WITHIN the noise floor -- do NOT
over-interpret. Confound: v24 ran on ondem-3, v28 on -0 (different nodes) -> node effects may contribute.
ROBUST (large, reproducible): default wait_complete prefetch (~87615 ms baseline) -> best_effort+lpm (~2000-2600 ms)
is a ~35-45x reduction. That is the real finding. Pivot: same-node (-0) REPEATS to get error bars, no single-run claims.

### v29-be-lpm-cgate8k [mechanism] 2519.2 ms. On node -0: cost-gate variants cluster ~2500-2600 (v28 gate4k 2586,
v29 gate8k 2519) -- ALL worse than v7 be+lpm 2010 (which ran on ondem-3). Node confound is now the prime suspect.
Decisive test = be+lpm NO-gate on -0 (v33): if ~2500 -> node -0 slower (gate neutral); if ~2010 -> gate hurts.

### *** v33-be-lpm-ctrlA (be+lpm NO gate, on -0) = 2539.8 ms — DECISIVE ***
On node -0: be+lpm NO-gate (v33 2539.8) ~= be+lpm+gate (v28 2586, v29 2519) -> COST-GATE IS NEUTRAL (no effect).
And be+lpm on -0 (2540) >> be+lpm on ondem-3 (v7 2010, v24 gate 1837.9) -> NODE -0 is ~25-30% SLOWER than ondem-3.
=> The ~40% "variance" is largely a NODE CONFOUND. Cross-node single-run comparisons are dominated by a ~25-30%
per-node speed effect; ALL my earlier fine-grained rankings are unreliable. Robust finding UNCHANGED: best_effort+lpm
(~2000-2600 ms, node-dependent) vs wait_complete baseline (87615 ms) = ~35-45x. Remaining repeats v34/v36 (be-alone
on -0) vs v33/v35 (be+lpm on -0) = clean same-node test of whether lpm's benefit is real above noise.

### *** v34-be-only-ctrlA (be-alone, on -0) = 2586.1 ms — lpm's benefit REFUTED as node confound ***
SAME-NODE (-0): be+lpm (v33 2539.8) vs be-alone/fcfs (v34 2586.1) = ~2% apart = WITHIN NOISE. lpm does NOT
meaningfully help! My earlier "lpm +38%" (v1 be-alone 3241 -> v7 be+lpm 2010) was ALSO a node confound (v1 on a
slow node, v7 on fast ondem-3). On -0 ALL configs cluster ~2520-2586: be+lpm 2540, be-alone 2586, +gate4k 2586,
+gate8k 2519 -> on a fixed node, best_effort perf is ~CONSTANT regardless of scheduling/cost-gate/timeout.
HONEST CONCLUSION FORMING: the ONE robust, reproducible lever is prefetch_policy=best_effort (skip synchronous
slow-disk prefetch that saturates the queue) -> ~35-45x over the wait_complete baseline (far beyond the ~25-30%
node variance). lpm / cost-gate / timeout / write-policy are all within node+run noise. Confirming with v35/v36 (n=2).

### INTEGRITY: v24-be-lpm-cgate4k RETRACTED in W&B (2026-07-05)
The 1837.9 ms was a single-run favorable-variance outlier (node ondem-3); exact repeat v28 (node -0) = 2586.4 ms.
Cost-gate is NEUTRAL (same-node controls v33/v28/v29). Correction already emailed. Point kept on curve but flagged
retracted so it is not read as a real best. My reproducible best-config = best_effort (+lpm, ~5% borderline).

### SAME-NODE(-0) n=2 CONTROLLED RESULT — lpm's benefit is REAL but MODEST (~8.5%)
be+lpm: {v33 2539.8, v35 2375.0} mean 2457.4  |  be-alone/fcfs: {v34 2586.1, v36 2782.4} mean 2684.3
Ranges DON'T overlap (be+lpm max 2540 < be-alone min 2586) -> lpm reliably beats fcfs by ~8.5% on the SAME node.
So lpm IS a genuine (modest) win on top of best_effort -- but the cross-node "lpm +38%" (v1 vs v7) was inflated ~4x
by the node confound. Same-node run-to-run spread ~7% (both configs). Cost-gate remains NEUTRAL (v28/v29 in the
be+lpm band). Honest ranking on a fixed node: best_effort (huge, ~35-45x) >> +lpm (~8.5%) > +cost-gate/dfs/write (0).

### v37-be-bal-ctrlA (best_effort + my balanced-batching mechanism, on -0) = 2746.0 ms — NEUTRAL
Within the be-alone range {v34 2586, v36 2782}. So balanced batching does NOT help on the same node; my earlier
"+7%" (v6 cross-node) was a node confound too. BOTH my novel mechanisms (balanced batching, cost-gate) are NEUTRAL
in this regime. The only real wins are CONFIG: best_effort (huge) + lpm (~8.5%). Honest: the regime is GPU-prefill-
bound once best_effort drains the queue, so scheduling/batching/gate tweaks can't add much -- the remaining lever is
cheaper prefill (frozen/lossy) or better caching (eviction/admission = off-limits for independence). v38 confirms n=2.

### BOTTLENECK ANALYSIS (v35 metrics, be+lpm on -0) — why mechanisms are neutral
throughput 2.55 req/s (< lambda 3.5 => still saturated, queue grows) | hit_rate 0.52 (~48% miss=recompute) |
hit tiers: device 59% / host 41% / DISK 0% | host_util 0.998 (FULL) | offload 252M tokens (written to disk, NEVER
read: l3_hit 0) | mean TTFT 2375 but median 1135, p99 17791 => MEAN IS TAIL-DOMINATED.
Interpretation: once best_effort drains the queue, the system is PREFILL-RECOMPUTE-bound AND host-capacity-bound
(host full -> churn -> disk -> skipped -> recompute). Levers that don't reduce recompute (scheduling/batching/gate/
layout) can't move the throughput floor -> that's why lpm gives only ~8.5% and balanced/cost-gate give ~0. Reducing
the 48% miss needs better caching (eviction/admission = off-limits for independence) or cheaper prefill (frozen/lossy).

### v38-be-bal-ctrlB = 2788.8 ms -> balanced n=2 {2746, 2789} mean 2767 CONFIRMED NEUTRAL (slightly negative vs
be-alone {2586,2782} mean 2684, ranges overlap). FINAL same-node(-0) ranking: be+lpm 2457 < be-alone 2684 <
be+balanced 2767 < be+cost-gate ~2550. Only lpm helps (~8.5%); my two mechanisms neutral-to-slightly-negative.
Next: v39/v40 page-size layout (last genuinely-different lever); if neutral, implement skip-L3-disk-write (bottleneck-
motivated: disk is write-only waste under best_effort -> 252M offload tokens for l3_hit=0; skipping is lossless).

### v39-be-page128 [config, layout] 3121.4 ms — NEGATIVE. page_size 128 > page64 (be-alone {2586,2782}) by ~16%.
Larger page -> coarser prefix-match granularity -> lower hit rate -> more recompute; the fewer/bigger-transfer
efficiency gain does NOT compensate. Layout page-size is the wrong direction. (v40 page256 expected worse still.)

### v40-be-page256 [config, layout] 3333.0 ms — NEGATIVE. Page-size sweep MONOTONIC: page64 ~2684 < page128 3121 <
page256 3333. Larger page consistently hurts (coarser prefix-match -> lower hit rate -> more recompute). Layout
page-size is a confirmed wrong direction; keep the default page64. Next: skip-L3-write mechanism (v43/v44).

### *** v43-be-lpm-skipL3-A [MECHANISM] 1147.4 ms — STRONG CANDIDATE WIN (SAME-NODE -0, -53% vs be+lpm) ***
be+lpm+SKIP_L3_WRITE (v43) vs be+lpm (v33/v35 {2540,2375} mean 2457) — ALL on node -0 (no node confound):
 ttft_mean 1147 vs 2375 (-52%) | throughput 3.40 vs 2.55 req/s (+33%, ~= lambda 3.5 -> queue stops growing!) |
 hit_rate 0.620 vs 0.519 | ttft_p99 9511 vs 17791 | tpot 240 vs 548 | offload_tokens NONE(skipped) vs 252M.
LOSSLESS: hit_storage_frac=0.0 in BOTH (disk never read under best_effort) -> skipping its writes changes no served
token. MECHANISM: the host->disk offload (252M tokens) is write-only waste under best_effort; it competes for
PCIe/CPU/host-pool with the essential H<->D load_back. Removing it frees those -> +33% throughput + higher host-hit
rate -> -53% TTFT. ~95x < v0_tuned. NOT EMAILED YET — confirming n>=2 same-node on a 2nd node (v44-v47) per the v24
variance lesson before claiming a new best.

### *** CONFIRMED NEW BEST: best_effort+lpm+SKIP_L3_WRITE ~1161 ms (n=2 same-node -0) — MECHANISM WIN ***
SKIP-L3-WRITE {v43 1147.4, v44 1175.6} mean 1161 (spread 2.4%) vs no-skip be+lpm {v33 2539.8, v35 2375.0} mean 2457
-- ALL on node -0 -> reproducible -53%, non-overlapping by ~2x, NOT an outlier (unlike retracted v24). ~94x < v0_tuned.
LOSSLESS (hit_storage_frac=0.0 in every run: disk is never read under best_effort, so skipping its writes changes no
served token). MECHANISM (my own, from bottleneck analysis): under best_effort the L3 disk tier is WRITE-ONLY; the
252M-token host->disk offload is pure waste competing with the essential H<->D load_back for PCIe/CPU/host-pool.
Skipping it (SGLANG_SKIP_L3_WRITE, self-guarded to best_effort) frees those -> throughput 2.55->3.40 req/s (+33%,
~=lambda 3.5 so the queue stops growing) + host hit_rate 0.52->0.62 -> TTFT halves. Upstream takeaway: don't offload
KV to a tier you never read. EMAILED. Further rigor: v45/v47 (no-skip) + v46 (skip) extend to n>=3 same-node.

### v45-be-lpm-noskip-n2 (no-skip be+lpm, -0) = 1833.3 ms — widens same-node no-skip spread.
no-skip be+lpm on -0 now {1833, 2375, 2540} mean 2249 (~31% run-to-run spread -- LARGER than the ~7-8% I estimated
from n=2; same-node variance across ALLOCATIONS/time is bigger). BUT the skip-L3-write win still HOLDS cleanly:
SKIP {1147, 1176} is below EVERY no-skip run (max skip 1176 < min no-skip 1833) -> non-overlapping -> robust
-36%(vs best no-skip 1833) to -53%(vs mean 2457). Honest note: report the skip win as ">=36% and reproducible",
skip is tight (~1161) while no-skip is noisy (1833-2540). v46 (skip n=3) + v47 (no-skip n=4) extend this.

### v46-be-lpm-skipL3-C = 1176.9 -> skip-L3-write n=3 {1147.4, 1175.6, 1176.9} mean 1167, TIGHT (~2.5% spread).
vs no-skip {1833, 2375, 2540} (noisy). skip is both LOWER and far more STABLE (removing disk-write IO contention ->
consistent runs). Win robust: every skip run < every no-skip run. Best config = best_effort+lpm+skip_L3_write ~1167ms
(~93x < v0_tuned). Hill-climb next: v48/v49 isolate skip w/o lpm; v50/v51 re-test balanced/cost-gate +skip (regime
shifted queue-bound -> near-GPU-bound at 3.4 req/s, so overlap/gate mechanisms may now help where they were neutral).

### v47 no-skip = 1803 -> ALLOCATION-LEVEL variance refinement (tightest control)
Node -0 ran TWO allocations: job 18297 (~2457 no-skip+lpm: v33 2540, v35 2375) and job 18328 (~1818: v45 1833,
v47 1803) -- a ~26% speed difference for the SAME physical node across allocations (thermal/co-tenant/disk-state
between allocations). So control WITHIN an allocation. skip-L3-write win holds within BOTH:
  18297: skip v43 1147 vs no-skip {2540,2375} -> -53%
  18328: skip {v44 1176, v46 1177} vs no-skip {v45 1833, v47 1803} -> -35%
=> reproducible -35%..-53% within-allocation (the cleanest possible comparison). On the current allocation 18328 the
baselines are: no-skip+lpm ~1818, skip+lpm ~1176 -> use these to read v48-v51 (all on 18328).

### v48-be-skip-nolpm-A (best_effort + skip_L3_write, NO lpm, on 18328) = 1214.1 ms
On allocation 18328: no-skip+lpm ~1818, skip+lpm ~1176, skip+NO-lpm 1214. => skip-L3-write is the DOMINANT
standalone mechanism (1214 vs no-skip 1818 = -33% even without lpm); lpm adds only ~3% ON TOP of skip (vs ~8.5%
without skip). In the post-skip near-GPU-bound regime (queue nearly drained at 3.4 req/s) scheduling order matters
less. Best config skip+lpm ~1176, but skip alone ~1214 captures nearly all of it. v49 confirms; v50/v51 test
balanced/cost-gate on top of skip.

### v49 skip-nolpm = 1138.8 -> skip WITHOUT lpm n=2 {1214, 1139} mean 1177 ~= skip+lpm ~1176 (alloc 18328).
=> lpm is REDUNDANT once skip-L3-write is applied (skip alone captures the full win). In the post-skip near-GPU-bound
regime, cache-aware scheduling adds ~0. So the mechanism (skip-L3-write) is the whole story; best config simplifies to
best_effort + skip_L3_write (~1177), lpm optional. v50/v51 test balanced/cost-gate on top; v52/v53 full disk-bypass.

### v50 skip+balanced = 1249.6 vs skip+lpm ~1176 (alloc 18328) -> balanced NEUTRAL/slightly-negative even post-skip.
My balanced-batching mechanism does not help in ANY regime tested (queue-bound OR near-GPU-bound). Confirmed dead.

### v51 skip+cgate = 1160.1 ~= skip+lpm ~1176 (alloc 18328) -> cost-gate NEUTRAL on top of skip too.
Nothing stacks meaningfully on skip-L3-write: skip+lpm 1176, skip-nolpm 1177, skip+balanced 1250, skip+cgate 1160.
skip-L3-write is the whole win. v52/v53 test the last companion: full disk-bypass (skip-write + skip-prefetch).

### v52-skip-both-A (FULL DISK-BYPASS: skip-write + skip-prefetch, be+lpm, alloc 18328) = 1094.5 ms — CANDIDATE
vs skip-write-only ~1176 (v44/v46 same alloc) -> -7%: my skip-L3-prefetch companion ADDS value on top of skip-write
(avoids the host-tier churn/eviction from the immediately-cancelled disk prefetch under best_effort). ~99x < v0_tuned.
LOSSLESS (disk neither read nor written under best_effort). n=1 -> v53 confirms before claiming (v24 lesson).

### *** CONFIRMED REFINED BEST: FULL DISK-BYPASS (skip-write + skip-prefetch) ~1094 ms (~99x < v0_tuned) ***
n=2 alloc-18328: full-bypass {1094.5, 1094.0} (spread 0.5ms!) < skip-write-only {1176,1177} < no-skip {1833,1803}
-- three non-overlapping tiers, same allocation. My TWO mechanisms compose: skip-L3-write (the ~2x) + skip-L3-prefetch
(a further reliable ~7%). Both LOSSLESS (best_effort reads/writes disk for 0 served tokens; l3_hit=0). Best config =
best_effort [+lpm, optional] + SKIP_L3_WRITE + SKIP_L3_PREFETCH. Upstream: under a never-read storage tier, skip BOTH
its writes AND its prefetch issue -- the latter also stops it evicting useful host KV for a buffer it discards. EMAILED.

### v54 skip-prefetch-ONLY = 1289.3 (alloc 18332) — prefetch-skip is a STANDALONE win too.
Rough decomposition (18328/18332 refs): no-skip ~1818 > skip-prefetch-only ~1289 (-29%) > skip-write-only ~1176
(-35%) > full-bypass ~1094 (-40%). Both mechanisms help alone and COMPOSE (write-skip the bigger, prefetch-skip
adds on top). v56 (no-skip on 18332) pins the exact same-alloc baseline; v55 repeats prefetch-only; v57 full-bypass.

### v56 no-skip (18332) = 2529.6 -> 18332 is a SLOW allocation (~2530 vs 18328 ~1818; allocation variance again).
Clean SAME-ALLOCATION decomposition, two allocations:
  18328 (fast): no-skip {1833,1803}~1818 > skip-write-only {1176,1177} (-35%) > full-bypass {1094,1094} (-40%)
  18332 (slow): no-skip 2530 > skip-prefetch-only {1289,1350}~1319 (-48%) > full-bypass v57 (pending)
=> BOTH mechanisms are large STANDALONE wins within-allocation (skip-write -35%, skip-prefetch -48%), and they
COMPOSE (full-bypass best on both). The absolute ms is allocation-dependent (~40% between allocations) but the
RELATIVE within-allocation wins are robust and consistent. Confirms: fully bypassing the never-read disk tier
(skip both write + prefetch) is the win; each half helps alone, together best.

### *** v57 full-bypass (18332) = 1096.4 -> FULL-BYPASS IS ALLOCATION-INVARIANT ***
Complete 18332 (slow alloc) same-alloc decomposition: no-skip 2529.6 > skip-prefetch-only {1289,1350} (-48%) >
full-bypass 1096.4 (-57%). STRIKING: full-bypass n=3 across TWO allocations = {1094.5, 1094.0, 1096.4} spread 2.4ms
(0.2%!) even though no-skip swings 1818(fast)<->2530(slow). => the skip mechanisms REMOVE the disk-contention variance
source; the system becomes purely GPU-prefill-bound (stable across allocations). So full disk-bypass both HALVES mean
TTFT and STABILIZES it. Final best: best_effort + skip_L3_write + skip_L3_prefetch = ~1095 ms (~99x < v0_tuned),
n=3 allocation-invariant. Both mechanisms are large standalone wins (skip-write -35%, skip-prefetch -48%) that compose.

### v58-fb-lbt1 (full-bypass + load_back_threshold=1, alloc 18333) = 1146.2 ms [prelim]
lbt=1 raises hit 0.622->0.638, throughput 3.44->3.47, median 581->570 (loads more small host hits, less recompute)
BUT mean not improved (1146; tail-dominated -- extra load_back may worsen tail). CROSS-ALLOC vs full-bypass ~1095;
need v60 (full-bypass control on 18333) to judge same-alloc. v59=lbt4.

### load_back_threshold VERDICT (full-bypass base, same alloc 18333) = NEUTRAL on mean.
lbt=1 (v58) 1146.2 | lbt=4 (v59) 1126.4 | default lbt=10 (v60) 1131.4 -- all within ~2% (noise). Lowering lbt loads
more small host hits (hit 0.622->0.638, throughput 3.44->3.47, median 581->570) BUT the extra load_back worsens the
tail, so the headline MEAN is unchanged. Honest negative on the mean; nuance = it trades GPU recompute for H<->D +
tail with no net mean gain. (Full-bypass on 18333 ~1131 vs 18328/18332 ~1095 = allocation variance.) Best stays
full-bypass ~1095-1131.

### v61-fb-wtsel (full-bypass + write_through_selective, new alloc 18336-on-0) = 1403.0 ms [prelim]
Above full-bypass ~1095-1131 -> write_through_selective (less-eager device->host offload) appears to HURT even under
full-bypass (lower host population -> lower hit rate -> more recompute; consistent with v20 negative). Need v62
same-alloc control to confirm. Default write_through (eager) remains best.

### v61/v62 (alloc 18336): write_through_selective CONFIRMED NEGATIVE under full-bypass.
full-bypass+wtsel (v61) 1403.0 vs full-bypass ctrl (v62) 1122.3 -> +25% same-alloc. Less-eager device->host offload
lowers host population/hit-rate -> more recompute (consistent with v20 without skip). Default eager write_through is
best in every regime. Full-bypass on 18336 = 1122 (within the ~1095-1131 allocation-invariant band). v63 no-lpm, v64 write_back next.

### v63 full-bypass no-lpm = 1145.1 ~= full-bypass+lpm (v62 1122.3) -> lpm REDUNDANT under full-bypass (confirmed,
consistent with v48/v49). skip mechanisms are the whole win; scheduling order adds ~0 once the queue is drained. v64 write_back last.

### *** v64-fb-writeback (full-bypass + write_back, alloc 18336) = 884.2 ms — STRONG CANDIDATE (new best?) ***
vs full-bypass+write_through ctrl (v62 1122.3) SAME ALLOC -> -21%. Metrics: hit_rate 0.627->0.732 (+10.5pp!),
throughput 3.46->3.52 (>= lambda 3.5 -> queue FULLY drains -> mean approaches median), median 586->505, p99 7722->6493,
tpot 232->175. LOSSLESS (hit_storage_frac=0.0 both; write_back persists KV to host on eviction). MECHANISM: eager
write_through offloads EVERY write device->host, churning the FULL host tier (evicting useful KV); write_back writes
only on device-eviction -> far less host churn -> higher host hit rate -> less recompute -> throughput reaches lambda.
REGIME-DEPENDENT: write_back was NEGATIVE without skip (v21 2819) but POSITIVE under full-bypass. ~123x < v0_tuned.
n=1 -> confirming n>=2 same-alloc (v65-v68) before claiming/emailing (v24 lesson).

### v65-fb-wb-A (full-bypass + write_back, alloc 18348) = 888.1 -> write_back n=2 = {884.2, 888.1} TIGHT (0.4%),
reproducible across 2 allocations, well below full-bypass write_through band (~1094-1145). Strong candidate new best
~886ms (~123x < v0_tuned). Awaiting v66 (write_through same-alloc control on 18348) for the clean same-alloc delta.

### *** CONFIRMED NEW BEST: full-bypass + write_back ~886 ms (~123x < v0_tuned) ***
TWO same-alloc confirmations, both -21%: 18336 wb 884.2 vs wt 1122.3 ; 18348 wb 888.1 vs wt 1130.0.
write_back n=2 {884.2, 888.1} (tight) NON-OVERLAPPING with write_through {1122.3, 1130.0}. LOSSLESS (hit_storage_frac
=0.0; write_back persists KV to host on device-eviction). MECHANISM: under full-bypass, eager write_through offloads
every write device->host, churning the FULL host tier (evicting useful KV); write_back writes only on eviction ->
host hit_rate 0.63->0.73 -> throughput reaches lambda 3.5 -> queue drains -> mean 1122->886. REGIME-DEPENDENT config
finding (write_back was NEGATIVE without skip: v21 2819). Best stack: best_effort + skip_L3_write + skip_L3_prefetch
+ write_back = ~886ms. EMAILED. v67/v68 extend n.

### v67 write_back = 895.4 -> write_back n=3 {884.2, 888.1, 895.4} mean 889, TIGHT (~1.2%), non-overlapping with
write_through {1122.3, 1130.0}. New best ~889 ms (~122x < v0_tuned) thoroughly confirmed. v68 = write_through n=3.
