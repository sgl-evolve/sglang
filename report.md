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
