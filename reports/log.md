## 2026-07-08 — session start (base_free, v0.25_ablations 2-tier)
- Fresh cell. Setup done: own clone evolve/base_free @ a334877e5, cu129 venv (torch 2.11.0+cu129, sgl_kernel, deep_gemm OK), 392 pkgs.
- check_env queued (job 18539). v0_official logged to W&B run base_free.
- Regime: 2-tier L1(2.35M)+L2(8.4M)=10.7M cap, no disk. Baseline p99 TTFT 6326ms (< 8s SLO), req/s 2.78, hit 0.62, host_util 1.0, evict 582M/load_back 298M = thrashing.
- Workload mapped: 1553 convs / 7163 turns / mean 4.6 turns; big doc prefix per conv; FIFO round-robin re-enqueue at λ=3 → queue-cycle eviction window between a conv's turns.
- Plan: build offline simulator (real dataset reuse + LRU 2-tier) to quantify eviction-before-reuse headroom & screen mechanisms before GPU eval.

## 2026-07-08 — v1 launched (WSAC admission control)
- Simulator (validated: prompt-tok exact, split 0.36/0.64, host_util 1.0) → ceiling hit 0.80 vs baseline ~0.60 = 21pp concurrency-eviction headroom. Turns 0-2 ~0 hit (docs evicted by wide FIFO waves). Gang-admission (bound active convs) recovers full headroom (G≤500→ceiling).
- v1 mechanism (commit dff3592e6, tag mechanism): WSAC — cap concurrent COLD-start prefills (SGLANG_WSAC_MAX_COLD), defer new-doc prefills beyond cap, prefer warm reuses, progress-guaranteed, lossless. Default off=stock.
- v1-wsac16 eval RUNNING on held node ondem-2 (cap=16, ratio=0.5, log=25). Monitoring live for admission dynamics (run_bs/wait/cold_active/deferred/host_util) to confirm the lever + calibrate.
- NOTE latency risk: cold-slot held prefill+decode (~32s, decode-bound) → sustainable cap ≈ 21; cap=16 may cause cold-TTFT backlog. Watching live.

## 2026-07-08 — WSAC bug fixed + wandb workflow + diagnostic relaunch
- BUG caught pre-serving: FCFS + UnifiedRadixCache has no fast-match → calc_priority leaves num_matched_prefix_tokens=0 → all reqs looked cold → over-throttle. FIXED (commit 8249dde29): call side-effect-free match_prefix_for_req in the gate. Killed the buggy v1-wsac16 run (node ondem-2 freed, GPU 0MiB). NOTE ops: pkill -f sglang SELF-MATCHES its own cmdline → use bracket trick pkill -f '[s]glang'.
- WANDB: online wandb.init TIMES OUT from login node (CommError, even 300s). FIX: WANDB_MODE=offline then `wandb sync <dir>` — sync WORKS. v0_official logged+synced to sgl-evolve/base_free (cloud curve point 0).
- Relaunched as DIAGNOSTIC (diag-baseline, MAX_COLD=0=stock scheduling, LOG=25, observe cold concurrency) on ondem-2 to ground-truth natural cold concurrency + wait-queue depth before choosing the cap. Server config confirmed: schedule_policy=fcfs, max_running_requests=None, decode cuda-graph max_bs 512, device KV pool 2.347M tok (=sim).

## 2026-07-08 — diagnostic results (diag-baseline, mechanism OFF) — LEVER CONFIRMED + latency constraint
Steady state (host_util=1.0): run_bs≈124 (≈max-conc 128), wait≈0-8, cold_active≈124 (~ALL reqs COLD),
prefill batches show #cached-token:0 → reuses MISSING (docs evicted) = the diagnosed thrash, ON HARDWARE.
**Device pool only ~47% used (1.1M/2.35M) while host_util=1.0** → device UNDER-utilized as a cache
(candidate v2: exclusive/less-aggressive-demotion tiering to use free device → +capacity, NO latency cost;
sim: host_cap 8.4M→hit 0.59, +device exclusive 10.16M→0.73).
Latency constraint: turns decode-bound (~32s), so full-turn cold-counting needs cap≳96 to sustain λ=3
(completion=cap/32s). ⇒ admission is SLO-limited to MILD throttle (124→~96). The 21pp hit headroom is only
partly recoverable under the p99≤8s SLO — this Pareto frontier IS what the goodput-under-SLO metric captures.
Launching v1-wsac96 (cap=96, sustainable) as primary mechanism test; investigating device-util for v2.

## 2026-07-08 — v1-wsac96 NEGATIVE (SLO) + diag control + PIVOT to exclusive tiering
- v1-wsac96 (admission cap=96): mechanism throttles correctly (run_bs 128→96, deferred~31) but live mean TTFT 12.7s >> 8s SLO at 4% through → KILLED. Cold-admission DEFERRAL can't recover headroom under p99≤8s (turns decode-bound → deferred cold reqs wait tens of s; worst in turn-0-heavy start). NOT logged (partial). Recorded as screened negative.
- diag-baseline (mech OFF + observe): hit 0.6205 ≈ v0_official 0.6217 (pipeline sane ✓), host_util 1.0, BUT p99 TTFT 24091ms vs baseline 6326ms, req/s 2.53 vs 2.78 → my OBSERVE instrumentation (match_prefix_for_req per waiting-req per pass) inflates TTFT 4×. LESSON: single-thread scheduler is CPU-sensitive → prefer CACHE-LAYER mechanisms over scheduler-loop ones. Compare all evals to v0_official (not diag).
- PIVOT: v2 = EXCLUSIVE tiering (capacity, latency-free, cache-layer not scheduler-loop). Screening write_back (cfg-writeback). OPS: added run_eval.sh retry-launcher (shared-pool free-flock≠free-node collisions → DRAM_TOO_LOW exit2 → retry).

## 2026-07-08 — v2 XTIER + cfg-writeback RUNNING (parallel, pool freed)
- After ~40min pool contention (4 cells×4 nodes, all high-GPU sibling evals), both grabbed nodes.
- cfg-writeback (write_back config, node 1-2) + v2-xtier (XTIER mechanism, node ondem-3) serving in parallel.
- v2-xtier XTIER CONFIRMED ACTIVE: env SGLANG_XTIER_LAZY=1 propagated; host tier populated (751K cached) which under lazy-backup can ONLY come from the proactive pass → mechanism working. Device-heavy hits 71% (vs baseline 40%) = exclusive-tiering signature. (XTIER_LOG diag not applied due to earlier relaunch hiccup — mechanism runs on defaults PERIOD=4,BATCH=128,WM_FRAC=0.2; assess via summary.json.)
- Early TTFT means healthy (cfg 1.57s, v2 1.65s — NOT blown up like WSAC 12.7s). Awaiting completion for final hit/p99/req-s vs v0_official 0.6217/6326ms/2.78.

## 2026-07-08 — RESULTS: exclusive tiering WINS (validated)
- cfg-writeback (write_back, CONFIG): hit 0.62→0.73 (+18%), p99 6326→4291ms (-32%), req/s 2.78→3.02 (+9%), out +9%. No stall. Logged W&B (config).
- v2-xtier (XTIER mechanism, LOSSLESS): hit 0.62→0.69 (+10%), p99 →4462ms (-29%), req/s 3.02 (+9%), MORE device-heavy (load_back +17% vs +29%). Logged W&B (mechanism). Below write_back's hit due to drop-recompute when proactive pass lags.
- INSIGHT (the contribution): inclusive write_through wastes L1 mirroring L2 → capacity-bound thrash; EXCLUSIVE tiering recovers ~18% hit / -32% p99 / +9% goodput, lossless, latency-free. Hit ceiling ~0.73 (exclusive-capacity limit). write_back sync-evict is NOT a stall in 2-tier (fast host) → edges XTIER here; XTIER async is more robust for slower/heavier regimes.
- v3-xtier-tuned launched (WM_FRAC 0.3/BATCH 256/PERIOD 2 → fewer drops → aim to beat write_back).

## 2026-07-08 — v3-xtier-tuned: over-backup HURTS (honest negative-tuning)
- v3 (WM_FRAC 0.3/BATCH 256/PERIOD 2, MORE proactive backup): hit 0.65 (+5%) < v2 0.69 (+10%) < write_back 0.73 (+18%). load_back +9% (fewest). Logged W&B (mechanism).
- INSIGHT: proactive backup OVER-backs-up → displaces host content prematurely (host always full → each premature backup drops a host item → recompute) → LOWER hit. write_back's backup-ONLY-on-eviction is OPTIMAL for exclusive tiering here; XTIER's async proactive design only helps where the backup tier is slow/eviction heavy (not this fast-host 2-tier).
- ⇒ To BEAT write_back: not more backup, but REUSE-AWARE DEVICE RETENTION (keep high-reuse docs in L1 → device hits → fewer load_backs → lower p99 at write_back's 0.73 hit). v4 candidate.

## 2026-07-08 — NEW BEST: v4-xtier-wm10 BEATS write_back on TTFT (headline SLO metric)
- XTIER WM_FRAC=0.1 (minimal proactive backup → maximally device-exclusive). vs v0_official:
  p50 525ms(-30%), p99 4067ms(-36%), mean 854ms(-25%), req/s 3.02(+9%), out 386 tok/s(+9%), hit 0.69(+11%).
- vs write_back (config): BETTER TTFT on all (write_back p50 585/-22%, p99 4291/-32%, mean 918/-20%). Same req/s.
- Mechanism: minimal backup → more DEVICE-exclusive KV → fewer load_backs (+17% vs write_back +29%) → lower TTFT.
  Trades ~4pp hit (0.69 vs 0.73) for fewer H→D transfers = right trade for p99-TTFT SLO. NOVEL engine-code
  mechanism beating the stock write_back flag on the headline (goodput-under-SLO), lossless. Logged W&B (mechanism).
- Sweep: WM_FRAC 0.1(BEST p99 4067) > 0.2(4462) > 0.3(4479) → less proactive backup = better (device-exclusivity↑, host-displacement↓).
- Next rigor: repeat v4 for error bars; knee sweep λ=4 to confirm goodput-curve shift.

## 2026-07-08 — v4b error-bar repeat (honest conclusion)
- v4/v4b (XTIER wm0.1) consistent: p50 525/539, mean 854/867, hit 0.69/0.69, load_back +17%/+19%. p99 4067/4486 (noisy tail).
- vs write_back (p50 585, mean 918, p99 4291, hit 0.73): XTIER wm0.1 RELIABLY beats wb on p50 (~8-10%) + mean (~6%) via fewer load_backs (device-exclusivity); COMPARABLE on p99 (within ±5% noise); ~4pp lower hit.
- HONEST headline: exclusive tiering (XTIER or write_back) is the WIN over baseline (p99 ~-30%, req/s +9%). XTIER = novel lossless engine mechanism realizing it with a tunable device-exclusivity knob (WM_FRAC), edging write_back on median/mean latency. p99 parity within noise.

## 2026-07-08 — ★ HEADLINE: knee sweep shows the goodput curve SHIFTS UP (measured)
Same-harness knee sweep (one load, λ∈{3,4}), XTIER-v4 (exclusive) vs fresh baseline (write_through):
- λ=3: XTIER p99 4390 / req/s 3.02  vs  baseline p99 5067 / 3.02  (XTIER -13% p99)
- λ=4: XTIER p99 **7794ms ≤ 8s SLO SUSTAINED** / req/s 3.67  vs  baseline p99 **9227ms > 8s SLO VIOLATED** / 3.52
⇒ XTIER sustains λ=4 under the SLO where baseline fails → max-sustainable goodput ~3.35→≥3.67 req/s (+~10%).
This is the charter's HEADLINE win (goodput-under-SLO curve shifts up), HW-measured. Knee harness needed GPU-idle+DRAM gates to survive the hostile shared pool.

## 2026-07-08 — CONCLUSION: XTIER near the lossless frontier
Two ceilings, both reached: (1) hit-rate ceiling 0.73 = exclusive distinct capacity L1+L2 (infinite=0.80 needs >cap, lossless-impossible); (2) device-hit ceiling — docs are radix ROOTS already retained by leaf-first eviction; demoted leaves are low-reuse QA tails → reuse-aware retention adds little. XTIER captures both → +~10% goodput-under-SLO, lossless. Further lossless gains need bigger physical tiers (out of budget). Contribution complete.

## 2026-07-08 — λ=5 knee: XTIER's exact knee pinned (headline refined)
Full XTIER knee (measured): λ=3 p99 4390 / λ=4 p99 7497-7794 (✅ ≤8s, 2 nodes) / λ=5 p99 11012 (✗). Baseline: λ=3 5067 / λ=4 9227 (✗).
⇒ max-sustainable goodput (p99≤8s): XTIER ~3.74-3.9 req/s (knee ~λ4.3) vs baseline ~3.35 (knee ~λ3.4) = +~12-16%. Headline goodput-curve shift MEASURED end-to-end. Contribution complete.

## 2026-07-08 — same-node A/B attempted; harness crash (transient node), abandoned
Custom A/B harness (ab_node.sh) crashed on write_back startup (sigquit — transient node issue; custom harnesses lack eval.sh's NCCL preflight). Freed node. XTIER-vs-write_back stands as characterized (reliable median/mean edge via structural load_back reduction; p99 parity; node-variance caveat noted). CONTRIBUTION COMPLETE: exclusive-tiering insight + XTIER mechanism + measured headline goodput curve (+12-16%, XTIER knee ~λ4.3 vs baseline ~λ3.4) + frontier analysis. All durable (W&B/report/memory/git).

## 2026-07-08 — same-node A/B (write_back half): clarified clean framing
Same-node (0-3) write_back: λ=3 p99 4909 / λ=4 p99 7718 (SUSTAINED, req/s 3.86) — matches XTIER knee (λ4 7497-7794), both beat inclusive baseline (9227, violated). ⇒ CLEANEST framing: EXCLUSIVE TIERING (XTIER mechanism OR write_back config) shifts the goodput curve up; mechanism & config comparable; XTIER = novel lossless engine realization (more robust for slow backup tiers, +device-hits). Same-node XTIER half stalled in harness teardown (pkill launch_server doesn't kill scheduler_TP → GPU not freed for 2nd server); XTIER numbers from dedicated knee runs. Node freed. CONTRIBUTION COMPLETE.

## 2026-07-08 — ★ DEFINITIVE same-node A/B complete (node 0-3): XTIER ≥ write_back
Same node, back-to-back: wb λ3 p99 4909 / λ4 7718(SUSTAIN); XTIER λ3 p99 4184(-15%) / λ4 7599(SUSTAIN). Both beat inclusive baseline (λ4 9227 VIOLATED). No node-variance confound. ⇒ XTIER mechanism matches-or-beats stock write_back config (clear λ=3 p99 edge -15%, λ=4 comparable), both realizing exclusive tiering. Resolves the earlier node-variance caveat. CONTRIBUTION now fully rigorous: novel lossless mechanism ≥ best config, same-node verified.
