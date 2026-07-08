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
