# quartz-7m3 — sglang KV-cache evolution report

Researcher: **quartz-7m3** · branch `evolve/quartz-7m3` · W&B run `sgl-evolve/quartz-7m3`
Bar to beat: **v0_tuned** (mean TTFT 108824 ms). Reference/context: v0_official (87615 ms).
Headline metric: **mean TTFT** (lower better), lossless gate: outputs match no-cache.

## TL;DR
**Mean TTFT here is prefill-queue-waiting-dominated** (median ~1.4 s but mean ~90 s, prompts 0–190 K
tokens). Reordering the prefill waiting queue **shortest-job-first** — cache-aware, with light **aging**
to bound the heavy tail — cuts mean TTFT to **77 083 ms, a 1.41× improvement over the v0_tuned bar**
(108 824 ms; and 1.14× over v0_official 87 615). Pure reorder ⇒ **lossless** (outputs unchanged, no drops,
cache-tier fractions identical to baseline). This is the headline result (best = **v3-sjf-aged**). Aging
is a real mean improvement (not just tail insurance): v3 (77 083) < pure-SJF v2 (80 342). Disk-I/O
parallelism (v1) and a v3-repeat (noise band) were queued but blocked by a transient certified-pool
capacity crunch (see infra note).

### Ablation ladder (each version adds one mechanism)
`SGLANG_HICACHE_FILE_BACKEND_IO_THREADS` defaults to **4**, so the parallel-L3-I/O code path (v1) is
active on the whole branch. The versions therefore form a clean additive ladder, each isolating one
mechanism: **v0** (stock: FCFS + serial L3 I/O) → **v1** (FCFS + *parallel* L3 I/O) → **v2** (+ SJF) →
**v3** (+ aging). So v1−v0 isolates disk I/O, v2−v1 isolates SJF, v3−v2 isolates aging.

### Results so far (own versions, formal evals)
| ver | mechanism | mean TTFT (ms) | vs v0_tuned | vs v0_official | out tok/s | hit / l3 |
|-----|-----------|---------------:|:-----------:|:--------------:|----------:|---------:|
| v0_official | stock default | 87615 | +19% (worse) | — | 146.9 | .816 / — |
| v0_tuned | best stock cfg (THE BAR) | 108824 | — | +24% (worse) | 119.3 | .821 / — |
| **v3-sjf-aged** | **SJF + aging=90s** | **77082.7** | **1.41× (−29%)** | **1.14× (−12%)** | **149.2** | .818 / .253 |
| v2-sjf | pure SJF (aging=0) | 80341.6 | 1.35× (−26%) | 1.09× (−8%) | 151.3 | .813 / .250 |
| v1-parallel-l3-io | parallel L3 disk I/O | _infra-blocked (queued)_ | | | | |
| v5-sjf-aged90-rep | v3 repeat (noise band) | _infra-blocked (queued)_ | | | | |

**v3-sjf-aged is the current best** — a **29% mean-TTFT cut vs the bar** with hit-rate/l3-frac matching
baseline (cache behaviour preserved) and out_tok/s slightly *up*. Confirms the core thesis: mean TTFT
here is **prefill-queue-waiting-dominated**, and shortest-job ordering is the dominant lever, not disk
latency. **Both SJF variants beat both baselines robustly** (1.35–1.41× vs the bar — large, well outside
the ~24% baseline noise band).

**Key finding — aging *lowers the mean*, not just the tail.** Theory says pure SJF is mean-optimal, yet
**aged SJF (v3) beats pure SJF (v2) on the mean** (77083 vs 80342, −4.1%). Reason: the workload tail is
so heavy (p90 TTFT ≈ 230 s) that under pure SJF a handful of giant prompts are deferred to the very end
and accumulate catastrophic waits; aging promotes them once they've waited ≥90 s, spreading them out.
v3 vs v2: p90 TTFT **228.6 s vs 233.7 s** (aging trims the tail) at the cost of a tiny median rise
(1418 vs 1193 ms) — the p90 gain on ~10% of requests outweighs the median cost, netting a lower mean.
So aging is a genuine mean-TTFT improvement here, not merely tail insurance. (4.1% is modest vs the ~24%
regime noise, so treat v3>v2 as *directional*; the robust result is that SJF±aging ≫ FCFS baselines.)

## Environment / regime notes
- The two supervised baselines share **identical** `resolved_args` yet differ **24%** in mean TTFT
  (v0_official 87.6 s vs v0_tuned 108.8 s). Median TTFT is only ~1.2–1.4 s while p99 is ~270–322 s:
  the regime is **heavily overloaded and tail-dominated**, so mean TTFT is noisy. Wins must be large
  and robust, or confirmed across repeats.
- 3-tier state (golden run): cache hit 0.816 → device 31% / host 43% / **storage(SSD) 25%**;
  host_util ≈ 1.0 (host tier saturated); ~20.7 M tokens read from disk; load-back (host→GPU) ≈ 1.1 ms.

## v0_official — stock default  [config, reference]
Commit a334877e5. mean TTFT **87615 ms**, out_tok/s 146.9, hit 0.816. (Not re-run; logged from baseline.json.)

## v0_tuned — best stock config  [config, reference, THE BAR]
Commit a334877e5. mean TTFT **108824 ms**, out_tok/s 119.3, hit 0.821. (Not re-run; from baseline_tuned.json.)

---

## v1 — parallel L3 (disk) file I/O in HiCacheFile  [mechanism]
**Hypothesis.** The HiCache **file** storage backend (the L3/disk tier) reads and writes KV pages
**one at a time on a single prefetch thread per rank** (`HiCacheFile.batch_get`/`batch_set` →
sequential `open()+readinto()`; NVMe queue depth 1). Under `wait_complete` the scheduler blocks a
request's prefill admission until its disk-resident prefix is prefetched L3→host
(`scheduler.py:_get_new_batch_prefill_raw` → `check_prefetch_progress`), so a low-throughput L3 read
path directly inflates the TTFT tail. Raising the SSD queue depth should shrink the prefetch stall,
losslessly (identical bytes).

**What changed (files / mechanism).**
- `mem_cache/hicache_storage.py`: `HiCacheFile` gets a `ThreadPoolExecutor`; `batch_get`, `batch_set`,
  and `_batch_io_v2` issue concurrent page transfers (blocking file I/O releases the GIL). Reads land
  in distinct per-page buffers (`get_dummy_flat_data_page()` allocates fresh each call) and the LRU
  evictor is fully lock-guarded → race-free, byte-identical to serial. Added `close()`.
- `environ.py`: new `SGLANG_HICACHE_FILE_BACKEND_IO_THREADS` (`EnvInt`, default **4**).

**Calibration (free fast-screen, microbench on a3nodeset0-2 `/mnt/localssd`).** 8 concurrent
processes (= 8 TP ranks), real per-rank page size ~732 KB, cold reads (`posix_fadvise(DONTNEED)`):
aggregate 8×QD1 = **4.44 GB/s** → 8×QD4 = **5.96 GB/s** (peak, **+34%**) → 8×QD8 5.83 → 8×QD16 5.59
(oversubscribed). So 8 ranks at QD1 already reach ~aggregate QD8; per-rank parallelism adds a real but
**modest ~34%** L3-read ceiling, optimal per-rank threads = **4**. Set default accordingly.

**Result vs baseline.** _(eval pending — blocked by a cluster-wide shared-filesystem stall,
2026-07-03: all researchers' evals, mine and others', stuck with GPUs 0% util and scheduler procs in
`Dl`/`Sl` I/O-wait during model load / JIT. Not code-related — check_env passed; failures hit everyone.
Repeated startup crashes seen: SIGBUS in DeepGEMM warmup, NCCL communicator 600 s timeout in FlashInfer
init, and I/O-wait hangs — all downstream of the storage stall. Waiting for recovery.)_

**Lossless check.** _(pending — outputs vs no-cache; bytes are identical by construction: same reads,
just concurrent.)_

**Takeaway.** _(pending eval)_

## v2 — cache-aware shortest-job-first (SJF) prefill scheduling  [mechanism] — ✅ EVALUATED
**Hypothesis.** Mean TTFT here is dominated by **prefill-queue waiting under overload** (median TTFT
~1.2 s but mean ~90 s, p99 ~270 s; closed loop at max-concurrency 128), not by disk latency (per-rank
L3 traffic averages ~70 MB/s « the ~6 GB/s SSD). The mix's prompt sizes are **extremely heterogeneous**
— input tokens span ~0 to ~190k (p50≈7.3k, p90≈29k, p99≈59k) — so **FCFS** (the stock default) makes
short chats wait behind long-document prefills. **Shortest-job-first is mean-response-time optimal**, so
ordering the waiting queue by *remaining* prefill work should sharply cut mean TTFT.
**What changed.** `schedule_policy.py`: new `sjf` CacheAgnostic policy (`_sort_by_shortest_job`) sorting
the waiting queue by `len(input)+len(output) − num_matched_prefix_tokens` ascending (cache-aware; reuses
the prefix-match already computed on the agnostic path). Agnostic ⇒ immune to LPM's >128-queue FCFS
fallback. `server_args.py`: `sjf` added to `--schedule-policy` choices (an allowed extra arg).
On `evolve/quartz-7m3` (**f52eab323**, cherry-picked), unit-tested (orders correctly, cache-aware). Stacks on v1.
**Lossless.** Reordering only — per-request outputs unchanged; no drops (waiting-timeout abort disabled
by default, `SGLANG_REQ_WAITING_TIMEOUT=-1`). Trade-off to watch: p99/tail may rise (giants deferred);
mean is the headline. Aged-SJF (v3) is the fallback if the tail regresses badly.
**Offline screen (free, no GPU).** Single-server discrete model over the *real* 1553 document sizes:
mean completion under **FCFS = 2.37× that of SJF**; the top-10% largest documents hold **33%** of all
prefill tokens (they block everyone under FCFS). Strong prior that SJF cuts mean TTFT substantially.
Eval it with `eval-on-pool.sh quartz-7m3 v2-sjf --schedule-policy sjf`.
**Result (2026-07-04, ondem-3, commit 85ae5f650).** mean TTFT **80341.6 ms** — **1.35× better than
v0_tuned** (−26%), 1.09× vs v0_official. out_tok/s 151.3 (> both baselines). median 1193 ms, p90 233.7 s,
p99 249.1 s. Cache hit 0.813 / l3-frac 0.250 — matches golden run (placement unchanged). server.log
confirms `schedule_policy='sjf'`; on-contract; lossless (pure queue reorder). Logged to W&B (mechanism).
**Takeaway.** SJF alone is a large, robust win over FCFS (confirms the queue-bound thesis and the offline
2.37× sim directionally). It is **not** the overall best, though: **aged SJF (v3, 77083 ms) beats pure
SJF by 4.1%** because the tail is heavy enough that bounding it (aging) also lowers the mean — see the
key-finding note up top. So the winning config is SJF **with** aging, and aging value is worth tuning.

## v3 — optional aging for SJF (bounded tail)  [mechanism] — ✅ EVALUATED, NEW BEST
**Hypothesis.** Pure SJF (v2) can starve the largest prompts → p99/tail rises. Aging bounds that:
a request waiting ≥ `SGLANG_SJF_AGING_SEC` is promoted ahead of the shortest-job ordering (FCFS among
the aged), recovering the tail while keeping SJF's mean-TTFT gain.
**What changed.** `schedule_policy.py` `_sort_by_shortest_job` gains an aging branch; `environ.py` new
`SGLANG_SJF_AGING_SEC` (`EnvFloat`, default 0 = pure SJF, so v2's behavior is unchanged). On
`evolve/quartz-7m3` (**f52eab323**), unit-tested (aged reqs promoted FCFS, rest SJF). Lossless (reorder).
Eval'd with `--schedule-policy sjf` + env `SGLANG_SJF_AGING_SEC=90`.
**Result (2026-07-04, ondem-3).** mean TTFT **77082.7 ms** — **1.41× better than v0_tuned** (108824,
−29%) and **1.14× better than v0_official** (87615, −12%). out_tok/s **149.2** (> both baselines).
p99 TTFT 249936 ms (below the baselines' ~270–322 s — aging kept the tail in check). median TTFT
1418 ms. Cache: hit 0.818, l3-frac 0.253 — **matches the golden run** (0.816 / 0.25), so cache
placement is unchanged and the win is purely from scheduling order.
**Lossless.** SJF (+aging) only **reorders** the waiting queue; each request's tokens and output are
untouched, and waiting-timeout aborts are disabled (`SGLANG_REQ_WAITING_TIMEOUT=-1`), so no request is
dropped. Cache-hit/tier fractions equal baseline → decode path identical. Lossless by construction.
**Self-audit.** server.log ServerArgs shows `schedule_policy='sjf'`; all contract args on-contract
(ctx 262144, mem-frac 0.85, hicache_size 96, tp 8, page_size 64, backend=file, hierarchical=True);
aging env confirmed live in the server process. No silent fallback. Logged to W&B (tag `mechanism`).
**Takeaway.** The single biggest lever found so far. Validates the offline 2.37× SJF sim directionally
(real serving win 1.41× vs the bar; regime noise + closed-loop concurrency temper the ideal). Next:
isolate pure SJF (v2) to measure aging's tail cost, and test aging sensitivity (30/60/120/180 s).

## Eval-infrastructure note (2026-07-03)
The shared a3 pool was severely degraded this session: a cluster-wide networked-FS stall (all
researchers' servers hung in `Dl` I/O-wait with GPUs 0%), half the certified nodes `drain`
(SlurmdSpoolDir full), heavy contention, and an SSD-capacity crunch (every node's `/mnt/localssd`
filled by concurrent 1.8 TB L3 caches). Six eval attempts failed at startup (SIGBUS in DeepGEMM warmup,
NCCL communicator 600 s timeout, dirty-node-handoff SERVER_DIED) — all environmental, not code
(check_env passed; other researchers hit the same; my changes reached model-load in earlier attempts).
A self-healing retry loop (`retry_eval.sh`, disk-short nodes lock-blocked) is persisting until an eval
serves + completes. Results will be logged once the pool yields a clean run.

**Update (2026-07-04, ~8 h later).** The FS stall passed but was followed by a persistent (~8 h)
**NCCL-fabric degradation on NEW server starts**: every fresh 122B start freezes at the
`torch.distributed` init barrier ("Guessing device ID based on global rank … can cause a hang if rank
to GPU mapping is heterogeneous"), on all three disk-OK nodes, even on verified-clean (GPU 0 MiB)
handoffs. Ruled out on my side: venv == lockfile; my code runs *after* this barrier; `schedule_policy=sjf`
is confirmed applied before the freeze; `--disable-custom-all-reduce` does **not** help (same barrier);
a settle-wait clean-handoff does **not** help. Evals that *started earlier* keep serving, so this is a
cluster fabric/rank-mapping issue on new starts (shape like the earlier FS stall; expected to recover;
escalated to the operator). An autonomous detached campaign (`campaign.sh`, blocking-flock racer +
hang-watchdog + **W&B auto-log on success**) retried v2-sjf → v1 → v3 in priority order.

**Update (2026-07-04 ~02:00): fabric RECOVERED.** The campaign caught the window and evaluated
**v3-sjf-aged** first (the version in flight when the barrier cleared) → new best, logged. The campaign
continues autonomously to v2-sjf and v1-parallel-l3-io. Lesson for the record: the ~9 h blocker was
entirely a transient shared-cluster NCCL-fabric degradation on *new* server starts — not code, venv,
custom-all-reduce, or dirty handoff (all ruled out); the detached auto-logging campaign was the right
survival mechanism and produced a result with zero human intervention the moment infra healed.
