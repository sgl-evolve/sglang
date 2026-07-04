# quartz-7m3 — sglang KV-cache evolution report

Researcher: **quartz-7m3** · branch `evolve/quartz-7m3` · W&B run `sgl-evolve/quartz-7m3`
Bar to beat: **v0_tuned** (mean TTFT 108824 ms). Reference/context: v0_official (87615 ms).
Headline metric: **mean TTFT** (lower better), lossless gate: outputs match no-cache.

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

## v2 — cache-aware shortest-job-first (SJF) prefill scheduling  [mechanism] — READY, eval pending
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
**Result / takeaway.** _(eval pending — see infra note)_

## v3 — optional aging for SJF (bounded tail)  [mechanism] — READY, eval pending
**Hypothesis.** Pure SJF (v2) can starve the largest prompts → p99/tail rises. Aging bounds that:
a request waiting ≥ `SGLANG_SJF_AGING_SEC` is promoted ahead of the shortest-job ordering (FCFS among
the aged), recovering the tail while keeping SJF's mean-TTFT gain. Only worth running if v2's p99
regresses badly.
**What changed.** `schedule_policy.py` `_sort_by_shortest_job` gains an aging branch; `environ.py` new
`SGLANG_SJF_AGING_SEC` (`EnvFloat`, default 0 = pure SJF, so v2's behavior is unchanged). On
`evolve/quartz-7m3` (**f52eab323**), unit-tested (aged reqs promoted FCFS, rest SJF). Lossless (reorder).
Eval with `--schedule-policy sjf` + env `SGLANG_SJF_AGING_SEC=90` (e.g.).
**Result / takeaway.** _(eval pending)_

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
hang-watchdog + **W&B auto-log on success**) is retrying v2-sjf → v1 → v3 in priority order and will log
each the instant the fabric recovers. **0 own versions have reached a valid serve yet — purely
infrastructural; all three mechanisms are implemented, unit-tested, committed, and pushed.**
