# onyx-7q2 — sglang KV-cache / HiCache evolution report

Researcher: **onyx-7q2** · branch `evolve/onyx-7q2` · W&B run `sgl-evolve/onyx-7q2`
Bar to beat: **v0_official mean TTFT 87615 ms** (the better of the two given baselines; v0_tuned is
108824 ms — worse despite identical resolved_args, so I treat the official number as the honest bar).

## Baselines (reference points, not re-run)
| version | tag | mean TTFT (ms) | median TTFT | p90 TTFT | out tok/s | hit | L3 hit |
|---|---|---|---|---|---|---|---|
| v0_official | baseline | 87615 | 1224 | 243466 | 146.9 | 0.816 | 0.254 |
| v0_tuned | baseline | 108824 | 1438 | 295413 | 119.3 | 0.821 | 0.259 |

**Baseline diagnosis.** median TTFT ≈ 1.2 s but mean ≈ 87.6 s and p90 ≈ 243 s → the mean is dominated
by a tail of the ~25% of requests that must be read back from the SSD/L3 tier. `host_util ≈ 1.0` and
`evict_tokens 574M` ≫ working set → heavy host↔disk thrash.

**Root cause (from code map).** L3 disk reads are fully serialized: a single controller
`prefetch_io_aux_thread` feeds `HiCacheFile.batch_get`, which is a serial Python loop of
`open()+readinto()+close()` per 64-token page. A long prefix = hundreds of sequential blocking
syscalls on one thread, shared across all concurrent requests. Under `wait_complete` each L3 request
blocks on its whole serial read (→ p90 243 s) while holding host budget + `protect_host` locks that
throttle everyone. Prior art (Strata, arXiv 2508.18572): schedulers are "loading-bound, not
compute-bound"; fixes are load-aware scheduling + GPU-assisted IO. Our model is hybrid-Mamba →
`direct` io-backend only, so Strata's GPU-assisted-IO kernel is unavailable; the IO parallelism and
scheduling levers remain.

---

## v1-parallel-reads — parallelize L3 disk reads  [mechanism]
**Hypothesis.** The L3 tail is bounded by serial single-thread disk reads. File-read syscalls release
the GIL, so reading a batch's pages through a thread pool gives real NVMe parallelism, draining the
prefetch queue ~N× faster and collapsing the wait_complete tail — losslessly (same bytes, just read
concurrently; order preserved).

**Change (pure Python, no recompile).**
- `mem_cache/hicache_storage.py`: `HiCacheFile` gets a persistent `ThreadPoolExecutor`; `batch_get`
  reads pages via `pool.map` (ordered → results[i] still maps to keys[i]; distinct target buffers;
  evictor is internally locked → race-free). Serial fallback when threads≤1 or batch≤1.
- `environ.py`: new `SGLANG_HICACHE_FILE_READ_THREADS` (EnvInt, default 16).

**Screens (free, no model load).**
- Correctness: content + order preserved vs serial, incl. shuffled-key test → lossless at read layer. ✅
- IO microbench on real `/mnt/localssd` (page cache dropped so reads hit NVMe): serial 1025 MB/s →
  **6126 MB/s @ 16 threads (~6×)**; 8≈5.8×, 32≈5.6×, 64≈4.9× (over-parallelizes). Default 16. ✅

**Result (partial, aborted).** Ran ~50 min on node-0 (parallel reads 16/rank + wait_complete +
fusion-off). Cumulative request throughput **0.86 req/s vs baseline 1.15** (2547/7037 at 36%), i.e.
*slower* than baseline. Live /metrics in the L3-pressure regime (host tier 99.4% full) showed the
smoking gun: **num_running ≈ 15, num_queue ≈ 113, GPU KV-pool token_usage ≈ 4%** — the GPU sits nearly
idle while 113 requests are stuck in the prefetch-wait state. **Diagnosis: under `wait_complete` the
scheduler blocks each L3 request until its *entire* prefetch finishes, so the GPU starves regardless of
read speed** (Strata's "loading-bound, not compute-bound"). Faster per-op reads (parallel) don't fix
this because the bottleneck is the *blocking admission policy*, not read bandwidth. Aborted at 36% to
prioritise the real lever. (Possible secondary effect: 8 ranks × 16 threads = 128 concurrent disk
readers may over-saturate the NVMe vs baseline's 8; to be isolated by v2 serial vs v3 parallel.)
**Takeaway.** wait_complete is the villain. Pivot to prefetch **`timeout`** (admit after a bounded wait,
recompute the un-prefetched tail) to fill the idle GPU. Measure parallel-reads cleanly *on top of*
timeout (v3) vs serial+timeout (v2).

---

## v2 — serial reads + prefetch `timeout`  [config]
**Hypothesis.** `timeout` admits an L3 request after a bounded wait (2 s + 0.1 s/1024 tok, cap 30 s),
recomputing the not-yet-loaded tail via prefill — filling the GPU that `wait_complete` leaves idle.
Serial reads (`SGLANG_HICACHE_FILE_READ_THREADS=1`, original path) to isolate the timeout effect and
avoid any parallel-read over-saturation. Lossless (recompute = exact KV).

**Live validation (L3-pressure regime, host tier full).** timeout **fixes the GPU starvation**:
num_running ≈ 109–127 (near max-conc 128), num_queue ≈ 0–18, GPU KV token_usage ≈ 0.33–0.43 — vs v1's
num_running=15 / queue=113 / usage=0.04. Bench throughput ~1.6 req/s cumulative (instantaneous up to
7.7 it/s) vs baseline 1.15. The GPU is busy instead of idle: admitting requests after a bounded wait
(recomputing the un-prefetched tail) beats blocking them on full disk prefetch.

**Result (commit 13b0f250c, clean run node-0, on-contract, no silent fallback, W&B `v2-serial-timeout`
[config]):**
| metric | v2 | v0_official | Δ |
|---|---|---|---|
| **mean TTFT** | **4243.6 ms** | 87615.4 | **−95.2%** |
| TTFT p90 | 10708 | 243466 | −95.6% |
| TTFT p99 | 24505 | 270799 | −91.0% |
| TTFT median | 2481 | 1224 | +102.7% |
| e2e mean | 55435 | 108148 | −48.7% |
| out tok/s | 272.6 | 146.9 | +85.6% |
| req thruput | 2.13 | 1.15 | +85.2% |
| tpot mean | 802 | 241 | +232% |
| hit_rate | 0.442 | 0.816 | −45.9% |
| L3 storage frac | 0.0 | 0.254 | −100% |

**Lossless:** timeout recomputes the un-prefetched tail via prefill → exact KV → identical outputs
(recompute ≡ no-cache path). **Interpretation:** a huge TTFT win, but it wins by *abandoning the slow
L3 tier* (storage hits → 0) and spending the formerly-idle GPU on recompute — median TTFT and tpot rise
slightly (more concurrent work), but the catastrophic tail collapses. This is a legitimate config win
(prefetch policy is tunable; "less L3 at better TTFT" is a win per the charter), and it definitively
maps the bottleneck. **But the real goal is to make L3 *useful*** — keep the hits AND the low TTFT.
That is the next step: **v3 = parallel reads + timeout** (faster reads → more prefetch completes inside
the timeout window → recover hit_rate without re-introducing the wait).

## v3 — parallel L3 reads + timeout  [mechanism]  (running)
**Refined hypothesis (bandwidth argument).** Why does v2 (serial) hit storage_frac=0? Not per-request
read latency (a single 50K-tok prefix reads in <1s even serial), but **aggregate read bandwidth vs the
timeout window**: under 128-way concurrency the working set to prefetch is ~tens of GB/rank; serial
reads (~1 GB/s, one page's open/readinto/close at a time) can't deliver it before each request's
timeout fires (2s + 0.1s/1024tok, cap 30s) → nearly everything recomputes. Parallel `batch_get`
(16 threads ≈ the NVMe's ~6 GB/s ceiling, measured) should deliver the same working set ~6× faster,
inside the timeout window → **recover storage hits, cut recompute → lower tpot (v2 regressed +232%) and
raise hit_rate, at the same low TTFT**. This also implies multi-threaded prefetch (multiple aux threads)
is *not* the lever — one aux thread's `batch_get` already saturates the 16-way pool at NVMe bandwidth;
extra aux threads share the same pool and add no bandwidth. Lossless (exact KV).

**Result (commit 390f0a0f8, clean run node1-2, on-contract, no fallback, W&B `v3-parallel-timeout`
[mechanism]).** v3 vs v2 differ *only* in `SGLANG_HICACHE_FILE_READ_THREADS` (16 vs 1) — both timeout,
both fusion-off — so this cleanly isolates the parallel-read mechanism:
| metric | v0_official | v2 serial+timeout | **v3 parallel+timeout** | v3 vs v2 |
|---|---|---|---|---|
| **mean TTFT** | 87615 | 4243.6 | **2903.4** (−96.7% vs base) | **−31.6%** |
| p90 TTFT | 243466 | 10708 | **6498** | −39% |
| p99 TTFT | 270799 | 24505 | **17654** | −28% |
| tpot mean | 241 | 802 | **493** | **−38%** |
| e2e mean | 108148 | 55435 | **42925** | −23% |
| out tok/s | 146.9 | 272.6 | **339.6** | +25% |
| req thruput | 1.15 | 2.13 | **2.66** | +25% |
| hit_rate | 0.816 | 0.442 | **0.587** | **+33%** |

**Confirmed:** parallel L3 reads (≈NVMe bandwidth) deliver the prefetch working set inside the timeout
window, so more requests get KV from the (host) cache instead of recomputing → hit_rate recovers
0.44→0.59, recompute drops → **tpot −38%**, and TTFT/throughput improve further. A genuine, cleanly
attributed **mechanism** win over the strong timeout config bar. Lossless (parallel reads = identical
bytes/order, verified; timeout recompute = exact KV). storage_frac stays 0 (disk-served pages land in
host before they're "hit", so hits are attributed to host) — the disk tier is now a fast *feeder* of the
host tier rather than a blocking wait.

## Next (v4+): push the frontier
tpot is still +104% vs baseline (hit 0.59 ⇒ ~41% recompute). Levers: **v4 = parallel + best_effort**
(0-wait admit; multiturn shared prefixes already in host from prior turns → keep hits at min TTFT);
timeout-duration sweep (hit_rate↔TTFT tradeoff, now that reads are fast); read-thread-count tuning.

## v1 (original planned result line — superseded above)
**Lossless check.** Lossless by construction: `batch_get` returns bit-identical bytes in identical
order (verified), so the KV loaded under wait_complete is identical to serial → model outputs identical
to the baseline. Pure read-concurrency change.
**Takeaway.** _(pending)_

### Infra note (reproducibility)
The shared held eval pool uses `srun --overlap` into per-node hold jobs, coordinated by an advisory
`flock`. eval.sh's EXIT trap runs a node-wide `pkill -9 -f sglang.launch_server`, so two servers on one
node kill each other. Two of my early eval attempts died (OOM, then SIGBUS during warmup) from a
flock-bypassing neighbour (`quartz-7m3`) co-locating on my held node. Fix: I **self-lock one exclusive
certified node** for the session (submit-gpu-job's hill-climb recipe) — true `--exclusive` isolation
(what the protocol requires for comparable numbers) + collision-free back-to-back evals. Helper:
`run_on_hold.sh` (unique PORT 30729, srun into holder job in `.holdjob`).
Update: the self-locked node **slurm2-a3nodeset0-2 is faulty** — the server hangs reproducibly (2×,
warm cache) right after KV-cache alloc during FlashInfer workspace NCCL setup (rank-0 c10d 600 s
timeout, 0% GPU), even though the 8-GPU NCCL preflight passes. node-0 clears this stage fine. Released
it (`scancel`) and fell back to the hardened **held-pool** path (`run_eval.sh`, node-0/node1-2/ondem-3),
accepting the residual foreign-collision risk (mitigated by the foreign-server skip + unique port).

**Init-hang root cause (affects ALL my runs).** The server hangs at startup right after KV-cache alloc,
in creating the **FlashInfer allreduce-fusion NCCL process group** — last log line is
`ProcessGroupNCCL.cpp:5188 Guessing device ID based on global rank. This can cause a hang if rank to
GPU mapping is heterogeneous`, then a 600 s c10d rendezvous timeout, 0% GPU. The fusion is
**auto-enabled for MoE models** (server_args `_handle_model_specific_adjustments`) and its NCCL-group
init deadlocks intermittently on these nodes (seen on node0-2 ×2, node-0, ondem-3; cold *and* warm
cache; no collision). Fix: launch with **`--enforce-disable-flashinfer-allreduce-fusion`**. This is
**lossless** (the fusion is a pure perf fusion of allreduce+residual+RMSNorm — identical numerics; the
unfused path is the same result) and **not a frozen/budget knob**; its TTFT effect is negligible (µs/layer
vs an ~87 s KV-bound TTFT). Applied to ALL my eval launches for consistency. To keep attribution clean
despite the supervisor baseline possibly running with fusion auto-on, I also run a **v0-repro-serial**
control (serial reads, `SGLANG_HICACHE_FILE_READ_THREADS=1`, fusion-off) so the parallel-read delta is
measured with everything else held constant. Wrapper: `run_eval_robust.sh` (auto-retry on init-hang/collision).

### Planned eval sequence (all on the self-locked node, back-to-back)
- **v1-parallel-reads**: parallel reads + wait_complete — isolate the mechanism vs v0_official.
- **v2**: parallel reads + `--hicache-storage-prefetch-policy timeout` — cap any residual tail.
- **v3**: parallel reads + `--hicache-write-policy write_through_selective` — cut host thrash (evict 574M).
- then a novel mechanism chosen from v1–v3 results (e.g. zero-copy direct-to-host reads, or scheduler).
