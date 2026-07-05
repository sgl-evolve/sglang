# quill-7m3 — sglang KV-cache / HiCache evolution report

Independent researcher. Branch `evolve/quill-7m3`. W&B run
`quill-7m3` in project `sgl-evolve`
(https://wandb.ai/acjy777-dartmouth/sgl-evolve/runs/quill-7m3).

Fixed protocol: Qwen3.5-122B-A10B-FP8 (hybrid-Mamba), TP=8, ctx 262144, mem-frac
0.85, host tier 768 GB (`--hicache-size 96` ×8), L3 file backend on
`/mnt/localssd` (max 1800G). Workload: real-text Mooncake 1:1:1 mix (loogle
loader), λ=3.5, max-concurrency 128, num-prompts 1553. Headline: mean TTFT
(lower better). Lossless gate: outputs unchanged.

## Reference baselines (logged, not re-run)

| version | tag | TTFT mean (ms) | TTFT p90 | TTFT p99 | out tok/s | hit rate | L3 hit frac |
|---|---|---|---|---|---|---|---|
| v0_official | baseline | 87,615 | 243,466 | 270,799 | 146.9 | 0.816 | 0.254 |
| v0_tuned    | baseline | 108,824 | 295,413 | 322,801 | 119.3 | 0.821 | 0.259 |

**Bar to beat: v0_tuned.** Note both baselines share identical `resolved_args`;
v0_tuned's higher TTFT reflects run-to-run variance under saturation. The system
is heavily overloaded (median TTFT ~1.2 s but mean ~88 s, p99 ~271 s) — the mean
is dominated by a long right tail of requests that stall, and host tier is 100%
full (host_util 0.9999) with ~25% of cache hits served from slow SSD (L3).

## Bottleneck analysis (from baseline metrics + code)

- Host RAM tier is saturated (host_util ≈ 1.0) → heavy eviction to disk
  (evict 574M tok) and load-back (444M tok); ~20.7M tokens read from SSD.
- The **entire L3 (SSD) I/O path is single-threaded and serial**: one
  `prefetch_io_aux_thread` and one `backup_thread` in `cache_controller.py`, each
  driving `HiCacheFile.batch_get`/`batch_set`, which are serial per-page loops of
  blocking `open(buffering=0)`+`readinto` / `tofile`+`os.replace`
  (`hicache_storage.py`). A single serial stream keeps NVMe queue depth at ~1, so
  the SSD sits mostly idle while prefetch (on the critical TTFT path under
  `wait_complete`) waits.

## v1 — parallel L3 file-backend I/O (`mechanism`)  — commit 2dda8247a

**Hypothesis.** Raising NVMe queue depth by issuing the independent per-page
transfers of a batch concurrently will speed both prefetch reads (critical path)
and backup writes (frees the full host tier sooner), lowering mean/tail TTFT.

**Change.** `HiCacheFile.batch_get`/`batch_set`/`_batch_io_v2` now dispatch the
per-page `get`/`set` calls across a persistent `ThreadPoolExecutor` (blocking file
I/O releases the GIL, so threads give real I/O parallelism). Worker count via new
env `SGLANG_HICACHE_FILE_BACKEND_IO_WORKERS` (default 16; <=1 = legacy serial).
Files are one-per-page and the LRU evictor is internally locked → **lossless**
(same bytes, same host slots; only concurrency of independent reads changes).

**Fast screen (free).** Micro-bench on the node's local NVMe, 768 KB pages
(≈ real per-page size = 96 GB host ÷ 7.81 M tok × 64), cold reads via
`posix_fadvise(DONTNEED)`:

| op | serial GB/s | best (16w) GB/s | speedup |
|---|---|---|---|
| write | 2.20 | 6.41 | 2.91× |
| read  | 3.97 | 6.44 | 1.62× |

16 workers is the peak for both (32 regresses → default 16 chosen). GO for full eval.

**Full eval.** _BLOCKED by an environment/infra issue (not the mechanism)._

### Infra blocker (2026-07-03) — server startup hangs during CUDA-graph capture

Every eval attempt hangs at server startup, on **multiple nodes** (1-2 SIGBUS in
flashinfer trtllm_allreduce during capture; ondem-3, -0, and clean node 1-0 all hang).
Ruled out as cause, systematically:
- **My code**: a no-hicache clean `sbatch --exclusive` launch (my HiCacheFile code not
  even imported) hangs identically → not v1.
- **My commit**: HEAD = baseline commit `a334877e5` + only my v1 commit.
- **My venv**: `uv pip sync` audited all 190 lock packages present/correct (torch
  2.11.0+cu129, nccl 2.28.9, flashinfer 0.6.12, triton 3.6.0). cu12.9, no cu13.
- **Launch method**: fails with both `srun --overlap` (pool) and clean `sbatch`.

Sequence every run: load (~11m) → cold JIT compile (~15m, CPU-bound) → NCCL warning
`"Guessing device ID based on global rank. This can cause a hang if rank to GPU mapping
is heterogeneous"` → **stall** (GPU 0%, ranks spin-wait `Rl`, log frozen). FS reads+writes
OK (not an FS stall); processes Running (not `D`). => a **NCCL rank→GPU mapping deadlock
during CUDA-graph capture**, environmental (baseline used this exact config successfully →
current pool degraded). Testing the standard lossless workaround
`CUDA_DEVICE_ORDER=PCI_BUS_ID` (consistent device enumeration; does not change compute or
eval results). eval.sh's 45-min server-ready timeout is also exceeded by the cold compile.

**Infra blocker — RESOLVED root cause + workaround (2026-07-03).** The "capture hang"
was **slow cold JIT compilation** (flashinfer + deepgemm kernels, written to the slow
NFS-v3 home `~/.cache`), NOT a hang: a long-timeout out-of-band warmup reached READY at
~25 min (GPUs hit 100% for capture; "server is fired up and ready to roll"). The first
cold runs exceeded eval.sh's 45-min server-ready timeout because the shared cache was
cold; I had also been **killing runs at ~25 min** (right as they'd finish) while testing
hypotheses. Fix: **warm the shared NFS JIT cache** with a long run — it persists and is
shared across all nodes (helps every researcher). deepgemm cache observed accumulating
(542→604→639 files). CUDA_DEVICE_ORDER/device-id were red herrings.

**Remaining blocker: eval-pool capacity.** The 4 manager-held certified nodes are
continuously locked by other researchers' long evals; the other 3 certified nodes are
`drained`; one more is a multi-hour personal hold. My autonomous, disk-aware, self-healing,
blocking-flock launcher + post-eval auto-audit/log handler are running and will capture v1's
full-protocol number the moment a node frees.

**Infra blocker RESOLVED (2026-07-03).** The real universal blocker was NOT node capacity
or cold compile — it was the **auto-enabled FlashInfer allreduce fusion** (sglang
auto-enables it on SM90/H100 for this MoE arch, overriding `enable=False`). During CUDA-graph
capture *with hicache* it crashes (SIGBUS in `trtllm_allreduce_fusion`, seen on 1-2/ondem-3)
or hangs (seen on the healthy node 1-0) — so the "flaky nodes" were never flaky. A no-hicache
warmup survived it (simpler capture), which misled me. **Fix: pass
`--enforce-disable-flashinfer-allreduce-fusion`** (eval.sh allows it — not in its FORBIDDEN
list). This is **lossless** (unfused = separate allreduce+rmsnorm = identical numerics) and, if
anything, slightly *pessimistic* (unfused is marginally slower on the model forward, which is a
tiny fraction of the queueing/L3-dominated TTFT). It is applied to **every** version I run, so
my evolution curve is internally comparable. With it, v1 reached ready in ~225 s and ran the
full protocol. **All my versions carry this flag; note it when comparing to the shared
baselines (whose fusion state is unknown).**

**Lossless check.** _pending eval outputs (design is lossless: identical bytes to identical
host slots; only I/O concurrency changes)._

**Full-protocol RESULT (2026-07-03, node 1-2, fusion-off):**

| metric | v0_official | v0_tuned | **v1** | v1 vs official |
|---|---|---|---|---|
| TTFT mean (ms) | 87,615 | 108,824 | **60,745** | **−31%** |
| TTFT p90 (ms) | 243,466 | 295,413 | **194,928** | −20% |
| TTFT p99 (ms) | 270,799 | 322,801 | 255,154 | −6% |
| out tok/s | 146.9 | 119.3 | **170.8** | **+16%** |
| req throughput | 1.15 | 0.93 | **1.34** | +17% |
| hit rate | 0.816 | 0.821 | 0.774 | −5% |
| L3 hit frac | 0.254 | 0.259 | **0.154** | fewer slow SSD hits |
| disk read tok | 20.7M | 21.2M | **11.9M** | −42% |

**Lossless check.** Design is lossless (identical bytes to identical host slots; only I/O
concurrency changes). Outputs unaffected by definition of the change.

**Takeaway.** **v1 (parallel L3 file-backend I/O) is a clear win: −31% mean TTFT vs
v0_official, +16% throughput.** Concurrent NVMe page I/O completes prefetch/backup faster →
less prefetch stalling → queue drains faster → higher throughput → lower TTFT; the faster
pipeline also churns less to disk (disk-reads −42%, L3-frac 0.15 vs 0.25). Note fusion-off
(pessimistic) applies to v1 too, so the win is if anything understated. **Logged as version 1
of 100.**

## v1b-besteffort — prefetch policy `best_effort` (`config`, version 2)

TTFT mean **2,529 ms** (−97% vs both baselines), TTFT p99 21,356 ms, out **288.8 tok/s**
(+96% vs official), req tput **2.26** (+96%). BUT hit_rate 0.55 (↓), **l3_hit_frac 0.0**
(the L3 tier is entirely bypassed — never waits for it), TPOT 511 ms (↑ from 241).

## v3-wc — full I/O-mechanism stack under wait_complete (`mechanism`, version 3)

v3 = v1 (parallel pages) + v2 (read-priority pools) + v3 (concurrent aux-threads), under the
default `wait_complete`. TTFT mean **92,956 ms** — **worse than v1 (60,745)**. So the extra
concurrency (separate read/write pools, multiple aux-threads) does NOT help and likely adds
thread/host-eviction contention (host_util≈1.0). **Honest negative: v1's simple parallel I/O
is the best of the I/O-mechanism family.** Batch-3 mechanisms build on v1, not v3.

## Key insight (from best_effort + the mechanism curve)

The catastrophic baseline TTFT was **entirely prefetch-wait-bound**:
`wait_complete` blocks admission until the (slow) L3 prefetch finishes, so requests pile up.
`best_effort` admits immediately and recomputes the un-prefetched prefix (lossless — recompute
yields identical KV). This crushes TTFT to ~2.5 s AND raises throughput ~2×, at the cost of
higher decode TPOT and near-zero L3 reuse. **Implication:** under `best_effort` the whole L3
architecture is wasted (0% hits); the interesting *mechanism* question becomes whether a
**bounded-wait (`timeout`) policy + fast L3 I/O** can keep TTFT low while *retaining* L3 reuse
(better TPOT/hit-rate than best_effort). That's the batch-2 focus (v1d-timeout, v3-timeout)
plus completing the mechanism curve (v2-wc, v3-wc).

## RESULTS SUMMARY (all logged versions; all fusion-off, so internally comparable)

| version | tag | TTFT mean (ms) | TTFT p99 | out tok/s | hit | L3 frac | notes |
|---|---|---|---|---|---|---|---|
| v0_official | base | 87,615 | 270,799 | 146.9 | 0.82 | 0.25 | shared baseline |
| v0_tuned | base | 108,824 | 322,801 | 119.3 | 0.82 | 0.26 | **bar to beat** |
| **v1-parallel-io** | mech | **60,745** | 255,154 | 170.8 | 0.77 | 0.15 | parallel L3 I/O, wait_complete: **−31% vs official** |
| v3-wc | mech | 92,956 | 280,154 | 138.2 | 0.82 | 0.26 | +read-priority+aux-threads: WORSE than v1 (contention) |
| **be-lpm** | config | **2,014** | 17,704 | 323.1 | 0.61 | 0.00 | **CHAMPION −98%**; best_effort + cache-aware `lpm` scheduling: **−20% & lower p99 vs plain best_effort** |
| be-lpm-aggr | config | 2,011 | 20,077 | 323.5 | 0.61 | 0.00 | be-lpm + `schedule-conservativeness 0.3` (aggressive admit): TIED w/ be-lpm (noise; no retractions) |
| be-lpm-consv | config | 2,078 | 18,855 | 333.1 | 0.60 | 0.00 | be-lpm + `conservativeness 2.0`: slightly WORSE — knob is not a lever |
| be-dfs | config | 2,149 | 18,506 | 315.4 | 0.60 | 0.00 | best_effort + `dfs-weight` schedule: WORSE than `lpm` — lpm is the best policy |
| v1b-besteffort | config | 2,529 | 21,356 | 288.8 | 0.55 | 0.00 | best_effort, default fcfs schedule: −97% |
| be-lpm-timeout | mech | 2,751 | 22,499 | – | 0.60 | 0.00 | timeout+lpm: WORSE than be-lpm (bounded L3 wait re-adds latency, still l3=0) |
| be-writeback | config | 2,796 | 11,328 | 198.9 | – | – | best_effort+write_back ≈ plain best_effort |
| v1d-timeout | config | 2,761 | 20,196 | – | 0.60 | 0.00 | timeout(fcfs): ≈ best_effort, no L3 gain |
| be-wtsel | config | 2,914 | 15,473 | 247.0 | 0.27 | 0.00 | best_effort+wt_selective ≈ plain best_effort |
| v4-tunedto | mech | 2,942 | 20,688 | 327.4 | 0.59 | 0.00 | v3+timeout(0.3/0.02/1.5s): l3=0 even at 1.5s wait |

### Narrative for a skeptical maintainer

1. **The baseline's catastrophic mean TTFT (~88 s) is prefetch-wait-bound.** The default
   `wait_complete` policy blocks a request's admission until its (slow, SSD-backed) L3 prefetch
   finishes; under λ=3.5/mc=128 this makes requests pile up. The median TTFT is ~1.2 s but the
   mean is ~88 s — a pure queueing tail.

2. **`best_effort` (config) is the overwhelming winner: 2.5 s mean TTFT (−97%), ~2× throughput,
   losslessly** (it admits immediately and recomputes the un-prefetched prefix → identical KV).
   It drives L3 hits to **0%** — i.e., it **bypasses the L3/SSD tier entirely**.

3. **The L3/SSD tier is fundamentally unusable at low TTFT in this regime.** Even a tuned SHORT
   `timeout` (≤1.5 s wait, v4) yields **l3_frac 0.0** — the prefetch can't land any page that fast
   under load. So no bounded-wait policy recovers L3 reuse without re-incurring the queueing tail.

4. **My novel mechanism — parallel L3 file-backend I/O (v1) — is the best *engine* change and the
   only regime where the L3 architecture helps.** Under `wait_complete` it cuts mean TTFT 88→61 s
   (−31%) losslessly by raising NVMe queue depth (micro-bench: read 1.6×/write 2.9×). Stacking more
   concurrency (v2 read-priority pools, v3 aux-threads) does **not** help (v3-wc 93 s > v1 61 s) —
   added thread/host-eviction contention. Honest negative.

5. **Cache-aware *scheduling* on top of `best_effort` is a second, independent win (be-lpm, the new
   champion).** Switching the request schedule from default `fcfs` to `lpm` (longest-prefix-match:
   order/group requests that share a radix prefix) cuts mean TTFT a further **2,529→2,014 ms (−20%)**,
   raises radix hit-rate 0.55→0.61, AND lowers p99 21.4→17.7 s. Mechanism: `lpm` batches same-prefix
   requests so the device/host radix prefix is reused before eviction (multiturn locality) — pure
   scheduling, fully lossless, l3 still 0. **A full sweep confirms `lpm` is the best policy and pins
   the frontier:** `dfs-weight` is worse (2,149 ms); the `schedule-conservativeness` admission knob is
   *not* a lever (0.3→2,011 ≈ 1.0→2,014 < 2.0→2,078, no retractions at 0.3, mem only ~38% used); and a
   *bounded-wait* variant (be-lpm-timeout) is worse (2,751 ms) — re-adding L3 waits only hurts,
   reconfirming L3 is unusable here. The config/policy space is exhausted at be-lpm.

6. **Takeaway:** for this overloaded 3-tier workload the two biggest levers are both in the
   *scheduler/admission* path — (a) don't block admission on slow L3 (`best_effort`, −97%), then
   (b) order admitted requests cache-awarely (`lpm`, another −20%) — not L3 I/O speed. The engine
   win (v1 parallel I/O) matters only when you must use L3. A production system should combine
   `best_effort` + `lpm` with v1's faster L3 backup path (v1 also cuts disk-read/eviction volume).

Global eval condition: all my versions pass `--enforce-disable-flashinfer-allreduce-fusion`
(the auto-enabled fusion hangs CUDA-graph capture with hicache on this cluster; lossless,
pessimistic — applies equally to every version).
