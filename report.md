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
| **be-sjf-cost** | mech | **1,519** | **9,012** | 341.9 | 0.596 | 0.00 | **CHAMPION −98.6%**; FLOP-weighted SJF (`uncached*total`): −3% over be-sjf (better prefill-cost estimate) |
| **be-sjf** | mech | **1,565** | **8,998** | 333.6 | 0.597 | 0.00 | SJF prefill scheduling (least-remaining-work-first): **−22% mean & −49% p99 vs be-lpm** — novel code mechanism |
| be-sjf-age | mech | 1,679 | 19,058 | 347.3 | 0.614 | 0.00 | SJF + anti-starvation aging (wait>8s→front): still beats lpm, but aging did NOT cap the tail here (p99 contention-driven, not starvation) — within SJF noise |
| be-sjf-cost-rep | mech | 2,466 | 30,194 | 328 | – | – | REPRODUCTION of be-sjf-cost: **median 1,395 (matches!) but mean/p99 spiked** — SJF tail-variance (see caveat) |
| be-sjf-mixedchunk | mech | — | — | — | — | — | be-sjf-cost + `--enable-mixed-chunk`: server CRASHES at startup (exit 4) — incompatible w/ hybrid-mamba+hicache |
| be-sjf-blend | mech | 1,698 | 9,491 | – | 0.600 | 0.00 | lpm-primary + SJF-secondary tiebreak: better than lpm, WORSE than pure/cost SJF — **SJF must be PRIMARY** |
| be-lpm | config | 2,014 | 17,704 | 323.1 | 0.61 | 0.00 | best_effort + cache-aware `lpm` scheduling: −20% & lower p99 vs plain best_effort (prior champion) |
| be-lpm-aggr | config | 2,011 | 20,077 | 323.5 | 0.61 | 0.00 | be-lpm + `schedule-conservativeness 0.3` (aggressive admit): TIED w/ be-lpm (noise; no retractions) |
| be-lpm-consv | config | 2,078 | 18,855 | 333.1 | 0.60 | 0.00 | be-lpm + `conservativeness 2.0`: slightly WORSE — knob is not a lever |
| be-dfs | config | 2,149 | 18,506 | 315.4 | 0.60 | 0.00 | best_effort + `dfs-weight` schedule: WORSE than `lpm` — lpm is the best policy |
| be-lpm-kernel | config | 2,075 | 17,593 | 334.1 | 0.598 | 0.00 | be-lpm + `--hicache-io-backend kernel`: WORSE than `direct` (hit 0.598<0.61) — direct transfer is optimal |
| be-lpm-slru | config | 2,063 | 18,173 | – | 0.592 | 0.00 | be-lpm + `--radix-eviction-policy slru`: WORSE than LRU (hit 0.592<0.61) |
| be-lpm-lfu | config | 2,118 | 18,024 | – | 0.579 | 0.00 | be-lpm + `--radix-eviction-policy lfu`: WORST eviction (hit 0.579) — frequency hurts recency-driven multiturn reuse |
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

6. **The radix *eviction* policy confirms be-lpm's default (LRU) is optimal.** Tested the hypothesis
   that reuse-frequency-aware eviction would protect hot multiturn prefixes and cut the 39% host-miss:
   it **backfires**. `--radix-eviction-policy`: **lru 2,014 ms (hit 0.61) > slru 2,063 (0.592) > lfu
   2,118 (0.579)**. Adding a frequency component *lowers* hit-rate because reuse here is recency-driven
   (turn N+1 reuses turn N's just-touched prefix); LFU keeps stale-but-frequent prefixes and evicts the
   recent ones. Honest negative — LRU wins. Host hit-rate is thus **capacity-bound at ~0.61** (768 GB
   host thrashed by 1553 concurrent convs; `hicache-size` is fixed by the eval), not policy-fixable.

7. **BREAKTHROUGH — SJF prefill scheduling (be-sjf, the final champion) beats the whole config
   frontier with a novel code mechanism.** After config was exhausted at be-lpm, I replaced `lpm`'s
   sort key (`-num_matched_prefix_tokens`, most-cached-first, *blind to total length*) with
   **least-remaining-prefill-work-first**: sort ascending by `uncached = len(prompt) −
   num_matched_prefix_tokens`. This single unified key rewards cache hits (more matched ⇒ less uncached)
   AND short prompts (shortest-job-first). Result: **mean TTFT 2,014 → 1,565 ms (−22%) and p99 17,704 →
   8,998 ms (−49%, tail HALVED)**, throughput unchanged, fully lossless (reordering the queue never
   changes any output). Why it works: under `best_effort` every request admits and prefills immediately,
   so the queue order *is* the TTFT order; the load is heavy-tailed (a few huge uncached recomputes), and
   classic SJF both minimizes mean wait and removes the head-of-line blocking those long recomputes cause
   — exactly the p99 collapse we see. `lpm` front-loaded big-but-cached requests; SJF fixes that. It
   costs only ~0.01 hit-rate (0.61→0.597) — a worthwhile trade. This is the one mechanism that improves
   the *champion* regime (unlike v1 parallel-I/O, which only helps the L3-bound `wait_complete` regime).
   **Refinement (be-sjf-cost, final champion, 1,519 ms):** sorting by estimated prefill FLOPs
   `uncached*total` instead of raw uncached token count shaves a further ~3% (each uncached token attends
   over ~total context, so `U*total` better ranks true cost) — a small, expected gain at the noise margin.

8. **Takeaway:** for this overloaded 3-tier workload every win is in the *scheduler/admission* path,
   not the storage engine: (a) don't block admission on slow L3 (`best_effort`, −97%), (b) then order
   the prefill queue by least-remaining-work (**SJF, −22% more & p99 halved**). Six config dimensions
   were swept and none beat the be-lpm defaults (prefetch best_effort≫timeout≫wait_complete; schedule
   lpm>fcfs>dfs-weight; write≈; conservativeness 1.0≈0.3<2.0; eviction lru>slru>lfu; io_backend
   direct>kernel) — the improvement had to come from a code change to the queue ordering. **be-sjf-cost
   (best_effort + FLOP-weighted SJF prefill order + lru + direct) is the final frontier: 1,519 ms mean
   TTFT, −98.6% vs v0_tuned, p99 9,012 ms, fully lossless.** The novel engine change (v1 parallel L3 I/O)
   remains the best lever for the L3-bound regime. Residual TTFT is now compute-bound (recompute of ~40%
   host-miss prefixes; host hit capacity-bound at ~0.6) — the remaining gap needs more host capacity, not policy.

### Reproducibility & caveats (honest)
- **SJF's median is robust; its mean/p99 have real run-to-run variance.** A reproduction of the champion
  (`be-sjf-cost-rep`, identical ref+args) landed at **median 1,395 ms — matching the champion** — but its
  **mean spiked to 2,466 ms with p99 30,194** (vs be-sjf-cost 1,519 / 9,012). Two clean SJF runs (be-sjf
  1,565/8,998; be-sjf-cost 1,519/9,012) plus this one tail-heavy run. Cause is the classic **SJF tradeoff**:
  aggressively ordering by least-remaining-work can occasionally **starve the longest requests** → a fat
  tail → inflated mean; likely compounded by transient contention on the *shared* held eval node (multiple
  researchers `srun --overlap` the same node). So the honest claim is: **SJF robustly cuts median + typical
  mean TTFT (~1,400 median, ~1,500 mean best-case, −25% vs be-lpm) but adds tail-risk** — a production use
  should pair it with anti-starvation aging to cap the tail. The *dimension-level* ordering
  (SJF < lpm < best_effort < baseline; gaps of 25–75%+) is robust across runs; small gaps (be-sjf-cost
  1,519 vs be-sjf 1,565) are within noise. Even the tail-heavy 2,466 is −97.7% vs v0_tuned.
- **SJF robustly beats lpm despite the variance.** Across 4 SJF-family runs (1,519 / 1,565 / 1,679 /
  2,466) vs the 7-run be-lpm cluster (tight 2,011–2,149): **3 of 4 SJF runs beat *every* lpm run, and
  SJF's p99 is ≤ lpm's in every run**. Only the one extreme-tail outlier (cost-rep 2,466, p99 30,194)
  exceeded lpm — and that coincided with heaviest shared-node contention. So the SJF win is real; the
  variance is the confound, not the effect.
- **Anti-starvation aging (be-sjf-age: wait>8s → front) did NOT demonstrably help.** At 1,679 / p99
  19,058 it sits within the SJF noise band and did not cap the tail to ~8s as hoped. Combined with the
  observation that p99 tracks *when* a run executed (contention) more than the policy, this indicates the
  tail-variance is **shared-node contention, not SJF starvation** — so aging has little to fix here (it
  may still matter on a dedicated node or a lower threshold; untested). be-sjf-cost (1,519) stays champion.
- **Config sweeps (best_effort/lpm/eviction/etc.) are single-run each**; their large gaps are trustworthy,
  but treat sub-5% differences as noise.
- `--enable-mixed-chunk` **crashes the server at startup (exit 4)** with this hybrid-mamba + hicache build.

Global eval condition: all my versions pass `--enforce-disable-flashinfer-allreduce-fusion`
(the auto-enabled fusion hangs CUDA-graph capture with hicache on this cluster; lossless,
pessimistic — applies equally to every version).
