# onyx-7q2 — sglang KV-cache / HiCache evolution report

## Executive summary (for a skeptical maintainer)

**The one validated, robust result: the HiCache storage-prefetch *policy* dominates mean TTFT.**
Switching from the default `wait_complete` to `best_effort` (0-wait admit + recompute the un-loaded tail)
takes **mean TTFT 87615 ms → 2330 ± 243 ms (n=3 reruns, −97.3%; best single 2013 ms), lossless**, on the
fixed 122B-A10B / 3-tier-HiCache / Mooncake-1:1:1 protocol. Throughput +85% (1.15→2.13 req/s), out
146.9→272.6 tok/s. **This is a config/tuning win using a stock sglang flag** (`--hicache-storage-prefetch-policy`),
not new engine code — and I'm labelling it honestly as such (the charter prizes novelty *beyond* tuning).

### ⚠️ Integrity correction (self-audit, supersedes all earlier "engine mechanism" claims)

Earlier versions of this report headlined a **"parallel L3 disk reads" engine mechanism** as the win, plus
a "frequency-aware eviction" ablation. **A post-hoc audit of the raw run metrics + the active code paths
shows all three of my engine changes were on dead or dormant paths and had ~zero effect on any eval.**
The measured differences between my versions were therefore driven by (a) the prefetch-**policy** flag and
(b) run-to-run hit-rate noise — *not* my code. Evidence:

1. **Active radix cache is `UnifiedRadixCache`, not `HiMambaRadixCache`.** server.log: `Tree cache
   initialized: impl=UnifiedRadixCache hybrid_ssm=True hierarchical=True`. My LFU/SLRU eviction edits went
   into `hi_mamba_radix_cache.py`, which is **never instantiated** for this model → **v16/v17 tested
   nothing** (they are best_effort-config replicates). **Retracted as invalid ablations.**
2. **L3/disk serves 0% of cache hits.** The tier-hit breakdown sums to 1.0 with storage = 0 in *every*
   run (device+host = 1.000; `hit_storage_frac = 0.0`), across both timeout and best_effort. L3 is
   *written* (`backuped_tokens_total{storage_backend=file}` ≈ 19M tok/rank = full working set) but never
   *read back* for hits. So my parallel-read change (`HiCacheFile.batch_get`, which *is* on the live file
   KV-read path) operates on a tier that contributes no hits → **negligible effect**. The old "v2→v3
   +31.6% parallel-read win" was **hit-rate variance** (v2 hit 0.442 → v3 hit 0.587), not the thread pool.
3. **The mamba extra-pool IO uses a *separate serial* path** (`batch_get_v2`/`batch_set_v2` →
   `_batch_io_v2`, a plain list comprehension) that my `batch_get`/`batch_set` change does not touch at
   all. Parallel writes (v15) were also off the TTFT critical path. **No TTFT effect.**

Net: **I have contributed no validated engine improvement yet.** The curve's real signal is the
policy-regime win above. The honest lesson (now in my notes): *verify a mechanism is on the active path
AND exercised — via a counter/log or the tier breakdown — before claiming it works.*

### Variance is tail-concentrated, not central (n=3 best_effort reruns + ~5 config-replicate runs)
The *median* TTFT is essentially deterministic — **1282 ± 38 ms (CV 2.9%)** — so the typical request
reproduces to a few percent. The *mean* carries all the run-to-run noise (**2330 ± 243 ms, CV 10.4%**)
because it inherits the cold-recompute tail (p99 15953/18095/18219 ms across the three v4 reruns). The
mean TTFT tracks the **hit rate**, which is itself stochastic under 0-wait admission (v4 hit
0.61/0.51/0.47 → 2013/2374/2603 ms). So the tail and the run-to-run noise are the *same phenomenon*:
stochastic GPU+host hit rate. The −97% policy-regime win is ~40× the mean's std — rock-solid; every
finer per-knob ranking is within this noise.

### Root cause (measured)
Under default `wait_complete`, an L3-hit request blocks in the prefetch-wait state until its entire
prefetch finishes; on this workload that starves the GPU (live `/metrics`: num_running≈15, num_queue≈113,
GPU KV token_usage≈4% — Strata's "loading-bound, not compute-bound"). The baseline mean TTFT (87.6 s) is
dominated by a ~240 s tail of such requests. `best_effort` removes the blocking wait (num_running≈120),
collapsing the tail; on the multiturn mix each turn's prefix is usually already warm in the **host** tier
(the disk tier is bypassed entirely — see point 2 above). **Also lossless:** any un-loaded tail is
recomputed via prefill = exact KV → identical outputs (recompute ≡ no-cache path).

### Startup-bug fix applied to all runs (real, lossless)
The FlashInfer allreduce-fusion NCCL group deadlocks intermittently on init for this MoE model (600 s
c10d timeout); `--enforce-disable-flashinfer-allreduce-fusion` avoids it — lossless (pure perf fusion,
identical numerics), not a frozen/budget knob, negligible TTFT effect. Applied to all my launches.

### Evolution curve (mean TTFT, honest attribution)
v0_official 87615 (bar) → v1 91721 (regression: `wait_complete` GPU starvation) → v2/v3 ~2900–4244
(`timeout`; the v2/v3 gap is **noise**, not the read thread-pool) → **v4 best_effort 2330 ± 243 (n=3),
−97.3% = the win**. v6/v7/v15/v16/v17 are **best_effort-config replicates** (their env-gated "mechanisms"
were no-ops); page-size (v9/v10) and write-policy (v11/v12) are real stock-flag configs, all within noise.
**16 eval points logged; exactly one real effect (prefetch policy).**

---

## Positioning vs the reference bar (Strata arXiv 2508.18572 + HiCache blog)
Both references target the regime I measured — *loading-bound, not compute-bound*. The honest positioning
now: **my result reproduces the HiCache blog's own `best_effort` admission recommendation** and quantifies
it on this protocol (−97%). I did **not** beat the SOTA with a novel mechanism. A notable *diagnostic*
finding worth a maintainer's attention: **on this workload the disk/L3 tier is pure dead weight** — it
absorbs the full 19M-tok/rank working set in writes but serves **0% of hits** under any prefetch policy,
because host RAM (768 GB) already holds the reused multiturn prefixes and best_effort never waits on disk.
That argues the interesting lever here is **GPU+host hit rate / eviction under host pressure**, not the
storage tier the earlier draft chased.

---

Researcher: **onyx-7q2** · branch `evolve/onyx-7q2` · W&B run `sgl-evolve/onyx-7q2`
Bar to beat: **v0_official mean TTFT 87615 ms** (better of the two baselines; v0_tuned 108824 ms is worse
despite identical resolved_args, so I treat the official number as the honest bar).

## Baselines (reference points, not re-run)
| version | tag | mean TTFT (ms) | median TTFT | p90 TTFT | out tok/s | hit | L3 hit |
|---|---|---|---|---|---|---|---|
| v0_official | baseline | 87615 | 1224 | 243466 | 146.9 | 0.816 | 0.254 |
| v0_tuned | baseline | 108824 | 1438 | 295413 | 119.3 | 0.821 | 0.259 |

**Baseline diagnosis.** median TTFT ≈ 1.2 s but mean ≈ 87.6 s and p90 ≈ 243 s → the mean is dominated by
a tail of requests that block on the SSD/L3 prefetch-wait. (Note the baseline reports L3 hit 0.254 — i.e.
the *default* `wait_complete` policy *does* read L3; `best_effort` trades those L3 hits away for 0-wait
admission, which is why storage_frac→0 in all my runs. That trade is the whole win.)

---

## The real finding: prefetch policy is the lever (measured data, corrected attribution)

All numbers below are real, on-contract eval results. What changed in this revision is the *attribution*
of the v2→v3 gap (noise, not code).

### v1 — parallel reads + `wait_complete` (aborted at 36%)  [attempted mechanism]
Ran ~50 min; cumulative throughput 0.86 req/s < baseline 1.15. Live /metrics: num_running≈15,
num_queue≈113, GPU KV usage≈4% — **GPU starvation under wait_complete**. Aborted. This correctly
identified `wait_complete` as the villain and motivated the pivot to `timeout`/`best_effort`. (The
"parallel reads" here were irrelevant — see integrity note.)

### v2 — `timeout` policy (commit 13b0f250c, W&B `v2-serial-timeout`)  [config]
| metric | v2 | v0_official | Δ |
|---|---|---|---|
| **mean TTFT** | **4243.6** | 87615.4 | **−95.2%** |
| p90 / p99 TTFT | 10708 / 24505 | 243466 / 270799 | −95.6% / −91.0% |
| out tok/s | 272.6 | 146.9 | +85.6% |
| hit_rate | 0.442 | 0.816 | −45.9% |
| L3 storage frac | 0.0 | 0.254 | −100% |

`timeout` fixes the GPU starvation (num_running≈109–127, queue≈0–18). Storage hits → 0: the win comes
from *abandoning the slow L3 tier* and spending the freed GPU on recompute. A legit config win.

### v3 — `timeout`, `READ_THREADS=16` (commit 390f0a0f8, W&B `v3-parallel-timeout`)  [was "mechanism"; NOW: config-replicate of v2]
| metric | v2 (thr=1) | v3 (thr=16) | Δ |
|---|---|---|---|
| **mean TTFT** | 4243.6 | **2903.4** | −31.6% |
| hit_rate | 0.442 | **0.587** | +33% |
| tpot mean | 802 | 493 | −38% |

**Corrected interpretation:** v2 and v3 differ only in `SGLANG_HICACHE_FILE_READ_THREADS`, which affects
only `HiCacheFile.batch_get` — a path that reads the L3 tier, which serves **0% of hits**. With storage
hits ≈ 0, the thread pool has nothing to accelerate. The entire v2→v3 improvement is **run-to-run
hit-rate variance** (0.442→0.587, exactly the kind of ±30% swing I later characterized), *misattributed*
in the earlier draft to the thread pool. **Not a mechanism win.**

### v4 — `best_effort` (commit 991317b1f, W&B `v4-parallel-besteffort`, ×3 reruns)  [config, CHAMPION]
| metric | v3 timeout | **v4 best_effort (best single)** | 3-rerun mean |
|---|---|---|---|
| **mean TTFT** | 2903 | **2013.5** | **2330 ± 243** |
| median TTFT | 1844 | 1309 | 1282 ± 38 |
| p99 TTFT | 17654 | 15953 | 17422 ± 1040 |
| hit_rate | 0.587 | 0.608 | 0.53 ± 0.07 |

**The real, robust win.** 0-wait admission removes the prefetch-wait entirely; the multiturn shared prefix
is already warm in host RAM. `best_effort ≫ timeout ≫ wait_complete ≫ baseline` is a regime effect far
larger than the ~10% mean noise. Lossless (recompute of any missing tail = exact KV).

## What the "mechanism"/sweep versions actually tested (corrected)

| version | intended change | reality | mean TTFT | status |
|---|---|---|---|---|
| v6-be-thr32 | read-threads 16→32 | env-gated `batch_get`; L3 serves 0% hits → **no-op** | 2863 | best_effort replicate (noise) |
| v7-be-thr8 | read-threads 16→8 | same dead lever → **no-op** | 2588 | best_effort replicate (noise) |
| v15-be-parwrite | parallel L3 writes (WRITE_THREADS=8) | `batch_set` off critical path; mamba writes use serial `batch_set_v2` → **no-op on TTFT** | 2995 | best_effort replicate (noise) |
| v16-be-lfu | LFU eviction (engine) | edited **dormant** `HiMambaRadixCache` → **no-op** | 2638 | **RETRACTED — invalid ablation** |
| v17-be-slru | SLRU eviction (engine) | edited **dormant** `HiMambaRadixCache` → **no-op** | 2799 | **RETRACTED — invalid ablation** |
| v9-be-page128 | page-size 64→128 | real stock flag | 3119 | real config, within noise |
| v10-be-page32 | page-size 64→32 | real stock flag | 2219 | real config, within noise |
| v11-be-selective | write policy → selective | real stock flag | 2842 | real config, within noise |
| v12-be-writeback | write policy → write_back | real stock flag | 2917 | real config, within noise |
| v13-grace | tuned `timeout` | real stock flag | 4161 | real config — confirms best_effort ≫ timeout |

**Silver lining:** because v6/v7/v15/v16/v17 were effectively identical to v4 (best_effort defaults),
they are **extra reproducibility samples of the champion config** — corroborating the ~10% mean-CV /
tail-concentrated variance finding (best_effort cluster spans ~2000–3000 ms with a stable ~1.3 s median).

**Real config axes (stock flags, all within noise vs v4):** page-size (concave-ish around 64),
write-policy (write_through ≈ selective ≈ write_back), and tuned-`timeout` (v13 ≈ v2, confirming *policy*
not its tuning is the lever). None beats best_effort; none is distinguishable from another at single-run
precision.

## Diagnostic finding worth upstreaming: the L3 tier is dead weight on this workload
Across all runs, the disk tier absorbs the full working set in writes (`backuped_tokens` ≈ 19M tok/rank)
but serves **0% of hits** (`hit_storage_frac = 0`; device+host tier fractions sum to 1.0). Host RAM
(768 GB) already holds the reused multiturn prefixes, and best_effort never waits on disk. So on this
protocol the SSD tier costs write bandwidth for no read benefit — the meaningful lever is **GPU+host hit
rate and eviction under host pressure** (`host_util ≈ 1.0`, `evict_tokens` ≈ 580M ≫ working set → heavy
host thrash), not the storage tier.

## Next steps (a *real*, validated mechanism this time)
Constraints learned: the active class is `UnifiedRadixCache`; the file KV-read path is
`_generic_page_get → HiCacheFile.batch_get` (live) but reads ≈0; the mamba pool IO is serial
`_batch_io_v2`. Any next mechanism must (1) target GPU+host hit rate / host eviction (where the misses
are), (2) be implemented in the **active** class, and (3) be **verified exercised** via a counter/log
before I claim anything. Candidate directions, in priority order:
1. **Host-eviction / admission policy in `UnifiedRadixCache`** to reduce eviction of soon-reused
   multiturn prefixes (the real miss source) — with an added counter proving the new path runs.
2. **Balanced/bundled batching (Strata's missing piece)** in the scheduler to cut prefill bubbles.
3. Only if a counter shows nonzero L3 reads under some policy: parallelize the *live* serial mamba path
   (`_batch_io_v2`) — otherwise it stays dead weight and is not worth touching.

---

### Infra notes (reproducibility) — still valid
- **Self-locked exclusive certified node** for back-to-back evals (submit-gpu-job hill-climb recipe);
  a disk-aware, GPU-preflighting `holder_watcher.sh` runs the queue, advances only on success, records
  bad nodes, and scancels the hold on queue-drain (good citizen). Unique PORT 30729.
- **Init-hang fix:** `--enforce-disable-flashinfer-allreduce-fusion` (FlashInfer allreduce-fusion NCCL
  group deadlocks intermittently on MoE init; lossless, negligible TTFT). Applied to all launches.
- **~30% run-to-run mean-TTFT variance** (tail/hit-rate driven; median stable). Regime wins are robust;
  single-run per-knob rankings are not — always multi-sample before ranking configs.
