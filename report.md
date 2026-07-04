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

**Why neither reference mechanism applies to my regime (studied both sources):** Strata's two
contributions are (1) *GPU-assisted I/O* (decoupled GPU/CPU layouts to fix fragmented transfers) — but
this hybrid-Mamba model forces `io-backend=direct`, so that kernel is **unavailable**; and (2)
*cache-aware scheduling* (overlap I/O stalls to move from loading-bound → compute-bound) — but
`best_effort`'s 0-wait admission **already** makes my runs compute-bound (num_running≈120), so its
headroom is small. The HiCache blog's data-plane wins (page-first layout, layer-wise overlap) are already
enabled in my config. So the SOTA offers no untried, well-fitting lever for the *already-best_effort*
regime — which is why the open question is whether the **storage tier can be made to contribute at all**
on clean disk (v19), and failing that, whether **host retention** can be improved beyond noise (v18).

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

## Diagnostic finding: the L3 read tier delivered 0 tokens — but possibly a disk artifact (needs a clean-disk run)
Across all my runs, the disk tier absorbs writes (`backuped_tokens` ≈ 19M tok/rank) but the per-request
prefetch log (`HiCache prefetch success ... loaded=N`, the active `UnifiedRadixCache` path) shows
**`loaded=0` AND `matched=0` and `completed_local=0` for EVERY one of the 11936 (v3) / 13312 (v4) prefetch
operations** — the storage prefetch loaded literally zero tokens, on every rank (not a cross-rank all_reduce
MIN artifact). Correspondingly `prefetched_tokens_total` is absent from the metrics and `hit_storage_frac=0`.
So the SSD **read** tier was completely non-functional; parallelizing reads was moot (nothing was read) —
independent of the dead-path bug.

**Strong prior that L3 is NOT structurally dead:** the provided **baseline (`wait_complete`) reports L3
hit fraction 0.254** — i.e. the storage read tier *does* produce hits with this exact model+workload when
the policy waits for the read. So my universal `loaded=0` is a property of my runs (best_effort/timeout on
disk-jammed nodes), not of the code. This shifts weight toward the disk-artifact hypothesis and makes v19
(clean disk) worth running.

**Why loaded=0 — two hypotheses, not yet distinguished:**
1. **Disk-space artifact (likely):** v4's server.log is **32% write-refusals** (`refusing ... to avoid
   OOM/ENOSPC`, 17433/54168 lines) — the node's `/mnt/localssd` sat at the 200 GB min_free watermark
   (foreign leftovers from prior jobs on the shared SSD). If the reusable long-doc prefixes (LooGLE is
   multi-question-per-doc → its long prefix IS reused across turns, overflows the 768 GB host, and needs
   L3) couldn't be written/retained, the prefetch finds nothing → loaded=0. On a **clean high-free-disk
   node** L3 might actually serve that long-context reuse — which is exactly the expensive recompute tail.
2. **Structurally useless:** L3 only ever caches one-shot cold prefixes that are never re-requested → 0
   hits regardless of disk or read speed.

**Decisive next eval (v19): `timeout` policy on a clean ≥2.5 TB-free node, then read `loaded=` in the log.**
If loaded>0 → the storage tier is real and disk-starvation crippled all prior runs (revives the storage
direction, correctly this time). If loaded=0 even on clean disk → L3 is fundamentally dead here and the
only lever is **GPU+host hit rate / host eviction** (`host_util ≈ 1.0`, `evict_tokens` ≈ 580M ≫ working
set → heavy host thrash). Either way it's a real finding. (Caveat on all prior numbers: they were measured
under disk-refusal pressure, so the storage tier was likely crippled throughout.)

## Quantitative analysis: hit rate is the lever (and how big a gain is needed)
Across 11 best_effort-cluster runs, **mean TTFT tracks hit rate: r = −0.85 (r² = 0.72)**, slope
**≈ −263 ms per +0.10 hit rate** (−2629 ms per unit). So mean TTFT is hit-rate-bound, confirming the tail
= misses. But two facts bound what's achievable:
- **Sensitivity is modest** (~11% of the 2330 ms mean per +0.10 hit), and **hit rate itself swings ±0.15
  run-to-run** at fixed config (observed range 0.26–0.61). So a mechanism must deliver a **large** hit-rate
  gain (≳+0.15) — or be measured over multiple samples — to clear the noise. A single eviction-policy run
  (v18) will most likely land within noise; I run it anyway to *honestly close* the eviction question that
  the invalid v16/v17 left open, but I don't expect it to be the win.
- **The one large opportunity is the discarded L3 reuse.** The baseline (`wait_complete`) served **0.254 of
  hits from L3**; best_effort throws all of it away (storage_frac 0) to get 0-wait admission. Recovering
  even part of that ~0.25 predicts ~−660 ms — comfortably above noise. The catch: L3 reads require *waiting*
  (which starved the GPU under wait_complete). `timeout` bounds the wait but still got 0 L3 hits — because
  the **live** read path couldn't deliver the working set inside the timeout window under 128-way concurrency.

**Corrected flagship mechanism (what my parallel-reads idea *should* have been):** parallelize the **live**
storage read path — the serial `_batch_io_v2` (mamba extra-pool) and confirm the KV `batch_get` threading —
and test it under **`timeout`** (where L3 is actually read), not best_effort (where it never is). If faster
reads deliver the prefetch working set inside the timeout window, L3 hits recover → hit rate rises toward
the baseline's 0.8 → TTFT could beat best_effort. This is the honest, on-the-active-path, right-policy redo
of the flagship idea; it must be verified exercised via an L3-read counter before any claim.

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
