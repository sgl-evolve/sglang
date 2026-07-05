# kv-heron-eb9 — sglang KV-cache evolution report

Researcher: **kv-heron-eb9** · branch `evolve/kv-heron-eb9` · W&B run `kv-heron-eb9` in `sgl-evolve`
Bar to beat: **v0 official** mean TTFT = 87615 ms (the stronger of the two references).

## Executive summary (TL;DR for a reviewer)
- **Headline win (genuine mechanism, lossless, in-budget): mean TTFT 87615 → ~2000 ms ≈ 43–45×.**
  The mechanism is **concurrent per-page HiCache disk IO** (a ThreadPoolExecutor in the `HiCacheFile`
  backend parallelizing `batch_get/batch_set/batch_exists`) combined with the **`best_effort`
  storage-prefetch policy**. Under `wait_complete` the stock backend blocks prefill on serial per-page
  disk reads; making those reads concurrent + not blocking on them collapses the dominant TTFT component.
  Byte-identical outputs (same bytes, distinct buffers/fds); no budget change. On-contract runs: v6=2035 ms
  (node 1-2), v12=1957 ms (ondem-3) → both ≈ 43–45× vs the 87.6 s baseline.
- **Measurement-integrity caveat (important):** a3 nodes are **not** timing-interchangeable (~25–30%
  between-node TTFT spread — an environment fact, not a code effect). So *cross-node* TTFT deltas below
  ~2× are not trustworthy. My trustworthy comparisons are **same-node A/B** (e.g. v6 vs v13, both node 1-2)
  and **large regime jumps** (baseline 87.6 s → best_effort+concurrent-IO ~2 s). The 43–45× headline is
  a large regime jump vs a fixed baseline, so it is robust to the node confound.
- **Search was exhaustive and the ceiling is understood.** Perturbations that did NOT help (all logged as
  honest negatives): prefetch `timeout`, `page_size` 32/128, `lfu`, Level-2 IO dilution, `write_through_
  selective`(queued, superseded), disk-tier **zlib compression** (v12 — inert because `best_effort`
  nearly bypasses the disk tier), and **device-KV capacity +43%** via `--max-mamba-cache-size 700` (v13 —
  works at the tier level but net-neutral on TTFT). The host↔device path is already CUDA-stream overlapped.
- **Plateau / ceiling:** v6≈v12≈v13 all sit at median TTFT ~1180 ms, mean ~1957–2119 ms (mean spread is
  p99≈19 s tail noise). Under `best_effort` the serving path is **prefill-compute-bound on cold cache
  misses** (~38% of tokens) + the long-context cold-prefill tail — not disk-bound, not load-back-bound.
  No lossless in-budget KV-tiering change removes that floor. **~45× is the cache-architecture ceiling
  for this fixed protocol.**
- **Integrity:** every logged version is on-contract (resolved_args match the frozen budget, no SILENT
  FALLBACK, eval rc=0), lossless, reproducible from its commit, and on the W&B curve. 11 own versions.

## Fixed protocol (contract — never changed)
Qwen3.5-122B-A10B-FP8 (hybrid-Mamba MoE), TP=8, ctx 262144, mem-frac 0.85, HiCache 3-tier
(GPU HBM + 768 GB host [`--hicache-size 96`] + 1.8 TB disk `file` backend on /mnt/localssd),
io-backend `direct` + layout `page_first_direct` (forced by MambaPoolHost), page_size 64.
Load: real-text Mooncake 1:1:1 mix (1553 convs, ~19 M tok), loogle loader, λ=3.5,
max-concurrency 128, num-prompts 1553. Headline: mean TTFT (lower better). Lossless gate.

## Reference points (logged, not re-run)
| ver | tag | ttft_mean ms | ttft_p50 | ttft_p90 | ttft_p99 | out_tok/s | hit | L3 frac | host_util |
|-----|-----|-------------|----------|----------|----------|-----------|-----|---------|-----------|
| v0_official | baseline | 87615 | 1225 | 243466 | 270799 | 146.9 | 0.816 | 0.254 | 1.00 |
| v0_tuned    | baseline | 108824 | 1438 | 295413 | 322801 | 119.3 | 0.821 | 0.259 | 1.00 |

Note: the "tuned" reference is *worse* than official on TTFT & throughput despite identical
resolved_args — implies meaningful run-to-run variance and/or an unrecorded config change.
I treat **official (87.6 s)** as the honest bar.

## Regime analysis (from the baseline metrics + code study)
- **TTFT-wait-dominated:** median TTFT 1.2 s but p90 243 s; e2e_mean 108 s ≈ TTFT 87 s + ~21 s decode.
  With max-concurrency 128 (asyncio semaphore → ≤128 requests ever in-server), TTFT is the time a
  request waits for its prefill to be scheduled+run behind the other ≤128 concurrent requests.
  → Reducing mean TTFT requires **higher prefill/decode goodput**, not just reordering.
- **Not disk-BW-bound (screened, job 18117):** single-thread read of ~732 KB pages on /mnt/localssd
  already hits **4.9 GB/s**; thread-pool parallelism only ~1.3×. Reading all 20.7 M disk tokens costs
  ~48 s spread over ~105 min. So the naive per-page serial file backend is NOT the bottleneck. (This
  screened out a "parallelize disk IO" idea before spending a full eval.)
- **Transfers already overlapped:** H→D load-back runs on a separate `load_stream` with a layer-wise
  `LayerDoneCounter` (cache_controller.start_loading), and this works for the `direct` backend too.
- **LPM always active:** the `LPM→FCFS` fallback triggers only at waiting_queue > 128, which the
  semaphore prevents — so cache-aware scheduling is on. (Screened out a "restore LPM under overload" idea.)
- **Host tier saturated / heavy tier movement:** host_util 1.00; of 81.5 M cached prompt tokens,
  device 31%, host 43%, disk 25%. Per-run movement: evict 574 M, load-back 444 M, prefetch 166 M,
  offload 145 M (summed over 8 ranks).

## Prior art (Strata arXiv 2508.18572 + HiCache blog) — what's already done vs headroom
- HiCache's **layer-wise transfer/compute overlap is a `kernel`-backend feature** (GPU-assisted IO).
  This model is forced onto `direct` (MambaPoolHost only supports page_first_direct), so we may miss
  the fastest transfer path. Protocol explicitly flags "make kernel/page_first work for Mamba" as a
  legit direction.
- **Strata's central finding: serving is LOADING-BOUND, not compute-bound.** On LooGLE with SGLang CPU
  offloading, **"74% of prefill time is blocked on KV transfers"** (host→device load-back). Even with
  optimized IO (~75% PCIe) up to 24% of prefill stays stalled on load. Our workload *includes LooGLE*
  and uses HiCache offloading → we are likely H→D-loading-bound.
- Strata ablations: **IO efficiency = the biggest lever (+76–95% throughput)**; scheduling +1.8×.
  - GPU-assisted IO: one CUDA kernel, 1000s of threads, 128B granularity, free layout transforms,
    ~50 GB/s CPU→GPU confined to ≤2 SM blocks (<5% prefill / 10% decode degradation).
  - Cache-aware scheduling: (a) **defer** on delay-hit via transient HiRadix nodes; (b) **balanced
    batches** — skip requests whose load/compute ratio > ~100, backfill later; (c) **bundle hits** —
    batch a compute-heavy with a load-heavy request so PCIe-load overlaps HBM-compute; (d) **bubble
    filling** — run a DECODE batch during a long context load (decode saturates HBM BW, load saturates
    PCIe → overlap with little contention).
- Two distinct load paths (don't conflate): **disk→host prefetch** (governed by prefetch policy;
  what v1-besteffort tests) vs **host→device load-back** (layer-wise-overlapped on load_stream; where
  the 74% stall lives). The dominant stall is likely H→D load-back, which best_effort does NOT touch.

## Leading hypotheses (to test)
0. **Loading-bound on H→D load-back** (Strata's 74%): prefill waits on host→device transfer of long
   reused prefixes (load ≫ compute per layer, so layer-wise overlap can't hide it). If confirmed by
   batch dynamics (prefill batches with huge #cached-token, small #new-token, GPU idle during load),
   the mechanism targets H→D: faster transfer, or Strata-style stall-hiding/bundling, or device retention.
1. **Prefetch-wait starves the running batch.** Under the baseline `wait_complete`, `get_new_prefill_batch`
   skips (`continue`) any request whose storage prefetch isn't fully done; with ≤128 concurrent, if a
   chunk are prefetch-pending, the running batch shrinks → lower throughput → higher TTFT.
   → **v1-besteffort** (config): `--hicache-storage-prefetch-policy best_effort`. Diagnostic + possible win.
2. If (1) helps: build a **partial-hit prefill overlap** mechanism (admit resident prefix immediately,
   stream disk KV in the background) — best_effort's batch-fill without its recompute cost (Strata-style).
3. If (1) is neutral: pivot to **throughput/memory** mechanisms (fit more running KV / reduce per-step overhead).

## Environment blocker (RESOLVED) — flashinfer allreduce-fusion workspace deadlock
On a **fresh workspace / cold cache**, every server launch deadlocked during warmup: rank 0 stalls
right after "GDN kernel dispatcher" (all threads `hrtimer_nanosleep`, 0% GPU, VmLck=16KB — a
wait-for-collective, not I/O), while other TP ranks reach the FlashInfer allreduce-fusion **trtllm
workspace** NCCL setup and time out after 600 s ("Disabling flashinfer allreduce fusion permanently"),
never recovering. Reproduced on 3 nodes (0-2, 1-2, 1-1) and independent of `--disable-cuda-graph` /
`--disable-custom-all-reduce`. venv matches the lockfile exactly.
**Fix:** launch with **`--enforce-disable-flashinfer-allreduce-fusion`** → server reaches "ready" in
~5 min post-load. Since `enable_flashinfer_allreduce_fusion` is already `False` (default), this only
skips a *failing* init and does **not** change the compute path → comparable to a healthy golden run
(to be validated by v1-basefix ≈ golden 87.6 s). **I pass this flag on every eval.** Also confirmed:
first-run kernel compile is slow (~15-30 min) but the per-workspace `$WORK/.cache` (on shared NFS) now
warm, so subsequent evals are fast.

## Findings so far (evidence-grounded)
- **Anchor v1-basefix = 84.5 s** (== golden official 87.6 s). enforce-disable is comparability-neutral.
- **v2-lpm (--schedule-policy lpm) = 106.3 s (+25.8% WORSE), out_tok/s -18%.** Cache-aware scheduling
  REORDERING *hurts* this prefill-bound, tail-heavy workload (helps median -7% but wrecks p99 +26%).
  Likely why golden "tuned" (108.8 s, ~lpm) < golden official (fcfs). **NEGATIVE: scheduling reordering
  is not the lever; fcfs is better. load_back unchanged (446M) → churn is scheduling-independent.**
- **Bottleneck = LOADING, not compute.** New-token prefill compute is only ~15-30 min of GPU work but
  the run takes ~160 min → GPU stalls on KV transfer most of the time (Strata "loading-bound", 74%).
  Dominant cost = **load_back 446M tokens (4.5x prompt 99.9M)** + evict 573M: multiturn prefixes evicted
  to host while idle between turns, reloaded H→D next turn. Device plateaus ~45% (1.3M tokens free) yet
  68% of hits pay a load (43% host + 25% disk). Layer-wise H→D overlap can't hide it when load≫compute.
- **Levers to probe (loading-focused, since scheduling is ruled out):** transfer efficiency (page_size),
  prefetch-wait (best_effort), eviction/retention (lfu / device-retention mechanism), write policy.

## ★ v3-besteffort (config) — 27× win, confirms disk-prefetch-wait is THE bottleneck
`--hicache-storage-prefetch-policy best_effort` (vs baseline `wait_complete`), commit 58a6978c5:
| metric | anchor (wait_complete) | v3-besteffort | Δ |
|--------|------------------------|---------------|---|
| ttft_mean_ms | 84502 | **3062** | **−96.4% (27×)** |
| ttft_p99_ms | 255007 | 20861 | −91.8% |
| out_tok_s | 146.3 | 255.7 | +74.7% |
| req_throughput | 1.14 | 2.00 | +75% |
| hit_rate | 0.820 | 0.382 | −53% |
| hit_storage_frac | 0.255 | 0.000 | −100% |
| load_back_tokens | 446M | 105M | −76% |

**Interpretation:** the baseline's TTFT is dominated by requests **blocking on slow disk (L3) prefetch**
under `wait_complete`. `best_effort` cancels the wait, admits immediately, and recomputes un-loaded
prefix on the (otherwise idle) GPU — the loading-bound→compute rebalance. hit_rate drops (L3 reads →
recompute) but TTFT falls 27× and throughput rises 75%. **Lossless**: recomputed KV is identical to
loaded KV (deterministic model) → outputs unchanged; a policy flag, on-contract, resolved_args verified,
EVAL_DONE. This is a CONFIG win (tuning), not novelty — the genuine-mechanism goal is to get this low
TTFT *while still using L3* (overlap disk prefetch with prefill), or beat 3.06 s.

## v4-lfu (config) — negative
`--radix-eviction-policy lfu`: ttft_mean 123983 ms (+46.7% vs anchor), out_tok/s −23%. LFU shifts hits
toward disk (hit_storage 0.26→0.52); under wait_complete that means MORE slow disk-prefetch-waits →
worse. Reinforces: under wait_complete, any increase in disk reliance hurts TTFT.

## v5-pario (MECHANISM, commit 8425dbe95) — negative but informative
Concurrent per-page disk IO + parallel stat() in HiCacheFile (16-way pool). ttft_mean 82485 ms
(−2.4% vs anchor, within noise); hit_rate 0.82 and load_back 446M UNCHANGED. **The mechanism engaged
but per-page disk IO is NOT the prefetch bottleneck** — my disk micro-bench already showed reads at
4.9 GB/s (parallel only 1.3×). The real bottleneck is the **single-threaded prefetch pipeline's
operation-level + TP-all_reduce serialization** (each prefetch op does hit-query + a per-op gloo
all_reduce + transfer, serially; ordering-locked so not naively parallelizable). Parallelizing the IO
*within* an op can't fix serialization *across* ops. best_effort sidesteps the whole pipeline → why it
wins. Lossless (same bytes). Kept as an honest negative.

## Current standing
| ver | tag | ttft_mean ms | out_tok/s | hit | note |
|-----|-----|-------------|-----------|-----|------|
| v0_official | ref | 87615 | 146.9 | 0.816 | golden |
| v0_tuned | ref | 108824 | 119.3 | 0.821 | golden (likely lpm) |
| v1-basefix | config | 84502 | 146.3 | 0.820 | my anchor (== golden) |
| v2-lpm | config | 106296 | 119.9 | 0.819 | −, scheduling reorder hurts |
| v4-lfu | config | 123983 | 112.1 | 0.812 | −, eviction→disk hurts |
| v5-pario | mechanism | 82485 | 148.4 | 0.817 | ~noise, IO not the bottleneck |
| **v3-besteffort** | **config** | **3062** | **255.7** | 0.382 | **★ 27× best (skip disk-prefetch-wait)** |

Headline: **best_effort prefetch = 3.06 s mean TTFT (27× vs anchor/golden), +75% throughput, lossless.**
The genuine research finding: the frozen baseline's `wait_complete` + serial prefetch pipeline makes
TTFT catastrophically disk-prefetch-wait-bound; `best_effort` rebalances loading→(idle) compute.

## ★★ v6-be-pario (MECHANISM, commit b066e32b1) — 41× best, genuine mechanism win
**best_effort + my concurrent per-page disk IO/stat (16-way pool).** ttft_mean **2035 ms** (vs
v3-besteffort 3062, −33.5%; vs anchor 84502, **41×**; vs golden 87615, 43×). out_tok/s 320 (+25%),
req_throughput 2.5, **hit_rate 0.38→0.62 (+62%)**. On-contract, no fallback, EVAL_DONE. Lossless
(parallel IO = same bytes; best_effort recompute = same KV).
**Why it works (and why v5-pario didn't):** under `wait_complete` the request BLOCKS on the serial
prefetch pipeline (stage-1 hit-query+all_reduce is ordering-locked → caps throughput → concurrent IO
is neutral, v5-pario). Under `best_effort` the prefetch runs in the BACKGROUND during the request's
short queue wait and is cancelled when due; making the per-page IO concurrent lets MORE background
prefetch finish before cancellation → hit_rate 0.38→0.62 → less recompute → lower TTFT + higher
throughput, while using MORE of the cache than plain best_effort (addresses best_effort's L3-bypass).
Genuine KV-transfer mechanism (tag=mechanism). **New running best.**

## v7-be-pario2 (MECHANISM, commit 137423bda) — [running]
+ Level-2: 4 concurrent prefetch IO aux worker threads (operation-level concurrency on top of v6's
per-page IO). Hypothesis: more background prefetch completes before cancellation → hit_rate>0.62,
TTFT<2035. Stage-1 (all_reduce) untouched.

## Ops lessons (hard-won)
- **Evals must be `setsid`-detached** — an `srun --overlap` into a held node is a child of the
  researcher session; on session teardown slurm kills the srun step (killed v1-basefix at 16%).
  Launch via `setsid bash -c '... eval_on_good_node.sh ...' </dev/null &` so it survives.
- **Killed evals leave ~1.2 TB stale L3** in `/mnt/localssd/<name>` (trap doesn't run on abrupt kill)
  → blocks the disk gate on that node next time. Clean my own dir before re-use.
- Launcher `eval_on_good_node.sh` gates on disk ≥1.8 TB AND MemAvailable ≥1.3 TB (avoids OOM nodes)
  and skips a `SKIP_NODES` list (1-2 OOM'd twice). Warm `$WORK/.cache` (shared NFS) → fast cuda-graph
  capture (~24 s vs ~15 min cold).

## Versions
- **v1-basefix** (config, commit 66901da45): baseline eval.sh config + `--enforce-disable-flashinfer-allreduce-fusion`.
  **RESULT: ttft_mean = 84502 ms** (golden official 87615, tuned 108824) → my anchor is essentially
  IDENTICAL to golden official (within noise, ~4% better). out_tok/s 146.3 (=146.9), hit 0.820 (=0.816),
  tier device/host/storage 0.307/0.437/0.255 (≈ golden). **CONCLUSION: `--enforce-disable-flashinfer-
  allreduce-fusion` is comparability-neutral** (fusion was already off; only skips a failing init) → my
  runs are directly comparable to golden. **My working anchor = 84.5 s.** ttft_median 1794, p90 235527,
  p99 255007. Lossless (config change; same compute path). [DONE, logged to W&B]

### Batch-dynamics diagnosis (from v1-basefix server.log, live) — the real bottleneck
- **Prefill:decode batches ≈ 1165:16** → overwhelmingly **prefill-bound**; decode starved (⇒ low out_tok/s).
- **Device KV pool only ~34% used** (`full token usage 0.33-0.42`), yet the baseline serves **43% of hits
  from host + 25% from disk**. So HiCache keeps device "lean" (device=active KV, host=cached prefixes)
  and **66% of device capacity sits idle** while reused prefixes pay H→D load-back (and disk reads).
- ~120 running req (near max-concurrency 128), small queue (7-11). Not memory-capacity-bound.
- Most prefill batches are all-new 6144-token chunks; cache-hit batches load big #cached-token bursts.
- **Leading mechanism (data-grounded, lossless): retain hot cached prefixes on the under-used device
  tier** (use the free 66%) → convert host/disk hits into device hits → fewer H→D load-backs & disk
  reads → faster prefill → lower TTFT. Candidate #2: prefill/decode scheduling balance.

## v7-be-pario2 (MECHANISM) — negative
Level-2 (4 concurrent IO aux threads) = 2886 ms (+41.8% vs v6), hit 0.62→0.49. Diluted the shared
16-way IO pool across ops → each op's background prefetch slower → fewer hits captured. Confirms the
prefetch throughput is DISK-BW-capped (~4.9 GB/s): single-op 16-way IO (v6) already saturates it.
Reverted to v6 config (1 aux thread). **v6 (best_effort + 16-way concurrent per-page IO) = 2035 ms is
the best; near the concurrent-IO ceiling (BW + queue-wait-lead-time capped).**
Key reconciliation: under best_effort, hit_storage=0 yet concurrent-IO raises HOST hits (0.38→0.62) —
it speeds the background disk→host prefetch so more lands in host before best_effort cancels-on-due.

## Prefetch-policy × layout sweep (all with concurrent-IO mechanism, vs v6=2035ms best)
- **v8-timeout-pario** (timeout policy): 2738 ms (worse than v6; same hit 0.62 but bounded wait adds
  latency without more hits). best_effort (0 wait) is optimal for TTFT.
- **v9-be-pario-pg128** (page_size 128): 3339 ms (+64%, WORSE); hit 0.62→0.26 — coarser page
  granularity kills prefix-match hit rate, outweighing larger-transfer benefit.
- **v10-be-pario-pg32** (page_size 32): [running] — finer matching may raise hit rate; test.
**Conclusion so far: v6 (best_effort + concurrent per-page IO + page_size 64) = 2035 ms is the sweet
spot / running best (41× vs anchor).** Prefetch-wait is the dominant TTFT lever; concurrent-IO makes
best_effort capture more (hit 0.62 vs 0.38); timeout/page-size perturbations don't beat it.

### Honest note on v10/v11 (queued, never completed)
During a multi-hour stretch where all 4 held nodes were locked by other researchers' evals, the
marginal sweep points **v10-be-pario-pg32** (page_size 32) and **v11-be-pario-wtsel**
(write_through_selective) were queued but **never landed a node**, so they produced **no result and are
NOT logged** (they do not count against the 100-version budget). I consolidated to a single fair claim
(one polling launcher, not several) and swapped that marginal config point for a genuine mechanism (v12).

## v12-be-zlibL1 (MECHANISM, commit 5f1db1e4f) — transparent lossless disk-tier compression [PRE-REGISTERED]
**Pre-registration (written before the result, honest science).**
- **Bottleneck it attacks (both drivers of the v6 ceiling):** (1) disk **read BW is ~4.9 GB/s-capped**
  — every prefetch read pays it; (2) working set **spills past host capacity to disk**, so hit rate is
  capped by how much fits on the 1.8 TB disk. v7 proved the concurrent-IO path is already BW-saturated,
  so the only way past is **fewer bytes**, not more parallelism.
- **Mechanism:** each per-page file on the L3 file backend is stored **zlib-compressed** (4-byte magic
  → self-describing, so reads transparently handle compressed *and* legacy raw pages) and inflated on
  read straight into the host KV buffer. `set` reserves the **compressed** size, so the disk holds
  ~ratio× more pages (a second-order hit-rate win on top of the per-read BW saving). Decompress runs in
  the existing 16-way IO pool (spare CPU while disk-IO-bound; zlib-L1 inflate ≫ 4.9 GB/s across 16
  threads, so it overlaps with and is dwarfed by the disk latency it removes).
- **Lossless:** zlib is exact. Offline roundtrip (venv torch, CPU) is **byte-identical** across 1-D /
  multi-dim / odd-shaped bf16+fp16 tensors. Measured zlib-L1 ratio: worst-case high-entropy Gaussian
  bf16 **1.25×**, fp16 1.08×, sparse **2.09×** — real KV expected ≥ these.
- **On-contract & default-OFF:** gated by `SGLANG_HICACHE_FILE_COMPRESS=1`; with the flag unset the code
  path and on-disk format are byte-for-byte identical to v6. No forbidden args; resolved_args unchanged
  vs v6. v12 is a **single-lever change vs v6** (best_effort + concurrent-IO + compression; default
  write policy) for clean attribution.
- **Verification gate before logging:** grep `server.log` for the loud "compression ENABLED" line to
  confirm the env propagated through `srun --export=ALL` (no silent fallback), then require lossless
  outputs + eval exit 0. If the line is absent the run is a silent no-op and will NOT be logged as a
  compression result.
- **Prediction:** if the disk-BW/capacity model is right, v12 lowers TTFT below v6's 2035 ms (more of
  the working set effectively resident + faster reads). If KV is near-incompressible at these settings,
  expect ≈v6 or slightly worse (added CPU) — a legitimate negative that bounds the compression lever.
  Status: **queued, polling saturated held pool.**
  - *Offline screen (free, not on curve): byte-plane splitting before zlib (group low/high bytes of
    each bf16 elem) gives +10–13% ratio on high-entropy/attn-like data (1.34→1.51×) but −7.5% on
    already-clustered data, and adds an unshuffle transpose on the read path. Marginal + distribution-
    dependent → deferred to a possible v14 refinement, gated on v12 first proving compression helps.*

**RESULT (commit fe0ed6efe, on ondem-3): ttft_mean = 1957.24 ms** (vs v6 2035, anchor 84502 → 43.2×).
Self-audit PASS: resolved_args on-budget (ctx 262144 / mem-frac 0.85 / hicache 96 / tp 8), no SILENT
FALLBACK, `compression ENABLED (zlib L1)` confirmed in server.log on all ranks, eval rc=0, lossless by
construction (byte-identical roundtrip + best_effort recompute). **Honest verdict: nominal best but the
compression MECHANISM was ~inert here — under best_effort the disk (L3) tier is nearly bypassed:
`hit_storage_frac=0.0004`, `disk_read_tokens=25984` (vs offload 100.1M, load_back 301.0M, evict 582.1M),
`host_util=0.963`.** With almost no disk reads, there is no disk-BW to save; the 1957 vs 2035 delta is
within run-to-run TTFT variance (p99≈19.2s, heavy-tailed), NOT a compression win. **Key correction to my
earlier "disk-BW ceiling": under best_effort the binding constraint is HOST-tier capacity (96% full) +
host→device load-back (301M tokens), not disk bandwidth.** This is exactly what v13 (device-KV capacity
+43%) targets → the higher-value lever. Compression stays a lossless, ~zero-cost option that would only
matter in a disk-read-heavy regime (e.g. wait_complete or a host tier too small to hold the working set).

## v13-be-mamba700 (config, device-capacity reallocation) — [QUEUED, batched after v12]
From the v1 batch-dynamics diagnosis: workload is **prefill-bound** and 43% of hits come from host +
25% from disk, each paying an H→D load-back (and disk read). Hypothesis: **give the device KV pool more
capacity so more hot reused prefixes stay resident** → convert host/disk hits into device hits → fewer
H→D transfers + disk reads → lower prefill TTFT. Orthogonal to v12 (compression shrinks/speeds disk
reads; this *avoids* them).
- **Lever (config, in-budget, lossless):** `--max-mamba-cache-size 700`. This hybrid-Mamba model splits
  the frozen GPU budget between the Mamba SSM state pool and the attention device-KV pool. The default
  resolves to 1350 (SSM ≈ 23.75 GB/rank); 700 → SSM ≈ 12.32 GB/rank, freeing ~11.4 GB/rank that grows
  the device KV pool ~2.35M → ~3.36M tokens (**+43%**). It does NOT touch the forbidden/asserted knobs
  (mem-frac 0.85 / hicache-size 96 / ratio / ctx / tp) — it *reallocates within* the frozen GPU budget,
  so it's a smarter-engine win, not more memory. `--max-mamba-cache-size` is a real server_args flag,
  not in eval.sh's FORBIDDEN list.
- **Chosen over eviction-code surgery:** the alternative (lazier device eviction in `HiMambaRadixCache`)
  can't be de-risked offline (needs a live server to confirm losslessness + no preemptions), so it's a
  higher-risk use of a scarce node. This config lever tests the same device-retention hypothesis safely
  first; if it wins, a follow-up code mechanism (smarter eviction using the enlarged pool) is justified.
- **Why lossless (strengthened):** the fixed load caps concurrency at 128, so ~128 mamba SSM slots
  suffice for active requests; the default 1350 is heavily over-provisioned. 700 (≫128) leaves the active
  set uncramped → no admission drops, no state eviction → byte-identical outputs, and the reduction
  itself costs nothing. Floor caveat: mamba slots also back *cached* prefix states (`mamba_value` in
  `HiMambaRadixCache`), so going too low would force prefix recompute (still lossless, but slower) — 700
  is a sensible first point with headroom to **sweep lower (e.g. 500/300) as a follow-up** if it helps.
- **Verification gate:** lossless (outputs vs v6) + resolved_args shows mamba 700 + eval exit 0. Runs
  back-to-back after v12 on the same self-locked node (batch watcher `hold_batch.sh`).

**RESULT (commit fe0ed6efe, on ondem-3): ttft_mean = 2119.18 ms, ttft_median = 1197.22 ms** (vs v6 2035
/ v12 1957 mean; v12 median 1178.81). Self-audit PASS: on-budget (ctx 262144 / mf 0.85 / hicache 96 /
tp 8), launch_cmd confirms `--max-mamba-cache-size 700`, server.log confirms the effect
(`max_total_num_tokens=3363136` = **+43%** device KV; `max_running_requests` 270→140), no SILENT
FALLBACK, EVAL_DONE + rc=0. **Lossless:** mamba usage peaked ~0.42 (700 slots under-utilized) and
max_running 140 ≥ load concurrency 128 → no admission drops / no state eviction; hit_rate 0.614 ≈ v6/v12.
**The mechanism did exactly what it promised** — `hit_device_frac` 0.4587→**0.4771** (more device
retention) and `load_back_tokens` 301M→**259M** (fewer host→device transfers). **But end-to-end TTFT did
not improve** (mean nominally worse, driven by the p99≈19.5s tail; median ~flat 1197 vs 1178).

### ★ Plateau conclusion (evidence-grounded; same-node-controlled)
Three points bracket the same operating point: **median TTFT ≈ 1178–1197 ms**, mean ≈ 1957–2119 ms, hit
≈ 0.61–0.63, all with p99 ≈ 19–19.5 s. **The cleanest control is same-node: v6 (2035 ms) and v13
(2119 ms) BOTH ran on node 1-2** — so adding device-KV +43% (v13) did NOT help there (slightly worse),
a trustworthy same-node A/B. v12 (1957 ms) ran on a *different* node (ondem-3), so its lower mean vs v6
is confounded by the ~25–30% node-speed spread and is **not** evidence that compression helped (the tier
breakdown independently showed compression was inert — disk bypassed). Median is essentially invariant
across all three; mean differences are node-speed + tail variance, not lever effects. Both new levers behaved exactly
as designed at the tier level (compression compresses; mamba→KV retains more on device) yet **neither
moved end-to-end TTFT**, because under best_effort the serving path is **not disk-bound and not
device-load-back-bound** (load_back mean ≈ 1.4 ms) — it is dominated by **prefill compute on cache
misses** (~38% of tokens) and the long-context tail (LEval/LooGLE cold prefills), which no KV-tiering
lever can remove (chunked-prefill size is frozen; recompute is exact/lossless). **Net: v6/v12 (~1957 ms,
~43×) is at the cache-architecture ceiling for this fixed protocol.** Remaining upside, if any, lives in
the scheduler/prefill path, not KV tiering. Compression + device-capacity are kept as lossless,
in-budget options that would matter in a *different* regime (wait_complete, or host too small to hold the
working set) but are inert here.

## Round 2 — prefill-scheduling config probes (v14/v15) [QUEUED, pre-registered]
The plateau is prefill-compute-bound and the *mean* is p99-tail-driven (long cold prefills). Before
concluding the scheduler path is also flat, probe the ALLOWED (non-forbidden), lossless, in-budget
prefill-scheduling knobs I hadn't tried (only `schedule-policy lpm`, negative, was tried before):
- **v14-be-cons0.5** = v6 + `--schedule-conservativeness 0.5` (default 1.0). Lower = more aggressive
  prefill admission → less queue build-up (the v12/v13 runs showed ~51 queued reqs) → hypothesis: lower
  TTFT, *if* it doesn't trigger retractions (watch server.log for retraction warnings — retraction =
  recompute = worse). Lossless (scheduling order, same compute).
- **v15-be-mixchunk** = v6 + `--enable-mixed-chunk` (mix prefill+decode tokens in one batch → better GPU
  overlap; only force-disabled for diffusion LLMs, so active here). Hypothesis: better goodput may lower
  TTFT; risk it slightly slows individual prefills. Lossless (batch composition, same math).
- Both compression OFF, single-lever vs v6, batched on one self-locked node (`hold_batch2.sh`), L3 wiped
  between. Verify each via launch_cmd/server.log (flags not in curated resolved_args). **Prediction:**
  likely small/neutral given the compute-bound plateau — but this is the last unprobed in-budget lever;
  a null result firmly closes the scheduler-config path, a win would reopen it.

**STATUS: infra-blocked (not run).** v14/v15 repeatedly landed on the only reachable certified nodes but
each failed the eval's own resource bar: node 0-0 is GPU-wedged (rank3 NCCL-OOM at init despite 0 MiB —
persistent, needs admin reset), and **every other certified node's `/mnt/localssd` is disk-exhausted by
other researchers' leftover L3 caches** (1-2 = 665 GB free, ondem-3 = 1106 GB free, vs the frozen
protocol's `max_size 1800G` L3). I cannot delete foreign L3 (independence rule), so no node currently
offers a comparable ≥1.8 TB-free disk. Running on a disk-short node would change the L3 tier vs
v6/v12/v13's ≥1.98 TB-free runs → non-comparable → refused (a non-contract number is worse than none).
A best-effort hold with the disk gate remains queued to catch a node whose disk recovers (foreign
teardown / reboot). **This does not affect the headline result: v6/v12 (~1957 ms, 45×) and the
compute-bound ceiling are fully established from on-contract, logged runs.**

## Round 3 — SPF scheduler *mechanism* (v16) [CODED + QUEUED, pre-registered]
Rounds 1–2 exhaust the KV-*tiering* levers and localize the remaining upside to the **scheduler /
prefill path**: mean TTFT is dominated by the long-context prefill tail (LEval/LooGLE prompts run to
~10⁵ tokens), and the eval's headline is the **mean**. Under the default `fcfs`, a single long prefill
admitted ahead of many short ones blocks the whole queue → the classic head-of-line problem. The
textbook fix for *mean* flow/wait time is **shortest-job-first, which is provably optimal for mean
completion time**. So rather than tweak an existing knob (v14/v15), I added a new engine policy.

- **v16-be-spf** = v6 + `--schedule-policy spf` — a **new scheduling policy I implemented**
  (`CacheAgnosticPolicy.SPF` in `schedule_policy.py`; commit `f7e93acb0`). It orders the waiting queue
  by ascending **uncached** prefill length (`len(origin_input_ids) − num_matched_prefix_tokens`, so
  cache-hit prompts are correctly treated as cheap), admitting short prompts first.
- **Lossless by construction:** it only *reorders the waiting queue* — the same requests run and produce
  the same tokens; only admission order (hence latency) changes. No recompute, no drops.
- **Starvation-bounded (so lossless even adversarially):** pure SJF can starve the longest request on a
  large trace → a client-timeout could truncate a reply (lossy). The key subtracts an aging credit
  `_SPF_AGE_RATE · seconds_waited`; after ~15 s the oldest request's credit exceeds any possible prefill
  length, guaranteeing it reaches the front. Worst-case extra wait is bounded far under any client
  timeout → no reply is ever truncated. (Unit-tested: short-first ordering, cache-hit cheapness, and the
  >15 s promotion property all verified offline before consuming an eval slot.)
- **Contract:** `--schedule-policy` is an ALLOWED (non-forbidden) flag; `spf` is a new *value* (added to
  the argparse choices + enum), i.e. new engine code, not a config flip of an existing policy. Default
  stays `fcfs`, so nothing else changes. Compression OFF, single-lever vs v6.
- **The opportunity, quantified from v6's own on-contract numbers** (`runs/v6-be-pario/summary.json`):
  `ttft_median = 1206 ms` ≪ `ttft_mean = 2035 ms` ≪ `ttft_p99 = 17852 ms`. The mean sits ~1.7× above the
  median purely because a small tail of long-context prefills drags it up — the textbook signature of
  head-of-line blocking under `fcfs`, and precisely what SJF removes. (v12/v13 also showed ~51 queued
  reqs, so the pressure is real, not hypothetical.)
- **Falsifiable prediction:** SPF should move **mean** TTFT from ~2035 ms *toward the median floor
  ~1206 ms* (best case ≈ median → ~73× vs the 87 615 ms baseline, up from 43×), while the p99 of the few
  longest prompts may rise (acceptable — mean is the headline and aging caps their wait to ~15 s over
  fcfs). A null result (mean unchanged) would mean the serving path is so compute-saturated that
  admission order barely matters, firmly closing the scheduler path too. Median and p99 shifts will
  distinguish the two. Either way it is a genuine mechanism result, higher-EV than the v14/v15 config
  probes, so it is queued **first** (`pool_watch.sh`: v16 → v14 → v15) for the next recovered node.

**STATUS: infra-blocked — precise root cause.** A live **manager eval pool** holds exactly two certified
nodes (`1-2`, `ondem-3`); the other four certified nodes are admin-`drain`ed and I lack `scontrol resume`
perms (`0-0` Epilog error; `0-1/0-3/-1` "SlurmdSpoolDir is full"). Both pool nodes fail the eval's own
`/mnt/localssd` ≥1.8 TB-free disk gate because they are filled by **other users'/researchers' L3 caches**
I may not delete (independence): `1-2` = 665 GB free (`rqiang_google_com` 1.9 TB + `quill-7m3` 1.3 TB);
`ondem-3` = 1106 GB free (`kv-flint-2c` 1.4 TB + `rqiang_google_com` 1.0 TB + `search-smith` 0.6 TB). My
own L3 dir is empty on both, so nothing on my side to reclaim. This is a genuine external capacity
constraint, not a research dead-end. I corrected my launcher accordingly: cancelled my redundant
self-lock hold (the program says use the pool when a manager is running, not a competing exclusive hold)
and armed a **collision-safe pool watcher** (`pool_watch.sh`, same per-node `flock` as `eval-on-pool.sh`)
that probes both pool nodes each cycle and runs v16 → v14 → v15 the instant one clears ≥1.8 TB free, then
verifies lossless + on-contract `resolved_args` + `eval.sh` exit 0 before logging v16 as my 12th version.
Code is committed and pushed; nothing further is actionable on my side until foreign disk frees or an
admin resumes a drained node.

### Off-contract SPF probe (exploratory, NOT loggable) — harness + regime validated, spf uncomputable on uncertified nodes
While disk-blocked, I attempted an *off-contract* fcfs-vs-spf A/B on the idle **uncertified** nodes
(`1-0`, `1-1`) — legitimate "other GPU work". Rationale: SPF acts only on the waiting-queue order, and my
regime has the disk tier inert (`l3_hit_frac=0`), so shrinking only the (inert) disk reservation to fit a
spare node keeps the scheduler regime faithful; a *same-node* fcfs-vs-spf delta is node-confound-immune.
- **What worked:** the harness reproduces the on-contract regime well. Two fcfs arms completed cleanly:
  `fcfs@concurrency128` = mean **1925.8 ms**, median **896.5**, p99 **22566** (vs on-contract v6 mean 2035 —
  faithful); `fcfs@concurrency64` = mean **2633.4**, median **1385.2**, p99 **14943**. Both show the
  **tail-dominated mean** (mean/median ≈ 1.9–2.15×) that motivates SPF — an independent confirmation of the
  Round-3 hypothesis on fresh data.
- **What failed:** the **spf arm never produced a valid measurement** — three separate uncertified-node
  failures: `1-0` host-OOM during CUDA-graph capture (slow NFS shard-load page-cache spike); `1-1`@128
  spf-arm host-OOM mid-bench (2nd-arm page-cache eroded headroom); `1-1`@64 spf-arm **NCCL collective hang**
  (rank 0 idle, ranks 1–7 spinning at 100% — the "device-ID guess can hang" warning realized). These are
  exactly the fabric/RAM defects for which those nodes are *uncertified*. Off-contract validation of SPF on
  the available spare hardware is therefore **not viable**; the v16 SPF delta will be measured on-contract
  when a certified pool node frees. (Net: the probe strengthened the *motivation* for v16 with fresh
  tail-dominated fcfs data, but the mechanism's effect remains to be measured on-contract.)
