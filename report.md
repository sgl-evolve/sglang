# kv-heron-eb9 — sglang KV-cache evolution report

Researcher: **kv-heron-eb9** · branch `evolve/kv-heron-eb9` · W&B run `kv-heron-eb9` in `sgl-evolve`
Bar to beat: **v0 official** mean TTFT = 87615 ms (the stronger of the two references).

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
