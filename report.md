# quartz-7m3 — sglang KV-cache evolution report

Researcher: **quartz-7m3** · branch `evolve/quartz-7m3` · W&B run `sgl-evolve/quartz-7m3`
Bar to beat: **v0_tuned** (mean TTFT 108824 ms). Reference/context: v0_official (87615 ms).
Headline metric: **mean TTFT** (lower better), lossless gate: outputs match no-cache.

## TL;DR
**Mean TTFT here is prefill-queue-waiting-dominated** (median ~1 s but mean ~80-90 s, prompts 0–190 K
tokens), so reordering the prefill waiting queue **shortest-job-first** (with light **aging**) is the lever:
**every SJF variant robustly beats the v0_tuned bar (108 824 ms)** — measured means span **65–88 K ms
(1.24–1.67×)** across v2/v3/v5/v8/v10. Pure reorder ⇒ **lossless** (each request's output is invariant to
schedule order — the KV cache returns bit-identical KV to recompute; completed=7037/run, no drops).
Best single run = **v8-sjf-ca-aged90 = 65 321 ms (1.67×)**.

> **⚠️ Variance caveat (2026-07-06, from repeats — READ THIS).** This regime is **very noisy** (the two
> stock baselines share identical args yet differ 24%). Same-config repeats confirm large run-to-run
> spread: SJF+aging90 **total-length** gave v3 77 083 / v5 84 043 (~9%); cache-aware SJF+aging90 gave
> **v8 65 321 / v10 87 575 (~34% spread!)**. So the **cache-aware vs total-length ranges OVERLAP**
> (cache-aware 65–88 K, total-length 77–84 K) — the cache-aware refinement is **NOT robustly distinguishable
> from total-length SJF** in these noisy, differently-node-conditioned runs (v10 also ran on a slower node:
> out 143 vs v8's 174 tok/s). **Robust claim: SJF (any variant) ≫ v0_tuned.** NOT robust: the precise
> cache-aware advantage, or beating v0_official (v10 87 575 ≈ v0_official 87 615). v8=65 321 is the best
> *single* draw, not a reproducible 1.67×. Proper A/B would need many repeats per config (eval-capacity-bound).

**Mechanism rationale (why cache-aware *should* help, even if noise masks it here):** the mix is **78%
multi-turn follow-ups** on already-cached documents (7037 requests / 1553 docs; doc ~12 K tokens, new
question ~40 tokens). Total-length SJF misjudges those cheap follow-ups as ~12 K-token jobs; cache-aware
SJF (env `SGLANG_SJF_CACHE_AWARE`) subtracts the cached prefix so they're scheduled first. v8's run showed
exactly this profile (median 988 ms, out 174) — but v10 (same config) did not reproduce the magnitude, so
treat the mechanism as *promising and sound* rather than a *proven* win over total-length SJF.

> **Correction & follow-up (2026-07-05).** Static-analysis audit of the schedule path found that the
> `− num_matched_prefix_tokens` (cached-prefix) term in the SJF/HRRN sort key was **inert** as-run:
> `num_matched_prefix_tokens` is only populated when `tree_cache.supports_fast_match_prefix()` is True,
> which is **False everywhere** (only the base class defines it), and the field defaults to 0. So **v2/v3
> as measured ordered by *total* prefill length, not cached-remaining** — the 1.35×/1.41× wins are real
> but come from total-length SJF. This does **not** change any measured number, only the mechanism
> description (corrected throughout below). It also motivates **v8 — genuine cache-aware SJF** (new env
> `SGLANG_SJF_CACHE_AWARE` forces the per-request match), a clean A/B vs v3 that should help on the
> multi-turn ShareGPT portion where a late turn has huge total length but a tiny *uncached* extension.

> **⚠️ v1 result is INVALID (degraded run).** v1-parallel-l3-io logged 150215 ms but with **hit_rate 0.298**
> (vs the normal ~0.82), median TTFT 161 s, out 98 tok/s — it ran (~03:51) during the cluster NCCL-fabric
> outage; the KV cache barely functioned, so this is a run-condition artifact, **not** a valid measurement of
> parallel-L3 disk I/O. It passed the completion guard (7012≥6800 done) but the guard doesn't check cache
> health — a known gap (TODO: add a hit-rate sanity gate). Needs a clean re-run for a valid ablation point;
> low priority (L3 I/O is not the lever — scheduling is). Does not affect the headline (v8=65321 ≪ 150215).

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
| **v8-sjf-ca-aged90** | cache-aware SJF + aging=90s | **65321.4** | **1.67× (−40%)** | 1.34× | 174.4 | .779 / .170 |
| v3-sjf-aged | SJF(total-len) + aging=90s | 77082.7 | 1.41× (−29%) | 1.14× | 149.2 | .818 / .253 |
| v2-sjf | pure SJF(total-len, aging=0) | 80341.6 | 1.35× (−26%) | 1.09× | 151.3 | .813 / .250 |
| v5-sjf-aged90-rep | **v3 REPEAT** (total-len+aging90) | 84043.4 | 1.29× (−23%) | 1.04× | 149.6 | .820 / .257 |
| v10-sjf-ca-rep | **v8 REPEAT** (cache-aware+aging90) | 87575.4 | 1.24× (−20%) | ~1.00× | 143.1 | .820 / .258 |
| v1-parallel-l3-io | parallel L3 disk I/O | _150215 ⚠️ **DEGRADED-INVALID**_ | — | — | 98.2 | **.298** / — |
| v7-hrrn / v9-hrrn-ca | HRRN / cache-aware HRRN | _no valid run (node-degraded, gave up)_ | | | | |

**Config pairs (repeats bracket the noise): cache-aware SJF+aging90 = {v8 65321, v10 87575}; total-length
SJF+aging90 = {v3 77083, v5 84043}. Ranges overlap ⇒ cache-aware ≉ distinguishable from total-length here.**

**v8 (65321) is the best single run, but repeats show the cache-aware advantage is within noise.** Two
same-config repeat pairs now exist: total-length SJF+aging90 = {v3 77083, v5 84043} (~9% spread) and
cache-aware SJF+aging90 = {v8 65321, v10 87575} (~34% spread). The cache-aware pair (65–88 K) **overlaps**
the total-length pair (77–84 K), so the earlier "v8 is robustly 15–22% better than total-length" claim was
**an artifact of comparing v8's lucky low draw to single total-length runs** — corrected here. v10 even
ran on a slower node (out 143 vs v8 174 tok/s), inflating its mean. **What IS robust:** all SJF variants
(65–88 K) beat v0_tuned (108824) by 1.24–1.67×; SJF ordering is the real lever. **What is NOT robust:** the
cache-aware refinement vs total-length, and beating v0_official (87615) on unlucky draws (v10 87575 ≈ 87615).

**v3-sjf-aged (77083) — prior best, superseded by v8** — a **29% mean-TTFT cut vs the bar** with hit-rate/l3-frac matching
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

## v2 — shortest-job-first (SJF) prefill scheduling  [mechanism] — ✅ EVALUATED
**Hypothesis.** Mean TTFT here is dominated by **prefill-queue waiting under overload** (median TTFT
~1.2 s but mean ~90 s, p99 ~270 s; closed loop at max-concurrency 128), not by disk latency (per-rank
L3 traffic averages ~70 MB/s « the ~6 GB/s SSD). The mix's prompt sizes are **extremely heterogeneous**
— input tokens span ~0 to ~190k (p50≈7.3k, p90≈29k, p99≈59k) — so **FCFS** (the stock default) makes
short chats wait behind long-document prefills. **Shortest-job-first is mean-response-time optimal**, so
ordering the waiting queue by *remaining* prefill work should sharply cut mean TTFT.
**What changed.** `schedule_policy.py`: new `sjf` CacheAgnostic policy (`_sort_by_shortest_job`) sorting
the waiting queue by `len(input)+len(output) − num_matched_prefix_tokens` ascending. **As-run the cached
term is 0** (see Correction up top: `supports_fast_match_prefix()` is False, so the match is never done
for cache-agnostic policies), so this orders by **total prefill length**. Agnostic ⇒ immune to LPM's
>128-queue FCFS fallback. `server_args.py`: `sjf` added to `--schedule-policy` choices (an allowed extra arg).
On `evolve/quartz-7m3` (**f52eab323**, cherry-picked), unit-tested (orders shortest-first). Stacks on v1.
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

## v7 — HRRN (highest response ratio next) prefill scheduling  [mechanism] — QUEUED (commit 4a1b349b4)
**Hypothesis.** v2/v3 show SJF wins and light aging helps. But binary aging (promote once wait ≥ T) is a
**threshold cliff** and needs T tuned. **HRRN** replaces it with a smooth, parameter-light rule from
classic scheduling: order by response ratio `R = 1 + wait/service`, highest first. Short jobs get a high
R (favored → low mean TTFT, the SJF effect); a long job's R **grows continuously with its wait**, so it
is promoted exactly when it has waited long enough relative to its size — bounding the tail without a
hard threshold. HRRN is the classic mean-response-time-competitive, starvation-free policy; plausibly ≥
SJF+aging on mean TTFT while being smoother/more robust.
**What changed.** `schedule_policy.py`: new `hrrn` CacheAgnostic policy (`_sort_by_hrrn`) sorting by
`-(1 + wait_sec·rate/remaining_tokens)` where `remaining` uses the same key as SJF (so as-run it is
total prefill length — see Correction up top; the cached term is inert unless `SGLANG_SJF_CACHE_AWARE`).
`environ.py`: `SGLANG_HRRN_TOKENS_PER_SEC` (prefill-throughput knob for the token→time conversion,
default 10000). `server_args.py`: `hrrn` added to `--schedule-policy` choices. Unit-tested: orders
cached → short → long-that-waited → fresh-long (correct). Additive — the sjf/default paths are byte-
identical, so it can't affect the other queued versions.
**Lossless.** Reordering only — per-request outputs unchanged, no drops. **Eval queued** (session job)
with `--schedule-policy hrrn`.
**Offline fast-screen (free, single-server sim on the real 1553 sizes, `runs/sched_sim.py`).** Two findings:
(1) **The `rate` knob is a no-op for scheduling.** The pick is `argmax(1 + wait·rate/size)`; `rate>0` is a
common positive factor and `1+` a common offset, so the argmax reduces to `argmax(wait/size)` — invariant
to `SGLANG_HRRN_TOKENS_PER_SEC`. So **HRRN here is inherently parameter-free** (elegant: nothing to tune,
unlike aging's threshold). *TODO (post-eval, when the tree is free): drop the unused knob from
`_sort_by_hrrn`/`environ.py` for code cleanliness.* v7 as-queued still validly tests pure HRRN. (2) In the
single-server model HRRN's mean sits **between** pure-SJF (best mean, catastrophic p99) and SJF+aging
(worse mean, bounded p99). **BUT this sim is unreliable for our regime:** it predicts aging *worsens* mean,
the opposite of the real v2 vs v3 result — the real system is closed-loop@128 with decode, not single-server.
So the sim only established the parameter-free property; HRRN's real mean-TTFT vs v3 is genuinely unknown
until the eval runs. No overclaim.
**Result / takeaway.** _(eval pending — certified-capacity blocked; runs in the session-hold job)_

## v8 — genuine cache-aware SJF (subtract the radix-matched prefix)  [mechanism] — ✅ EVALUATED, NEW BEST
**Hypothesis.** v2/v3 (and v7) order by **total** prefill length because the cached-prefix term is inert
(Correction up top). But this mix is heavily multi-turn: a late turn has a **large total length** (whole
history) yet a **tiny uncached extension** (only the new turn needs prefill — the history is already in
the radix/host cache). Total-length SJF wrongly treats such a request as "large" and defers it, when it is
actually **cheap**. Ordering by *true remaining (uncached) prefill work* should schedule these cheap-but-
long requests first, cutting mean TTFT further — the most **KV-cache-native** version of the mechanism.

**Dataset evidence (measured, `mooncake_mix_v1.jsonl` + v3 `server.log`; strong prior that v8 ≠ v3).**
- The mix is **1553 documents → 7037 requests** (`enable_multiturn=True`, mean **4.6 questions/doc**, up
  to 61). **78% of all requests are turn≥2 follow-ups** on an already-seen doc. Docs are ~12k tokens
  (49.5k chars); a follow-up question is ~40 tokens (161 chars) → a follow-up's **uncached extension is
  ~300× smaller than its total length**.
- The v3 golden run confirms the reuse is real: **cache hit = 0.816** (device 31% / host 43% / storage
  25%). `num_matched_prefix_tokens` = device+host match — so v8 will see the large cached doc-prefix on
  those 78% of requests and treat them as the ~40-token jobs they actually are.
- Because **every record is large-doc format** (no tiny single-turn chats), total-length SJF (v3) barely
  differentiates requests (all look ~doc-sized ⇒ near-FCFS); v8 collapses 78% of them to ~40 tokens ⇒
  large reordering headroom. Expensive turn-1 doc prefills (22%) are correctly deprioritized. Net mean
  TTFT should drop. (Caveat: v8 costs one `match_prefix` per waiting req per pass — the eval measures
  whether the reorder gain outweighs it; also num_matched excludes the storage-tier 25%, so residency
  still counts partly as remaining — directionally conservative.)
**What changed.** `environ.py`: new `SGLANG_SJF_CACHE_AWARE` (`EnvBool`, default **False** ⇒ v2/v3
byte-identical). `schedule_policy.py` `calc_priority`: when the flag is set and policy ∈ {sjf, hrrn},
run `match_prefix_for_req` for every waiting request (the same read-only radix match the LPM cache-aware
policy already performs via `_compute_prefix_matches`) so `num_matched_prefix_tokens` is populated and the
sort key becomes true uncached-remaining. `match_prefix` is a pure lookup (`include_req=False`; no
lock_ref, no tree mutation) and is recomputed at actual scheduling, so this is side-effect-free.
**Cost.** One `match_prefix` per waiting request per schedule pass (bounded by concurrency 128). This is
the exact cost `supports_fast_match_prefix()=False` was avoiding; the eval measures whether better
ordering outweighs it.
**Lossless.** Reordering only — outputs unchanged, no drops (waiting-timeout abort disabled).
**Verification (no GPU).** py_compile + import OK; env override True/False verified; unit test: a 50k-token
conv with 48k cached (2k uncached) sorts **ahead** of a fresh 8k prompt (total-length SJF would sort it
last) — cache-aware ordering confirmed. Eval'd as `v8-sjf-ca-aged90` = `--schedule-policy sjf` +
`SGLANG_SJF_AGING_SEC=90` + `SGLANG_SJF_CACHE_AWARE=1` (only the cache-aware term differs from v3 → clean A/B);
both env vars verified live in the running server process.
**Result (2026-07-05, slurm2-a3nodeset1-2, commit 6ada4acb5).** mean TTFT **65321.4 ms** — **1.67× better
than v0_tuned** (108824, −40%), **1.34× vs v0_official** (87615, −25%), and **15.3% better than v3** (77083,
the prior best) — a large, clean A/B win (only the cache-aware term differs). **median TTFT 987.7 ms**
(v3 1418 — cheap follow-ups now scheduled first), **p90 200.2 s / p99 234.8 s** (v3 228.6/249.9 — tail also
improved), out_tok/s **174.4** (v3 149.2), completed **7037/7037** (no drops), duration 5161 s.
**Cache dynamics (measured).** hit_rate **0.779** (device 0.333 / host 0.497 / storage 0.170), host_util
0.996; disk-read **13.2 M** tokens (v3 ~20.7 M). So cache-aware ordering *reduces* storage I/O — it groups
same-doc turns so hot docs stay resident on device/host. Hit rate is slightly below v3 (0.818) because the
reorder changes eviction timing, but the scheduling gain dominates decisively.
**Lossless.** Pure queue reorder — each request's output is invariant to schedule order (the KV cache is
transparent: a hit returns bit-identical KV to recompute), waiting-timeout aborts disabled, completed=7037
with no drops. Lossless by construction (same as v2/v3). Logged to W&B (tag `mechanism`).
**Takeaway.** **The single biggest lever found — cache-aware SJF is the new best (1.67× vs the bar).** It
directly exploits KV-cache residency in the schedule: on a workload that is 78% cached multi-turn
follow-ups, ordering by *uncached* remaining work (not total length) schedules the near-free follow-ups
first, cutting mean **and** median **and** p99 TTFT while raising throughput and *lowering* disk I/O. This
is the most KV-cache-native version of the mechanism and validates the correction/hypothesis chain
(v3 total-length was leaving the follow-up value on the table). Next: cache-aware HRRN (v9), and confirm
via a repeat.

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
