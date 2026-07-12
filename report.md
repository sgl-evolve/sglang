# Research report — **kleinrock** (sglang KV-cache, v0.31)

**Researcher name:** `kleinrock` (Leonard Kleinrock — queueing theory; goodput@SLO under open-loop
Poisson load *is* a queueing problem). Branch `evolve/kleinrock`; W&B run `kleinrock` in `sgl-evolve`.
Clean base commit `a334877e5`.

## The bar (self-reminder)
Top-venue ONLY. Need a **genuinely novel primitive** (not eviction policy, not exclusive tiering, not
SRPF/SJF, not a config flip), a **non-obvious generalizing insight**, and **PC-proof evidence** (clean
curve shift, ablations, error bars, lossless, attributable to the mechanism not de-saturation). A
rigorous bounded **negative/impossibility** result also counts.

## The eval (v0.31, frozen)
Full-decode Poisson **rate sweep** λ∈{3,5,7,10}, WARMUP burst (300 convs) then sweep with **NO flush**
(warm steady-state), NUMP=1553 (~19M tok ≫ L1+L2 ~10.7M → real pressure, hit ~0.62), max-concurrency
256. **Headline = goodput@SLO = max req/s with p99 TTFT ≤ 8 s.** 2-tier (L1 GPU + L2 768GB host, no L3).
v0.31 FIXES the two v0.3-era eval flaws: under-pressure (now NUMP=1553) + cold-start metastability (now
warmup + warm steady-state).

Baseline.json (old single-point λ=3, cold): p99 TTFT 6326ms, p50 750ms, req/s 2.78, hit 0.6217,
host_util 0.9999, load_back_tokens 298M, evict_tokens 582M. host DRAM is SATURATED → genuine pressure.

## Landscape / prior art in this codebase (what NOT to redo)
- Stock scheduling policies present: LPM, DFS_WEIGHT, FCFS, LOF, RANDOM, ROUTING_KEY. (SRPF/WSAC/PGAC are
  NOT in my clean base — they were prior-campaign code in the dirty fork tree.)
- Known dead-ends/baselines (from the field & prior campaigns): eviction-policy tuning (LRU≈Belady),
  exclusive L1↔L2 tiering (~+13pp hit but config-equivalent = a *baseline*), admission gating (PGAC/WSAC
  reportedly catastrophic/neutral), recurrent/Mamba state (non-lever). Cache-residency positives on the
  official eval are largely closed.
- **The one charter-endorsed, un-exploited code gap:** transfer/compute overlap for **L2↔L1**. In stock
  sglang, host→device **load-back is SYNCHRONOUS + on-demand at admission** (scheduler forward path);
  **prefetch exists ONLY for the L3→L2 storage tier** (`prefetch_from_storage`), which is DISABLED in our
  2-tier config. So there is NO overlap of L2→L1 load-back with compute. `load_back_tokens`=298M at λ=3
  → a lot of data moves on-demand; under load this can convoy on the single load stream.

## Workload model (from the trace + loogle loader)
1553 docs → 1553 multiturn conversations. Each conv = **turn-0** ("Input: <full doc> Question: <Q0>",
cold, HEAVY-TAILED prefill: doc tokens p50 7.4K / p90 29K / **max 191K**) + **turns 1..N** (short Q_i,
warm — reuse the doc+prior-QA prefix, small new prefill; may need L2→L1 load-back of the doc if evicted
between turns). Total ~19.2M tok ≫ cache 10.7M. Hit ~0.62 = mostly INTRA-conversation doc reuse
(turns 1..N reuse turn-0's doc). Cross-rate re-runs are mostly COLD again (LRU over 1553 docs ≫ cache →
a conv's own doc is evicted by the time the next rate re-runs it) — so no-flush warmth helps intra-conv,
not turn-0 re-runs.

**Tail structure (the crux):** goodput@SLO = max λ with p99 TTFT ≤ 8s. p99 TTFT ≈ prefill time of a
~p99-length (≈29K-tok) doc **+ queueing delay**. Old cold λ=3 point: p99=6326ms ≈ a 29K-doc prefill → so
baseline goodput@SLO ≈ 3 (λ=3 passes), with headroom to push to 5/7 by cutting the **queueing-delay**
component of p99 at higher λ (the docs themselves don't get longer; the wait does). This is a
Pollaczek–Khinchine tail: heavy-tailed prefill service time S (docs) → large E[S²] → the tail is
queueing-delay-amplified variance, NOT mean hit rate.

## Accurate trace stats (Qwen tokenizer; tools/trace_stats.json)
- convs=1553 (1015 with a long doc, 538 empty-input ShareGPT-style short chats); requests(turns)=7163,
  mean 4.6 turns/conv (p90=10, max=61).
- **turn0 doc tokens (non-empty)**: p50=16.7K, p90=34.7K, **p99=65.6K, max=192.5K** (heavy-tailed).
- **follow-up Q tokens: p50=16, mean=37** (TINY new prefill).
- answer/output tokens: p50=29, p90=386, mean=129 (short decode).
- ⇒ **Extreme compute-to-context asymmetry**: a follow-up computes ~16 new tokens but needs a
  16K–192K-token prefix resident. Follow-up TTFT is TRI-modal: instant (prefix on device) /
  load-back-latency (prefix in L2) / full doc-recompute (prefix evicted). Baseline host_util=1.0 +
  582M evict tok ⇒ evictions happen ⇒ the tail likely has BOTH cache-immune cold turn-0s AND
  cache-relevant follow-up load-backs/recomputes. Decomposition decides positive-vs-impossibility.

## Strategy under SEVERE compute scarcity
1 cycling certified node for the whole v0.31 cell (v0.3 holds 3/4); round-robin base/valiant/me ⇒ ~1 eval
per ~10h for me. So: (a) make each eval maximally informative (baseline; then a diagnostics-instrumented
mechanism run that logs per-request tail composition); (b) LEAN toward the rigorous
characterization/impossibility (needs few runs, noise-robust) while staying alert for a positive if the
tail has a big, attributable cache-relevant component that a NOVEL (non-CachedAttention, non-exclusive-
tiering) mechanism can move.

## ⚠️ EVAL BEHAVIOR FINDING — per-rate flush (verify empirically)
`bench_serving.py:447` posts `/flush_cache` UNCONDITIONALLY at the start of each `benchmark()` call
(`if "sglang" in backend`), with NO flag to disable. `eval.sh` invokes bench_serving ONCE PER RATE. So
despite eval.sh's "NO flush between rates / warm steady-state" comment, **each rate is COLD-started**
(warmup + prior-rate cache wiped). Consequence chain (to verify from baseline):
- Each rate = independent open-loop run at λ on the same 1553 convs, cache cold→warms over the run.
- Docs are UNIQUE per conv; turn-0 (doc prefill) is therefore ALWAYS a cold MISS within a rate. Hit
  ~0.62 = purely INTRA-conv reuse (turns 1..N reuse turn-0's doc). No cross-conv/cross-rate doc reuse.
- **p99 TTFT (= goodput@SLO gate) is dominated by cold turn-0 long-doc prefills** (heavy-tailed, up to
  191K tok). These are CACHE-IMMUNE (unique, flushed) → goodput@SLO is bounded by raw prefill THROUGHPUT
  on cold heavy-tailed work, NOT by hit-rate or load-back.
- **This predicts SLOP (load-back overlap) is NEUTRAL on goodput@SLO** — load-back only helps warm
  follow-up turns (p50/mean), which don't gate the p99. Likely why siblings found cache mechanisms
  neutral on this metric. Candidate CONTRIBUTION reframe: a rigorous *bounded impossibility* — "in a
  flush-per-rate, unique-doc open-loop benchmark, the SLO tail is cache-immune; hit-rate/movement gains
  are orthogonal to goodput@SLO" — UNLESS a scheduling lever reduces cold-prefill queueing delay at the
  knee. MUST verify tail composition from baseline before committing.

## Hypothesis (leading, pending baseline data)
Under open-loop load, goodput@SLO is a **tail** (p99 TTFT) metric. The tail is set by (a) HoL blocking
from long cold prefills and (b) the **non-overlapped, on-demand L2→L1 load-back** for cache hits, which
convoys under load. A **queue-driven speculative load-back** — start the host→device transfer for
soon-to-be-admitted cache-hit requests *while they wait*, overlapped with running-batch compute, under a
memory-pressure-aware revocable budget — should hide transfer latency and shift the curve, losslessly.
**Go/no-go depends on whether load-back is a real tail cost under load** (measured in v0-stock sweep).
Fallback if load-back is negligible: rigorous characterization of what sets goodput@SLO (candidate
bounded-impossibility for lossless KV-movement).

---

## Versions

### ★ DIRECTION DECIDED (from v0-stock data): BOUNDED IMPOSSIBILITY ★
Full baseline curve (v0-stock, commit 9a7fbe472, churn py **net 0** = genuinely stock; W&B logged):
| λ | req/s | out_tok/s | p50 TTFT | **p99 TTFT** | hit |
|---|-------|-----------|----------|--------------|-----|
| 3 | 2.85  | 365       | 1023ms   | **14062ms**  | .677|
| 5 | 3.67  | 470       | 1033ms   | **25449ms**  | .666|
| 7 | 4.07  | 521       | 1024ms   | **33314ms**  | .660|
| 10| 4.21  | 538       | 1043ms   | **42008ms**  | .656|
**goodput@SLO = 0** (no rate ≤ 8s p99). peak req/s 4.21 (knee ~3.5), peak tok/s 538.
**Tail decomposition (analyze_run.py + calibrate.py):**
- **p50 DEAD FLAT ~1.0s across all λ** — the cache serves the median (warm follow-ups) perfectly,
  load-independent. **p99 explodes 14→42s** — the tail.
- **Load-back is FAST: p99 ≤ 8.6ms, mean ~1.3ms** (despite 330M load-back tok/rate, 40% of hits from
  host). ⇒ **L2→L1 transfer is NOT on the critical path → SLOP escape hatch is RULED OUT** (nothing to
  overlap). This is the decisive kill of the one positive candidate.
- eviction ~2.4B tok cumulative (massive host churn) — but p50 flat + hit stable ⇒ eviction/recompute is
  NOT gating the tail either.
- Calibrated **P≈20.5K tok/s** (peak). Measured p99 is **6–8× above** the perfect-cache P-K prediction
  (λ3 pred 2.4s vs meas 14s) — the tail is a HEAVY-TAIL + SERIALIZATION phenomenon (cold turn-0 docs
  65K–192K tok, serialized one-chunked_req-at-a-time), NOT hit-rate or transfer.
**CONCLUSION:** The KV cache is a **mean/median optimizer, not a tail optimizer**. goodput@SLO (a p99
metric) is **structurally 0 and cache-immune** on this heavy-tailed cold-prefill workload: the p99 is set
by unique cold turn-0 document prefills (14% of requests, lossless-irreducible) + their serialized
queueing; neither higher hit rate nor faster movement (both already near-ideal: p50 flat, load-back <9ms)
can touch it. This is a genuinely-new bounded impossibility that EXPLAINS the field's repeated negatives.
**Controls to nail it (cheap, config-only):** v1-writeback (hit↑ +13pp per prior work → predict goodput@SLO
still 0) [job 19490 QUEUED]. Consider a v0-stock replicate for error bars.

### ★ Queue-timeline decomposition (server.log, existing data — resolves structural-vs-flush)
Per-batch timestamps + #queue-req reveal EACH rate has: (a) a COLD-START RAMP at rate start (~2-4 min:
queue spikes to **maxQ 90-184**) then drains to **meanQ 2-8** as cache warms; (b) **big-cold prefills
sustained throughout** (~300-660/2min, ~constant — every turn-0 is a unique cold doc). So p99 = TWO
cache-immune components: (a) cold-start queue ramp (flush artifact) + (b) heavy-tailed cold turn-0 prefill
(structural). HONEST NUANCE: the flush's cold-start ramp is a MAJOR p99 contributor; in warm steady-state
(meanQ 2-8) the p99 would be lower (~cold-doc prefill 3-9s, near the SLO). BUT within the FROZEN eval the
flush is fixed for all configs AND cold turn-0 is cache-immune ⇒ no lossless KV mechanism raises
goodput@SLO on the frozen eval regardless. The impossibility (on the frozen eval) holds; framing must
credit the flush's role (paper §5/§7) — do NOT overclaim "pure structural."

### v0-stock (config) — baseline rate sweep — ✅ DONE (job 19436, ondem-3, ~2.5h)
FIRST DATA (λ=3): req/s 2.85, out_tok/s 365, TTFT **p50 1023ms, p99 14062ms**, hit 0.677.
Warmup (cold, discarded): p99 48284ms. ⇒ **λ=3 already FAILS the 8s SLO → baseline goodput@SLO = 0**
(higher rates worse). p50 1.0s vs p99 14s = the heavy-tailed cold-turn-0 + cold-start-per-rate tail.
CONFIRMS: per-rate flush → cold-start; p99 dominated by cache-immune cold long-doc prefills. Awaiting
λ=5/7/10 + calibration. This is strong empirical support for the bounded-impossibility direction.

CURVE (partial): λ3 req/s2.85 p50 1.02s **p99 14.1s** hit.677 | λ5 req/s3.67 p50 1.03s **p99 25.4s**
hit.666. **KEY PATTERN: p50≈1s (excellent — warm follow-ups hit) but p99 catastrophic & rising
(14→25s).** knee ~3.5 (λ5 achieved 3.67<5=saturated). ⇒ **goodput@SLO=0**; the cache moves the
MEDIAN/mean but NOT the SLO tail (cold heavy-tailed turn-0 + saturation queue). This is the crisp
impossibility: goodput@SLO is structurally 0, cache-immune. Awaiting λ7/10.

### v0-stock (config) — baseline rate sweep — QUEUED
Hypothesis/change: stock 2-tier config swept λ∈{3,5,7,10} to get the real throughput–latency curve +
goodput@SLO baseline + per-rate load-back/host-util diagnostics. This anchors all comparisons.

## Mechanism design (ready to implement, gated on baseline data)
**Primary — SLOP: Speculative Load-back Overlap under Pressure (queue-driven L2→L1 KV prefetch).**
Today load-back is layer-pipelined with the *admitting* iteration's forward (`LayerDoneCounter.wait_until`)
but STARTS only at admission — never overlapped with prior decode iterations, and the sync
alloc/evict sits on the admission path. SLOP:
- Hook `_add_request_to_queue`/`_prefetch_kvcache` (scheduler.py:2305/2280): on queue entry, run
  match_prefix; if host hit ≥ threshold AND device-prefetch budget available → start the H2D load-back NOW
  (overlaps with the running batch's decode compute).
- Admission (scheduler.py:2861 loop): mirror the L3 storage-prefetch skip pattern (line 2882) — if the
  speculative load-back is still in flight, `continue`; when done, admit with KV already resident (no sync
  load_back, no admission-path eviction).
- **Pressure-aware, revocable budget** (the hard part / novelty): prefetch only up to a reserved device-KV
  budget B so it NEVER evicts running-batch KV; revoke under pressure. Prefetch depth ≈ λ × load-back
  latency (Kleinrock: prefetch exactly the reqs admitted within the transfer window).
- Novelty vs prior art: the L2→L1 prefetch tier does not exist in stock (only L3→L2
  `prefetch_from_storage`); memory-pressure-aware revocable speculative load-back driven by the
  waiting-queue lookahead is new; ties queueing dynamics to the KV memory hierarchy. Distinct from
  Strata/Mooncake (L3/remote prefetch), LMCache/CacheGen (compression), AttentionStore (disk).
- **Go/no-go:** only if v0-stock shows load-back on the critical path under load (load_back_duration p99
  grows with λ; load-back tokens material).

**Fallback A (load-back negligible):** dual-class prefill-budget reservation — reserve a slice of the
per-iteration prefill token budget for short/warm (cached) requests so a long cold prefill can't HoL-block
them; SLO-feedback-tuned. NOT SRPF (bandwidth reservation, not global reorder).
**Fallback B (tail is irreducible cold prefill):** rigorous bounded-impossibility — goodput@SLO is set by
heavy-tailed cold-doc prefill service time; no lossless KV-movement mechanism shifts it (cache helps mean,
not the tail). Establish with the curve + tail decomposition + ablations.

## Queueing model of the prefill tail (tools/queue_model.py) — motivation core
M/G/1 P-K model of the prefill server (rate P tok/s), driven by the real trace. Per-request prefill-token
demand under two cache bounds: perfect (follow-ups=Q≈16 tok; only turn-0 pays the doc) vs none (recompute
full context each turn). Results:
- **none-cache is UNSTABLE (ρ>1) at every λ, every P** (E[tok]=13.7K) → caching is essential just for
  stability. perfect-cache E[tok]=2671, p99=39.2K (the cold turn-0 doc).
- **Predicted perfect-cache goodput@SLO = 3 / 7 / 10 at P = 15K / 25K / 40K tok/s.** Goodput is set by P
  and the real-vs-perfect gap.
- Tail is QUEUEING-dominated: p99 TTFT ≈ Wq + p99_service; Wq ∝ λ·E[tok²]/(2P²(1−ρ)) — heavy-tail
  variance E[tok²] and saturation (1−ρ) drive it; own-service floor p99_doc/P is <8s at plausible P.
- **Calibration target = P.** Old baseline (λ=3, p99=6.3s) sits ABOVE perfect-cache prediction
  (1.9–3.8s) ⇒ the real imperfect cache leaves a measurable gap (recompute/load-back/cold-ramp). Size of
  that gap, from the baseline curve, decides POSITIVE (close the gap → raise goodput) vs IMPOSSIBILITY
  (gap is irreducible cold-turn-0 variance). Overlay predicted-vs-measured = the paper's motivation figure.
- Novel-lever hint: since Wq ∝ E[tok²], **variance reduction** of prefill work may beat hit-rate (mean)
  gains for the tail — but SRPT/SRPF HURT here (they starve the big turn-0 docs that ARE the p99). Room
  for a genuinely new tail-aware primitive, TBD from data.

## KEY design insight — chunked prefill serializes heavy docs → p99 ≈ K·t
Only ONE `chunked_req` at a time (scheduler.py:1026) and a full 6144 chunk exhausts the per-iter prefill
budget ⇒ concurrent heavy turn-0 docs prefill ~one-at-a-time to completion (serialized). So p99 TTFT ≈
**K·t** (K = #heavy prefills queued at the knee, t = mean heavy-prefill time). Consequences:
- Serialize is actually near-OPTIMAL for p99 (round-robin/parallel chunking makes ALL heavy docs finish
  late ⇒ worse p99). So SRPT/parallel-chunk both hurt. Confirms the tail is queueing-structural.
- t and the COLD turn-0 count are irreducible (unique docs, lossless) ⇒ that part of p99 is a hard floor
  (impossibility component).
- **The ONE lossless lever = reduce K by preventing follow-up RECOMPUTES** (evicted active-conv docs that
  get re-prefilled add to K). Worth MORE than raising hit rate: it cuts the *count of heavy prefills*, not
  just mean tokens (E[X²] driver). Mechanism candidate: **conversation-liveness-aware residency** — under
  pressure evict COMPLETED-conversation docs before ACTIVE-conversation docs (semantic, not LRU recency);
  differentiate from CachedAttention (no think-time; open-loop; pressure-driven eviction protection, not
  think-time prefetch) and from exclusive-tiering (this is an eviction-VICTIM-selection by liveness, not a
  placement policy). GATE: baseline must show follow-up recompute is a material tail contributor (else the
  impossibility floor dominates and residency can't move goodput@SLO).
- ⚠️ CRITICAL CAVEAT (pushes toward impossibility): the server has **no conversation-liveness oracle** —
  open-loop turns arrive as independent HTTP requests with no conversation ID; the server sees only token
  prefixes. So it can only PREDICT reuse from access patterns (recency/frequency/depth) = exactly the known
  eviction signals, and stock LRU already exploits recency (active convs = recently accessed). ⇒ Without an
  oracle, K-reduction cannot beat LRU losslessly. Combined with irreducible cold turn-0 prefills, this is
  the backbone of a **bounded impossibility**: *no lossless KV mechanism materially raises goodput@SLO on
  this benchmark class.* A positive would require the data to reveal a NON-eviction lever (e.g., a genuine
  load-back-convoy transfer bottleneck that overlap removes) — verify from baseline load_back histogram.

## Related work / novelty positioning (governs which direction is publishable)
- **AttentionStore / CachedAttention (OSDI'24)** — hierarchical KV caching for MULTI-TURN convs with
  scheduler-aware prefetch of a conversation's KV ahead of its next turn (layer-wise pre-load, positional
  overlap). ⚠️ **This is very close to "conversation co-residency + load-back prefetch".** So a plain
  conversation-aware L2→L1 prefetch is likely NOT novel → SLOP/co-residency must differentiate SHARPLY
  (open-loop Poisson, NO think-time gap, memory-pressure-aware REVOCABLE prefetch tied to queue depth) or
  it's incremental.
- **Strata (arXiv 2508.18572)** — multi-tier (GPU/CPU/SSD) KV with cross-tier prefetch+placement; slack
  in the slow tiers. Distinct from my hard device-memory-contended L2↔L1 focus / open-loop tail.
- **Mooncake (FAST'25)** — disaggregated P/D, KVCache-centric global cache, SLO-aware early rejection
  under overload. Related to a tail-scheduling fallback; I'm single-node 2-tier, no disaggregation, and I
  do not reject requests (lossless goodput).
- **LMCache / CacheGen** — KV compression/streaming for cross-request reuse (lossy-ish). Different lever;
  I stay lossless.
- **Queueing theory (Kleinrock; Pollaczek–Khinchine; Harchol-Balter tail scheduling; SRPT)** — my
  analytical frame: goodput@SLO is a P-K tail set by E[S²] of a heavy-tailed prefill service time.
- **Conclusion:** the CLEAREST-novel contribution given prior art is a **rigorous bounded-impossibility /
  characterization** — *"under a flush-per-rate, unique-document open-loop benchmark, the goodput@SLO tail
  is dominated by cache-immune cold prefills; hit-rate & KV-movement gains are provably orthogonal to
  goodput@SLO (cache helps mean, not the SLO tail)"* — which also EXPLAINS the field's repeated negatives.
  A positive (SLOP/co-residency) is only publishable if baseline data shows load-back/recompute is a
  MATERIAL tail contributor at the knee AND I differentiate from CachedAttention. Decide from data.

## Controls (impossibility evidence)
- **v1-writeback** (`--hicache-write-policy write_back`, config; job 19490 QUEUED): tests lever-1 (hit rate
  ↑ +13pp per prior work → predict goodput@SLO still 0). Flag-only, lossless.
- **v2-srpf** (READY in worktree `evolve/kleinrock-srpf`, 2 commits — added SRPF schedule policy +
  argparse choice; import+enum validated): tests lever-4 (tail-optimal scheduling). Stock default =
  **fcfs**; SRPF = shortest-remaining-prefill-first. Predict goodput@SLO still 0 / WORSE (SRPF starves the
  long cold turn-0 docs that ARE the p99). Lossless (reordering only). Merge into main clone AFTER
  v1-writeback runs (keep main tree stock until then), then `eval-on-pool.sh kleinrock v2-srpf
  --schedule-policy srpf`.
- Lever-2 (movement) already covered empirically (load-back p99 <9ms). Lever-3 (residency) partly by
  write_back + argued (no liveness oracle).

## Formal submissions
_(none yet)_

## Ops notes
- Path bug: `eval.sh` computes WORK under `workspace/sgl/v0.3_ablations/research/...` (stale v0.3
  copy-paste) but setup clones under `workspace/sgl/v0.31/research/...`. Fixed WITHOUT editing frozen
  eval.sh via symlink `v0.3_ablations/research -> ../v0.31/research` (matches the already-present
  `base ->` symlink the manager/base-cell used).
- ⚠️ Eval runs the WORKING TREE via PYTHONPATH at run time → keep `python/` STOCK until the queued
  baseline job (19436) actually runs, else the "baseline" isn't stock. Engine edits only after v0-stock lands.
- Compute contention (ROOT CAUSE, 02:47Z): v0.3 manager holds 3/4 certified nodes (0-3,1-2,ondem-2) with
  24h sleep-infinity — and `squeue -s` shows ONLY `.batch/.extern` steps on them → **RESERVED-BUT-IDLE**
  (no evals running). Meanwhile ≥4 v0.31 evals (kleinrock/valiant/wilkes/base) starve on the 1 cycling
  node (ondem-3). This is WASTED capacity / a fleet resource-allocation problem — fix is manager/supervisor
  (release idle v0.3 holds OR v0.31 mgr establishes a shared flock pool). As a researcher: do NOT srun into
  another campaign's holds (collision+fairness), do NOT monopolize ondem-3 with a session hold (unfair to 3
  siblings). Keep fair one-shot exclusive queue; slurm fair-share should raise my priority (I've used 0
  compute). Study while queued.
