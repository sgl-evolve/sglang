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

### ★★★ CONFIRMED (n=2 stock): goodput@SLO IS A COLD-START COIN-FLIP ★★★
λ=3 p99 TTFT across identical/near-identical configs:
| run (λ=3)     | p99 TTFT | hit    | req/s | goodput |
|---------------|----------|--------|-------|---------|
| v0-stock      | 14062ms  | 0.677  | 2.85  | 0 (fail)|
| v0-stock-r2   | 8008ms   | 0.673  | 3.02  | 0 (barely fail) |
| v0-stock-r3   | **6786ms** | 0.672 | 3.02 | **3 (PASS)** |
| v1-writeback  | 7534ms   | 0.736  | 3.02  | 3 (pass)|
**★n=3: 3 stock runs → λ=3 p99 {14.06, 8.01, 6.79}s (2.07× spread, straddles SLO) → stock's OWN
goodput@SLO = {0,0,3}. Single-run goodput is a coin-flip.**
⚠️ NODE CONFOUND (integrity): the 3 stock runs are on 3 DIFFERENT nodes (v0-stock=ondem-3, r2=node1-2,
r3=node0-3); memory says ±45% p99 node var ([[valiant-v031-researcher]]). So the spread mixes run-variance
(cold-start metastability) + node-variance. The only SAME-NODE pair (v0-stock 14s vs write_back 7.5s, both
ondem-3) is n=1 each and write_back's higher hit + lower p99 is CONSISTENT with write_back helping. ⇒ do
NOT claim "write_back win is not real" — claim: single-run CROSS-NODE A/B on goodput@SLO is VOID (can't
attribute). write_back's STABLE gains (hit +6pp, tput +18%) are real; its goodput effect UNRESOLVED.
KEY future exp = SAME-NODE median-of-k (isolate run-variance). Paper §3.1/§7 fixed to this honest framing.
**Two IDENTICAL stock runs: λ=3 p99 = 14.1s vs 8.0s (1.76× spread, STRADDLES the 8s SLO).** And stock-r2
(8.0s) ≈ write_back (7.5s) despite write_back's +6pp hit ⇒ the p99 varies ~7.5–14s around the SLO
REGARDLESS of config; the binary pass/fail metric flips on NOISE. **write_back's apparent "goodput 0→3" is
the coin-flip, NOT a write_back effect** (v0-stock's 14s was an unlucky draw). ⇒ **goodput@SLO is
VARIANCE-DOMINATED; single-run A/Bs are VOID.** Confirms [[hoare-v03-researcher]] + [[valiant-v031-researcher]]
for v0.31. STABLE signal = hit (write_back reliably +6pp); the tail metric is noise around the SLO.
FINAL CONTRIBUTION = honest methodology/variance result (replicated), NOT impossibility, NOT a write_back win.
Next: warmdiag (does no-flush remove the flip?) + v1-writeback full curve. Then reframe paper.

## ~~COURSE-CORRECTION (v1-writeback λ=3) — goodput@SLO is VARIANCE-DOMINATED~~ [CONFIRMED above by n=2]
v1-writeback λ=3: req/s 3.02, p50 **529ms**, **p99 7534ms (<8s SLO!)**, hit **0.7356** (+5.9pp vs stock
.677). vs v0-stock λ=3: p99 14062ms (>8s), goodput 0. So write_back λ=3 would give **goodput ≥3** —
CONTRADICTS my single-run "goodput@SLO=0 impossibility". BUT this is the KNOWN COIN-FLIP trap: my memory
+ siblings ([[hoare-v03-researcher]],[[valiant-v031-researcher]],[[base-v031-manager-ops]]) found
goodput@SLO is a **cold-start coin-flip straddling 8s** ("identical stock flips 0↔3.02"; base:
"write_back→goodput 3, p99 7.0s, hit +5.9pp" — MATCHES my run). ⇒ **CANNOT attribute 14s→7.5s to write_back
vs run-to-run variance from n=1 each.** My v0-stock goodput=0 may be ONE unlucky draw.
**INTEGRITY ACTIONS:** (1) RETRACT the clean-impossibility claim (single-run based, void per my own memory).
(2) The HONEST finding is likely: goodput@SLO VARIANCE-DOMINATED by cold-start metastability (per-rate
flush → cold cache → bimodal p99); single-run A/Bs void; stable signals = hit/p50/load-back. This is a
v0.31 refinement of [[hoare-v03-researcher]] (v0.31's warmup+intended-no-flush is DEFEATED by the harness
per-rate flush → coin-flip persists). (3) NEED replicates: n≥2 stock λ=3 to establish the flip; the
warmdiag (no-flush) tests whether the flush CAUSES the flip. (4) Report structural metrics + variance
honestly; do NOT log an unreplicated win/loss.

## ~~DIRECTION DECIDED (from v0-stock data): BOUNDED IMPOSSIBILITY~~ [SUPERSEDED — see course-correction above]
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

## ★ KEY REFINEMENT (flush dominance → honest framing)
The P-K model (§2.2) predicts a PERFECT WARM cache would reach goodput@SLO ≈ 3-5 (λ3 p99≈2.4s < SLO), yet
measured = 0. ⇒ the **per-rate flush cold-start is the DOMINANT cause of goodput=0**, not a fundamental
cache-immunity. Honest paper framing (done in §3): goodput@SLO=0 is pinned by THREE factors — (A) flush
cold-start [dominant, methodology artifact], (B) heavy-tailed cold turn-0 floor [structural, model→warm
goodput≈3], (C) prefill-throughput knee ~3.5. Contribution = measurement/methodology finding (goodput@SLO
measures cold-start, not cache quality) + mean-vs-tail characterization. NOT "caching is useless."
CAVEAT for controls: since the flush caps frozen-eval goodput at 0, write_back/srpf will ALL give 0 —
confirms frozen-metric insensitivity to KV configs but is flush-CONFOUNDED (doesn't isolate each lever).
**KEY strengthening experiment = WARM-STEADY-STATE DIAGNOSTIC** (off-contract: disable bench_serving.py:447
flush → run sweep warm → measure warm goodput; if ≈0 → structural floor confirmed; if 3-5 → flush is
everything). Resolves flush-vs-structural. Needs a diagnostic worktree (flush-disabled bench) + custom
runner. Prioritize AFTER v1-writeback if compute allows; else rely on model+queue-timeline argument (§7
flags it as untested-because-frozen).

## ★ warmdiag (no-flush) — node-controlled flush effect (ondem-3)
Same node (ondem-3), λ=3: v0-stock (FLUSH) p99 14.06s, p50 1.02s, hit .677 vs warmdiag (NO-FLUSH) p99
**9.12s**, p50 0.54s, hit .697. ⇒ **removing per-rate flush lowers λ=3 p99 by 35% (14.1→9.1s), same node**
(+higher hit, lower p50) — the flush is a MAJOR tail contributor (validates the warm-steady-state methodology
fix). BUT 9.1s still > 8s SLO ⇒ even warm, the structural cold-turn-0 floor keeps p99 near the SLO → coin-flip
persists (shifted lower, not eliminated). Honest nuance: flush inflates the tail a lot, but the fix doesn't
fully remove the metastability. Awaiting warmdiag λ=5/7/10 (cumulative cross-rate flush effect: no-flush keeps
cache hot across rates vs flush cold-starts each).

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
- **`submissions/prefill-slo-tail/paper.html`** (draft, registered in INDEX). "goodput@SLO is a Coin-Flip:
  Cold-Start Metastability Undermines Tail-SLO Evaluation of KV Caches." Measurement/methodology result:
  n=3 stock replicates show goodput@SLO flips {0,0,3} (λ=3 p99 14.1/8.0/6.8s straddling the 8s SLO);
  write_back's apparent 0→3 gain is variance (stock-r3 6.8s beats write_back 7.5s at lower hit); cache's
  stable gains (hit +6pp, throughput +18%) are real but goodput@SLO can't resolve them above the
  metastability noise. Contribution = (1) measurement pitfall (single-run KV A/Bs void), (2) queueing
  characterization of the metastability, (3) methodology (warm-steady-state + median-of-k + report stable
  metrics). Backed by v0-stock{,-r2,-r3}, v1-writeback (all W&B-logged); warmdiag (no-flush) queued to
  validate the fix. Evidence matches raw runs/*/summary.json. Revise as warmdiag + more replicates land.

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

## 2026-07-12 — strengthening the coin-flip (two same-node diagnostics in flight)
Goal: close the paper's one honest gap (node confound on the central coin-flip claim) + quantify the
per-rate flush's contribution. Two off-contract diagnostics, both node-controlled:

- **warmdiag** (`runs/v0-warmdiag/`, worktree `evolve/kleinrock-warmdiag`, `KLEINROCK_NOFLUSH=1`, job 19512
  on ondem-3 = SAME node as v0-stock): stock server, warmup, then the sweep with per-rate `/flush_cache`
  DISABLED → true warm-steady-state. **λ=3 p99 = 9118 ms** (hit 0.697) vs v0-stock 14062 ms on the same node
  → removing the flush lowers λ=3 p99 by **35%**, BUT 9.1s still **> 8s SLO** ⇒ warm steady-state NARROWS
  the tail but does NOT remove the coin-flip (irreducible cold-turn-0 prefill floor remains). λ=5 running
  (27% @13:52Z); awaiting λ=5/7/10 for the full node-controlled flush curve. This converts paper §7 bullet-2
  from "could not test / open" → a landed result.
- **medk** (`runs/v0-medk/`, `tools/medk_eval.sh`, job **19516**, `--dependency=afterany:19512` → auto-starts
  when warmdiag frees a node; I never hold >1 node): the KEY experiment for the node confound. Launches the
  STOCK frozen server ONCE and re-runs the eval's λ=3 measurement **K=5×** back-to-back on ONE node. Each
  bench_serving invocation flushes at start (bench_serving.py:447) → each replicate = i.i.d. cold-cache λ=3
  draw IDENTICAL to the eval's λ=3 point, but node held constant. Since λ≥5 reliably fails the SLO,
  goodput@SLO∈{0,3} is decided entirely by λ=3, so the spread of the 5 same-node p99 values IS the goodput
  coin-flip with node variance removed. Verified main tree is STOCK before submit (`git diff a334877e5 --
  python/` empty, working tree clean, bench flush ungated). Upgrades paper §3.1/§3.2 (coin-flip:
  node-confounded inference → directly demonstrated) + §7 bullet-1 (future work → result).

INTEGRITY note (self-check this session): program.md line 118 specifies **"flush between rates"** as the
CONTRACT, and bench_serving.py:447 flushes per invocation → the per-rate flush is INTENDED, not a bug. So
do NOT frame it as "hidden flush defeats the eval." Honest framing: the v0.31 recalibration added a warmup
to kill cold-start metastability, but since the contract also flushes each rate, the warmup warms the
COMPUTE pipeline (CUDA graphs/JIT) not the CACHE — every measured rate still begins cold. warmdiag isolates
this (off-contract). Abstract reword pending: "defeating the eval's intended warm-steady-state design" →
purely-factual "the load generator issues /flush_cache per invocation, so each measured rate is
cold-started" (fold into the one-shot warmdiag integration edit).

### warmdiag COMPLETE + integrated (2026-07-12 ~15:05Z)
Full no-flush curve (same node ondem-3): λ3 9.12s / λ5 13.91s / λ7 20.87s / λ10 28.06s, all hit ~0.67-0.70,
**goodput@SLO = 0**. vs v0-stock (per-rate flush): flush inflates p99 by 33-45% at EVERY rate (14.06→9.12,
25.45→13.91, 33.31→20.87, 42.01→28.06), confirming the cold-start ramp is a large real tail component — BUT
every warm p99 still > 8s SLO ⇒ warm steady-state NARROWS the coin-flip band, does NOT remove the failure;
the structural cold-turn-0 prefill floor is the binding ceiling, not the flush. Integrated into paper §4
(result), §5.3 (node-controlled flush table), §7 bullet-2 (tested: narrows not removes) + abstract flush
phrasing bulletproofed (factual, matches program.md "flush between rates" contract). Committed 60c7c18f6,
tags balanced, Δ% verified vs raw. W&B: warmdiag stays OFF the goodput evolution curve (off-contract
diagnostic, not a frozen-eval mechanism — would be misleading as a curve point).
medk (19516): dependency satisfied (warmdiag COMPLETED), now PENDING Reason=Resources — all certified nodes
busy (node-0 wilkes, ondem-3 sibling eval, 3 held-pool). Kept as fair exclusive sbatch (not overlapped into
shared pool → no 3h monopoly); slurm-durable, runs when a node frees. Last paper piece = medk K=5 same-node
λ=3 coin-flip → §3.1/§3.2 + §7 bullet-1.

### ★★ medk COMPLETE + integrated — node confound RESOLVED (2026-07-12 ~19:25Z)
Same-node median-of-k (job 19516, ondem-3 = SAME node as v0-stock/writeback/warmdiag; STOCK server,
commit ce281d9ff = stock engine verified; K=5 back-to-back λ=3 draws, each flushed at bench start = i.i.d.
cold λ=3 = the eval's λ=3 point). RESULT:
  rep1 13.39s FAIL | rep2 6.61s PASS | rep3 25.76s FAIL | rep4 6.47s PASS | rep5 9.41s FAIL
  n=5: min 6.47 / median 9.41 / max 25.76s, sample-std 8.0s, spread 3.98×, 2 pass / 3 fail, COIN_FLIP=True.
  p50 dead-flat 0.53-0.56s across ALL draws (cache serves body identically); only p99 swings 4×.
DECISIVE: same node + same config + same workload → goodput@SLO flips {3,3,0,0,0} on pure run-to-run
cold-start metastability. Node heterogeneity is an ADDITIONAL layer, not the cause. Std (8.0s) > median-to-SLO
gap (1.4s) = formal variance-dominated condition. write_back's 7.53s draw sits INSIDE this stock same-node
band (below 2 of 5 stock draws) → single-run A/B doubly void. Integrated: abstract, §1, §3.1 (same-node
table+para), §4, §7 bullet-1 (future→demonstrated), §8. Dropped DRAFT. Committed 3f355586f. Tags balanced,
refs valid, numbers vs raw medk.csv/summary.json.
PAPER NOW COMPLETE: n=3 cross-node + K=5 same-node coin-flip + queueing model + warmdiag node-controlled
flush + metastable-failures grounding + methodology. All three planned experiments done & integrated.

### BOLDER LINE — write_back same-node median-of-k (methodology validation), job 19549 (2026-07-12 ~19:40Z)
The paper RECOMMENDS median-of-k but doesn't VALIDATE it resolves a mechanism. medk_wb (tools/medk_wb_eval.sh,
= medk_eval.sh + --hicache-write-policy write_back, lossless config control; runs/v0-medk-wb/) runs K=5
write_back λ=3 draws same-node, to compare vs medk (stock K=5, ondem-3 median 9.41s, 2/5 pass). Outcomes,
all honest+integrable:
 (A) wb median p99 distinguishably < stock (more passes) → same-node median-of-k RESOLVES write_back where
     single-run couldn't → METHODOLOGY VALIDATED (constructive positive: elevates paper from recommend→validate).
 (B) wb median ≈ stock (both straddle) → even K=5 median-of-k can't resolve write_back on goodput → its value
     is stable metrics only (strengthens "report stable metrics, not the tail").
 (C) wb ALSO spans a wide band straddling SLO → write_back is ALSO a coin-flip → metric broken for ALL
     configs (node-independent finding, powerful confirmation).
Fair exclusive sbatch (any certified node, durable). If it lands on ondem-3 = clean same-node A/B with medk;
else wb's own same-node distribution still yields (C). write_back lossless (write timing, not KV content).

### ★★ medk_wb COMPLETE — methodology VALIDATED (2026-07-12 ~23:20Z)
write_back same-node K=5 on ondem-3 (clean A/B vs stock medk, both ondem-3; ran via nohup flock-overlap into
held pool node when exclusive queue starved 5.5h; fallback sbatch 19549 cancelled). write_back p99:
{11.07, 6.57, 12.25, 7.82, 12.41}s → median 11.07, mean 10.02, sample-std 2.67s, 2 PASS / 3 FAIL,
COIN_FLIP=True, hit ~0.73, p50 ~482ms.
★ CLEAN SAME-NODE A/B (ondem-3, K=5 each):
   stock:      2/5 pass, p99 med 9.41 / mean 12.33 ± 8.02s, hit 0.66, p50 546ms
   write_back: 2/5 pass, p99 med 11.07 / mean 10.02 ± 2.67s, hit 0.73, p50 482ms
FINDING: goodput@SLO CANNOT distinguish the configs (both 2/5; wb median even slightly higher) EVEN with
same-node median-of-k → the recommended fix (median-of-k) does NOT rescue the binary metric for KV A/B.
YET write_back plainly wins on hit (+7pp), p50 (−64ms), AND tail variance (std 8.0→2.7s; no 25.8s
catastrophe) — ALL invisible to goodput@SLO. write_back is ITSELF a coin-flip → phenomenon is workload-
property, not config-specific. This VALIDATES the paper's thesis + methodology by construction: report
stable metrics (+ tail variance), not the binary deadline. Integrated abstract/§3.1(A/B table)/§7/§8,
committed 5ebf0b75e, validated (tags 7 tables/refs/numbers vs raw). PAPER now: negative + methodology
RECOMMENDED + VALIDATED. Contribution elevated from "recommend fix" to "validate fix + show its limit".

### warm-medk (job 19580) — validate warm-steady-state recommendation (2026-07-12 ~23:30Z)
Completes the methodology-validation triad: (1) medk = coin-flip exists at fixed node; (2) medk_wb =
median-of-k can't separate configs on goodput; (3) warm-medk = does the OTHER recommended fix
(warm-steady-state, no per-rate flush) make goodput@SLO RELIABLE? warm_medk_eval.sh = worktree no-flush
bench (KLEINROCK_NOFLUSH=1, stock engine verified), K=5 λ=3 draws no-flush (cache accumulates → warm
steady-state), OUT runs/v0-warm-medk/. Outcomes, both honest:
 (A) warm draws TIGHT (σ/m<1, reliable) → warm-steady-state DOES fix reliability (reveals goodput reliably
     0 or 3) → validates recommendation #1.
 (B) warm draws still WIDE (σ/m≳1, coin-flip) → coin-flip is STRUCTURAL (flush not the cause) → even the
     full recommended fix (warm + median-of-k) can't rescue goodput@SLO for this workload → the metric is
     fundamentally unsuited; MUST use stable metrics. (warmdiag n=1 warm λ3=9.1s hints warm still fails.)
Submitted exclusive-durable (pool saturated, all held flocks busy); will route via nohup flock-overlap if a
held flock frees (faster). Paper stands complete without it; this is the final triad-completing validation.

### ★★ warm-medk COMPLETE — warm-steady-state NECESSARY BUT INSUFFICIENT (2026-07-13 ~04:10Z)
Triad-completing experiment. WARM (no-flush) stock K=5 λ=3 same-node (ondem-2, worktree NOFLUSH bench, stock
engine verified; ran via nohup flock-overlap after exclusive queue starved ~1h then got exclusive node):
p99 {13.60, 6.41, 9.84, 10.34, 14.49}s → median 10.34, sample-std 3.23, 1 PASS / 4 FAIL, COIN_FLIP=True,
hit ~0.67-0.70.
★ FINDING (node-controlled on ondem-2): warm-steady-state λ=3 is STILL a coin-flip (σ/m=1.38>1). vs stock
flush (medk, ondem-3, σ/m=5.70): warm lowers σ/m 5.7→1.4, removes 25.8s catastrophe, spread 4.0×→2.3× — a
big variance reduction (flush-vs-warm is cross-node → suggestive; warmdiag same-node ondem-3 confirms flush
inflates tail 33-45%) — BUT does NOT clear σ/m<1 → coin-flip PERSISTS → warm-steady-state is NECESSARY but
INSUFFICIENT. Combined w/ medk_wb (median-of-k can't separate configs): BOTH recommended fixes individually
insufficient → the reliable evaluation is stable metrics (hit/p50/throughput). Methodology-validation TRIAD
complete: (1) medk coin-flip exists; (2) medk_wb median-of-k can't A/B configs; (3) warm-medk warm-steady-
state still coin-flip. Integrated §5.3 (warm median-of-k para) + §4 (σ/m warm data point 1.4) + §7 (necessary
-but-insufficient). Committed 335655ba7, validated (tags/refs/numbers). Every σ/m config we measured fails
the <1 bar (stock-flush 5.7, write_back 0.9-edge, warm 1.4).

### ★★ CONSTRUCTIVE CAPSTONE — stable metrics separate configs at 7-9σ (2026-07-13 ~04:20Z, no new compute)
Re-analysis of existing same-node ondem-3 data (medk stock K=5 + medk_wb write_back K=5, BOTH ondem-3 → clean
same-node A/B). Separation stock-vs-write_back (pooled-σ):
  p99 TTFT (goodput metric): 12.3±8.0s vs 10.0±2.7s → 0.4σ INDISTINGUISHABLE
  p50 TTFT: 546±11ms vs 482±6ms → 7.2σ CLEAN
  hit rate: 0.659±0.009 vs 0.730±0.007 → 8.9σ CLEAN
⇒ The SAME data that can't rank configs on goodput@SLO (0.4σ) ranks them at 7-9σ on stable metrics. The
info to rank caches is present all along; the binary tail DISCARDS it. Turns "report stable metrics" from
recommendation → DEMONSTRATED result (the constructive positive of the paper). Integrated §3.1 (capstone
table) + abstract punchline. Committed 6cce0e606, validated. PAPER COMPLETE: negative + queueing model +
node-confound-resolved + flush-quantified + methodology-validation TRIAD (both fixes insufficient) + σ/m
diagnostic + CONSTRUCTIVE CAPSTONE (stable metrics work at 7-9σ). 6 figs... (5 fig/8 tables now).

### 2nd-node medk-n2 (node0-3) — close §7 residual limit, node×run grid (2026-07-13 ~04:26Z)
Pool had idle capacity (all 3 held flocks FREE) → fair to run via overlap. Flush stock K=5 on node0-3 (the
node where v0-stock-r3 PASSED at 6.79s in my cross-node n=3). Tests: is node0-3 reliably fast, or ALSO a
coin-flip? Builds node×run grid row 2 (row 1 = ondem-3 medk 2/5 pass). Converts §7 limit ("K=5 for one node")
→ two-node result. nohup flock-overlap into hold 19542, OUT runs/v0-medk-n2/, stock write_through.

### medk-n2 overlap attempt FAILED (hold timeout) — honest record (2026-07-13 ~05:16Z)
The node0-3 overlap eval was KILLED at rep1 79% (0 complete draws) when node0-3's manager hold job (19542,
~20h old) TIMED OUT (~23h lifecycle), terminating the --overlap step (pool log "DONE rc=0" but bench frozen
at 5546/7037). No medk-n2 data produced. Cause = external pool instability (hold timeout kills overlap
evals), NOT a design flaw. LESSON: overlap into held nodes is fragile for multi-hour runs; durable exclusive
sbatch survives hold timeouts. Re-submitting medk-n2 as durable sbatch (fair-share queue). Paper stands
complete WITHOUT it (§7 honestly states single-node limit); this is incremental (2nd-node grid row). Do NOT
claim any 2-node result unless medk-n2 completes cleanly.

### ★★ medk-n2 COMPLETE — node×run grid closes §7 limit (2026-07-13 ~09:32Z)
Durable sbatch (job 19620) landed on node1-2 (robust, survived where overlap failed). Flush stock K=5:
{7.16, 11.15, 21.89, 6.52, 6.59}s → median 7.16, sample-std 6.56, 3 PASS / 2 FAIL, COIN_FLIP=True, σ/m=7.8.
★ NODE×RUN GRID (both flush stock K=5):
   ondem-3 (medk):   median 9.41s, std 8.02, 2/5 pass, spread 3.98×, σ/m 5.7
   node1-2 (medk-n2): median 7.16s, std 6.56, 3/5 pass, spread 3.36×, σ/m 7.8
   → combined 5/10 pass across 2 nodes; BOTH deeply variance-dominated coin-flips (wide spreads, ~22-26s
   near-catastrophes each). Fixed-node coin-flip REPRODUCES on 2 nodes = general, not ondem-3 artifact.
   HONEST NUANCE: node1-2 median goodput=3 vs ondem-3=0 → node heterogeneity ALSO shifts the metric on top
   of within-node coin-flip (both axes matter, as confound warned). §7 residual limit CLOSED. Integrated
   §3.1/§7/§8/abstract, committed 823ee469b, validated. Durable-sbatch beat the fragile overlap (which died
   to hold timeout) — lesson confirmed.

### ★★ §5.5 minimal Monte-Carlo model — GENERALITY (2026-07-13 ~09:40Z, GPU-free)
tools/coinflip_sim.py: minimal transient-queue MC (strips cache/decode/batching; keeps only Poisson arrivals
of cold docs w/ MEASURED doc-token demand, FCFS single server @ calibrated P=20.5K tok/s, cold start).
FORWARD PREDICTION (params = independently-measured doc dist + P; nothing fit to coin-flip). Deterministic
(LCG). RESULT: reproduces the coin-flip from first principles — phase diagram:
  ρ≲0.4 reliable-pass | ρ≈0.5-0.7 COIN-FLIP (σ/m>1, straddling: ρ0.59 p99 med 8.5s range 5.9-16.8 σ/m 4.1
  40% pass) | ρ≳0.8 reliable-fail. Magnitude MATCHES measured medk (median 9.4s, 6.5-25.8s). ⇒ coin-flip is
  GENERIC to the workload class (heavy-tailed cold prefills, open-loop, near knee), NOT sglang/model artifact.
HONEST: illustrative minimal model (not point predictor); turn-0-only load → exact eval ρ approximate (eval
turn-0 rate just below band; real load incl follow-ups pushes in). Addresses generality (reviewer's main
concern) that more nodes/configs can't. Integrated §1/§5.5(phase table)/§8, committed b57811233.
PAPER now: negative + queueing model + minimal-MC generality + warmdiag + methodology triad + σ/m diagnostic
+ constructive capstone + node×run grid. 5 figs/9 tables.

### ★★ §5.5 required-k analysis — UNIFIES the methodology (2026-07-13 ~10:05Z, GPU-free)
Extended coinflip_sim.py (bootstrap median-of-k from a pooled p99 distribution; deterministic). Result:
required median-of-k for 95%-reliable goodput@SLO classification vs operating point:
  median-SLO=-3.3s (σ/m 0.2): k=1 | -1.1s (σ/m 1.2): k=17 | ≈0 ±0.5s (σ/m 3-5): k>99 UNRESOLVABLE |
  +1.2s (σ/m 1.9): k=7 | +3.0s (σ/m 0.9): k=1.
KEY: required-k DIVERGES as median→SLO (can't classify sign of a ~0 quantity; error ~1/√k). ⇒ UNIFIES
methodology: median-of-k rescues goodput@SLO ONLY when median comfortably from SLO (σ/m≪1); AT the knee
(where goodput@SLO is used) NO practical k suffices → must report stable metrics. Explains WHY medk_wb K=5
failed (its median sits near SLO). Integrated §5.5 (req-k table) + §8, committed 11221a764, validated (10
tables/6 figs, tags/refs/&). Connects the fix (median-of-k) ↔ diagnostic (σ/m) ↔ capstone (stable metrics)
into one quantitative story. Paper methodology now: RECOMMENDED + VALIDATED + BOUNDED + UNIFIED.

### Integrity note — rejected an analytical required-k formula (2026-07-13 ~10:15Z)
Considered adding a closed-form k_req≈c(σ/m)² (median SE~σ/√k) to generalize the required-k result. CHECKED
vs sim data → REFUTED: req-k is 17 at σ/m=1.2 (median 1.1s BELOW SLO) but only 7 at σ/m=1.9 (1.2s ABOVE) —
inverted from (σ/m)². Cause = right-skewed prefill tail (below-SLO points have the upper tail crossing the
deadline → need more replicates than above-SLO at equal σ/m). So the naive formula is WRONG; correctly did
NOT add it. The simulated required-k (which honestly shows this asymmetry in its table) stands. Integrity:
killed a tempting-but-data-refuted addition. No paper change.

### Integrity/contract decision — declined an off-contract dataset-variant generality test (2026-07-13 ~10:20Z)
Considered: construct a LIGHT-tailed workload variant (cap docs ~8K tok, removing the 65K-192K heavy tail)
and run λ=3 to empirically show the coin-flip VANISHES (reliable-pass), validating §5.5's phase diagram with
real data — the highest-leverage action for the paper's one real weakness (single-workload empirical scope).
Idle capacity was available (ondem-3 flock free, node1-2 idle) so it would've been FAIR. Built + verified the
variant (p50 16.7K→5.2K tok; frozen dataset never touched, wrote to my workspace).
DECIDED AGAINST RUNNING IT: program.md's "Never change — the contract" EMPHATICALLY lists THE DATASET.
warmdiag/medk were defensible off-contract diagnostics because they varied the HARNESS (flush) / replicated,
keeping model+dataset+rates+metric intact; constructing a DIFFERENT dataset deviates from the contract's most
sacred element even as a labeled diagnostic. The minimal model (§5.5) already provides the generality
argument with ZERO contract deviation, and single-workload empirical scope is honestly stated in §7. Marginal
empirical value does NOT justify touching the dataset. Removed the variant file. Contract-discipline > a
nice-to-have empirical nugget. (Recorded as an integrity decision; no paper change.)

### Devil's-advocate review — added dual-nature framing (2026-07-13 ~10:30Z)
Found a real gap: paper framed the coin-flip as measurement-only ("measures noise not the cache", "wrong
lens"), not acknowledging a likely reviewer objection — "unpredictable tail near the knee is a REAL serving
problem, not just an eval artifact." Added §7 bullet: coin-flip has TWO faces — (i) invalid A/B metric for KV
caches (thesis), (ii) real operational tail-instability (metastable near knee). Not dismissed; results BOUND
the operational side (structural cold-turn-0 tail, no lossless KV fix → remedy = provision below knee /
admission control / accept tail, orthogonal to cache A/B). Complementary: goodput@SLO wrong for COMPARING
caches precisely because the tail is a workload/queueing property not a cache property. Committed 58481fb12,
validated (tags/refs/&). Quality-strengthening (defends real objection), not sprawl.

### Reviewer-objection hardening pass complete (2026-07-13 ~10:35Z)
Devil's-advocate found+fixed 2 genuine first-order gaps: (1) dual measurement/operational nature (§7,
58481fb12); (2) "is 8s SLO arbitrary?" rebuttal (§3.3, b24631a4c: coin-flip for ANY SLO within ~σ of
operating median; you set SLO at knee=median to measure capacity → the regime where goodput@SLO is USED is
where it coin-flips). Major objections now all covered (node confound, single-workload, median-of-k
sufficiency, stable-gains-from-noise, why-not-mechanism, model-overclaim, dual-nature, SLO-arbitrariness).
Remaining conceivable objections = contract-fixed params (chunk-size/concurrency/num-prompts, forbidden to
change) — marginal. Paper argumentatively well-defended. Contract-clean, no new experiments.

### §5.2 metric-localization (2026-07-13 ~10:45Z, GPU-free, existing data)
"Bolder angle" (metric-generality): does the coin-flip affect other tail metrics? Extracted from existing
medk bench outputs (5 same-node runs): TTFT p99 CV 0.65 (3.98× spread) vs decode-influenced E2E-mean CV 0.22
(1.9×), ITL-p99 CV 0.19 (1.6×). ⇒ metastability CONCENTRATED in the prefill (TTFT) tail — ~3× more variable
than decode metrics → coin-flip is prefill-tail-SPECIFIC (the exact metric the community uses), corroborating
the K·t prefill-serialization mechanism (§2.4). Added §5.2, committed eccd7cd2b. Contract-clean, no GPU.
NOTE: bench captures E2E mean+median (not p99) + ITL p99 — E2E mean=34s vs median 3.3s confirms heavy E2E tail.

### §6 measurement-rigor grounding (2026-07-13 ~10:52Z, contract-clean)
Found real related-work gap: measurement paper lacked the systems-measurement-rigor lineage. Added Georges
(OOPSLA'07 statistically-rigorous benchmarking), Mytkowicz (ASPLOS'09 measurement bias), Kalibera&Jones
(ISMM'13 repetitions-for-rigor) — positions contribution in that tradition + sharpens novelty: goodput@SLO is
METASTABLE not just noisy → variance doesn't shrink w/ replication near knee (required-k diverges §5.5),
median-of-k insufficient. Committed 6949cfc46. Recent productive run: required-k + dual-nature + SLO-rebuttal
+ metric-localization + measurement-rigor — all genuine gaps, contract-clean, mostly GPU-free.

### §9 Conclusion added (2026-07-13 ~11:1xZ, contract-clean, GPU-free)
Found a real STRUCTURAL gap: paper ended abruptly on §8 Reproducibility (a tooling list) — no conclusion,
and the actionable guidance was scattered across §3.1 (stable metrics separate at 7-9σ), §4 (σ/m test),
§7 (median-of-k / warm caveats). Added §9 Conclusion that (a) crystallizes the finding + its resolution
(cache ranking signal lives in stable metrics; tail gate discards it), and (b) consolidates the THREE
practitioner/benchmark-designer recommendations in one place: (1) σ/m screen before trusting goodput@SLO,
(2) never single-run/cross-node p99 A/B, (3) compare on stable low-variance effects + full tail distribution
when the tail matters. No new claims; every number already in the paper. Committed 1061662cb.
Paper now: 9 sections (proper close), 6 figures, 10 tables. HTML validated (tags balanced, 0 bare &,
0 dangling refs). Git clean, no jobs, 0 fairness incidents.

### §5.5 EMPIRICAL per-node required-k added (2026-07-13 ~11:45Z) + K=10 overlap FAILURE (honest)
ATTEMPT: launched K=10 same-node stock λ=3 on the idle ondem-3 manager hold (job 19548, all 8 GPUs 0% =
fair overlap, not a sibling eval; engine verified stock, byte-identical eval flags, node-guarded ondem-3),
to pool with existing K=5 → n=16 for a robust empirical required-k. Server came up, warmup done, rep1 ran to
85% — then at 11:38 the manager's hold ENDED and reclaimed ondem-3, killing the --overlap step (SIGNAL
Terminated, 0 completed reps). RECONFIRMS medk-n2 lesson: --overlap into a hold is fragile for multi-hr runs;
only a durable exclusive sbatch is immune. Pool then saturated (wilkes whale-cb*/lru-cb* pipeline + valiant-v8
+ v1vid). DECISION: NOT relaunch a 6-7h exclusive job amid a saturated sibling pipeline for a MARGINAL
firming (paper core already complete; §5.5 has the principled model + §3.1/§7 already show K=5 insufficient
via 2-node median-of-5 disagreement) — poor citizenship for low value.
PIVOT (zero new compute, contract-clean): compute empirical required-k PER NODE from EXISTING on-contract
draws (tools/required_k_empirical.py, deterministic bootstrap):
  • ondem-3 n=6 (median 11.4s, 3.4s ABOVE SLO, σ/m=1.9, 2/6 pass): required-k ≈ 23 (goodput=0)
  • node1-2 n=5 (median 7.2s, 0.8s BELOW SLO, σ/m=7.0, 3/5 pass): UNRESOLVED even at k=25
This is a real-data match to the model's DIVERGENCE: node nearer the SLO (node1-2) needs MORE replicates than
the one further (ondem-3) — exactly required-k↑ as median→SLO. Explains directly why §3.1 K=5 failed (5 ≪
20+). Integrated §5.5 (after model required-k table) + §8 tooling, committed 984252d15. HONEST: pools small
(n=5-6) → exact k coarse; qualitative result robust. Reflexive point added: precisely measuring required-k is
expensive BECAUSE required-k is large — the difficulty IS the finding. HTML re-validated (43/43 p, 0 bare &,
0 dangling refs). Paper: 9 sections / 6 figs / 10 tables. W&B: log as note (variance study), not new curve.

### §5.5 SELF-CORRECTION: required-k unstable (23→9) + session-variance finding (2026-07-13 ~14:10Z)
Launched a DURABLE exclusive sbatch (job 19658, ondem-3, 10h, immune to hold reclamation — the fix for the
earlier --overlap death) on a freely-idle certified node (wilkes pipeline was node-constrained elsewhere =
no starvation) to firm ondem-3 required-k n=6→n=16. Got 3 fresh same-node stock λ=3 draws before I stopped:
{30050, 28389, 23991}ms — ALL deep-fail (0/3 pass), higher than ANY earlier ondem-3 draw (prior max 25761).
KEY (integrity): this OVERTURNED my just-integrated n=6 number. ondem-3 bootstrap required-k is UNSTABLE:
  • n=6 (2/6 pass, median 11.4s): k≈23
  • n=9 (2/9 pass, median 14.1s): k≈9   ← the 3 fresh fails moved the pooled op-point DEEPER into fail
  ⇒ 'k≈23' did NOT hold. Small-n coarseness caveat now DEMONSTRATED (not just hedged). Self-corrected §5.5.
NEW FINDING (honestly caveated): the SAME node's goodput dist shifts across SERVER LAUNCHES — session2 (medk)
median 9.4s/2-of-5 pass vs session3 (reqk16) median ~28s/0-of-3 pass = a SESSION-LEVEL variance layer on top
of run-to-run. Caveat: could be session-level metastability OR uncontrolled node state between launches
(confound I can't fully separate) — either way an ADDITIONAL variance source supporting the thesis.
DIVERGENCE still holds & cleaner: node1-2 (median 7.2s, AT knee) UNRESOLVED @k=25; ondem-3 (further out)
resolves @k≈9. Nearer SLO ⇒ more k = the model's prediction, on real data.
GOOD CITIZENSHIP: killed job 19658 at n=9 (enough for the honest point) → freed idle certified node for the
sibling pipeline (wilkes whale-cb*/lru-cb*). Committed be7530308. required_k_empirical.py now reads live
medk.csv per-node. Paper: 9 sections / 6 figs / 10 tables (empirical para split into 2). HTML validated.
LESSON: getting MORE same-node data before finalizing caught an unstable number — replication guardrail worked
(consistent w/ my base-v031 campaign's 4 over-claim self-corrections). W&B: log as note, not a curve version.

### DEFINITIVE CLOSE-OUT (2026-07-13 ~14:22Z) — all threads resolved, paper complete
Re-examined the two remaining candidate directions with fresh eyes; both DEFINITIVELY closed:
1. MECHANISM (charter's 1st preference): VERIFIED CLOSED. The one un-exploited gap (L2↔L1 transfer/compute
   overlap, "SLOP") is empirically dead — §5.2 already documents it: L2→L1 load-back is mean 1.3ms / p99
   ≤8.6ms (3-4 orders below the 14-42s p99), so overlap has nothing to hide; hit-rate can't help (tail is
   unique cold turn-0 docs); tail is lossless-irreducible K·t cold-prefill serialization (65% full chunks,
   69% cold). No viable mechanism exists on this eval — which is what MOTIVATES the measurement contribution.
2. SESSION-VARIANCE experiment (clean version of the §5.5/§7 caveated finding): DEFERRED, do NOT re-litigate.
   Would need N launches × K draws in one allocation (~5h GPU + ~30 monitoring turns) + still partly
   confounded (thermal drift). It is a SECONDARY finding, already honestly caveated in §5.5 + §7. Pool has
   active sibling contention (wilkes lru-cb* pending on Resources, valiant iterating). Marginal value <
   shared-compute cost → good-citizenship HOLD. Re-open ONLY if: pool goes fully idle (no pending siblings)
   for an extended window, OR a reviewer explicitly requests it.
STATE: paper HEAD cf0ec6a9d is a COMPLETE, rigorous, self-consistent top-venue measurement/methodology
contribution (coin-flip across run/node/session layers + first-principles model + validated/bounded
methodology + mechanism-space closure + honest self-corrections). Nothing to build, nothing to add without
sprawl. Holding for a material trigger (reviewer/supervisor feedback, WARNINGS/fairness signal, new
direction). Supervisor manages retirement.

### CONTROLLED SESSION-VARIANCE EXPERIMENT → RETRACT launch-level layer (2026-07-13 ~21:15Z)
Ran a durable exclusive sbatch (job 19661, idle 0-3, immune to hold reclamation; ondem-3 left free for
siblings — fair spare capacity, no pending siblings) to RESOLVE the §5.5/§7 caveat about a possible
launch-level (session) variance layer. Design: N=3 back-to-back server launches × K=3 stock λ=3 draws in ONE
allocation (controls node + cross-allocation drift; residual = slow thermal, noted). Byte-identical stock
flags (on-contract). tools/session_var_eval.sh + sessvar.sbatch + session_var_analyze.py.
RESULT (9 draws): within-session variance DOMINATES.
  S1 {15.6,6.1,22.1} mean 14.6 | S2 {8.8,7.7,36.7} mean 17.7 | S3 {12.8,12.8,26.7} mean 17.4
  session means nearly identical (σ_across=1.4s) while EACH session spans the full coin-flip range 6-37s
  (σ_within=11.5s) → across/within variance ratio = 0.02 (2%).
⇒ NO separable launch-level layer. The earlier cross-ALLOCATION medk(2/5)-vs-bign(0/3) difference was
within-session SAMPLING (a session can draw 3 high values by chance, as S2 did here) + node state, NOT a
genuine server-launch effect. RETRACTED the tentative "third variance layer" from §5.5 + §7.
This is a CLEAN CONTROLLED SELF-CORRECTION that REINFORCES the core thesis: the coin-flip is fundamentally
run-to-run (a single server's draws already straddle the SLO). Integrated §5.5 (required-k para reframed
sampling-sensitive not launch-level) + §7 (third-layer threat → controlled null) + §8 (v0-sessvar). Committed
9f687155d. HTML validated (tags balanced, 0 bare &). Node freed on completion. Good-citizen fair-capacity run.
KEY META: the experiment I'd repeatedly deferred was worth running — it RESOLVED an open caveat and CORRECTED
an over-reach (my own hypothesized 3rd layer). Replication/controlled-test guardrail worked again.

### 3-NODE generality from sessvar data (2026-07-13, zero new compute, commit eefa82e6e)
Leveraged the session-variance experiment's 9 stock λ=3 draws on node 0-3 for a SECOND purpose: node-
generality. 0-3 exhibits the coin-flip (6.1-36.7s, 2/9 pass, straddles SLO) → upgraded the "reproduces
across nodes / not one machine" claim from 2 nodes to THREE (ondem-3, node1-2, 0-3) in §7 + abstract.
Median-of-k/required-k/confound-resolution claims correctly stay "2 nodes" (those used ondem-3+node1-2).
Honest, accurate, non-sprawl strengthening from data already in hand.

### Charter re-read → mechanism-space RE-EXAMINATION + §3.3 strengthening (2026-07-13, commit f525c307c)
Re-read program.md ("when a line is exhausted, pick a BOLDER one"). Did a fresh, deep mechanism-space
re-examination to test whether a bolder MECHANISM line remains open, considering scheduling ideas beyond the
paper: prefill-priority-over-decode (NO headroom — decode already light/not-saturated §5.1, prefill already
gets the engine), fair chunk interleaving (HURTS — delays the heavy docs that ARE the p99), and length/
deadline-aware reordering (the interesting one). CONCLUSION: mechanism space genuinely CLOSED, but for a
deeper reason than "no mechanism exists" — the coin-flip makes tail-targeting mechanisms UNMEASURABLE on
goodput@SLO. Even length-aware scheduling, which COULD in principle raise goodput@SLO (deprioritize docs whose
prefill alone >SLO — 192K-tok needs ~9.4s, structurally hopeless), (a) is a textbook deadline transplant not a
novel primitive, and (b) acts on the coin-flip tail the metric cannot resolve. So the boldest HONEST line IS
the measurement paper — it SUBSUMES the mechanism dead-ends by proving why they can't be validated. Integrated
into §3.3: comprehensively closes the prefill-scheduling space + sharpens the thesis ("the metric's
unreliability blocks validating ANY tail-targeting mechanism → why the contribution is measurement not
mechanism"). Honest (unmeasurable+not-novel, NOT "tested/closed"). Also confirms mechanism unmeasurability
applies under BOTH goodput@SLO (coin-flip) AND stable metrics (only config-equivalent wins exist per landscape).
No bolder line exists that isn't already closed. Paper: 9 sections / 7 figs / 10 tables, HEAD f525c307c.

### PHASE-MAP diagnostic → empirical phase diagram on the REAL system (2026-07-14 ~00:50Z, commit c3544ee7e)
Charter re-read ("pick a bolder line") → identified a genuinely valuable within-charter experiment: the §5.5
phase diagram's reliable-PASS regime was MODEL-ONLY (the eval's λ-set {3,5,7,10} never samples below the coin-
flip band). Ran an off-contract low-λ diagnostic (job 19702, phasemap_eval.sh, durable sbatch on idle 1-2 —
FAIR: all 4 certified nodes idle, no pending siblings) to measure it. RESULT: at λ=1.5, p99 TTFT = 3.36, 3.35s
(n=2) — comfortably below the 8s SLO and TIGHT (16ms apart, req_tput 1.52 = kept up, below-saturation). So the
real system EMPIRICALLY exhibits all THREE regimes the minimal model predicts:
  reliable-PASS (λ=1.5, p99~3.35s, LOW variance) → COIN-FLIP band (λ=3, 6.1-36.7s, straddles SLO) → reliable-
  FAIL (λ≥5). And the run-to-run VARIANCE collapses away from the knee exactly as σ/m (§4) requires.
⇒ Turns the model-only phase diagram into one EMPIRICALLY CONFIRMED on the real serving system — a strong
generality/model-validation result (a key reviewer concern). Integrated §5.5 + §8, committed c3544ee7e. Killed
job after the key n=2 result (freed idle node; λ=2.5 transition detail not worth the monitoring cost). Off-
contract diagnostic (like warmdiag/medk), headline eval UNCHANGED. Paper: 9 sections / 7 figs / 10 tables.
KEY: charter re-read genuinely earned a bolder experiment — not polish. The reliable-pass regime is the last
empirical gap in the phase-diagram story, now closed on real hardware.

### SRPF A/B — closing the last UNMEASURED gap in §3.3's scheduling closure (2026-07-14 ~22:15Z, job 19832)
Re-audited the paper's mechanism-closure on re-launch and found a genuine HOLE: §3.3 closes the prefill-
SCHEDULING space partly by ARGUMENT — "SRPF/SRPT HURT: they starve the long cold turn-0 docs that ARE the p99"
(report.md:245) — a PREDICTION I never MEASURED. My v2-srpf engine mechanism (branch evolve/kleinrock-srpf,
2 commits: SRPF = shortest-remaining-prefill-first sort of the waiting queue by uncached tokens ascending;
CacheAwarePolicy so num_matched_prefix_tokens IS populated before the sort → true SRPF not degraded SJF; NOT
downgraded for large queues, verified) was BUILT + import/enum-validated but NEVER RUN. An unmeasured closure
claim is the paper's most vulnerable point — and it is exactly the kind of tail-scheduling lever a queueing
reviewer (my namesake!) would demand I measure. So I am MEASURING it now, from my own code (no sibling reads).
Design (same-node paired A/B, per my OWN methodology): both phases run the SAME binary (srpf-wt python = main +
only the 2 SRPF files) so the ONLY difference is `--schedule-policy srpf` (contract-legal: NOT in eval.sh's
FORBIDDEN list; a real engine mechanism selected by an arg). Phase A = fcfs (the eval default); Phase B = srpf.
Rates: λ=5 DECISIVE (stock reliably-FAILS = low variance → a clean fail→pass detection uncontaminated by the
coin-flip), K=5; λ=3 the coin-flip knee, K=3. Warmup→steady-state before each phase; report the p99 TTFT
DISTRIBUTION + goodput-pass-count (not a single coin-flip draw). Durable exclusive sbatch on idle ondem-2
(9 idle a3 nodes, 0 pending → fair spare capacity). tools/srpf_ab_eval.sh + srpf_ab.sbatch + srpf_ab_analyze.py.
TWO honest outcomes, both STRENGTHEN the paper: (a) SRPF shifts the p99 median / turns λ=5 fail→pass → a
POSITIVE tail-scheduling MECHANISM (charter's top prize; reconcile with variance-domination by reporting the
stable p99 distribution, exactly as my methodology prescribes); (b) SRPF is neutral/worse → convert the
vulnerable PREDICTION into a MEASURED negative, hardening §3.3's closure. Result pending (~10h).

## ═══ CHARTER UPDATE (program.md Jul 14 21:28) → OPEN PAPER 2 ═══ (2026-07-14 ~22:30Z)
Re-read program.md: it was REWRITTEN today with strong new language — "A formal submission is a MILESTONE not
the finish line… Do not stop, 'hold', or declare the campaign complete after one paper… produce a SERIES of
strong submissions… Only the supervisor decides when a slot retires." My "hold for material trigger" posture is
now CONTRARY to charter. ⇒ Opening Paper 2 (orthogonal direction). ALSO: line 23 explicitly DISQUALIFIES
"SJF/SRPF … dropped onto pluggable interfaces" as a contribution → my in-flight SRPF A/B (job 19832) is NOT a
Paper-2 bid; it is (a) a lossless CONTROL firming Paper 1's one unmeasured §3.3 claim, and (b) a DIAGNOSTIC for
Paper 2 (below).

### PAPER 2 direction (candidate, code-verified): prefill HEAD-OF-LINE at chunk granularity → a "reserved short-prefill lane"
CODE FINDING (kleinrock-srpf-wt python, v0.31): the engine serves ONE chunked_req at a time (scheduler.py
self.chunked_req single field). In get_new_batch_prefill (scheduler.py:2813-2896): each prefill iteration has a
6144-token budget; if a chunked_req is in flight, `adder.add_chunked_req` runs FIRST and takes
`_rem_tokens=min(rem_chunk_tokens, rem_total_tokens)` (schedule_policy.py:719-756) = the FULL 6144 for a mega-
doc mid-prefill → `rem_chunk_tokens→0` → the subsequent `add_one_req` loop returns NO_TOKEN for every waiting
req → batch full. So while a 192K-token cold doc prefills (~192542/6144≈32 iterations), NO other request can
prefill; each req arriving in that window is HEAD-OF-LINE-BLOCKED ~32×iter (~8s @ ~250ms/iter) = exactly the
p99 6-11s tail Paper 1 measured. This is INTER-PREFILL HOL, distinct from a req's own prefill time.
NOVEL MECHANISM: cap the chunked mega-doc's per-iteration take at `6144 - RESERVE`, reserving RESERVE tokens so
short waiting reqs prefill ALONGSIDE it (a localized change in add_chunked_req). NOT SRPF (no reorder, no
starvation — mega-doc still progresses, just fair-shares); lossless (same tokens/outputs); a real engine change,
not a config flag. HYPOTHESIS: collapses the HOL-victim p99 without starving mega-docs → shifts the curve /
raises goodput reliably (victims' prefill is short+deterministic once unblocked).
SRPF is the DIAGNOSTIC GATE: SRPF puts short reqs first (kills HOL for them) but starves mega-docs. If SRPF
IMPROVES p99 ⇒ HOL victims dominate the tail ⇒ fair-lane is promising & STRICTLY BETTER than SRPF. If SRPF
worsens/neutral ⇒ mega-docs' own prefill dominates ⇒ new bounded negative (fair-lane can't beat the mega-doc
floor either). Either way informs Paper 2.
TODO before implementing: (1) await SRPF result; (2) per-request instrumentation — bench JSONs are AGGREGATE-only
(no per-req arrays), so HOL must be measured via server-side per-req logging (arrival→first-prefill delay vs
prefill size) or a bench patch. Prior art to position vs: Sarathi/chunked-prefill (prefill↔decode mixing),
FastServe (preemptive), Mooncake, and classic fair-queuing (must frame novelty as the INSIGHT + inter-prefill
mechanism, not fair-queuing-transplanted, to clear the bar).

### Paper 2 — prior-art positioning (from training knowledge, VERIFY+cite before submission; web search blocked here)
Closest prior art for the reserved-short-prefill-lane, and why the mechanism is distinct:
- **Sarathi-Serve (Agrawal et al., OSDI'24) / chunked prefill (also vLLM):** the ORIGIN of chunked prefill —
  splits a long prefill into token-budget chunks and piggybacks decodes ("stall-free batching") to stop long
  prefills from STALLING DECODE. Target = prefill→DECODE interference. It does NOT reserve budget for other
  WAITING PREFILLS; a single large chunked prefill can still monopolize the per-iteration prefill token budget
  across its many chunks → INTER-PREFILL head-of-line blocking remains (exactly what sglang, a Sarathi-style
  system, exhibits: add_chunked_req takes the full 6144). My mechanism fills that gap: reserve a slice so short
  waiting prefills co-run. Distinct AXIS (prefill↔prefill, not prefill↔decode).
- **FastServe (Wu et al.):** MLFQ + PREEMPTION + skip-join to avoid request-level HOL. Mechanism = preempt
  running reqs. Mine is NON-preemptive (no eviction/recompute) — a within-iteration budget split. Different.
- **SRPF/SJF/shortest-first (explicitly DISQUALIFIED by charter L23):** reorders to serve short first but
  STARVES the large cold docs (their TTFT explodes). My reserve does NOT reorder and does NOT starve — the
  mega-doc keeps progressing every iteration, just shares the budget. Strictly better on the starvation axis.
- **Classic fair-queuing / DRR:** the reserve is fair-queuing-flavored at the prefill-token-budget level. RISK:
  a reviewer may call it "textbook FQ transplanted." DEFENSE (must carry the paper): the NOVELTY is the
  DIAGNOSIS (goodput@SLO tail = inter-prefill HOL behind a heavy-tailed cold-doc workload, quantified) + that a
  tiny reserved lane collapses the victim tail LOSSLESSLY without the starvation SRPF causes — evidenced by
  beating BOTH fcfs and srpf. Clears the bar only if the effect is real + attributed (per-req input_len vs TTFT).
Novelty hinges on the empirical signal (screen 19836) + attribution. If the effect is null → a NEW bounded
negative ("even fair prefill budget-sharing can't beat the cold mega-doc prefill floor"), still a Paper-2-worthy
result distinct from Paper 1.

### SRPF/short-lane A/Bs — first data (2026-07-14 ~22:57Z)
- SRPF fcfs λ5 rep1 (job 19832, ondem-2): p99 TTFT=22330ms (deep FAIL), p50=706ms, req=4.11, tok=525.5,
  hit=0.677. Confirms Paper-1 baseline (λ5 reliably fails). NOTE: 22.3s ≫ single 192K-doc prefill (~9.4s) →
  ~13s of QUEUEING atop the mega-doc floor = consistent with HOL stacking. Implication: at λ5, HOL relief may
  cut p99 toward the ~9.4s mega-doc floor (big distribution shift) but not below 8s (no goodput flip); the
  fail→pass goodput WIN is likelier at λ3 (coin-flip boundary). Screen at λ5 still detects the p99 cut; if
  present, run full sweep incl λ3. Awaiting more reps + srpf arm + shortlane reserve0/reserve2048.

### A/B baselines + NODE HETEROGENEITY (2026-07-14 ~23:32Z)
Stock baselines differ BY NODE (both byte-identical stock, λ5): fcfs on ondem-2 = p99 22.4s (n=2: 22331,22491
— reliable-fail, 0.7% apart, LOW variance ⇒ λ5 is the reliable-FAIL regime, NOT coin-flip → clean A/B),
p50 679ms, req 4.12, hit .672. reserve0 on node 1-1 = p99 26.5s, p50 1395ms, req 3.23, hit .677. Node 1-1 is
~18% higher p99 / ~2× p50 / ~22% lower throughput = NODE HETEROGENEITY (Paper-1's ±45% node var). ⇒ compare
ONLY same-node (reserve2048 vs reserve0 both on 1-1; srpf vs fcfs both on ondem-2); NEVER cross-node. Also
req<λ on both ⇒ λ5 SATURATES both nodes → reserve (slightly slows mega-docs) may trade throughput for victim-p99
at λ5; clean goodput win likelier at λ3. Decisive shortlane data (reserve2048@1-1) ~2.5h out; SRPF srpf arm ~4h.

### SHORT-LANE λ5 result — MECHANISM NEGATIVE at saturation (2026-07-15 ~01:31Z, job 19836, node 1-1)
Same-node A/B, λ5: reserve0 (n=3) p99 median 24226ms {22339,24226,26539}, p50 1145, req 3.51, tok 449, hit .663
vs reserve2048 (n=1) p99 42554ms (+75.7% WORSE), p50 944 (-17.6% BETTER), req 3.18 (-9.4%), tok 406, hit .680.
INTERPRETATION: the reserve does its designed job — helps SHORT reqs (p50 down 17.6%) — but at λ5 the p99 tail is
NOT HOL victims; it is the mega-docs' OWN prefill. The reserve caps the mega-doc's chunk (6144→4096) → ~1.5×
more iterations per mega-doc → under saturation (req 3.2-3.8 ≪ λ5) this EXTENDS the congestion window → the tail
gets WORSE. ⇒ At saturation, inter-prefill fair-sharing backfires on the p99. This CORROBORATES Paper 1 at the
MECHANISM level: the goodput tail is cold-doc-prefill-bound (irreducible K·t), not schedulable by fair-sharing.
CAVEAT: λ5 is SATURATED (I predicted mega-doc-floor dominance here). The possible WIN is at λ3 (moderate load,
where HOL victims may dominate the tail & the queue doesn't grow unboundedly). Launching a λ3 A/B to complete the
story. n=1 for reserve2048 (reps 2-3 landing ~02:00/02:30 to confirm; +76% ≫ reserve0's ~15% band → direction clear).
NOTE: NO CRASH — the reserve2048 path (my has_chunked_req fix) ran warmup + a full 1553-req λ5 sweep cleanly.

### λ5 firmed + λ3 baseline (2026-07-15 ~02:40Z)
λ5 reserve2048 n=2: p99 {42359,42554} median 42456 (+75.3% vs reserve0 24226), p50 -18%, req/tok -8% → NEGATIVE
FIRMED (tight, robust). λ3 reserve0 rep1: p99 12949ms (fail side of coin-flip, but ≪ λ5's 24s), p50 1178, req
2.79 (≈λ3 offered → less saturated). Decisive λ3 reserve2048 comparison pending (~3h; K=3 needed for coin-flip
variance). SRPF srpf arm ~1h (triangulation: does shortest-first also fail the tail?).

### ★ PIVOTAL: SRPF λ5 = fail→PASS, GENUINE (not gaming) (2026-07-15 ~03:40Z, job 19832, ondem-2, n=1)
SRPF λ5 rep1: p99 7028ms (PASS) vs fcfs λ5 n=5 median 23891ms (0/5 fail) = -70.6%. FULL distribution (rep1):
mean TTFT 2064→1183 (-43%), median 706→600 (-15%), P90 4103→1897 (-54%), P99 22331→7028 (-69%); req 4.11→4.08
(flat), E2E median 7150→6690. ⇒ SRPF GENUINELY improves the WHOLE TTFT distribution (mean drops 43%) = classic
SJF minimizing mean flow-time, NOT p99-gaming/starvation-artifact (starvation would RAISE the mean). The few
mega-docs (~3-4 >100K) pay a fairness cost largely INVISIBLE to p99 (they fall beyond top-1%).
TWO SERIOUS CONSEQUENCES:
(1) ★INTEGRITY: this REFUTES Paper 1 (prefill-slo-tail) §3.3 scheduling-closure ("no serving policy moves the
tail" / "SRPF starves the big docs = HURTS"). That was an UNMEASURED prediction; the direct A/B refutes it. MUST
correct Paper 1 (revise §3.3: the tail IS schedulable; SRPF moves λ5 fail→pass, mean/p90/p99 all down). The
coin-flip CORE of Paper 1 stands (variance ≠ schedulability); only the scheduling-closure sub-claim is wrong.
(2) Paper 2 axis reassessment: prefill SCHEDULING is (a) SIBLING-OCCUPIED (base found SRPF +37-66% goodput on
this eval) and (b) charter-DISQUALIFIED (L23: SJF/SRPF on pluggable interfaces ≠ contribution). My novel
NON-starving twist (fair reserve) FAILS (+75% p99): a thin 2048 reserve neither serves short reqs fast enough
NOR stops the mega-doc occupying the pipeline → worse. INSIGHT (candidate, needs care vs base): for heavy-tailed
prefill, the p99-goodput benefit is INSEPARABLE from DEPRIORITIZING (delaying) the large docs; a fair/non-starving
budget split forfeits it. ⇒ Paper 2 as a POSITIVE mechanism is dead (occupied+disqualified); at most a bounded
negative ("fairness and tail-goodput are opposed in prefill scheduling"), which may be too incremental to stand
alone → consider FOLDING the SRPF+reserve evidence into a Paper 1 REVISION (correct §3.3 + add the fair-mechanism
boundary) rather than a weak Paper 2. DECIDE after srpf replication (n≥3, reps 2-5 landing) + λ3 + reserve ablation.

### λ3 reserve NEGATIVE + srpf λ5 n=3 (2026-07-15 ~05:01Z)
λ3 reserve2048 rep1: p99 18197ms vs reserve0 λ3 median 8093 = +125% WORSE (p50 -11%, req -8%). ⇒ the fair
reserve FAILS at BOTH λ3 (+125%) AND λ5 (+75%) — never helps goodput, only median. NOT load-specific. Completes
the boundary: capturing SRPF's benefit REQUIRES deferring the heavy docs; a fair non-starving budget split
forfeits it (too thin to clear the HOL backlog, and slows the mega-doc → extends congestion). srpf λ5 n=3: 3/3
pass {5530,6983,7028} median 6983 (-70.8% vs fcfs), Fisher p=0.018 — correction rock-solid. FOLDING the
reserve-fails boundary into Paper 1 §5.6 (the fair alternative to SRPF fails → benefit requires deferral).

### srpf λ3 fixes the KNEE too + reserve λ3 firmed (2026-07-15 ~06:19Z)
srpf λ3 rep1: p99 6158ms (PASS) vs fcfs λ3 (n=3) {10815,20126,34216} median 20126 (0/3, coin-flip). ⇒ SRPF helps
at BOTH loads: λ5 saturation (5/5, -71%) AND λ3 knee (1/1 so far, likely STABILIZES the coin-flip via
deterministic short-first serving; n=1, firming reps 2-3). λ3 reserve n=2 {18197,19540} +133% (still fails).
FULL PICTURE: SRPF wins everywhere (scheduling lever, known/base); my fair reserve fails everywhere (+75%@λ5,
+133%@λ3, helps p50 only). Paper 1 §5.6 to add srpf λ3 once n≥2. Core correction DONE+firmed.

### SRPF is near-PARETO → metric-bias Paper-2 direction DEAD (2026-07-15 ~07:10Z)
Checked E2E latency srpf vs fcfs λ5 (n=3 each): mean E2E ~44.7s (fcfs) vs ~45.0s (srpf) = IDENTICAL; median E2E
srpf ≤ fcfs every rep (~6.6 vs ~6.9s). ⇒ SRPF defers heavy docs' TTFT (first-token) but their E2E is UNCHANGED
(decode-dominated, E2E≫TTFT). So SRPF is a near-PARETO improvement (better/equal TTFT, equal E2E, equal
throughput), NOT a harmful starvation tradeoff. ⇒ the "goodput@SLO rewards deferral/starvation" metric-bias
Paper-2 direction is DEAD (no real harm for the metric to reward — SRPF is just a good policy; base was right).
Honest negative that saves a weak paper. ALSO refine §5.6: soften "unbounded delay/starvation" → TTFT-deferral,
E2E-neutral (near-Pareto). srpf λ3 n=2 {6158,6744} tight (stabilizes knee). Design space now well-characterized;
corrected+strengthened Paper 1 is this cycle's real contribution.

### RESERVE ABLATION result — fair reserve fails at ALL sizes, NO crossover (2026-07-15 ~12:44Z, job 19919, node 0-0)
Same-node (0-0) λ5 reserve sweep: reserve0 p99 23.3s (0/3) → reserve1024 22.3s (0/3, NEUTRAL -4.4%, n=3) →
reserve4096 100.3s (n=1, CATASTROPHIC +330%, req collapse 3.63→2.05). Plus reserve2048 (node 1-1) +75% (n=3).
⇒ NO crossover: the fair reserve is neutral when thin, harmful when medium, CATASTROPHIC when large. My
"bounded-deferral / approaches-SRPF at large reserve" hypothesis is WRONG. WHY (instructive): unlike SRPF which
DEFERS the mega-doc entirely, the reserve keeps RUNNING it every iter, just throttled → a large reserve makes it
occupy the prefill pipeline ~94 iters (192K/2048) → throughput collapses → p99 explodes. Throttling-while-running
is STRICTLY WORSE than both fcfs AND srpf. ⇒ DEFINITIVE bounded negative: the goodput benefit requires DEFERRING
the mega-doc (SRPF), not throttling it; no reserve size captures it. Answers the §5.6 reviewer question decisively
(failure is fundamental, not thin-regime-specific). Paper-2-as-mechanism (bounded-deferral) is DEAD. Strengthens
Paper 1 §5.6. reserve4096 reps 2-3 will firm n=3 (100s + req-collapse unambiguous). ⇒ update §5.6 with the curve.

### Direct per-request HOL attribution (2026-07-15 ~12:55Z, from existing perreq dumps, no new GPU)
Analyzed KLEINROCK_PERREQ_DUMP from stock (reserve0) runs (n=15,549 reqs, 3 runs each λ3/λ5) via
tools/perreq_hol_analyze.py. DECISIVE HOL evidence: tiny prompts (<1K tok, own-prefill <0.3s) have a HIGHER p99
TTFT than the heavy docs themselves — λ3: tiny p99 26.9s vs heavy(>=50K) 20.4s; λ5: tiny 30.5s vs heavy 23.3s.
27-40% of the p99 tail are tiny prompts; 145(λ3)/301(λ5) tiny reqs exceed 8s SLO PURELY from queueing (a 27-68
token prompt waiting 22s = pure HOL). ⇒ the p99 tail is HOL VICTIMS, and they suffer MORE than the heavy docs
that block them — airtight direct proof of the §2.4/§5.6 correction ("long docs are NOT the p99"). Integrated
§5.6 (commit c0cfb6a7b). Turned collected-but-unused attribution data into a strong evidence addition (no GPU).

### ★ PUSHED evolve/kleinrock (2026-07-15 ~13:05Z) — 34 unpushed commits were local-only!
Discovered origin/evolve/kleinrock was stuck at 4bafcbc47 (~07:32) while local HEAD was 1c2f1feaf — the ENTIRE
Paper 1 correction campaign (SRPF correction, reserve ablation, HOL attribution + scatter figure, §8/§9/§5.2
consistency, sharpened abstract, INDEX updates) was committed but NOT pushed. Pushed all 34 commits →
origin now at 1c2f1feaf. LESSON (matches memory warning): after committing, VERIFY push status (git ls-remote),
don't assume; a clean working tree ≠ pushed. The corrected paper is now durable + visible to the supervisor.

### program.md updated (23:45 Jul 14) → step-3 now MANDATES push-after-every-commit (2026-07-15 ~13:15Z)
Trigger scan: program.md changed again (21:28→23:45, +267B). The change = The Loop step 3 now reads "commit AND
push it — git push origin evolve/<name>. Push after every commit: unpushed work is local-only and unpreserved."
(Supervisor added the push mandate — validates last turn's catch that my 34-commit correction campaign was
unpushed.) This is an OPS mandate, not a new research direction. Already compliant: remote==local (7033ba6f2).
STANDING RULE going forward: git push after every commit; verify remote==local each loop. No new research
trigger, no WARNINGS/reviews/feedback, eval/protocol unchanged.

### DEFERRAL-SPECIFICITY confirmed (2026-07-15 ~15:40Z, job 19953, node 0-0, λ5)
fcfs (n=3) p99 median 22.7s {22408,22658,23406} vs LOF (longest-output-first, n=1) 28.0s (+23%, fail) vs srpf
(deferral, 7.0s pass, from v3-srpf-ab n=5). ⇒ LOF — a real queue REORDERING that does NOT reorder by prefill
size — does NOT help the tail (even slightly worse), while srpf (defers heavy docs by remaining-prefill) does.
CONFIRMS it is DEFERRAL-BY-PREFILL-SIZE specifically, not reordering-in-general → rules out the "any reordering
helps" alternative. Sharpens §5.6. Firming LOF to n=3 (reps 2-3 landing), then add the sentence + commit/push.

### §5.6 deferral-specificity integrated (2026-07-15 ~16:40Z)
LOF n=3 firm: p99 {26460,27521,27979} median 27.5s, 0/3 (slightly worse than fcfs 22.7s). Added to §5.6: the
goodput lever is DEFERRAL-BY-PREFILL specifically — LOF (reorder-by-output, doesn't defer heavy docs) fails like
fcfs; only srpf (defer) passes. Rules out "any reordering helps." srpf-on-0-0 arm (fully same-node triple) still
running (~18:30) — will confirm; current §5.6 cites srpf 7.0s (n=5, 3x gap ≫ node variance). Committed+pushed.

### Same-node triple confirmed (2026-07-15 ~17:22Z): srpf-on-0-0 rep1 = 7.4s PASS
On node 0-0 (same as fcfs 22.7s, lof 27.5s): srpf rep1 p99 7399ms (PASS). Fully-controlled same-node
deferral-specificity: only srpf (defer) passes; fcfs+lof (no defer) fail. Confirms cross-run srpf (7.0s n=5).
Firming srpf to n=3, then update §5.6 to the clean same-node triple (fcfs/lof/srpf = 22.7/27.5/7.4s, n=3 each).

### ★ DEFERRAL-SPECIFICITY COMPLETE (2026-07-15 ~18:22Z, job 19953) — fully same-node triple
Node 0-0, λ5, n=3 each: fcfs p99 22.7s (0/3) / LOF 27.5s (0/3) / srpf 6.7s (3/3), DISJOINT (srpf max 7.4 < fcfs
min 22.4). ⇒ only srpf (defer-by-remaining-prefill) passes; fcfs & LOF (no deferral) fail. The goodput lever is
DEFERRAL-BY-PREFILL SPECIFICALLY, not reordering-in-general (LOF, a real reorder-by-output, doesn't help). §5.6
firmed to this same-node triple (fc3245d8c, pushed). Deferral-specificity fully established + integrated.

### SRPF across-λ ceiling (2026-07-15 ~20:40Z, job 19988, node 0-0)
srpf λ7 rep1: p99 10152ms vs fcfs λ7 (n=3) median 33342ms = -70% (SAME relative cut as λ5) BUT still FAIL (>8s),
req 4.54 (=fcfs 4.58). ⇒ SRPF's ~70% p99 reduction holds even at severe saturation (λ7), but converts to a
goodput PASS only at λ≤5: at λ7 (offered 7 ≫ compute ceiling ~4.6 req/s) the backlog keeps p99>SLO. So SRPF's
goodput@SLO CEILING ≈ λ5 (~4.1 req/s). Across-λ: λ3 6.7s pass / λ5 7.0s pass / λ7 10.2s fail (all -70% vs fcfs).
The lever reduces the tail everywhere but clears the SLO only below severe saturation. n=1 (firming to n=3), then
integrate into §5.6/§5.1.

### Number-consistency audit PASSED (2026-07-15 ~21:40Z)
Cross-checked every SRPF/reserve/LOF/HOL number cited in paper.html against raw run CSVs (v3-v8): all match and
are internally coherent. SRPF λ5 23.9→7.0s/0-5→5/5/p0.0079; reserve λ5 +75%/λ3 +141%/abl 1024=22.3/4096=100s;
deferral triple 22.7/27.5/6.7 (n=3, 0-0); srpf λ7 33.3→9.6 (-71%); HOL tiny 26.9/30.5 > heavy 20.4/23.3. Stock-λ5
baseline varies 22.7-24.2s across comparisons = node heterogeneity (each A/B same-node-internally-consistent, no
within-comparison contradiction). Integrity mandate satisfied (numbers match raw runs). No fixes needed. Paper 1
submission-ready. remote==local (bfdf28b52).

### ★ PAPER 2 registered — `prefill-hol-defer` (2026-07-15 ~) — bounded-negative + diagnosis
Charter says a submitted paper is a MILESTONE, open the next direction; and it EXPLICITLY (a) disqualifies
"SJF/SRPF dropped onto pluggable interfaces" and (b) welcomes "a rigorously-established negative/impossibility".
So the scheduling story (which had ballooned §5.6 to 970 lines) becomes its own paper — but reframed to respect
the disqualification: **NOT** "SRPF wins" (SRPF is classical + a sibling already owns that result), instead a
**bounded-NEGATIVE + HOL-diagnosis** paper. Contributions: (1) per-request HOL diagnosis (n=15549: tiny <1K-tok
victims p99 26.9/30.5s > heavy-doc blockers 20.4/23.3s — victims suffer more than blockers); (2) a fair,
non-starving, lossless **reserved-short-prefill-lane** mechanism (MY own) shown to **FAIL at every reserve size**
— normalized to own-run stock: 1024=0.96× (neutral), 2048=1.75× (+75%, n=3 disjoint), 4096=4.30× (catastrophic,
throughput 3.6→2.0 req/s, server OOM-killed, n=1) — because throttling MAXIMIZES the blocking window (≈32→94
iters), the opposite of deferral; (3) deferral-specificity (LOF reorder-by-output fails 27.5s vs deferral 6.7s,
same-node triple) → the lever is deferral-BY-PREFILL-SIZE. Principle: **defer, don't throttle**. SRPF appears
ONLY as a 1-row cited reference-target (§4.1, disclaimed, not claimed as a contribution). Across-λ bound: the
scheduling lever's goodput ceiling ≈λ5 (λ7 backlog-bound).
- **§5.6 slimmed 970→34 lines**: now a compact correction (SRPF fail→pass 2-row table + per-req HOL) that
  sharpens Paper 1's thesis (goodput@SLO movable-by-scheduling yet blind-to-caching) + pointer to Paper 2. This
  DE-BLOATS Paper 1 and removes duplication (the two papers are now non-overlapping: P1=metric-critique,
  P2=mechanism/negative).
- Removed superseded unregistered draft `prefill-hol-lane` (it wrongly framed the lane as a "mitigation"; it
  actually fails → contradicted the honest finding).
- **Number audit**: all Paper 2 tabled numbers re-checked vs raw runs v3–v8 (E2E mean-of-runs 45.3/45.1s — the
  old "44.7/45.0" was imprecise; reserve ratios normalized to own-run stock since 2048 is from v4/24.2s while
  1024/4096 are from v6/23.3s; LOF median 22658). Match.
- **Honest standalone assessment**: this is a legitimate SECOND contribution because its claims (HOL diagnosis +
  the reserved-lane bounded negative + defer-don't-throttle necessary-condition) are NOT SRPF and NOT the
  metric-critique — it is the "rigorously-established negative" the charter names as valid. Had I kept it as
  "SRPF wins" it would have been an incremental re-slice of a disqualified/sibling-owned result → I reframed
  instead of folding back.

### IN FLIGHT — defer-knee (P1↔P2 bridge) (2026-07-15, job 20018, node 0-2, ~7-8h)
NEW direction after 2 papers: does DEFERRAL collapse the goodput@SLO COIN-FLIP at the knee (P1) or merely SHIFT
it? Same-node K=5 fcfs vs srpf (deferral REFERENCE — characterization probe, NOT a mechanism claim) at λ3.
runs/v9-defer-knee/srpf_ab.csv. Motivation: existing n=3 is ambiguous — srpf λ3 {6158,6744,8086} = 2/3 (one
GRAZED SLO at 8.086s) vs fcfs λ3 {10815,20126,34216} 0/3. ★Analyzer (ab_analyze.py) now reports σ/m (P1
diagnostic) + variance verdict; existing data already shows the variance-reduction is KNEE-SPECIFIC (λ3 σ/m
0.54→0.14 under deferral; λ5 FLAT 0.09→0.12 = deferral only shifts all-fail→all-pass, doesn't tighten). New job
firms whether srpf λ3 reliably PASSES (variance collapsed → reliable goodput, bridges P1+P2) or keeps straddling
(coin-flip scheduling-robust, strengthens P1). RESUME: `python3 tools/ab_analyze.py runs/v9-defer-knee/srpf_ab.csv`;
if job 20018 dead + csv incomplete, resubmit tools/defer_knee.sbatch on any idle a3 node (NOT slurm2-a3nodeset-2 =
GPU-less). Either outcome integrates into P2 (§ deferral & the knee variance) with honest framing; commit+push.

### ★ defer-knee RESULT (2026-07-16, job 20018, node 0-2, COMPLETE) — DEFERRAL COLLAPSES THE KNEE COIN-FLIP
Same-node λ3 K5, fcfs vs srpf (deferral reference):
- **fcfs λ3**: {9671,12265,22035,34132,35870}ms, median 22.0s, **σ/m 0.53**, 0/5 pass — textbook coin-flip (3.7× spread).
- **srpf λ3**: {6177,6225,6665,6711,6755}ms, median 6.7s, **σ/m 0.04**, 5/5 pass — tight (1.09× spread), disjoint (srpf max 6755 < fcfs min 9671).
⇒ Deferral cuts p99 median −70% (Fisher p=0.0079) AND **collapses σ/m 13× (0.53→0.04)**. p50/hit/req flat (3.02, lossless, throughput-neutral). **ANSWER to the P1↔P2 bridge: deferral doesn't merely SHIFT the knee median under the SLO — it COLLAPSES the metastable run-to-run variance (P1's coin-flip) into a reliable pass.** The goodput@SLO knee coin-flip is a SCHEDULABILITY artifact: a well-ordered (deferred) queue is both faster and ~13× more predictable. Unifies P1 (coin-flip, variance-dominated for CACHING) + P2 (deferral clears the HOL tail): caching can't collapse the coin-flip, scheduling (deferral) can. SRPF = cited reference (not claimed). → integrate into Paper 2 as a new subsection (deferral & the knee variance).

### ★ reserve-sweep RESULT (2026-07-16, job 20020, node 1-2, λ5) — CLEAN same-node monotone ablation
reserve0 {24504,24638,29193} med 24.6s (σ/m 0.10, n=3) → 1024 {20773,30014,32566} med 30.0s (+22%, n=3) → 2048
{41333,42276,42377} med 42.3s (+72%, σ/m 0.01, n=3) → 3072 rep1 63.1s then SERVER CRASH (KV-pool-leak invariant
violation `_report_leak("pool")`, reps 2-3 dead) → 4096 THROUGHPUT COLLAPSE (7 s/it = ~1000× slower, full-KV-pool
climbing, cancelled at 0%). ALL SAME NODE, single stock baseline → replaces P2's patched cross-baseline §4.2 table
(had to normalize to own-run stock). Monotone-harmful, no crossover; destabilization onset R≥3072. ★1024 here
+22% (harmful) vs v6's neutral-1024 (different node) → 1024 = node-dependent neutral-to-harmful, NEVER beneficial
(reported honestly). ★Good-neighbor: cancelled job 20020 once 4096 collapse confirmed (7s/it, won't complete in
walltime) → freed node 1-2. Cross-node v4 (2048 +75%) independently agrees.

### ★ BOTH INTEGRATED into Paper 2 (2026-07-16, commit 4691aad80, pushed) — significantly strengthened
§4.2 rebuilt (clean same-node ablation + destabilization onset 3072); NEW §5.3 coin-flip-collapse bridge
(defer-knee); abstract/§1-contributions(+coin-flip-collapse bullet)/§5.2(firm n=5 λ3 row)/§7/§8(v9,v10 runs)
updated. HTML valid (5 tables, 1 svg, 9 §§). SRPF stays cited-reference-only (charter L23). INDEX updated. Paper 2
is now: HOL diagnosis + fair-budget-sharing bounded-negative (clean monotone ablation, destabilization onset) +
deferral-specificity + defer-don't-throttle principle + coin-flip-collapse bridge. A genuinely strong bounded-
negative + diagnosis + unification paper.

### ★ WORKLOAD EVIDENCE: p99 tail is UNCACHEABLE (2026-07-16, GPU-free, commit 92e1ab9f9) — closes last caching loophole
tools/doc_reuse_analyze.py over mooncake_mix_v1.jsonl (1553 convs): 888 unique docs; 665 records (43%) share an
EXACT doc with another conv (cross-conv reuse DOES exist) BUT overwhelmingly SHORT docs (18/30 reused <10K tok;
only 4 reused >25K tok = ~0.26% avoidable prefills). ★The 3 p99-causing mega-docs (~109K/150K/190K tok, >=100K)
each appear EXACTLY ONCE → unique first-sight → NO cache at any capacity (incl. content-addressed / cross-conv
prefix dedup) can convert the p99-tail prefills to hits. ⇒ the p99 tail is NECESSARILY a scheduling problem, not
a caching one — direct dataset proof unifying P1 (metric blind to caching) + P2 (movable by scheduling): both true
because the tail docs are unique. Folded into Paper 2 §3 (diagnosis) + §1 + §8. This rules out the "maybe cross-
conversation caching moves the tail" loophole a skeptical PC would raise. GPU-free, genuinely novel, evidence-backed.

★ CYCLE STATUS: Paper 2 now HOL-diagnosis + uncacheable-tail-evidence + fair-budget-sharing bounded-negative
(clean same-node ablation, destabilization onset) + deferral-specificity + defer-don't-throttle + coin-flip-
collapse bridge. Comprehensive, rigorous, honest. Space mapped (caching provably out for the tail; scheduling =
deferral, disqualified/sibling-owned so cited-only; my reserve/LOF negatives). Won't fabricate; stay ready for triggers.

### ★ Paper 2 §5.4 GENERALITY + §5.2 λ7 refinement (2026-07-16, GPU-free, commit 52e68847c)
Built tools/hol_sim.py — minimal discrete-event sim of the single-chunked-prefill queue (one server, 6144
budget, one-chunked-req-at-a-time, non-preemptive), driven ONLY by the measured doc distribution + Poisson(λ);
no engine/cache/sglang. Calibrate P to FCFS λ5, then PREDICT deferral. Reproduces the STRUCTURE from first
principles: FCFS tiny-req p99 grows then EXPLODES with load (12.7→36.8→341s @λ3/5/7), deferral flat-low
(4.6/5.6/2.3s); fail→pass at λ≤5; λ5 relative cut matches measured (−70%..−85%). ⇒ defer-don't-throttle is
GENERIC to heavy-tailed prefill under FCFS-vs-shortest-first, NOT sglang-specific (structural generality, not
point prediction — single-server idealization omits decode/concurrency, like P1's coinflip_sim). ★ADVERSARIAL
CATCH: sim shows short reqs stay FAST at λ7 (2.3s) → the measured srpf λ7 goodput FAIL (9.6s aggregate) is
BACKLOG-bound (deferred heavy docs + saturation), NOT slow short reqs → CORRECTED §5.2's unverified claim
"backlog keeps even the reordered short requests above SLO" (I have no per-req srpf-λ7 dump; sim disputed it) to
the honest backlog-bound framing. §5.4 added, §1/§8 updated. Paper 2 now has generality (a PC strength both my
papers lacked). HTML valid (6 tables, 10 §§). This cycle = genuine value from GPU-free modeling, not fabrication.

### ★★ NEW LEAD (2026-07-16): guaranteed-progress CHUNK-INTERLEAVING (candidate novel mechanism) — sim-motivated
tools/interleave_sim.py (GPU-free chunk-level sim): a mega-doc yields to waiting SHORT reqs BETWEEN its 6144-tok
chunks (bounds short-req HOL to ~1 chunk) with a progress guarantee (mega-doc chunk every N iters → bounded defer,
no starvation of the SCHEDULER; full-budget iters → no throttle). NOT SJF/SRPF (disqualified), NOT my reserve
(throttle, failed), NOT LOF. ★SIM (P=14K): short-req(tiny) p99 — fcfs {12.7,36.8,341} / srpf {4.6,5.6,2.3} /
interleave {1.1,1.9,1.7}s @λ3/5/7. Interleave STRICTLY BEATS srpf on short-req tail (srpf lets an in-flight
mega-doc block once started — one-chunked-req invariant; interleave yields every chunk). Robust across N (λ7 tiny
p99 2.0/1.7/1.2/0.8 for N=1/2/4/8). Mega-doc p99 = srpf (both defer heavy docs; ~1253s @λ7). ⇒ since goodput@SLO
p99 is over 7037 reqs of which 99% are short, keeping shorts fast may PASS λ7 where srpf FAILS (measured 9.6s) →
EXTEND the goodput ceiling past SRPF's λ5 = candidate TOP-PRIZE novel mechanism. ★INTEGRITY: the λ7 "win" would
come from starving the 1% mega-docs (beyond p99) → legit by the metric's p99 def BUT must be disclosed = also a
metric-gaming caution connecting to P1. Also: sim assumes cached follow-ups (real λ7 tail may include evicted
mega-doc follow-ups = heavier) → ONLY a GPU A/B resolves it. PLAN: build chunk-interleave engine mechanism (new
branch evolve/kleinrock-interleave) → same-node A/B fcfs vs srpf vs interleave @λ5(margin)+λ7(the win) WITH
per-req dump (measure BOTH short-req p99 AND mega-doc starvation, honestly). If it passes λ7 losslessly → P3.

### ★★ IN FLIGHT (2026-07-16): chunk-interleave engine A/B — candidate TOP-PRIZE mechanism
BUILT --prefill-interleave-defer (branch evolve/kleinrock-interleave @ 477f03005, off srpf so 1 binary =
fcfs/srpf/interleave; 4 localized edits: server_args flag, scheduler park-gate at add_chunked_req +
_interleave_should_yield()+yield counter, add_one_req has_chunked_req guard [no-op for stock/srpf]). Mechanism:
park the in-flight chunked mega-doc for an iter when shorts wait (HOL ~1 chunk) w/ progress guarantee (max_yield=2
→ mega-doc advances ≥1 chunk/3 iters, no starvation; full-budget iters, no throttle). Distinct from SRPF (pins
started chunked_req), reserve (throttle), LOF. Launched jobs 20162 (il-l7, node 0-0, λ7 K3) + 20163 (il-l5, node
0-1, λ5 K3), 3 arms fcfs/srpf/interleave, isolated caches. runs/v11-interleave-{l7,l5}. ★DECISIVE: does interleave
PASS goodput@SLO at λ7 (aggregate p99<8s) where SRPF fails (9.6s)? Sim predicts interleave short-req p99 {1.1,1.9,
1.7}s@λ3/5/7 << srpf. If it passes λ7 losslessly → novel mechanism EXTENDS the goodput ceiling past SRPF = P3
(with honest mega-doc-starvation disclosure + metric-gaming caution connecting to P1). If neutral/heavier-than-sim
→ characterized. RESUME: analyze runs/v11-interleave-{l7,l5}/srpf_ab.csv via ab_analyze.py; verify LOSSLESS (stock
outputs match) if interleave wins; if a job died + csv incomplete, resubmit the sbatch on any idle a3 node (NOT -2).

### ★ CHUNK-INTERLEAVE = NEGATIVE (2026-07-16, job 20162, node 0-0, λ7) — over-defers the mega-docs
Same-node λ7: fcfs {12.2,33.9,35.0}s med 34s (0/3) / srpf {9.6,9.6,8.4}s med 9.6s (0/3) / INTERLEAVE rep1
**90.1s (0/3), CATASTROPHICALLY WORSE** than both. Mechanism runs (no crash — park-gate+guard correct under
full λ7 load), keeps SHORT reqs fast (per sim) BUT the AGGREGATE p99 (what goodput@SLO measures) is dominated by
the OVER-DEFERRED mega-docs: parking a mega-doc every ≤N=2 iters defers it MORE than srpf (which runs it to
completion once started) → mega-doc TTFT explodes (~90s at λ7 backlog). ★The GPU-free interleave_sim was
MISLEADING — it tracked TINY-req p99 (~1.7s, optimistic) not the aggregate metric; the real goodput tail is the
mega-docs, which interleaving worsens. ⇒ chunk-interleaving does NOT beat srpf; it's WORSE than both baselines.
No N helps: large N→fcfs-like, small N→over-defer; srpf's defer-at-admission-then-run-to-completion is the sweet
spot. HONEST NEGATIVE (charter-valid). ★SHARPENS defer-don't-throttle: deferral works ONLY as srpf does it;
repeated/chunk-level deferral (interleaving) OVER-defers heavy docs → wrecks the aggregate tail. Awaiting reps
2-3 + λ5 to firm; then fold as a characterized negative into Paper 2 §4 (a 3rd failed alternative to SRPF, after
reserve[throttle] and LOF[wrong-key]: interleave[over-defer]). SRPF remains the unique deferral sweet spot.
LESSON: sim must measure the SAME metric as the eval (aggregate p99), not a proxy (tiny-req p99).

### ★ CHUNK-INTERLEAVE NEGATIVE — CONFIRMED at BOTH loads (2026-07-16)
Complete same-node A/B (fcfs/srpf/interleave):
- λ5 (node 0-1): fcfs {25.9,23.3,24.0} med 24s 0/3 / srpf {6.8,6.2,6.4} med 6.4s **3/3 PASS** / interleave rep1 **92.8s FAIL**.
- λ7 (node 0-0): fcfs {12.2,33.9,35.0} med 34s 0/3 / srpf {9.6,9.6,8.4} med 9.6s 0/3 / interleave rep1 **90.1s FAIL**.
⇒ interleave is CATASTROPHICALLY worse (~90s) at BOTH loads — 2 independent confirmations. At λ5 it DESTROYS the
goodput srpf cleanly achieves (6.4s pass → 92.8s fail). DECISIVE negative (~10-15× worse; over-defers mega-docs).
Good-neighbor: 2 jobs pending → let l7 il rep2 finish (n=2 λ7) then CANCEL both to free nodes. n=1-per-load +
cross-load consistency = firm enough for a bounded negative (report n honestly, as with reserve-4096 n=1).
INTEGRATION: Paper 2 §4 gains interleave as the 4th failed point mapping the deferral design space — the sweet
spot is UNIQUE (SRPF defer-at-admission-then-run): under-defer=FCFS, throttle=reserve, wrong-key=LOF,
over-defer=interleave ALL fail. + methodology lesson (sim must measure the eval metric, not a tiny-req proxy).

---
## DIRECTION 3 (2026-07-16): L1↔L2 KV MOVEMENT — is transfer/overlap a goodput@SLO lever?
Orthogonal to P1(caching)/P2(scheduling). Charter-hinted ("transfer/compute overlap that hides L1↔L2
latency under load"). Hypothesis: under high-λ eviction pressure a waiting short's prefix is evicted to L2;
when it's finally scheduled (after a mega-doc clears) it stalls on L2→L1 restore → an anticipatory
prefetch-during-mega-doc-compute could hide it (novel overlap primitive) OR movement is a non-lever.

CODE PATH (my clone): `cache_controller.load()` only ALLOCS+ENQUEUES to load_queue; real DMA runs later in
`start_loading()` on `load_stream`, waited per-layer via cross-stream wait_event DURING batch N's forward
(overlaps within-batch), but NOT pre-issued during the PRIOR (mega-doc) batch → overlap GAP exists (verdict B).
`load_back_duration_seconds` (Prometheus) wraps ONLY the enqueue; NO actual-DMA timer existed.

EVIDENCE (all from the eval's OWN artifacts, GPU-free unless noted):
- (A) ENQUEUE: restore avg ~1.3ms p99≤8ms; evict avg ~0.9ms p99≤7ms — negligible. VOLUME huge: λ7 restores
  1.04B tok / evicts 1.84B tok (vs only 109M NEW tokens recomputed) → ~10× more KV MOVED than PRODUCED
  (heavy L1↔L2 thrashing) yet ~1ms enqueue.
- (C) EXPOSED (client TTFT, real runs): non-queued cache-hit reqs @λ5, TTFT FLAT vs prompt_len:
  <1K→585ms, 15-40K→506ms, 40-100K→673ms → a 100K cached-context restore adds ≤~100-170ms over the ~500ms
  sched floor, DESPITE ~3 restores/req. Exposed restore ≤~170ms ≪ the seconds-scale p99 knee.
- (D) COST HIERARCHY: recompute ~38-63K tok/s aggregate; restore tok/s pending (B, KL_DMA_TIMING run 20210).
- (B) ACTUAL DMA: added env-gated CUDA-event timing (commit 3b2756b28, KL_DMA_TIMING, off=lossless);
  instrumented sweep job 20210 on node 1-0 → runs/v12-dma-timing (read [KLDMA] from server.log).

DECISION GATE: if actual DMA p99 small (expected) → movement is a NON-LEVER; the max benefit of any
prefetch/overlap mechanism is bounded above by the exposed restore (≤~170ms) ≪ 8s SLO → no need to build it
(ceiling-bound impossibility argument). → PAPER 3 (focused measurement study): "transfer latency is NOT the
bottleneck for compute-bound long-context prefill; tiering's value is CAPACITY not latency-hiding; reframes
the transfer-overlap machinery of HiCache/LMCache/AttentionStore/Strata for this regime" + unified lever map
(caching[P1]/scheduling[P2]/movement[P3]). If DMA surprisingly large AND section C somehow misleading → build
anticipatory prefetch + A/B. Tool: tools/movement_analyze.py (A/B/C/D reproducible).

### Paper 3 (kv-tiering-movement) DRAFT status + RESUME (2026-07-16)
DRAFT: submissions/kv-tiering-movement/paper.html (committed; NOT in INDEX yet). Sections A(enqueue+volume,
Table1-2)/C(exposed TTFT flatness, Table4)/D(cost hierarchy ~40×) COMPLETE from existing artifacts. §5.5 has the
airtight 2-part ceiling bound (worst-case full un-overlapped DMA from §5.2 + common-case ≤170ms exposed).
PENDING: §5.2 actual-DMA numbers (REPLACE_P99/REPLACE_AVG/REPLACE_P50/REPLACE_MAX/REPLACE_N/REPLACE_RESTORE_TOKS
in Table3 + abstract + §5.4 REPLACE_RATIO + §5.5 REPLACE_P99). 11 REPLACE_ tokens total.
★RESUME: KL_DMA_TIMING run = job 20211 → runs/v12-dma-timing. Harvest [KLDMA] lines from server.log:
  grep KLDMA runs/v12-dma-timing/server.log | tail -3   (cumulative; last line = final distribution)
Fires once start_loading crosses 500 calls (merges ~40 restores/call → crosses during λ5; full sweep ~2h).
Also: python3 tools/movement_analyze.py runs/v12-dma-timing/server.log  → fills B + D ratio.
Then fill the 11 REPLACE_ tokens, recompute REPLACE_RATIO = restore_tok_per_s / ~14000 (per-stream prefill),
re-validate HTML, register INDEX (append 1 line), update memory, push. If job 20211 died before λ5 (check sacct),
resubmit: sbatch --exclusive --mem=0 --gres=gpu:8 -w <idle a3, NOT -2> --wrap "export KL_DMA_TIMING=1; bash <eval.sh> kleinrock v12-dma-timing".
⚠️ --mem=0 is MANDATORY (plain --exclusive gives only 8944M → 122B OOM-kills scheduler; learned the hard way, job 20210).

### Paper 3 §5.2 harvest — CORRECTED target (2026-07-16): job 20221, v13-dma-timing, node 1-0
Killed 20211 (threshold 500 tripped too late — start_loading merges at batch level, ~350 calls/2-rates).
Lowered threshold 500→25 (commit 98b0c5712) → relaunched job 20221 → runs/v13-dma-timing (KL_DMA_TIMING=1,
--mem=0). KLDMA now fires early in λ3 (~25 start_loading calls). HARVEST:
  grep KLDMA runs/v13-dma-timing/server.log | tail -3      (last = cumulative distribution)
  python3 tools/movement_analyze.py runs/v13-dma-timing/server.log   (fills B + D ratio)
Then fill 11 REPLACE_ tokens in submissions/kv-tiering-movement/paper.html (Table3 §5.2 + abstract + §5.4
REPLACE_RATIO + §5.5 REPLACE_P99); RATIO = restore_tok_per_s / ~14000 (per-stream prefill); re-validate HTML;
register INDEX (1 line); update memory; push. If 20221 died: check sacct; resubmit same cmd (--mem=0 MANDATORY).

### §5.2 harvest — job/node UPDATE (2026-07-16): now job 20226, node -0 (v13-dma-timing)
20221 on node 1-0 OOM'd during 122B load DESPITE --mem=0 (node 1-0 available RAM ~1310G = right at eval.sh's
1300G gate edge; load peak exceeds it). Node -0 is PROVEN (ran 20211 to λ5 fine w/ --mem=0). Relaunched → job
20226 on -0, v13-dma-timing, threshold=25. HARVEST unchanged: grep KLDMA runs/v13-dma-timing/server.log | tail -3
then python3 tools/movement_analyze.py runs/v13-dma-timing/server.log; fill 11 REPLACE_ tokens; register INDEX.
★OPS: node slurm2-a3nodeset1-0 OOM-kills the 122B load even with --mem=0 (RAM too tight); prefer node -0.

---
## DIRECTION 4 (2026-07-16): THE DECODE TAIL — the metric hides the real tail
★NOVEL FINDING (GPU-free, from existing stock bench jsons + server.log): the eval's E2E p99 is ~400 SECONDS
(λ3 446s→λ10 392s), of which DECODE is 89-97% (TTFT is only 3-11%). TPOT median ~220ms (normal) but TPOT p99
~3.5s and ITL p99 ~4.5s → severe decode STALLS. ★ORTHOGONAL TO SRPF: same-node λ5 n=5, SRPF fixes TTFT p99
22-28s→5.5-7.8s but leaves TPOT/ITL/E2E p99 UNCHANGED (itl ~4.4-4.9→4.2-4.6s). So the decode tail is NOT
prefill-ordering — my entire P1/P2/P3 series (all TTFT) MISSED it. ★CAUSE = decode STARVATION under prefill-greedy
scheduling: stock server.log has 34071 Prefill batches vs 1621 Decode batches (21:1), with runs of up to 661
CONSECUTIVE prefill batches (256 decode seqs present, not advancing) → decode stalls seconds during prefill bursts.
NO retraction/preemption events (not retraction-driven). ★THESIS: goodput@SLO (TTFT) HIDES the real tail — in
long-context multiturn the decode tail (E2E p99 ~400s) is ~50× the TTFT tail and is decode-starvation, a distinct
axis. Likely a prefill-decode Pareto tradeoff (decode-friendly scheduling costs TTFT). NEXT: read scheduler
prefill-vs-decode decision; GPU A/B stock vs decode-friendly (measure BOTH TTFT-goodput AND decode/E2E tail) to
quantify the tradeoff. Candidate mechanism must be novel+lossless (not mixed-chunk config = disqualified); framing
may be a CHARACTERIZATION/tradeoff paper (charter-valid). Data: runs/v3-srpf-ab/bench_{fcfs,srpf}_l5_r*.json.

### P4 experiment LAUNCHED (2026-07-16): job 20231, v14-mixedchunk (--enable-mixed-chunk full sweep)
Compare to v0-stock (existing, all λ). RESUME: when 20231 done (runs/v14-mixedchunk/, ~2-2.5h; --mem=0, node -0):
  python3 - <<'PY'  # decode-tail + goodput Pareto, both arms
  import json,glob
  for tag,patt in [("stock","runs/v0-stock/bench_r%s.json"),("mixedchunk","runs/v14-mixedchunk/bench_r%s.json")]:
    print(tag); 
    for r in [3,5,7,10]:
      try:
        d=json.load(open(patt%r)); print(r, "ttft_p99",d['p99_ttft_ms'],"itl_p99",d['p99_itl_ms'],"tpot_p99",d['p99_tpot_ms'],"e2e_p99",d['p99_e2e_latency_ms'],"reqtput",d['request_throughput'])
      except Exception as e: print(r,e)
  PY
Then: goodput@SLO(TTFT≤8s) each arm from curve.csv; the PARETO = does mixedchunk cut ITL/TPOT p99 (decode tail)
and at what TTFT-goodput cost? If clear → same-node stock arm (rigor, coin-flip control) then write P4
CHARACTERIZATION paper "The Hidden Decode Tail" (honest: mixed-chunk is a known config = NOT my mechanism; the
contribution is the hidden-tail MEASUREMENT + the prefill-decode Pareto + the metric-blind-spot thesis). If
mixedchunk does NOT help (or the tail is saturation not starvation) → the decode tail is more fundamental →
re-scope. ⚠️ --mem=0 MANDATORY; node 1-0 OOMs the 122B load; harvest at run END (don't babysit).

### P4 starvation depth quantified (GPU-free, stock server.log)
Consecutive-prefill-run length (# prefill batches decode waits through): p50=5, p90=89, p99=229, max=661
(n=1289 runs). Prefill:decode batch ratio 21:1 (34071:1621). Clean decode-stall = client-side ITL p99 ~4.5s /
TPOT p99 ~3.5s (bench); present at λ3 → STARVATION not saturation. CAVEAT: raw server-log inter-decode gaps
(p99 41s) are idle/flush-contaminated — use ITL (client) + run-length (structural), not raw gaps, in the paper.

### P4 KEY RESULT (2026-07-16, GPU-free from stock replicates) — the metric is BLIND to the decode tail
★★ v0-stock-r3 λ3: p99 TTFT=6786ms → goodput@SLO **PASSES** (≤8s) — yet E2E p99=**346896ms (347s)**, ITL p99=4121ms,
TPOT p99=3624ms. THE HEADLINE METRIC REPORTS SUCCESS WHILE E2E IS 347 SECONDS. Decode tail is STABLE across all 3
stock replicates × both λ (itl_p99 4.1-5.1s, tpot_p99 2.8-4.3s, e2e_p99 347-446s) regardless of TTFT pass/fail →
a separate, INVARIANT pathology, orthogonal to BOTH the TTFT coin-flip (P1) and the prefill-ordering lever (P2/SRPF).
⇒ P4 THESIS STRENGTHENED to a METRIC-BLINDNESS result (not just config-critique): goodput@SLO (TTFT) is not only
unreliable (P1 coin-flip) but INCOMPLETE — it hides a decode tail ~50× larger that dominates real E2E latency.
This is a genuine measurement/critique contribution (like P1's methodology) REGARDLESS of the mixed-chunk fix
outcome (job 20231). Table for paper: {run, λ, ttft_p99, goodput, itl_p99, tpot_p99, e2e_p99} from v0-stock{,-r2,-r3}.

### P4 §5 experiment 1: --enable-mixed-chunk CRASHES (2026-07-16, job 20231, v14, node -0)
mixed-chunk (the textbook decode-starvation fix: co-batch decode with prefill chunks) CRASHED at λ3 (4390/7037)
with `ValueError: pool memory leak detected! [full] total=2347648 available=5312 evictable=2342592` (also [mamba]).
SAME KV-pool-leak failure mode as P2's reserved-lane at large reserve → batching-composition changes destabilize
this hybrid-MoE + hierarchical-cache + long-context config. ⇒ §5 finding: the decode tail is NOT cheaply fixable —
the standard mixed/stall-free-batching mitigation destabilizes the server here. Node freed (scancel). NEXT: try a
non-batch-changing decode-friendly knob (schedule-conservativeness / prefill-delayer) for a clean Pareto, else
§5 = "obvious fix destabilizes; a proper decode-QoS fix needs scheduler redesign (future work)". P4 core
(metric-blindness) stands regardless.

---
## DIRECTION 5 (2026-07-16): decode-QoS mechanism `--decode-starvation-bound` (P4 §5 future-work → built)
Motivated by P4: decode tail = prefill-first starvation; the textbook fix (mixed-chunk) CRASHES (pool leak).
BUILT a decode-QoS primitive (commit 3171982c2): force a pure-decode batch after K consecutive prefill batches
when decode is pending → bounds decode starvation / ITL tail WITHOUT changing batch composition (no pool-leak).
scheduler.py get_next_batch_to_run: force_decode gate + _consec_prefill_batches counter; server_args
decode_starvation_bound (default 0 = byte-identical stock = lossless). LOSSLESS: only reorders forward passes.
★EXPERIMENT: job 20238, v15-decodeqos-k4 (--decode-starvation-bound 4, full sweep, --mem=0 node -0). Compare to
v0-stock: does it (a) NOT crash, (b) cut ITL/TPOT/E2E p99 (decode tail), (c) at what TTFT-goodput cost = the
prefill-decode PARETO. RESUME: harvest runs/v15-decodeqos-k4 via decode_pareto.py (A=v0-stock B=v15-decodeqos-k4);
if it cuts the tail cleanly → this is a WORKING decode-QoS fix → UPGRADE P4 §5 (critique→critique+solution+Pareto),
optionally sweep K∈{2,8} + same-node stock arm for rigor. If neutral/crash → learn + keep P4 as-is. ⚠️ if
decode-QoS is deemed "classical fair-scheduling" (charter L23), frame as the Pareto CHARACTERIZATION + the
robust-simple-fix-that-avoids-mixed-chunk's-crash, not a novel-primitive claim. ⚠️--mem=0 mandatory; harvest at end.

### Direction 5 decode-QoS: v1 (K=4, job 20238) CRASHED (pool leak) → FIXED (7091d4957) → v16 (job 20239)
v15/20238 crashed in WARMUP: pool memory leak [full] available=4736, protected=577792. ROOT CAUSE: my v1 built
the prefill batch via get_new_batch_prefill() (which ALLOCATES KV) then DISCARDED it when forcing decode →
leaked reserved slots (protected KV). FIX: decide force_decode BEFORE building the prefill batch (skip
get_new_batch_prefill entirely when forcing → no allocate-then-discard) + guard `self.chunked_req is None`
(never interrupt a mid-flight chunked mega-doc → don't strand its protected KV). Lossless at bound=0 (stock path
untouched). Relaunched v16-decodeqos-k4 (job 20239, node -0, --mem=0). ★NOTE on coverage: the chunked_req guard
means a mega-doc's own ~31-chunk prefill isn't interrupted (its ~13s decode stall stays, = p99.9), but the
common 661-run of SEPARATE small prefills IS bounded to ≤K → should cut ITL p99 (the metric). HARVEST v16:
decode_pareto.py (A=v0-stock B=v16) + batch_ratio.py (v16 log: run-length capped ~K? decode share up?) + verify
completed=7037 (full trace = lossless-served). If cuts tail + no crash → same-node stock arm → P4 §5 upgrade.

### decode-QoS leak fix VALIDATED (2026-07-16): v16 survived warmup past v1's crash point
v16 (job 20239, K=4, leak-fixed 7091d4957) reached 11:34 elapsed in warmup with NO pool-leak — past the ~10:16
point where v1 (20238) crashed. ⇒ the allocate-then-discard leak is fixed; decode-QoS runs cleanly. Sweep in
progress (λ3 ~40min, full ~2h). HARVEST when done: decode_pareto.py A=v0-stock B=v16-decodeqos-k4 (ITL/TPOT/E2E
p99 cut? goodput/TTFT cost?) + batch_ratio.py runs/v16-decodeqos-k4/server.log (run-length capped ~K=4? decode
share up from 4.5%?) + completed=7037 (lossless-served). If tail cut + no crash → same-node stock arm on -0 →
P4 §5 UPGRADE (critique → working decode-QoS solution + prefill-decode Pareto).

### decode-QoS MECHANISTIC PREVIEW (2026-07-16, v16 partial log via batch_ratio.py)
v16 (K=4, guard=chunked_req-None) vs stock: decode share 4.5%→11.8% (2.6×), prefill:decode 21:1→7.5:1,
consecutive-prefill runs p50 5→3, p90 89→10 (COMMON runs capped ~K), BUT p99 229→214, max 661→388 (tail NOT
capped). ★WHY: the chunked_req guard (don't force decode mid-chunked-prefill, added to avoid the leak) disables
decode-QoS whenever a doc >6144 tok is chunk-prefilling — and most first-turn docs are chunked (p50 16.7K), so
long chunked-prefill stretches still starve decode → runs escape to 388. ⇒ decode-QoS REDUCES starvation
(decode share 2.6×, common runs bounded) but can't ELIMINATE it under the safe guard. HONEST §5: partial fix +
the chunked-prefill limitation (removing the guard = cap ALL runs but risks the protected-KV pool-leak the guard
prevents). Await v16 ITL/TPOT p99 (bench) for the actual tail-cut magnitude; decode share 2.6× suggests real
but partial improvement. OPTION if v16 ITL-cut disappointing: v17 WITHOUT guard (allocate-then-discard leak is
already fixed independently; interrupting chunked prefill MAY be safe since existing chunked_req stash/resume
machinery handles it) — but test carefully (risk re-crash). Don't kill running v16.

### decode-QoS λ3 RESULT (2026-07-16, v16 K=4 vs stock-r3, lossless: completed 7037/7037)
ttft_p99 6786→29367ms (+333%, PASS→FAIL); itl_p99 4121→3942 (-4%, FLAT); tpot_p99 3624→9087 (+151% WORSE);
e2e_p99 346896→235619 (-32%); tpot_MEDIAN 99.5→45.3 (-54%); req/s 3.0→3.0. ★INTERPRETATION: decode-QoS is a
genuine prefill-decode PARETO but NOT a decode-TAIL fix. It runs decode earlier → MEDIAN decode −54% + E2E p99
−32%, but WRECKS TTFT-goodput (+333%, breaks the SLO) and does NOT cut the p99 decode TAIL (itl flat, tpot p99
worse). WHY: the p99 tail stalls happen DURING heavy-doc chunked prefills, which the chunked_req guard doesn't
interrupt (per batch_ratio: runs escape to 388 during chunked prefills). ⇒ §5 STORY (stronger + honest): the
metric-hidden decode tail RESISTS fixing — mixed-chunk CRASHES (pool leak), and safe decode-QoS trades away
TTFT-goodput for median/E2E gains WITHOUT cutting the p99 tail (which is bound to un-interruptible heavy-doc
prefills). No cheap fix; the tail is a genuinely hard open problem. This REINFORCES P4's thesis. ★NOTE: ttft
+333% (29s) exceeds the stock coin-flip band (6.8-14s) so the TTFT-degradation is real despite cross-node.
Let v16 finish for λ5/7/10 confirmation, then integrate as P4 §5 (Pareto + tail-resists-fixing).

### decode-QoS λ5 cross-load (2026-07-16, v16, lossless 7037/7037) — LOAD-DEPENDENT, refines §5
λ5 K=4 vs stock: ttft_p99 25449→23145 (-9%, both FAIL), itl_p99 5099→4341 (-14%), tpot_med 312→225 (-28%),
e2e_p99 400269→336714 (-15%). ★CONTRAST with λ3: λ3 (unsaturated, PASS case) ttft +333% (PASS→FAIL) / itl -4%
(FLAT); λ5 (saturated, both fail) ttft -9% / itl -14% (modest cut). ⇒ decode-QoS is a genuine prefill-decode
PARETO knob, LOAD-DEPENDENT, NEVER a clean win: at the unsaturated goodput-PASS case it BREAKS goodput without
cutting the tail; at saturation it modestly cuts the tail (itl -14%) but goodput is already lost. NO operating
point fixes the p99 tail while preserving goodput. ★§5 FIX: don't over-claim "itl flat/doesn't cut the tail"
universally (λ5 shows -14%); frame as the load-dependent Pareto (λ3 = break-goodput-no-tail-cut; λ5 =
modest-tail-cut-but-goodput-gone). Await λ7/λ10 then update Table 3 (add λ5, +λ7/λ10) + refine §5 prose.

### decode-QoS λ7 cross-load (2026-07-16, v16, lossless 7037/7037)
λ7 K=4 vs stock: ttft_p99 33314→23355 (-29%), itl_p99 4915→4427 (-9%), tpot_med 350→271 (-22%), e2e_p99
399583→339832 (-14%). ★FULL PATTERN (λ3/5/7): ttft +333%/-9%/-29%; itl -4%/-14%/-9%. ⇒ decode-QoS BREAKS
goodput exactly where it's ACHIEVABLE (λ3 unsaturated, PASS→FAIL, tail untouched) and only helps modestly in the
SATURATED regime (λ5/λ7) where goodput is ALREADY LOST. "Helps where it doesn't matter, hurts where it does."
Reinforces "no operating point cuts the tail while preserving goodput." Await λ10 (~30min) → final 4-rate Table 3
+ refine §5 prose (the wrecks-TTFT is λ3-specific; saturated regime shows modest ttft+tail improvement but goodput moot).

---
## DIRECTION 6 LEAD (2026-07-16): the "pool leak" blocking decode-tail mitigation is likely a FALSE-POSITIVE check bug
Explore verdict (my clone, stock sglang code): invariant_checker `available+evictable+protected+session_held+
uncached==total` (line 76) FIRES under mixed-chunk/reserve/decode-QoS-no-guard with protected as the EXACT excess
(available+evictable≈total already). evictable & protected are DISJOINT by design (mamba_radix_cache inc/dec_lock_ref
886-919 move tokens between them). ⇒ full_evictable_size_ COUNTER has DRIFTED to over-count by ~protected → likely
a FALSE-POSITIVE (counter bug, physical allocator fine — crash is an invariant ASSERT not a CUDA OOM). Candidate
root cause: len(key) vs len(value) inconsistency (delete uses len(key) @1286/1303, lock uses len(value) @887/918)
or _split_node (1147-1178, splits a locked node, no size update). ★IF false-positive → fixing it could UNBLOCK
mixed-chunk/decode-QoS-no-guard → potential FIRST CLEAN decode-tail fix (charter TOP PRIZE, resolves P4 §5 open Q).
★PLAN: (1) pinpoint the drift bug (fix it = SAFE path, correct accounting → guaranteed lossless) OR run with
strict-check=warn (SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0) to test if the run completes; (2) VERIFY
LOSSLESSNESS rigorously (compare greedy outputs to stock — cardinal rule; a counter-drift is benign/lossless, a
real corruption is not) before claiming ANY positive; (3) measure decode tail (mixed-chunk should cut it Sarathi-
style). Contribution if it pans out = the BUG-FIX (engine correctness, not a config flip) that unlocks decode-tail
mitigation. ⚠️HIGH-RISK (losslessness); do NOT claim positive without output-equivalence proof. Genuine lead, not fabrication.

### Direction 6 analysis refinement (2026-07-16, GPU-free source trace)
RULED OUT: _split_node (mamba_radix_cache 1147-1178) is CORRECT — splits conserve value + lock category (both
halves inherit lock_ref), no size update needed. len(key)==len(value) (key/value sliced identically) → the
delete-uses-len(key) @1286/1303 vs lock-uses-len(value) @887/918 inconsistency is HARMLESS. ★REFINED HYPOTHESIS:
the leak check is an IDLE-invariant (available+evictable+protected+session_held+uncached==total) run DURING BUSY
via self_check_during_busy (scheduler.py:1553, gated by SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY; raises via the
DURING_IDLE flag). Under mixed-chunk's higher concurrent prefill+decode, transient mid-operation state (tokens
allocated-but-not-yet-inserted, or in-flight) makes the idle-invariant transiently false → false-positive abort.
⇒ FALSE-POSITIVE (transient), not a real slot leak. ★DIAGNOSTIC job 20257 (v17-mixedchunk-nocheck, IDLE=0 →all
leak checks warn-not-raise, --exclude bad node 1-0): if it COMPLETES the sweep → false-positive confirmed →
pursue the fix (make the busy check tolerate transient state, or account for in-flight) + VERIFY losslessness
(output-equivalence vs stock) → then mixed-chunk unblocked = decode-tail fix. If OOM/abnormal → real leak → P4 firmed.

### Direction 6 refinement #2 (2026-07-16): the check is on_idle → drift manifests AT IDLE (counter bug)
CORRECTION to refinement #1: the firing check is scheduler.py:3531 on_idle, which runs ONLY when is_fully_idle()
(in-flight==0) → uncached=0 is CORRECT there. self_check_during_busy (256) properly accounts in-flight via
uncached=Σ(allocated-cache_protected) (248). So the crash (uncached=0) is the IDLE check at a transient-idle
moment mid-sweep, and the invariant failing by ~protected means the full_evictable_size_ COUNTER genuinely
drifted even AT REST (not a transient-busy artifact). ⇒ a real counter-accounting drift triggered by mixed-chunk
ops. KEY: is it (a) counter-only (physical allocator `available` is the true free count → pool fine → LOSSLESS,
check is over-strict) or (b) the inflated evictable counter misleads eviction (real harm)? DECIDED BY diagnostic
20257 (queued, cluster saturated 10 alloc/2 idle): completes losslessly → (a) counter bug, fix = correct the
drift → unblocks decode-tail mitigation; OOM/abnormal → (b) real. Pinpoint-on-confirm: add a debug walk of the
LRU list summing actual ref==0 tokens vs the counter to locate the drift site empirically.

### Direction 6 refinement #3 (2026-07-16): built-in tree-sanity check disambiguates the diagnostic
mamba_radix_cache.py:402-410 sanity_check() ALREADY walks the LRU list and asserts full_evictable_size_ (counter)
== sanity_check_evictable_size() (physical ref==0 walk). Runs in on_idle at line 3547, AFTER the pool invariant
(3539); it's a plain assert (NOT gated by SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE). ⇒ diagnostic 20257 (IDLE=0
→ pool check warns) now cleanly disambiguates: (a) crashes at tree-sanity assert (409, "evictable size X != lru
list Y") → COUNTER DRIFT (fixable false-positive; fix = find/fix the drift site so counter==physical); (b)
COMPLETES past on_idle → counter consistent + pool-invariant over-strict → benign/LOSSLESS (fix = correct the
pool invariant to model whatever it misses); (c) OOM → REAL over-commit (fundamental → P4 firmed). No more source
speculation needed — the empirical run decides. 20257 queued (cluster saturated 10 alloc/2 idle; next-priority).
