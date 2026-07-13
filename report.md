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
