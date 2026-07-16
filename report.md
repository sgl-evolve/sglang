# Researcher: floyd

**Branch:** `evolve/floyd`
**Clone:** `workspace/sgl/v0.31/research/researchers/floyd/`
**Base commit:** `a334877e5` (stock sglang)

> **⚠ CURRENT STATE (2026-07-15): 4-PAPER BODY, lossless design space mapped+bounded. This log is
> chronological — early "Thesis"/"Mechanism" prose (admission control / "retraction cliff") is SUPERSEDED.**
> ONE thesis: *goodput@SLO for long-doc multiturn serving is a prefill-scheduling problem bounded by compute,
> not a KV-cache problem.* The λ3 goodput coin-flip is **head-of-line blocking** (small turns behind a few big
> cold docs' back-to-back chunked prefill), not a retraction cascade (stock = 0 retractions).
> **Papers (see submissions/INDEX.md):** P3 `goodput-is-scheduling` = capstone map (whole-request SRPF, textbook
> = the lever; goodput 0/3→4.0@λ5, SLO-tail-bound below raw ceiling ~4.7). P2 `corpus-bound-goodput` = caching
> mirage (LRU=Belady=0 avoidable). P1 `cca-prefill-admission` = admission NEGATIVE (deferring cold prefills is
> self-defeating; the tail IS them). P4 `rpb-chunking` = my novel chunk-level Reserved-Prefill-Budget mechanism,
> a NEGATIVE (fixed hurts λ5 via ~54% reserve waste; adaptive zero-waste is neutral → chunk-level reservation
> can't beat SRPF at any impl). Every KV data-movement/retention axis is a bounded non-lever. All papers
> hostile-reviewed + integrity-clean. **★2026-07-15 SRPF-win significance FIRMED:** artifact inventory found
> **6 (not 3) plain-SRPF λ5 passes** (r1/r2/r3/full on node 0-3; ctl4/ctl5 on node 1-1; all verified
> `schedule_policy=srpf`, RPB off) → **λ5 SRPF 6/6 vs stock 0/5, Fisher p=0.0022** (was 3/3, p=0.018); the win
> is **cross-node ROBUST** (SRPF passes on 0-3+1-1, stock fails on 0-3+ondem-3; 9.5s gap ≫ ±45% node var), so
> cross-node pooling is a robustness check not a weakness. Same-node stock controls on 0-3 running (job 20005)
> → strictly-same-node **4/4 vs 0/4, p=0.014** (zero pooling). No remaining novel lossless lever (CP-for-hybrid
> unavailable v0.31). The detailed running log follows; latest results near the end.

## Research Direction (Paper 1): Cache-adjusted prefill admission control

**Diagnosis (trace-driven, `analysis/trace_diag.py`).** The FIXED workload = 1553 multiturn
conversations (leval/loogle long-doc QA + sharegpt chat), replayed in file order at Poisson λ,
7163 total turns. Turn-0 embeds the whole document (huge cold prefill, p90 ~29K tok); turns 1..N
reuse it via the radix cache. Under an idealized LRU cache at the real 2-tier capacity (~10.7M tok):

- **Prefill cost is extremely bimodal & concentrated.** Per-turn cache-adjusted prefill work:
  p50 = 26 tok (near-free reuse), p99 = 40,579 tok, max = 190,944 tok — a **~1500× spread**.
  The **top 5% of turns carry 50%** of all prefill work; **top 10% carry 78%**; **top 20% carry 99%.**
  70% of turns are near-free reuse (≤26 tok).
- **Caching is fundamentally bounded here.** Of all prefill work, **75% is COLD** (first-ever
  document prefill — unavoidable, no cache can remove it) and only **~24% is AVOIDABLE**
  evicted-recompute. Since eviction is already near-optimal (LRU≈Belady) and transfer is cheap
  (~1.6ms), residency mechanisms have little headroom left. The real lever is **how the
  heterogeneous prefill work is SCHEDULED**, not how much is cached.

**Thesis.** The goodput@SLO metric is a *tail* metric (max req/s with p99 TTFT ≤ 8s). Its known
instability (a metastable "coin-flip") is a **control instability**: the scheduler admits prefill
by raw token budget / request count, treating a 26-tok reuse and a 40K-tok cold prefill as
comparable, so a burst of cold prefills — a *few requests* but the *bulk of the work* — transiently
overloads the prefill pipeline, drives the KV pool into the **retraction cliff** (running decode
requests get retracted → re-prefilled → more overload), and the queue never recovers within the run.

**Mechanism (the new primitive): work-bounded prefill admission in cache-adjusted currency.**
Bound the *outstanding cache-adjusted prefill work* (Σ over admitted-but-not-yet-first-token
requests of `input_tokens − radix_matched_prefix`) below a budget B chosen so pipeline latency
(outstanding_work / prefill_throughput) stays ≤ SLO. A cheap reuse turn is admitted freely even
under pressure; an expensive cold prefill is gated until the pipeline has drained. This directly
bounds the TTFT tail and breaks the retraction cascade — **losslessly** (it changes only *when*
prefills run, never KV contents or outputs).

**Why novel (vs prior art & stock).** (1) The control *currency is cache-adjusted compute work*,
not request count (SRPF/SJF order but don't bound), not raw tokens, not current pool usage (the
stock `PrefillDelayer`, which targets decode-batch fragmentation, is off in the eval and cache-blind).
(2) The insight — *prefill cost varies 1500× due to cache residency, so admission must be metered in
residency-adjusted work to stay off the retraction cliff* — is non-obvious and generalizes to any
prefix-cached long-context serving. Positioned against Strata (balanced batching), Mooncake
(overload rejection), SRPF. **Controls to run:** stock `PrefillDelayer` (config), naive SRPF,
raw-token budget (ablate the cache-adjustment) — the win must be attributable to the cache-adjusted
currency, not to any config flip.

**Honest fallback.** If the mechanism proves config-reachable or the 75%-cold ceiling is truly
immovable, the paper becomes a *bounded-impossibility*: for long-doc multiturn serving, goodput@SLO
is cold-prefill-compute-bound and no lossless KV mechanism shifts it beyond the ~24% evicted fraction.

---

## Versions

### v0_official (supervisor baseline, λ=3 only)
- hit_rate=0.6217, TTFT p50=750ms, p99=6326ms, req/s=2.78, tok/s=355; load_back 1.6ms (NOT bottleneck).

### v0-stock (my baseline, ondem-3, job 19730)
- **Curve (all p50 fine ~1s; all p99 FAIL 8s SLO):**
  | λ | req/s | tok/s | p50 | **p99** | hit |
  |---|---|---|---|---|---|
  | 3 | 2.87 | 368 | 1056 | **11787 FAIL** | 0.678 |
  | 5 | 3.66 | 468 | 984 | **17372 FAIL** | 0.667 |
  | 7 | 4.00 | 512 | 1020 | **33961 FAIL** | 0.662 |
  | 10 | (pending, saturated ~1h) | | | | |
- **goodput@SLO = 0** this run (no rate passes). Peak req/s ~4.0 (system capacity), peak tok/s ~512.
- **Key observation:** stock hit **full KV-pool usage 1.00 (18 events)**; pool usage BIMODAL
  (22% ≥0.90 ↔ 18% ≤0.05). p50 fine, p99 blown — tail = a few pool-saturation/large-cold-prefill
  events. This run is on the FAIL side of the λ=3 coin-flip (base sibling saw λ=3 p99 range 6.5↔23.7s).

### v1-cca (first-look screen, node1-2) — WEDGED (JIT race), re-running serially
- **Mechanism-level evidence (node-independent, valid):** with CCA on, KV-pool usage was
  **capped at 0.70** (0× at 1.00) vs stock's 10× at 1.00. The preventive watermark gate fires
  (`cca[defer:5455 gated-pass:1288 force:283]`), sustains 70–99 running reqs at 0.20–0.36 pool
  usage, and prevents the saturation spikes — exactly as designed. Confirms the mechanism
  engages the live path and controls pool pressure.
- **Wedged** ~2.5 min into λ=3 measurement: concurrent floyd evals share the workspace
  flashinfer JIT cache over NFS → JIT race hang (a known OPS hazard, not a logic bug; the gate
  was functioning normally up to the freeze). Killed, not logged. **Re-run SERIALLY** (no
  concurrent floyd eval) on ondem-3 after v0-stock finishes → clean same-node A/B pair.

### Live-scheduler findings (from baseline server.log, shape the interpretation)
- **KV device pool ≈ 2.4M tokens.** Pool usage is **BIMODAL**: hundreds of samples at 0.00–0.03
  (lulls) AND hundreds at 0.91–1.00 (near-saturation), incl. 18 events at exactly 1.00. This
  oscillation is the direct signature of the metastable instability; CCA damps it (capped 0.70).
- **λ=3:** KV pool spikes to 1.00 in bursts (bursty arrivals → server queue + pool saturation →
  TTFT tail). This is CCA's clearest lever.
- **λ=5:** system is **client-concurrency-capped** (running-req pinned at max-concurrency 256),
  KV pool avg ~0.40 (still spikes to 1.00), **mamba usage 0.75**, server queue ~0. With an empty
  server queue, CCA has little to defer at λ≥5 → CCA likely INERT at high rates; the binding
  constraint there is concurrency/throughput/mamba, not the KV-pool.
- **Implication:** CCA's expected contribution is **stabilizing the λ=3 goodput coin-flip**
  (FAIL→reliable PASS by preventing transient pool saturation), NOT lifting goodput to higher
  rates (concurrency-bound). An honest, still-publishable framing IF λ=3 reliably passes.
  Must verify same-node, n≥2, vs the baseline coin-flip.

### v1-cca (recompute-currency) — NEGATIVE + crash
- Currency = input − device_prefix − host_hit_length (recompute cost). **HARMFUL:** where stock has
  **0 retracts**, recompute-CCA drove the pool to **0.99 → 518 retracts → watchdog CUDA crash.**
- **Why:** it throttles cold prefills but NOT big L2→L1 load-backs (recompute≈0 for reuse turns), so
  it shifts load toward decode-concurrency that overruns the pool → retraction cascade. The recompute
  currency MISSES the device-footprint of load-back. (3 recompute/early runs also hit NCCL hangs.)

### v2-cca-df (device-footprint currency) — mechanism FIXED, measurement in progress
- Currency = input − device_prefix (counts the load-back KV). **Caps the pool** (validated: peak 0.55
  during the cold warmup burst, **0 retracts**) — the fix works at the control level. But 3/3 CCA runs
  so far FAILED to complete on ondem-3 (hangs at pool 0.10 / 0.17, one crash at 0.99); ondem-3 may have
  degraded over hours. Re-testing on a health-checked node (stock-r2 first, then v2-cca-df, same-node).

### ★ Same-node coin-flip CONFIRMED (my data, n=3)
STOCK λ=3 p99 same-node (ondem-3): **{11787 FAIL, 6327 PASS, 8124 FAIL}** — n=3, **1/3 pass**,
mean ~8746ms, range 6.3–11.8s straddling the 8s SLO (1.86× spread). goodput {0, 3.02, 0}.
v4-lb λ=3 (8556) sits at the stock MEAN ⇒ neutral. (v4 replicates firming the variance.) goodput@SLO is a metastable
coin-flip even same-node ⇒ single-run A/Bs are VOID; any CCA claim needs n≥3–6 replicates per arm
(compare p99 DISTRIBUTIONS, Levene/MWU). stock-r2 also: pool peak 1.00, **0 retracts** (confirms
stock handles saturation via prefill-stall, no cascade). ondem-3 is HEALTHY (stock ran clean) ⇒ the
3 prior CCA hangs/crash were CCA-related or transient, not node degradation.

### ⚠ Emerging thesis reframe (honest)
Stock has **0 retracts** and completes — its goodput=0 is a **cold-prefill-compute p99 tail**, NOT a
retraction cascade. So CCA's premise ("prevent the retraction cliff") is only half-right: there is no
stock retraction to prevent. And **deferring** expensive prefills is *misaligned* with goodput@SLO,
because the SLO tail IS the expensive cold prefills — delaying them can only hurt their TTFT. Likely
outcome: a **rigorous bounded-negative** — goodput@SLO here is cold-prefill-compute-bound; no
lossless *deferring* admission/scheduling mechanism improves it (device-footprint CCA caps the pool
but cannot beat the ~6s largest-cold-doc compute floor). Needs one clean device-footprint CCA curve
to measure (in progress).

### v3-cca (device-footprint + DETERMINISTIC valve, stable) — DECISIVE NEGATIVE
- First CCA run to complete stably (deterministic-count valve fixed the TP-desync hang).
- **λ=3: p99 = 56157ms — CATASTROPHIC (5–9× WORSE than stock {11787, 6327}).** p50 fine (1126ms),
  req/s 3.02, hit 0.722 (even higher). 0 retracts, pool capped ~0.92.
- **Why (measured confirmation of the misalignment):** CCA DEFERS expensive prefills (cold docs +
  big load-backs). But the p99 tail IS exactly those expensive prefills. Deferring them (up to the
  valve, 200 passes) multiplies their TTFT → p99 explodes to 56s. The cheap reuse turns (p50) are
  unaffected. So capping the pool via deferral cannot help a tail-of-expensive-prefills metric — it
  directly inflates the tail.
- **Bracketing argument:** aggressive gating → catastrophic (56s); minimal gating (high watermark /
  small valve) → converges to stock (no benefit). No (watermark, valve) setting where deferring the
  tail-causing prefills helps. CCA is dominated by stock across its parameter space.

### v4-cca-lb (load-back-only gating, protect cold prefills, valve=30) — NEUTRAL
- λ=3 p99 = **8556ms** (n=1). vs v3-df 56157 (protecting cold prefills avoids the catastrophe), but
  within stock's coin-flip band {6327, 11787} and still FAILS (>8000). Pool capped 0.54, 0 retracts,
  defer 1920 (vs v3's 21591 — gates only load-backs). ⇒ load-back gating is ~NEUTRAL: pacing reuse
  load-backs neither craters nor clearly helps, because stock's collapse is NOT retraction/pool-driven
  (0 retracts) — capping the pool addresses a non-bottleneck. Firming n≥3.
- **v4-lb full (n=1):** λ=3 8556 (FAIL, neutral), λ=5 **13528** vs stock λ=5 {17372, 23130} — load-back
  gating may REDUCE the λ=5 tail (~13.5s vs ~20s), but still >8s ⇒ goodput still 0. Suggestive
  tail-reduction at higher rate; firming n≥3 to confirm vs noise.
- **Complete CCA picture:** deferring cold prefills = CATASTROPHIC (they ARE the tail); deferring only
  load-backs = NEUTRAL at λ=3, maybe mild λ=5 tail-reduction, but NO goodput@SLO improvement (never
  clears 8s). No admission-control variant improves the headline.

### ★★ PIVOT: v4-cca-lb REDUCES the goodput VARIANCE (n=3 confirmed)
- **v4-lb λ=3 p99 = {8556, 8549, 7767}** (n=3, std ~370ms, mean 8291, 1/3 pass) vs stock
  {6327, 8124, 11787} (std ~2272ms, mean 8746, 1/3 pass). **CCA-lb is ~6× tighter variance** and
  slightly lower mean — the coin-flip is largely collapsed to a cluster RIGHT AT the 8s SLO boundary.
  (The n=2 "7ms" tightness was luck; n=3 std ~370ms, still ~6× < stock.)
- Pass-rate tied at n=3 (1/3 each), BUT v4's cluster hugs the SLO ⇒ a config that shifts it reliably
  <8000 flips it to reliable-pass (goodput 0→3) while stock stays a coin-flip. THAT is the win to find.
  Load-back-only gating **collapses the coin-flip to a stable ~8.55s** — a ~1000× variance reduction.
  This is a POSITIVE mechanism effect (novel: pacing L2→L1 load-backs removes the metastable
  stall-clusters that cause stock's high-variance FAIL runs), analogous to the known de-dup
  variance-reduction but via prefill admission.
- **The catch:** the stable point (8.55s) is JUST above the 8s SLO → reliably FAILS λ=3, so goodput
  reliably 0 (vs stock's lucky 1/3 pass). Variance reduction landed on the wrong side of the SLO.
- **The opportunity:** if a config shifts the stable point <8000ms, CCA-lb converts the coin-flip into
  a RELIABLE pass → goodput reliably 3 (a real WIN). ⇒ sweep watermark/threshold to find it.
- Confirm low variance at n≥3 (v4-r3); then watermark sweep (0.75, 0.90) + threshold.
- **λ=5 NOISY (honest correction):** v4-lb λ=5 p99 {13528, 11186, 23281} (n=3) — r3 landed 23281 (in
  stock's range), so the earlier "λ=5 tail reduction" was partly luck; λ=5 mean ~16s vs stock ~21s but
  high variance. The clearer effect is the **λ=3 variance reduction** (std ~370 vs 2272), but pass-rate
  is UNCHANGED (1/3 both). ⇒ v4-lb is trending NEUTRAL on goodput (tightens λ=3 near the boundary, no
  pass-rate gain). The sweep (v5-w90/v6-w95/v7-th16k) is the last chance for a config that reliably
  clears 8s; absent that, the honest result is the NEGATIVE + a suggestive variance-reduction observation.

### v5-w90 (watermark 0.90) λ=3 — modest positive, NOT a clean flip (HONEST, n=2)
- v5-w90 λ=3 = **{7789 PASS, 8243 FAIL}** (n=2) — ALSO a coin-flip, but centered LOWER (mean 8016) than
  v4-w85 (8291) and stock (8746). So watermark 0.90 lowers the mean further but still straddles 8000.
- **Honest reassessment (no over-claim):** CCA-lb does NOT reliably flip goodput. What it DOES:
  lowers λ=3 p99 MEAN (stock 8746 → CCA ~8016–8291) and REDUCES variance (std 2272 → ~370), nudging
  the coin-flip toward passing but not decisively clearing the 8s SLO. Same PATTERN as the known
  capacity-dedup variance-reduction result, but via a NOVEL mechanism (load-back prefill admission).
- Path to a defensible claim: firm ONE config (v5-w90) to n≥5-6 + stock n≥5-6; Levene (variance) +
  Mann-Whitney (location) on λ=3 p99. Claim "reduces tail mean+variance" IF significant — NOT a
  goodput flip. (base guardrail: 4 over-claims caught by replication — do not repeat.)

### ★★ FINAL (stock n=5, CCA-lb n=5): CLEAN NEGATIVE — variance-reduction is goodput-neutral/negative
- stock λ3 {6327,6391,7765,8124,11786} mean 8079 std 1988, **3/5 pass**.
- CCA-lb λ3 {7767,7789,8242,8549,8555} mean 8181 std 348, **2/5 pass**.
- CCA-lb tightens the p99 spread ~5.7× (real behavioral effect) BUT: (a) NOT statistically significant
  (Levene p=0.15, Fligner p=0.16 — underpowered on the heavy-tailed coin-flip even at n=5); (b)
  goodput-NEUTRAL-to-NEGATIVE — CCA-lb passes LESS often (2/5 vs 3/5) because tightening removes the
  lucky-LOW passes (6327, 6391, 7765) along with the unlucky-high fails. Mean unchanged (8181≈8079).
- ★∴ **CLEAN NEGATIVE: NO admission-control variant improves goodput@SLO.** Load-back gating trades the
  coin-flip for a predictable near-SLO value — no pass-rate gain (slight loss). The variance-reduction
  is an honest behavioral OBSERVATION, NOT a contribution. Firming further is pointless (Levene trending
  ~0.15 not →0.05; pass-rate stock≥CCA-lb). Paper = rigorous negative + characterization + TP-lesson.

### (superseded) picture (stock n=4): variance-reduction REAL but goodput-NEUTRAL
- stock λ3 {6327, 6392, 8124, 11787} n=4: mean **8158**, std 2216, 2/4 pass.
- CCA-lb {8556,8549,7767,7790,8243} n=5: mean **8181**, std 348, 2/5 pass.
- ⇒ CCA-lb reduces the p99 SPREAD ~6.4× (std 2216→348; range 5460→789ms) BUT mean is EQUAL (8158≈8181)
  and pass-rate is EQUAL/slightly-worse (2/4 vs 2/5). **The variance-reduction does NOT improve
  goodput** — it trades an unpredictable coin-flip for a predictable near-8s value, still ~50% failing.
- **∴ Paper conclusion = NEGATIVE: prefill admission control (any variant) does not improve goodput@SLO.**
  Load-back gating's variance-reduction is a descriptive side-observation (makes p99 predictable, not
  better). Report the ~6.4× spread reduction DESCRIPTIVELY (Levene underpowered at feasible n on the
  heavy-tailed coin-flip: p≈0.11 at n=4/5). Do NOT frame as a goodput win.

### Stats (superseded): variance-reduction SUGGESTIVE, not yet significant
- stock λ3 p99 (n=3): mean 8746, std **2272**. CCA-lb pooled v4+v5 (n=5): mean 8181, std **348**.
- **Levene p=0.10 (variance) — NOT significant** at n=3/5 (stock variance driven by the 11787 outlier;
  small n → unstable variance estimate). MWU p=1.0 (means overlap). pass@8s: stock 1/3, CCA-lb 2/5.
- ⇒ The ~7× std ratio is visually striking but needs n≥6 (both arms) to firm (base needed n=6-12).
  Confirm campaign → stock n=4, CCA-lb n=7; re-test then. HONEST: report as SUGGESTIVE unless Levene<0.05.
- ROBUST findings (paper's core, decisive at current n): the deferring-negative (56s), the recompute
  crash (0→518), the characterization (coin-flip/bimodal/75%-cold/0-retract), the TP-desync fix.

## Contribution = rigorous NEGATIVE + characterization + TP-lesson (+ suggestive variance-reduction)
**Thesis:** goodput@SLO for long-document multiturn 2-tier serving is a **cold-prefill-compute-bound
metastable coin-flip**, and **prefill admission control that defers expensive prefills cannot improve
it (and can catastrophically harm it)** — because the SLO tail IS the expensive cold prefills.
Evidence: (1) cost bimodal 1500×, 75% cold, top-10%-turns=78%-work; (2) pool bimodal (metastable);
(3) stock goodput coin-flip {0, 3.02} same-node, **0 retracts** (collapse ≠ retraction, it's
prefill-compute queueing); (4) CCA measured negative — recompute-currency induces 0→518-retract crash
(cost-misestimation via ignored load-back footprint); device-footprint currency (correct, stable)
craters λ=3 p99 to 56s. (5) Systems by-product: a per-rank wall-clock scheduling decision desyncs TP
ranks' prefill batches → NCCL hang (must be deterministic / all-gathered).

## Formal Submissions

**Paper 1 — `submissions/cca-prefill-admission/paper.html` — SUBMITTED (v1).** Clean negative +
characterization + compute bounds (§2.3) + two-regime engine evidence (§2.4) + TP-desync systems
lesson. Registered in `submissions/INDEX.md`.

## ★ Post-submission deepening: scheduling axis rigorously CLOSED → design space CLOSED

After paper 1, I probed the last open control axis — **scheduling** (reordering the waiting queue) —
to decide whether a genuinely novel paper 2 exists. It does not; the axis is closed.

- **Offline oracle** (`analysis/oracle.py`): trace-driven discrete-step sim of sglang chunked prefill
  (1-new-seq × 6144-tok budget/step), calibrated to the measured saturation throughput 4.22 req/s.
  Compares FCFS / SRPF(small-first) / SRPT / LPF / EDF / SLACK(tail-protecting). **Structural finding
  (robust):** small-first ordering (which the stock cache-aware `lpm` policy approximates) minimizes
  median TTFT but pushes the heavy cold-doc tail (= the p99) outward; tail-protecting orders bound it —
  BUT none lifts throughput past the ~4.2 req/s wall. ⚠ The sim is **fidelity-limited in absolute
  latency** (it does not model prefill/decode step interleaving; sim p99 ran 10–30× high) → I use it
  ONLY as structural corroboration, never for absolute goodput claims.
- **Direct engine evidence (load-bearing, from `runs/v0-stock/server.log` + metrics):**
  - Prefill is **serial+chunked**: nearly every prefill step is `#new-seq:1, #new-token:6144` → a
    190.9k-tok cold doc occupies ~31 consecutive prefill steps (its TTFT floor is physical).
  - **λ=3: `#queue-req≈0`** (empty queue) → scheduling ORDER and admission are BOTH inert; goodput is
    pure cold-doc collision variance (the coin-flip). No reordering can help below the knee.
  - **λ≥5: `#queue-req` 20–183 (deep) BUT `full token usage` 0.96–0.98 + input tput ~39k tok/s** →
    simultaneously compute- AND KV-memory-bound. Reordering is meaningful here and the sibling's SRPF
    reaches ~4.1 req/s = within ~3% of the 4.22 work-conservation ceiling (§2.3). The residual gap is
    the compute wall, not a missing policy.
- **Ground-truth calibration** (`runs/v0-stock` full sweep): achieved req/s saturates 2.87→3.66→4.00→
  **4.22** at λ=3→5→7→10 (arrival ≫ achieved above λ3); new-prefill compute ≈ input_tput×(1−hit) ≈
  **~19k tok/s** at saturation — matches §2.3's C independently. hit 0.677→0.659 across rates.

**Conclusion:** the control-policy design space for goodput@SLO on this frozen contract is CLOSED —
admission = negative (paper 1), residency/caching = bounded (fleet: LRU≈Belady), reordering = inert
below the knee + ≤3% of the compute ceiling above it (sibling SRPF already there). The only lever that
could raise the 4.22 wall is reducing cold first-turn prefill COST (context/sequence parallelism,
compression, quantization) — all either lossy or frozen by the contract (`--tp 8`, model, KV budget).
Paper 1 deepened with §2.4 to make this the definitive characterization; no fragile/derivative paper 2.

## ★★ PAPER 2 — `corpus-bound-goodput` — SUBMITTED v1 (caching axis; commit 3e3a36543)

Opened a NEW direction (caching/document-reuse) and found a distinct, rigorous negative:
**the document-reuse mirage.** The workload LOOKS like a caching goldmine (45% of conversations reuse a
document another already prefilled; 2 documents serve 41% of conversations) but yields ZERO online
caching headroom:
- **Corpus structure** (`analysis/doc_reuse.py`): 888 unique docs / 1553 convs; DEGENERATE popularity —
  858 docs (97%) used exactly once, only 2 hot (538×, 100×); hot-doc reuse distance median=2 convs.
- **PROOF (trace replay @ CAP 10.7M):** single-pass **LRU = LFU = Belady = 0 avoidable re-prefill** →
  LRU is Belady-OPTIMAL; every miss is a first-sight unique-doc prefill (cold floor 18.4M tok,
  policy-invariant). Hot docs trivially resident; singletons have no reuse to capture.
- **The 55M "avoidable" is a benchmark artifact** of the 4×-rate replay (same convs, no flush), and it is
  online-UNCAPTURABLE (LFU 54M ≈ LRU 55M; only clairvoyant Belady 23M recovers it).
- **Eviction knob DEAD on the active path:** `HiMambaRadixCache.evict()` hardcodes `last_access_time`
  (LRU) and ignores `--radix-eviction-policy` (wired only into non-hier `RadixCache`) → no confounded
  A/B needed; negative holds by proof AND construction.
- **Corpus-bound model + control-invariance:** goodput ≤ C/mean-unique-work (~4.2); p99 ≥ max-unique-doc/C
  (~5–11s); invariant across ALL 3 control axes (caching=here, admission=paper1, scheduling=oracle).

## DIRECTION 3 (RESOLVED — bounded negative, folded into paper 2 §5.1, commit fea49708f)

**Context-parallel prefill — the sole compute-side lever against the p99 cold-doc floor — is
architecturally UNAVAILABLE for the Qwen3.5 hybrid GatedDeltaNet model.**
- Live check (job 19827, cancelled): `--enable-prefill-cp --cp-strategy zigzag` launches but the server
  reports `attn_cp_size=1` → CP is an INERT no-op for this model.
- Code: the CP auto-enable (`attn_cp_size = tp//dp`) is gated to `is_deepseek_dsa(hf_config)`
  (DeepSeek 3.2 / GLM-5 DSA) in `server_args.py`; qwen3_5 is not in the arch list. `qwen3_5.py` has ZERO
  CP code — only `qwen3_moe.py` (pure-attention MoE) and the DSA backends implement prefill-CP.
- ★INTEGRITY CORRECTION (commit f482245e9): CP is an IMPLEMENTATION GAP, not a fundamental barrier.
  I briefly overclaimed "linear-recurrence can't be context-parallelized." That is WRONG: CP splits the
  SEQUENCE, so O(n) per-token work (proj/MoE/linear-attn) parallelizes across ranks, and the GatedDeltaNet
  recurrence admits a chunked cross-device state scan (O(cp) serial combine, as in sequence-parallel
  Mamba). Prefill would parallelize near-linearly.
- FLOP breakdown (`analysis/prefill_flops.py`, config: 36 GDN + 12 full-attn layers): O(n²) attention is
  21% / 49% / 73% of prefill FLOPs at the median / p99 / max document → the biggest cold docs (the p99
  floor) are ATTENTION-DOMINATED, so parallelizing their prefill would cut the tail materially.
- So CP WOULD help — it is just NOT WIRED for the qwen3_5 hybrid model in sglang v0.31 (attn_cp_size stays
  1; auto-enable gated to `is_deepseek_dsa` + pure-attention Qwen3-MoE). A hybrid-model prefill-CP
  implementation is a plausible FUTURE lever against the p99 floor (large engineering effort; not this work).

Under the current engine + frozen contract, the p99 cold-doc floor stands: no serving-policy lever moves
it (papers 1, 2), and the one compute-side lever that could (CP) is unavailable for this hybrid model.
goodput@SLO is corpus- and hardware-bound in this system. Both "CP blocked by fixed tp" and
"linear-recurrence can't be CP'd" were wrong framings — corrected.

## ★ Paper-2 empirical grounding (eviction/retention A/B) + phase boundary (2026-07-14 session)

**Integrity corrections this session** (found by hands-on verification, not armchair):
1. Active cache = **UnifiedRadixCache** (not HiMambaRadixCache, which is DORMANT). My first hot-pin edit
   was a dead path (flag True but inert) → cancelled wasted run 19829.
2. Eviction knob is **LIVE** on the active path (`full_component.drive_eviction` orders by
   `eviction_strategy.get_priority`; strategy from `--radix-eviction-policy`, kv_cache_builder.py:219).
   Paper 2 §3.3+abstract "dead knob" claim was WRONG (misattributed to dormant HiMamba) → FIXED
   (commit e5f7af397). Reverted redundant dead-path hot_doc_pin code (692e4364a); native `SLRUStrategy`
   IS the retention steelman (protect nodes with hit_count≥threshold).

**§3.4 phase boundary (commit 107fd9157):** caching headroom = fraction of reuses with LRU stack-distance
> cache horizon H=CAP/mean-doc≈517 docs. Measured: 99.8% of 665 reuses within H (p50=2,p90=11,max=519) →
mirage regime, ~0% headroom. Generalizes: predicts exactly when caching WOULD help (large working set,
long reuse distance). `analysis/reuse_distance.py`.

**Eviction/retention A/B (job 19830, ondem-2, screen λ3,5, live --radix-eviction-policy):**
- Hypothesis: since single-pass LRU=Belady (§3.1) + 99.8% reuses within horizon (§3.4), NO eviction
  (LFU) or retention (SLRU=protect reused nodes) policy beats stock LRU on hit_rate or goodput@SLO.
- Method: v-evict-lfu (LFU) + v0-stock-evictctl (LRU, same-node) + v-evict-slru (SLRU), compare hit+goodput.
- OUTCOME: LFU partial run CORROBORATED degradation — at λ=3 (where stock LRU queue≈0) LFU built a DEEP
  prefill queue (queue-depth p50≈7, p90≈91) → LFU degrades, does not improve. Killed the ~4h campaign
  after this clear signal to conserve the shared node (LRU=Belady proof is dispositive anyway). §3.3
  now rests on proof+phase-boundary, corroborated by the partial LFU observation. Did NOT run stock-ctl/SLRU.
- This gives paper 2 its rate-sweep empirical leg (program paper §5 spec).

## ★★ PIVOTAL CORRECTION (head-of-line): scheduling IS the lever, not caching/admission

Hands-on re-examination (arithmetic + calibrated sim) overturned my earlier "λ3 p99 = prefill floor /
cold-prefill-compute-bound / control-invariance" claims in BOTH papers.
- **Arithmetic (robust):** λ3 p99 (6–11s) is NOT a per-request prefill floor — only <0.1% of turns
  prefill ≥6s (p99-work ~1–2.5s; max-doc ~5s @39k tok/s). So the 6–11s p99 is QUEUE-WAIT.
- **Mechanism:** HEAD-OF-LINE blocking — small turns stuck behind the back-to-back chunked prefill of a
  few large cold docs (scheduler runs chunked_req without yielding, verified scheduler.py:2838-2841).
- **Sim (`analysis/hol_sim.py`, per-step 6144-tok greedy-fill):** reproduces the stock λ3 coin-flip (p99
  mean 7507ms, runs 5.6–11.0s, straddles SLO) and shows **shortest-prefill-first collapses it to ~4031ms,
  all 5 seeds pass (−46%, coin-flip removed)**; round-robin interleave is WORSE (my idea inferior).
- **Consequence:** CACHING (paper 2 mirage) and ADMISSION (paper 1 negative) cores STAND, but the
  "no-lever/compute-bound/control-invariance" framing was WRONG — **SCHEDULING (shortest-prefill-first)
  IS the goodput lever** (relieves λ3 head-of-line + λ5 throughput=concurrent SRPF). Both papers reframed:
  P1 = "admission is the wrong tool, scheduling the right"; P2 retitled "cache retention is the non-lever,
  prefill scheduling is the lever." Commits cf6c60f94 (P1), 297bde123 (P2), 62bd9c28d (INDEX).
- **Honesty note:** SRPF/SJF is a TEXTBOOK scheduling policy (the charter explicitly excludes it as a
  contribution); my contribution is the head-of-line DIAGNOSIS + the admission/caching NEGATIVES. Papers
  now cite SRPF as textbook prior art (no sibling attribution) and back the reordering claim with my OWN
  GPU run.
- **Lesson:** my earlier oracle.py over-serialized (1-turn-per-step), wrongly suggesting small-first
  "pushes the tail out"; hol_sim.py (correct 6144-packing) shows the opposite. Verify sim fidelity before
  drawing scheduling conclusions.
- **Direct trace (`analysis/hol_trace.py`):** in the stock run, the waiting queue is **9.6× deeper**
  during big-cold-doc prefill chunks (#new-token≥6144, #cached-token=0: mean 11.9, p90 33, max 183) vs
  other prefill steps (mean 1.2, p90 0) — head-of-line confirmed directly from the real run, independent
  of the sim.

### ★ GPU A/B DONE (job 19834, node 0-3, my own run): SRPF λ3 CONFIRMS head-of-line relief
- **v-srpf-r1 λ3:** p99 TTFT **5892ms**, p50 519ms, req_tput 3.02, hit 0.673 — **below the 8s SLO**,
  below the stock coin-flip mean (8079) and 4/5 of its draws {6327,6391,7765,8124,11786}. Matches the sim
  prediction (srpf_np 6205ms). ⇒ reordering (serve-small-first) relieves the λ3 head-of-line tail, exactly
  as diagnosed. n=1 (corroborative; diagnosis rests on arithmetic + trace, not this one run).
- **v-srpf-r1 λ5:** p99 TTFT **5850ms at 4.16 req/s** — BELOW SLO. Stock λ5 = 17372ms (stable fail,
  ≤1.08× var). ⇒ **goodput@SLO 0/≤3 (stock coin-flip) → 4.16 (srpf) = the compute ceiling**, purely by
  reordering prefills (cache untouched). n=1 but λ5 stock is stable-fail so the 3× gap is robust. This is
  my OWN full GPU sweep confirming scheduling is the lever at BOTH regimes — no sibling citation needed.
- **Logged to W&B** (`sgl-evolve/floyd`, v-srpf-r1, tag=mechanism, churn py+206, goodput 4.16).
- **★ SAME-NODE CONTROL FIRMED (job 19854, node 0-3, `runs/v-stock-srpfctl`):** stock (fcfs) λ3 p99
  **7612ms** (lucky coin-flip draw, passes), λ5 p99 **22746ms** (stable FAIL) → stock goodput@SLO=3.02.
  vs same-node srpf λ3 5892/λ5 5850 → 4.16. **Clean same-node A/B: λ5 gap 3.9× (22746 vs 5850), NO
  cross-node confound; +38% goodput.** Reference v0-stock λ5 17372 agrees (stock fails λ5 robustly).
  Logged W&B (v-stock-srpfctl, tag=config). ⇒ P3 draft→**submitted**. Campaign now running srpf r2/r3
  (n=3 firming for λ3). ★New: `analysis/slo_sensitivity.py` — scheduling decisive for SLO∈[6,41]s
  (+38% @8s), NOT an 8s artifact; `analysis/gen_curve_svg.py` — goodput-curve figure (in P2/P3).
- Integrated into BOTH papers (commits above); SRPF cited as textbook, my GPU sweep as the confirmation.
- **Other axes bounded this session:** Mamba-state pool NEVER binding (max usage 0.77 vs attn-KV 1.0,
  `analysis/mamba_pressure.py`) ⇒ hybrid-asymmetry is a non-lever too. Design space for KV-cache goodput
  levers is closing hard: residency=mirage, admission=neg, transfer=cheap, Mamba=non-binding; only
  prefill SCHEDULING (textbook) and compute-side cost reduction remain.

### DIRECTION SCREENED-OUT (negative, no GPU spent): head-of-line fast-lane
- **Hypothesis:** relieve head-of-line WITHOUT reordering by reserving R tokens of the per-batch chunk
  budget for waiting small turns while a big cold doc is mid-chunk (chunk_cap on add_chunked_req).
  Distinct primitive from SRPF (budget-partition vs queue-reorder); lossless.
- **FAST-SCREEN (`hol_sim.py` fastlane policy, λ3, 5 seeds):** stock 3/5 (7507ms); fast-lane R∈{1k,2k,3k}
  **4/5 (~6310ms)** — beats stock; but non-preempt SRPF **5/5 (6205ms) STRICTLY DOMINATES it**.
- **Verdict: NEGATIVE, screened out (no GPU burned).** Structural reason: fast-lane still gives the big
  cold doc B−R tokens/step, so it is a *partial* SRPF that converges to SRPF only as R→B (where it IS
  SRPF). It cannot beat the textbook policy ⇒ not a contribution (charter: a gain a textbook policy also
  gets is not a contribution). REINFORCES the papers: SRPF is the *efficient* realization of head-of-line
  relief; budget-partition variants are dominated. Fast-screen discipline (sim before GPU) paid off.

### DIRECTION 3 (screened): hit-rate≠goodput "condition 2" — REFUTED (negative)
- Hypothesized caching hit-rate headroom doesn't help goodput because the p99 tail is all COLD.
  `analysis/goodput_vs_hitrate.py`: 4-pass tail is ~75% AVOIDABLE (big docs recur across passes) →
  refuted; a better cache COULD cut big-prefill COUNT. Decoupling holds only within-rate (single-pass
  LRU=Belady, vacuous). DROPPED. Kept the TRUE finding: p99 per-request prefill floor = largest UNIQUE
  doc, policy-invariant (supports P2 corpus-bound).

### DIRECTION 3 (real, DONE): P2 §3.4 phase-boundary VALIDATED across constructed corpora (P2 v3)
- `analysis/phase_boundary.py`: LRU−Belady avoidable gap is a STEP FUNCTION at stack-distance = H
  (0 below = mirage/LRU-optimal, full-reuse above = caching helps); crossover exactly at H. Makes the
  mirage a sharp, PREDICTIVE, corpus-general criterion (addresses the #1 generality concern). Folded
  into P2 §3.4 as a validation table (v3, committed).

### DIRECTION 3 (last axis): prefill/decode split — NON-LEVER (negative)
- `analysis/pd_schedule.py`: at saturation 99% of forward steps are PREFILL (decode starved to 1%).
  goodput@SLO is TTFT-based → spending compute on prefill (more first-tokens) is already optimal;
  shifting to decode lowers goodput, throttling prefill = admission (P1 neg). Throughput 4.22 ==
  ceiling 4.2 → compute-hard, no macro-bubbles (big log gaps = protocol flushes). Added to P1 §2.4.

### ★ DESIGN-SPACE MAP NOW COMPLETE (goodput@SLO, this workload/eval)
Every KV-cache/serving axis characterized:
| axis | verdict | evidence |
|---|---|---|
| residency / eviction | non-lever (mirage, LRU=Belady) | P2 §3, phase-boundary |
| admission | negative (deferring worsens HOL) | P1 §4 (CCA) |
| transfer (L1↔L2) | non-binding (~1.6ms) | P1 |
| Mamba-state pool | never binding (max 0.77) | mamba_pressure.py |
| fast-lane chunk-partition | dominated by SRPF | hol_sim (screened) |
| prefill/decode split | TTFT-optimal (99% prefill) | pd_schedule.py |
| **prefill SCHEDULING** | **THE lever (0/3→4.16)** | **SRPF sweep (textbook)** |
| compute | hard ceiling ~4.2 req/s | bound.py, matches measured |
My contributions = the two axis-negatives (P1 admission, P2 caching) + the head-of-line DIAGNOSIS
(multi-method: arithmetic+trace+sim+GPU) + the phase-boundary generality. SRPF (the lever) is textbook,
cited not claimed. The space is mapped; a genuine NEW lossless mechanism would have to beat the compute
ceiling (CP/quant — lossy or contract-frozen) — none exists losslessly. Continue: firm the goodput A/B
(campaign n=3 + same-node control in flight), keep papers bulletproof, stay honest; supervisor retires.

## ★★ INTEGRITY CORRECTION (2026-07-15): empty-doc-hash bug inflated document-popularity stats
Found while building `analysis/phase_boundary_realcorpora.py` (multi-corpus generality) + an adversarial
PC self-review of P3 (via subagent). TWO issues fixed:

**(A) Adversarial-review integrity fixes to P3** (commit e-review): (1) the "predictive model" was CIRCULAR
— `C_eff=15000` is back-solved from the measured ~4.2 saturation (`bound.py` "back out the effective rate"),
so `g_ceil=C_eff/E[W]` reproduces 4.2 by construction. Deleted the "genuine cross-validation not a tautology"
claim; reframed g_ceil as a SELF-CONSISTENCY check; promoted the PARAMETER-FREE lever decision (f<T, h, bimodal)
as the real contribution. (2) SRPF is n=1 — removed language implying r2/r3 data exists; headline now rests on
stock λ5 STABLE-fail (n=5, 17-24s) vs single SRPF pass 5.85s (3-4× gap a coin-flip can't fake). (3) goodput
∈[4.16,<7) not "=ceiling". (4) "provably"→per-row evidence strength. (5) sim is λ3-only scope.

**(B) empty-doc-hash bug** (commit 14e59304b): `reuse_distance.py`/`doc_reuse.py` hashed the 538 empty-`input`
ShareGPT chat records to ONE fixed md5 → counted as 538 spurious "reuses of one document." This produced the
WRONG "2 hot docs serve 41%", "45% shared", "888 unique", "reuse distance median 2" claims in P2/P3.
- FIX: `if not doc: continue` (document-reuse analysis is doc-bearing convs only).
- TRUE structure: 1553 convs = **1015 doc-bearing (887 unique) + 538 chat (no doc)**. **ONE** hot doc (100
  convs = 6% of all / 10% of doc-bearing), NOT two serving 41%. 97% singletons (was correct). Only **~10% of
  convs re-use a document** (4% of turn-0 work), NOT 45%. Reuse stack-dist p50=10/p90=197/max=518 vs H=516 →
  **99.2% captured, 0.8% headroom** (was p50=2/p90=11/99.8%). Hot-doc recur gap median 6. doc-hit 13% single /
  13-14-51% 4-pass (was 43/68%, inflated).
- **CORE MIRAGE UNCHANGED & STRONGER**: single-pass LRU=LFU=Belady=0 avoidable STILL holds; cold floor 18.4M
  unchanged; 4-pass online-uncapturable still holds. The workload has even LESS real cross-doc reuse than
  claimed → caching is even more of a mirage.
- Corrected P2 abstract/§1/§2/§3/§3.1/§3.4 + P3 abstract-bullet/§6. Added `phase_boundary_realcorpora.py`:
  **4 real long-doc corpora (mooncake/LEval/LooGLE×2) ALL in mirage** (≤0.8% headroom) → generality is now
  multi-corpus, not n=1 (answers the #1 reviewer concern). This multi-corpus analysis is what caught the bug.
- LESSON: hashing a possibly-empty field silently collapses all empties to one key → spurious reuse. Always
  guard `if not <field>`. Multi-corpus/cross-check analyses catch single-trace artifacts.

## ★★ ADVERSARIAL-REVIEW INTEGRITY FIXES (2026-07-15, P1 + P2, via subagents)
Ran hostile-PC subagent reviews on P1 (CCA) and P2 (mirage) — both found reject-level integrity issues, all
verified against artifacts and fixed (no fabrication left):
- **P2:** (1) cited an "LFU/**SLRU** A/B" — SLRU was NEVER run (only in jobs_slru.txt) → removed all SLRU-run
  claims. (2) the live **LFU** run never completed a benchmark (curve.csv header-only, no summary.json) →
  reframed "degrades" to queue-depth-only w/ MATCHED stock baseline (LFU overall mean 24.7 vs stock 7.8);
  LRU=Belady proof is the load-bearer. (3) ceiling "exactly the measured saturation" → self-consistency
  (C_eff back-solved); reconciled E[W]=3594 vs 4.3k; sensitivity band. (4) "~4s all-passing" was the
  PREEMPTIVE sim; deployed non-preempt is ~6s (matches GPU 5.9s) → relabeled. (5) cited Mattson1970/Denning
  (stack-distance is classic). (6) 4 real corpora: 2 are pure-singleton (0 reuse) — stated honestly. (7)
  reconciled doc-hit 13% (cross-conv) vs server 0.68 (within-conv). Updated stale docstrings.
- **P1:** (1) the "recompute-currency → **518-retract crash**" had NO preserved artifact (v1-cca crashed
  before writing one; no run dir, no server.log w/ cca_ignore_loadback) → REMOVED the table row + all
  specific claims; reframed as an excluded early unstable run (qualitative hazard only). Device-footprint
  (v3-cca, artifact exists) carries the catastrophe. (2) CCA-lb "n=5" POOLED W=0.85 {7767,8549,8556} and
  W=0.90 {7790,8243} → split; dropped pooled "5.7×/2/5" variance claim. (3) SRPF corroboration was
  cross-node (0-3 vs ondem-3) → replaced w/ clean same-node A/B (control on 0-3: λ3 7.6 / λ5 22.7). (4)
  ceiling: dropped circular C=4.2×mean; lead w/ independent input_tput×(1−hit)≈19k; band ~4.2-5. (5) "no
  valve escapes" → by-construction. Verified reproductions (trace_diag concentration, hol_trace 9.6×,
  Levene 0.15) all hold.
- LESSON: **preserve server.log for crashing runs** (else the finding is uncitable); never pool distinct
  configs into one "n"; a run that took effect ≠ a run that produced a metric (check summary.json exists);
  hostile-PC subagent review is HIGH-YIELD (caught what self-review missed). FUTURE: optionally re-run
  recompute-currency capturing server.log to restore that finding; run SLRU (jobs_slru.txt) if desired.

## Direction 4 SCREENED-OUT (negative, no GPU): "SRPF worsens E2E/TPOT" metric-tension — REFUTED
Hypothesis: prefill-first + SRPF (TTFT-optimal) might starve decode → worse E2E/TPOT (a metric-tension paper).
Fast-screen from existing run bench json (stock v0-stock vs srpf v-srpf-r1):
- λ3: TTFT 11787→5892 (srpf better); TPOT 3465→4330 (srpf slightly WORSE); E2E 437790→322022 (srpf BETTER).
- λ5: TTFT 17372→5850 (better); TPOT 3093→3130 (~same); E2E 410625→335694 (srpf BETTER).
⇒ REFUTED: SRPF does NOT worsen E2E (it improves it); TPOT only marginally worse at λ3. No clean tension.
The real observation (E2E p99 ~5-7 MIN regardless of policy) is a saturation/open-loop-rate-sweep artifact,
OFF the fixed TTFT metric. Not a paper. Value-add: added an honest P3 limitations note that SRPF's TTFT win
is NOT bought at E2E/TPOT's expense (preempts the reviewer "what about E2E" question). Fast-screen-before-
build discipline again avoided a false-hypothesis paper.

## ★★ FULL n=3 FIRMING COMPLETE (2026-07-15) — headline scheduling win firmed
Campaign jobs_srpf.txt fully done (r1, v-stock-srpfctl, r2, r3). SRPF now **n=3 at both rates, all 6 runs
clear the 8s SLO**:
- λ3: {5892, 6039, 7387}ms — 3/3 pass (vs stock λ3 coin-flip 3/5 {6327,6391,7765,8124,11786}).
- λ5: {5850, 6576, 7457}ms — 3/3 pass, req/s {4.16,4.07,4.05} (vs stock λ5 STABLE-fail 0/5, 17-24s).
- Overall srpf range 5.85-7.46s; genuine run-to-run spread (worst 7.46s = 0.5s under SLO) but EVERY run
  passes. Reported honestly (no cherry-picking best run). goodput@SLO: stock ≤3 (λ3-bound) → srpf ~4.1.
- All logged to W&B (v-srpf-r1/r2/r3 mechanism, v0-stock/v-stock-srpfctl config). Integrated across P1/P2/P3.
**RECOMPUTE-CURRENCY RE-RUN: DECLINED** — it's a known-server-crashing config; re-running on the SHARED pool
risks a CUDA-coredump/COMPLETING-wedge that disrupts sibling cells (fairness first). It's a secondary P1
finding and P1 is honest without it (cleanly excluded, qualitative hazard noted). Not worth the neighbor risk.

**CAMPAIGN STATE: COMPLETE & INTEGRITY-CLEAN.** 3 submitted papers (P1 admission-neg+TP-lesson, P2
caching-mirage+phase-boundary, P3 capstone-map+diagnosis+predictive-model), all survived 2 adversarial-review
rounds, design space fully bounded, n=3-firmed headline, multi-corpus generality, honest E2E scoping,
complete related-work, reproducible, W&B current. Supervisor decides retirement; I hold in monitor mode.

## ★ Active-research find (2026-07-15): DECODE-SIDE head-of-line — unifies the diagnosis
Rather than passive monitor mode, actively investigated the E2E catastrophe. `analysis/hol_decode.py`
(v0-stock server.log): during big-cold-doc prefill chunks, running requests wait **24.5s between decode
batches (p90 37s, max 182s) vs 1.8s otherwise = 13.6× decode stall**. ⇒ the SAME head-of-line mechanism
(big cold docs monopolizing prefill) has TWO victims: (1) TTFT of waiting small turns (9.6× deeper queue,
hol_trace), (2) decode/ITL of running requests (13.6× stall → E2E minutes). SRPF fixes the TTFT side
(reorder) but NOT the decode stall (serializing a big cold prefill is inherent) → sharpens the case for
context-parallel prefill (§5.1) as the full fix. Added to P3 §4 (two-victims), reconciled §8 E2E note
(mechanism not open-loop artifact), + Fisher significance of the goodput win (λ5 p=0.018, pooled p=0.011,
`analysis/goodput_stats.py`). All committed + pushed (evolve/floyd @ 03a21af76). Papers stronger; core intact.

## ★ CONTRACT-COMPLETE FULL SWEEP (2026-07-15): goodput@SLO = 4.0, resolved
Ran the SRPF policy over the **full fixed contract λ{3,5,7,10}** (job 19901, `runs/v-srpf-full`), not just
the λ3/5 screen. Result (p99 TTFT / achieved req/s):
- λ3 **6503ms** / 3.02 — PASS
- λ5 **7823ms** / 4.00 — PASS
- λ7 **10318ms** / 4.48 — FAIL (SLO)
- λ10 **13964ms** / 4.72 — FAIL (SLO)

⇒ **goodput@SLO = 4.0 req/s** (the λ5 rate; SRPF's p99 crosses the 8s SLO between λ5 (7.8s) and λ7 (10.3s)).
This *resolves* the earlier `[4.16, <7)` range to a point, and **refines** two things the λ3/5 screen left
open:
1. **Goodput is SLO/tail-bound, NOT raw-compute-bound.** SRPF's *raw* throughput keeps climbing past λ5 —
   4.48 (λ7), **4.72 (λ10)** — while its p99 crosses the SLO. So the binding constraint at higher load is the
   head-of-line *tail* crossing 8s, not a compute wall. This STRENGTHENS the head-of-line thesis (§4): the
   tail is what caps goodput at every regime.
2. **The raw-throughput ceiling is ~4.7, not ~4.2.** SRPF reaches 4.72 req/s at λ10 (within-run measured).
   The earlier "~4.2 ceiling" was a sub-knee estimate; the full sweep measures it directly. C_eff back-out
   updated 15k→~17k; band ~4.2–4.7. ⚠ NOTE (integrity): stock's own full sweep reached 4.22 at λ10 but on a
   DIFFERENT node/day (v0-stock 07-14 vs v-srpf-full 07-15) — so I do NOT claim "SRPF raises the ceiling
   +12% vs stock" (cross-node, ±45% var, void by my own rule). I only claim the measured raw ceiling is ~4.7;
   whether SRPF raises it above stock's would need a same-node full sweep (not run — would be gilding, and the
   claim isn't load-bearing: the headline goodput=4.0 rests on the within-run v-srpf-full curve).

The same-node n=3 A/B (node 0-3: srpf λ3 {5.9,6.0,7.4}/λ5 {5.85,6.58,7.46} vs stock control 7.6/22.7) still
provides the *rigorous* λ3,5 comparison + Fisher significance; v-srpf-full (node 1-2) provides the
*contract-complete curve*. Both cited; goodput headline = 4.0.

**Integrated across all 3 papers** (P3 abstract/fig/table/§5/§6.1, P1 §2.3/§4, P2 §4/fig): goodput 4.0,
raw ceiling ~4.7 measured, goodput SLO-tail-bound below it. Figure regenerated data-driven from
`runs/v-srpf-full/curve.csv` (`analysis/gen_curve_svg.py`). W&B logged (v-srpf-full, tag=mechanism).
All papers re-verified well-formed. Committed + pushed to evolve/floyd.

## ★ NEW DIRECTION (Paper 4): RPB — Reserved-Prefill-Budget chunking (2026-07-15)
A paper is a checkpoint, not the end. My P3 map bounds *whole-request* scheduling (SRPF, textbook) and the
FCFS *fast-lane* (dominated). The ONE scheduling sub-axis it does NOT close — and the exact thing a skeptical
PC would poke at the impossibility claim — is **chunk-level / preemptive** prefill scheduling.

**The residual SRPF leaves.** `--schedule-policy srpf` sorts only the WAITING queue. But the scheduler adds
the in-flight `chunked_req` to the batch FIRST every step (scheduler.py:2904; `add_chunked_req`,
schedule_policy.py:727), and it takes `min(cand_len, rem_chunk_tokens)` = the whole per-batch chunk budget.
So once a big cold doc's chunking starts, it runs ~31 chunks (190K/6144) back-to-back and small turns
arriving *mid-prefill* still head-of-line block — SRPF cannot preempt an in-flight chunk. This residual GROWS
with rate (more arrivals land during a big-doc chunk-run), i.e. it bites hardest at λ5 (the goodput-setting
rate, where deployed SRPF is a marginal 7.8s pass).

**Mechanism (RPB).** When a big doc is mid-chunk AND requests are waiting, cap the in-flight chunk at
(1−r)·budget and reserve r·budget for the shortest waiting turn(s) (the waiting queue is SRPF-sorted just
before the adder). Bounds BOTH tails with no starvation: small-turn TTFT ≤ ~1 chunk-time; big-doc slowdown =
1/(1−r) (bounded). Distinct from pure preemptive SRPT (starves big docs) and my screened FCFS fast-lane
(no in-flight cap, FCFS bigs). Flags `--enable-rpb-chunking --rpb-reserve-frac` (default 0.25).

**Novelty (config-equivalence check PASSED).** Not a flag: `add_chunked_req` gives the in-flight doc the whole
budget; `enable_mixed_chunk` mixes *decode* into prefill (not waiting prefills); `enable_dynamic_chunking`
tunes chunk *size* and is gated `pp_size>1` (we run PP=1 → off); `chunked_prefill_size` is a global knob (a
smaller value slows everyone, doesn't reserve for smalls). TP-safe (static frac + deterministic SRPF sort →
identical across ranks; learned from the CCA wall-clock desync hang). Lossless (identical tokens, resliced).

**Impl:** server_args.py (2 flags) + schedule_policy.py (PrefillAdder param + the cap in add_chunked_req) +
scheduler.py (thread param). +34 lines. py_compile OK. Commit 04940e7cd, pushed.

**Sim screen (`hol_sim.py`, calibrated @λ3):** RPB25 λ3 p99 **6205→4340ms (−30% vs deployed srpf_np)**,
matching the preemptive-SRPF idealization (4031) WITHOUT full preemption — i.e. RPB captures the benefit
whole-request SRPF leaves on the table. rpb50 4846 (r=0.25 better — less big-doc slowdown). λ5 sim
over-serializes (documented artifact) → GPU is the arbiter. Since srpf_np already PASSES λ3 (5.9s GPU), the
λ3 win is a margin/robustness gain; the test that matters is whether the same ~30% reduction at λ5 turns the
marginal 7.8s pass into a robust pass.

**GPU A/B (in flight, job 19915→v-rpb25, node 1-1, screen λ3,5):** baseline `--schedule-policy srpf`
(v-srpf-ctl4) vs `srpf + RPB25` (v-rpb25), same node, serial. DELTA = the RPB mechanism (srpf held fixed).
Decision rule: RPB lowers λ5 p99 meaningfully (→robust pass) = firm to Paper 4; RPB ≈ srpf = rigorous
NEGATIVE that closes chunk-level scheduling (strengthens P3 into an airtight impossibility). VERDICT: TBD.

### RPB prior-art positioning + honest novelty assessment (2026-07-15, pre-result)
Web search blocked (org policy); positioning from knowledge (cutoff Jan 2026 covers these).
- **Sarathi-Serve (OSDI'24)** — chunked prefill + stall-free batching fills the leftover per-batch budget
  with *decodes* (= sglang `enable_mixed_chunk`). Targets prefill-blocks-DECODE, NOT big-prefill-blocks-
  small-PREFILL. RPB fills the reserved slice with waiting *prefills*. Orthogonal/complementary.
- **FastServe (2023)** — MLFQ, whole-request iteration-level PREEMPTION for head-of-line. RPB does not
  preempt (no starvation, no re-prefill cost); it co-progresses a budget fraction. Different point in the
  design space (reservation vs preemption).
- **WFQ / DRR / GPS (classic fair-queueing)** — reserve bandwidth per flow. This is RPB's closest ancestor:
  RPB reserves prefill-token-budget for the "short-job flow." So the IDEA (reservation) is classic.
- **SRPF/SJF** — textbook whole-request reordering (my P3 result; a stock `--schedule-policy` flag).

**Honest novelty verdict (governs how I write this up):** RPB's core idea = fair-share/reservation applied
to the chunked-prefill token budget. The charter excludes "textbook policies dropped onto EXISTING pluggable
interfaces" (e.g. SJF via --schedule-policy) — RPB is more than that (a NEW engine mechanism in add_chunked_req,
not an existing interface, motivated by the specific chunk-level HOL diagnosis that whole-request SRPF leaves
a residual). Systems venues DO accept classic-scheduling-applied-to-new-systems-problem when the diagnosis is
sharp + mechanism well-engineered + eval convincing (FastServe=MLFQ→LLM, Sarathi=chunking→LLM). So the
DECISION on standalone-Paper-4 vs fold-into-P3 depends on the GPU magnitude:
  - RPB wins BIG at λ5 (marginal→robust, clear curve shift): candidate standalone systems paper, framed as
    "diagnosing + closing chunked-prefill head-of-line," with fair-queueing cited as the mechanism ancestor
    (honest) and the DIAGNOSIS + engine mechanism + curve as the contribution.
  - RPB wins marginally / neutral: fold into P3 as the completion of the scheduling axis (positive refinement
    OR bounded negative) — NOT an over-claimed standalone novelty. Integrity first.

### RPB A/B — baseline landed (2026-07-15, node slurm2-a3nodeset1-1, screen λ3,5)
Same-node SRPF baseline (v-srpf-ctl4): λ3 p99 **6973ms** (req/s 3.02, hit 0.678) PASS; λ5 p99 **6580ms**
(req/s 4.11, hit 0.666) PASS with **1.4s margin**. ★NODE-VARIANCE NOTE: node 1-1's SRPF λ5 is a COMFORTABLE
pass (6580ms), not the marginal 7.8s seen on node 1-2 (v-srpf-full) — the known ±45% cross-node p99 variance.
⇒ on THIS node the RPB test is a p99-REDUCTION test (can RPB lower 6580→lower?), not a marginal→robust flip.
The same-node A/B is still clean (RPB vs SRPF both on 1-1); a p99 reduction here would generalize to unluckier
nodes where SRPF λ5 is marginal. v-rpb25 next (auto-submitted by the campaign driver).

### RPB A/B — λ3 landed (v-rpb25 vs v-srpf-ctl4, same node 1-1)
| rate | base p99 | RPB p99 | Δ | tok/s | hit |
|------|----------|---------|-----|-------|-----|
| 3 | 6973ms | **6300ms** | −673 (−9.7%) | 386.7→386.7 (0%) | 0.678→0.677 |
★KEY: RPB lowers λ3 p99 ~10% (right direction, matches sim sign), **throughput IDENTICAL** (tok/s 386.7 both →
NO reserve waste; the head-of-line premise holds — small turns are always waiting during big-doc runs, so the
reserve is fully used; resolves my implementation concern, no adaptive-reserve fix needed), **LOSSLESS** (hit
matched Δ0.001). BUT −9.7% is WITHIN same-node run-variance (±~12% at λ3) → suggestive not conclusive → n=3
replicates needed. Effect size on GPU (−10%) < sim (−30%). λ5 (decisive) running.

### ★ RPB A/B COMPLETE — DECISIVE (v-rpb25 vs v-srpf-ctl4, same node 1-1, screen λ3,5)
| rate | base p99 | RPB p99 | Δp99 | base tok/s | RPB tok/s | Δtok | SLO |
|------|----------|---------|------|-----------|-----------|------|-----|
| 3 | 6973 | 6300 | −673 (−10%) | 386.7 | 386.7 | 0% | PASS→PASS |
| 5 | 6580 | **9287** | **+2707 (+41%)** | 525.1 | 480.1 | **−8.6%** | **PASS→FAIL** |
★★RPB is a NEGATIVE at the goodput-setting rate λ5: p99 +41% CROSSES the SLO, throughput −8.6% (req/s
4.11→3.75). OPPOSITE of λ3 (where it helped −10%, throughput-neutral). MECHANISM (clear, important): at higher
load the waiting queue holds MEDIUM turns (> the 1536-tok reserve); the crash-guard correctly refuses to
truncate them, so the reserved slice goes UNUSED → wasted budget → throughput loss; near the saturation knee
(λ5) that throughput loss pushes the tail over the SLO. At λ3 the queue is tiny-turn-dominated (fit reserve) →
no waste → helps. hit matched (lossless holds). ⇒ FIXED-fraction chunk-level reservation trades throughput
for sub-knee tail-relief, and at the near-knee goodput rate the throughput cost DOMINATES → net negative.
NEXT: reserve-size sweep (rpb10 smaller=less waste, rpb50 larger=more waste) to characterize the tradeoff +
establish the negative rigorously across reserve sizes. Likely conclusion: chunk-level scheduling reservation
does NOT improve goodput over whole-request SRPF (closes the last scheduling sub-axis; strengthens P3).

### RPB negative — MECHANISM quantified (analysis/rpb_waste.py, server.log prefill batch sizes)
Baseline (no RPB): big-doc prefill steps mean **6111 tok**, 97% full 6144 chunks. RPB r=0.25: big-doc steps
mean **5320 tok**, ~50% capped ≤4700 with ~0 small turns added → **~54% of the 1536-tok reserve WASTED** on
capped steps. ⇒ RPB cuts effective prefill capacity ~13% on big-doc steps. λ3 (slack): absorbed (overall
tput unchanged 386.7) + small-turn interleave helps (−10% p99). λ5 (near knee): capacity loss crosses
saturation → tput −8.6% (525→480) → queue → p99 +41% SLO FAIL. ROOT CAUSE of the negative = the reserve is
wasted whenever the shortest waiting turn exceeds it (frequent as load rises), and the goodput-setting rate
is exactly where you can't afford lost prefill capacity. Reserve sweep (rpb10/rpb50, running) tests the
size tradeoff: smaller reserve = less waste = less λ5 harm; larger = more harm (expected monotonic).

### RPB reserve sweep — rpb10 CRASHED (transient pool leak), rpb25 clean stands (2026-07-15)
rpb10 (r=0.10) crashed mid-λ3: `pool memory leak detected! total=2347648 available=2112 evictable=2345728`
→ server Killed → partial run (4255/7037 turns, 1886s) → curve garbage (req/s 2.26, p99 7679, hit 0.000). **DISCARD.**
★INTEGRITY: the leak hit ONLY rpb10 — baseline (v-srpf-ctl4) and rpb25 had **0 leak messages** and completed
cleanly (7037 turns, EVAL_DONE). So the main negative (rpb25 λ5 +41%/−8.6% tput) is a CLEAN run, NOT confounded.
The crash is likely TRANSIENT (not deterministic-RPB): rpb25 caps MORE aggressively (4608 vs 5530 → more chunk
boundaries) yet was clean, and r=0.10 has FEWER boundaries — opposite of a capping-induced leak; matches the
intermittent pool/NCCL events seen on long node runs (node 1-1 ran ~5h continuously). rpb50 (r=0.50) running =
leak-robustness check at the MOST-capping config (if clean → confirms transient). Then adaptive RPB (decisive).
Reserve sweep now = rpb25 (clean) + rpb50 (pending); rpb10 point lost to the crash (re-run optional).

### Adaptive RPB — mechanism confirmed working (mid-λ3, v-rpbA25)
Adaptive big-chunk mean **5993 tok (89% full 6144)** vs fixed-rpb25's 5320 (44% full): adaptive caps only ~11%
of big-chunk steps (only when a small turn fits the reserve), so prefill capacity is ~PRESERVED (5993 vs
baseline 6111, −2% vs fixed-rpb's −13%). ⇒ adaptive should AVOID the λ5 throughput-loss SLO-fail. Open Q: does
the selective interleave (the ~11% p99-relevant cases) still HELP the tail (WIN) or is it NEUTRAL? GPU deciding.

### Adaptive RPB λ3 landed — captures the benefit, strictly beats fixed (v-rpbA25, node 1-1)
adaptive λ3 p99 **6303ms** ≈ fixed-rpb25 6300 (both −10% vs base 6973); tput identical (386.65); lossless
(hit 0.6755). Adaptive gets the SAME λ3 benefit with only ~11% capping (vs fixed's 56%) → the beneficial
interleaves ARE the selective ~11% (small turn waiting behind big doc); fixed-rpb's extra 45% capping was pure
waste. ⇒ adaptive strictly dominates fixed at λ3. DECISIVE λ5 running (fixed failed there via waste; adaptive
should preserve tput — WIN if p99<6580, neutral if ≈6580).

### ★★ DECISIVE: adaptive RPB = NEUTRAL at λ5 → the negative is AIRTIGHT (v-rpbA25, node 1-1)
| rate | base p99 | fixed-rpb25 | adaptive p99 | adaptive tput | adaptive SLO |
|------|----------|-------------|--------------|---------------|--------------|
| 3 | 6973 | 6300 (−10%) | 6303 (−10%) | 386.6 (=) | PASS |
| 5 | 6580 (tput525) | 9287 (+41%, tput480 FAIL) | **6894 (+5%)** | **520 (−1.0%)** | **PASS** |
★Adaptive REMOVES the waste (tput 520 vs fixed 480 → preserved; no SLO fail) but is NEUTRAL at λ5 (p99 6894 ≈
base 6580, +5% within ±12-20% run-noise; goodput 4.07≈4.11). Helps ONLY the sub-knee λ3 tail (already passes).
EXACTLY as §7 predicted: the λ5 p99 is set by MEDIUM/LARGE waiting turns, not the tiny turns the reserve serves
→ zero-waste reservation recovers NEUTRALITY, not a win. lossless (hit matched). ★★AIRTIGHT NEGATIVE: chunk-level
budget reservation cannot improve goodput@SLO over whole-request SRPF at ANY implementation — fixed HURTS (waste),
adaptive NEUTRAL. Scheduling axis definitively closed at both granularities. The adaptive control also proves the
fixed-RPB λ5 failure was CAUSED by the waste (removing it removes the failure), strengthening the mechanism claim.

### n=2 firming — baseline landed (node 1-1); adaptive-neutral framing pends the adaptive replicate
n=2 same-node baseline: λ3 {6973, 6393} (±4.3%), λ5 {6580, 6411} (±1.3%, TIGHT). ⇒ fixed-RPB λ5 9287 (+41%) is
unambiguously real (far outside the tight band). BUT adaptive λ5 6894 (n1) is +5-7.5% ABOVE the tight n=2 band —
so "neutral" vs "small real regression" is NOT yet settled; depends on v-rpbA25-r2 (running after v-rpb25-r2).
Prior n=3 same-node SRPF λ5 {5850,6576,7457} (±14%) WOULD contain 6894 → genuinely ambiguous. ★HONESTY: let the
adaptive replicate decide the framing — if rpbA25-r2 ≈ baseline (6400-6600) → adaptive was a high draw = neutral;
if ≈6900 → adaptive is a SMALL real λ5 regression (~+6%), reframe from "neutral" to "no goodput gain + small tail
cost" (still supports the core negative: reservation doesn't help goodput). Either way the CONCLUSION holds
(chunk-level reservation can't improve goodput); only the adaptive λ5 sub-claim's precise wording is at stake.

### n=2 firming update — fixed RPB λ3 is within coin-flip noise (the "−10% help" was a lucky draw)
fixed RPB λ3 n=2: {6300, 7646} (span ~19%, tput 386.7/386.8 identical). vs baseline λ3 {6973, 6393}. ⇒ at λ3
fixed RPB is WITHIN the coin-flip/metastable noise band — the n=1 "−10% help" (6300 vs 6973) was a lucky LOW
draw. HONEST n=2 reframe: fixed RPB has NO robust sub-knee benefit; the only robust effect is the λ5 harm
(+41%, mechanistic throughput drop). This SHARPENS the negative (RPB gives no robust benefit anywhere + hurts
the goodput rate). Must update P4 §5: soften "helps λ3 −10%" → "λ3 within coin-flip noise (no robust effect)".
Awaiting fixed r2 λ5 (confirms the robust harm) + adaptive r2 (framing decider). NOTE λ3 metastability was my
own core thesis — single-run λ3 comparisons unreliable — so this is expected, not surprising.

### n=2 firming — fixed-RPB λ5 HARM CONFIRMED robust (2026-07-15)
fixed RPB λ5 n=2: {9287, 9927} (mean 9607, BOTH fail 8s SLO), tput {480, 485} vs baseline {6580,6411}/{525,539}.
⇒ the +48% p99 / −9% throughput harm is ROBUST (not a fluke) — both fixed runs fail via reserve waste. Core
negative n=2-solid. λ3 effects are within coin-flip noise (both). REMAINING: adaptive replicate (v-rpbA25-r2,
next) decides adaptive-λ5 wording — adaptive r1 λ5 6894 vs baseline mean 6496 (+6%); if r2 ≈6500 → neutral, if
≈6900 → small real regression. Either way the CONCLUSION holds (chunk-level reservation can't improve goodput:
fixed robustly hurts, adaptive at best neutral).

### ★★ P4 FINALIZED — n=2 complete, airtight negative (2026-07-15)
Full n=2 same-node (node 1-1), p99 mean [runs]:
| rate | baseline | fixed RPB | adaptive RPB |
|------|----------|-----------|--------------|
| λ3 | 6683 [6973,6393] | 6973 [6300,7646] | 6744 [6303,7185] — all within coin-flip noise |
| λ5 | 6496 [6580,6411] PASS | **9607 [9287,9927] FAIL (+48%, tput −9%, ROBUST)** | 7205 [6894,7516] PASS (goodput=baseline req/s~4.1, p99 +11% small cost) |
★CONCLUSION (n=2, airtight): chunk-level budget reservation CANNOT improve goodput@SLO over whole-request SRPF at
ANY implementation — fixed robustly HURTS (both λ5 runs fail via ~54% reserve waste → −13% capacity), adaptive
(zero-waste) is GOODPUT-NEUTRAL (both pass at baseline req/s, no gain, small +11% p99 cost). λ3 = coin-flip noise
(the n=1 "−10% help" was a lucky draw — firming corrected it). Scheduling axis CLOSED at both granularities.
Paper fully updated (abstract/intro/§5 tables+figure n=2 error bars/§7/§8), well-formed, committed 37dd7e1f6,
W&B-logged (v-srpf-ctl5, v-rpb25-r2, v-rpbA25-r2). ⇒ P4 is now at P3's rigor (n=2 + error bars + hostile-reviewed).
RPB DIRECTION COMPLETE. Whole lossless design space mapped+bounded across 4 papers. No remaining novel lossless lever.

### ★ RPB page-alignment BUG fixed + reserve-size tradeoff MEASURED (2026-07-15)
★BUG: r=0.10 crashed REPRODUCIBLY (2/2) with "pool memory leak detected" — root cause = my RPB cap wasn't
page-aligned (6144−int(0.10·6144)=5530, 5530%64=26 → non-page-aligned chunk extent corrupts the paged KV
allocator accounting). r=0.25 (4608) & r=0.50 (3072) are %64-aligned by luck → clean, so rpb25/adaptive results
UNAFFECTED. FIX = page-align the cap (rpb_cap//page_size*page_size), commit 46ca24f72. ⇒ the paper's "transient
crash" claim is WRONG — it's a fixed alignment bug. Validated: fixed rpb10 completes 7037/7037 clean, 0 leaks.
★MEASURED r-TRADEOFF λ5 (n=1 each, + throughput monotonicity = robust): baseline 6496ms/532tput (PASS) →
r=0.10 8000ms/512 (borderline fail, +23%) → r=0.25 9607ms/482 (FAIL, +48%) → r=0.50 (pending). Harm scales
MONOTONICALLY with r (p99 up, tput down) — confirms §7's analytical argument EMPIRICALLY. Even the smallest
reserve (r=0.10) hits the SLO boundary ⇒ NO fixed r>0 avoids the λ5 harm (only zero reserve=baseline, or
adaptive zero-waste=neutral). Strengthens the negative. TODO: rpb50 completes curve; then correct P4 §5
(transient→alignment-bug) + upgrade §7 (analytical→measured) + regen figure.

### Reserve sweep COMPLETE — r=0.50 OOMs (large reserve over-admits); tradeoff established by r∈{0.10,0.25}
rpb50 (r=0.50) crashed during warmup: "RuntimeError: Prefill out of memory" — the large reserve (3072 = half the
6144 budget) lets the waiting-queue loop admit too many concurrent turns → KV/prefill OOM. Extreme reserve,
clearly on the harmful end; not pursued further (whether intrinsic over-admission or a fixable accounting gap, it's
an impractical operating point). ⇒ MEASURED r-tradeoff (fixed RPB, λ5, node 1-1): baseline 6496 PASS → r=0.10
8000ms/tput512 (borderline fail) → r=0.25 9607ms/tput482 (fail) → r=0.50 OOM-unstable. Harm (p99↑, tput↓) is
MONOTONIC in r; even the smallest reserve (r=0.10) hits the SLO boundary; large reserve is unstable. ⇒ NO fixed
r>0 helps goodput (confirms §7 analytically-argued tradeoff EMPIRICALLY). Adaptive (zero-waste) remains the only
non-harmful reservation, and it's goodput-neutral. This CLOSES the reserve-size question empirically.

---

## ★ 2026-07-15 (Pass, post-P4): SRPF-win significance FIRMED to n=6 (self-audit of my own artifacts)

**Trigger.** Re-examining whether any legitimate high-value action remained, I inventoried my own SRPF/stock
runs instead of assuming the paper's numbers. Lesson from the P4 reserve-sweep (probing my own claims finds
real things) applied to the flagship's headline.

**Discovery — the flagship UNDER-reported its own win.** The paper claimed SRPF λ5 = 3/3 (r1/r2/r3), p=0.018.
But I have **6** independent plain-SRPF λ5 passes, all verified `schedule_policy='srpf'` (ctl4/ctl5 also
`enable_rpb_chunking=False`, i.e. RPB inert — they were the RPB-off control arm):

| run | λ5 p99 (ms) | node | note |
|---|---|---|---|
| v-srpf-r1 | 5850 ✓ | 0-3 | SRPF A/B |
| v-srpf-r2 | 7457 ✓ | 0-3 | SRPF A/B |
| v-srpf-r3 | 6576 ✓ | 0-3 | SRPF A/B |
| v-srpf-full | 7822 ✓ | 0-3 | full contract sweep |
| v-srpf-ctl4 | 6580 ✓ | 1-1 | RPB-off baseline |
| v-srpf-ctl5 | 6411 ✓ | 1-1 | RPB-off baseline |

Stock (fcfs) λ5: 5/5 FAIL — v0-stock 17372 / r2 23130 / r3 23587 / r4 23772 (ondem-3) + srpfctl 22746 (0-3).
Nodes recovered by correlating run mtimes with the campaign node-lock log (`runs/campaign_*.log`).

**Result (`analysis/goodput_stats.py`, rewritten as reproducible source-of-truth):**
- **λ5 decisive: SRPF 6/6 vs stock 0/5 → Fisher one-tailed p=0.0022** (was 0.018 — ~10× stronger; reporting
  only 3 of 6 passing runs was an integrity gap, now closed).
- both rates pooled: 12/12 vs 4/11 → p=0.0013.
- λ3: 6/6 vs 4/6 → p=0.23 (coin-flip regime; distributional, not count-separable — unchanged conclusion).
- **Cross-node ROBUSTNESS (reframes the old "pooling weakness"):** SRPF passes on BOTH nodes tested (0-3 n=4,
  1-1 n=2); stock fails on BOTH (ondem-3 n=4, 0-3 n=1). Treatment gap (min stock 17.4s vs max SRPF 7.82s =
  9.5s) ≫ ±45% cross-node p99 var ⇒ not a node confound. Cross-node pooling is a robustness check the effect
  survives, not a weakness.
- Same-node 0-3 A/B: SRPF 4/4 (5.85–7.82s) vs stock 0/1 (22.7s) — clean separation, but p=0.20 with a single
  stock control alone; pooled p=0.0022 carries significance, direction unambiguous same-node.

**GPU follow-up (job 20005, node 0-3, `analysis/jobs_stock_n03.txt`).** To firm the STRICTLY-same-node result
(the one methodological caveat my own protocol flags) to significance with ZERO cross-node pooling: 3 stock
full-sweep controls on node 0-3 (which already holds the 4 SRPF passes). Expected 3 fails (stock λ5 fails 5/5,
≤1.08× spread) → 0-3 becomes **SRPF 4/4 vs stock 0/4, Fisher p=0.014**. If any stock run PASSES on 0-3, that is
itself a major node-dependence finding. Serial via campaign.sh (TIMEOUT=4:00:00 for full-sweep summary.json).
*[Integrate λ5 points into P3 §5 same-node sentence when they land.]*

**Committed** `6e1c2ecf3` (zero-GPU corrections across P3/P1/P2 + goodput_stats.py + campaign.sh + jobs file;
pushed). P4 references de-counted. This is completeness/integrity strengthening of the flagship headline, on
free-node GPU — not a new lossless direction (none remains). Charter-faithful "survive a skeptical PC".

### 2026-07-15 (waiting-turn QA while controls run): related-work + reproducibility + cross-ref audit
While job 20005 (same-node stock controls) runs, did non-GPU submission-hardening:
- **P3 §7 related work STRENGTHENED** (commit 47977bf4d) — was thin for a top-venue systems paper.
  Added scheduling prior art (Orca=iteration batching substrate; Sarathi-Serve=chunked prefill, the exact
  mechanism §4 diagnoses; **FastServe**=closest prior art, preemptive MLFQ for head-of-line) with crisp
  positioning: FastServe schedules against UNKNOWN decode length (feedback-queue approx); our lever is
  prefill heterogeneity KNOWN exactly at arrival → exact non-speculative SRPF, no preemption. Plus Marconi
  (hybrid SSM-attn prefix caching = our model class) and DistServe/Splitwise (P/D disaggregation; our §3
  co-located split already TTFT-optimal). Contribution reiterated = map + diagnosis + criterion, NOT the policy.
- **Fixed a dangling §5.1 cross-ref** in P3 §8 (P3 has no §5.1; P2's real §5.1 left intact).
- **Cross-ref audit (all 4 papers): CLEAN** — every §N ref resolves (P3 §3.4 is a valid *companion-paper* ref).
- **Reproducibility spot-check:** all 25 cited scripts/engine-paths exist; hol_sim.py/phase_boundary.py/
  goodput_stats.py run (rc=0) and reproduce their paper claims.
- P1/P2/P4 related-work confirmed appropriately focused (admission/caching/chunk axes) — no gaps.

### 2026-07-15 (cont.): same-node significance FIRMED — caveat CLOSED (p=0.029, zero pooling)
The same-node stock controls on node 0-3 (jobs 20005/20019) completed 2 of 3 runs (stopped after #2 —
p=0.029 already resolves the caveat; released the node rather than hold it ~2.5h for #3's marginal p=0.014):
- **control-1** full sweep: λ3=31922 / λ5=23540 / λ7=34390 / λ10=41041 ms (all FAIL; λ7/λ10 match v0-stock's
  34/41s cross-node). λ3=31.9s diagnosed = clean metastable queue-95 blowup, 0 retracts, healthy warmup 5.1s.
  Curve NON-monotonic (λ3 31.9 > λ5 23.5) = textbook λ3-coin-flip bad draw, corroborates metastability thesis.
- **control-2**: λ3=23514 / λ5=23464 ms (both FAIL; cancelled λ7/λ10 — had the λ5 I needed).
- ⇒ **STRICTLY same-node (node 0-3, ZERO cross-node pooling): SRPF 4/4 (5850/6576/7457/7823) vs stock 0/3
  (22746/23540/23464) → Fisher one-tailed p=0.029.** Closes the flagship's last methodological caveat with
  real same-node data. The extra controls also strengthen the POOLED λ5 to 6/6 vs 0/7 (p=0.0006, was 0/5
  p=0.0022) and both-rates-pooled 12/12 vs 4/15 (p=0.0001). λ3 stock pooled → 4/8 (the 2 controls drew into
  the metastable tail), p=0.07, still coin-flip/distributional (decisive win stays λ5).
- Integrated across P3/P1/P2 + goodput_stats.py (reproducible source of truth). Commit 764008f36, pushed.
- OPS lesson: `pkill/pgrep -f <pattern>` where the pattern appears in the current command line SELF-MATCHES
  and kills the running shell (exit 144). Use a regex bracket trick (`campaign[.]sh`) or kill by explicit PID.

### 2026-07-16: reproducibility audit COMPLETE across all 4 papers' core proofs
Verified (rc=0, each reproduces its paper claim) the load-bearing analysis script of every paper:
P1 hol_sim (HOL diagnosis); P2 doc_reuse (single-pass LRU==LFU==Belady==0 avoidable = the mirage) +
phase_boundary (generality); P3 mamba_pressure (attn-KV→1.0 while mamba max 0.77 = never binding) +
bound (C_eff 15k→~4.2 derived, consistent w/ measured ~4.7 raw ceiling) + goodput_stats (Fisher);
P4 rpb_waste (rpb25 chunk mean 5327 vs base 6111 → ~54% reserve wasted). Artifact is verified-runnable.

### 2026-07-16: ★CORRECTION (fresh hostile-PC review caught a node-attribution error I introduced)
Re-reviewed the flagship (materially changed since the last review: n=6, same-node firming, §7). The reviewer
(general-purpose subagent, artifact-backed) reproduced all stats EXACTLY but flagged one real error:
- **The strictly-same-node p=0.029 relied on v-srpf-full being on node 0-3 — but that was MTIME-INFERRED, not
  logged.** I checked the slurm `sacct` NodeList (ground truth): **v-srpf-full (job 19901) ran on node 1-2, NOT
  0-3.** My earlier mtime-correlation was wrong. ⇒ strictly same-node (0-3) is **SRPF 3/3 (r1/r2/r3) vs stock
  0/3 → Fisher p=0.05** (marginal, at threshold), NOT 4/4 vs 0/3 p=0.029.
- Silver lining: SRPF passes on **THREE** nodes (0-3 n=3, 1-2 n=1, 1-1 n=2), not two → cross-node robustness is
  STRONGER. Reframed §5: decisive = pooled 6/6 vs 0/7 p=0.0006 (node-independent) + 3-node robustness; same-node
  p=0.05 = confound-free corroboration, not sole anchor. All node data now from sacct (stated in-paper).
- Reviewer also verified a TPOT overclaim: §8 "p99 TPOT within noise" was WRONG — SRPF TPOT is +25% (λ3 4.3s vs
  stock 3.5s). Corrected to "trades small per-token slowdown for large E2E win (322 vs 438s)". Fixed.
- Corrected everywhere (P3 §4/§5/caption/appendix/repro, P1 cross-ref, goodput_stats.py) + P2 caption. Commit af0f01a96.
- ★OPS LESSON: node identity = `sacct -j <jobid> -o NodeList` (ground truth), NEVER mtime-correlation with
  campaign logs (I got v-srpf-full wrong that way). Reviewer verdict was weak-accept; this was its #1 fix.

### 2026-07-16 (cont.): sacct-verified the COMPANIONS' same-node claims too (error class did NOT propagate)
After the P3 v-srpf-full node error, checked whether the same mtime-inference bug affected P1/P4 same-node A/Bs:
- **P4 (RPB) same-node A/B: ALL on node 1-1** (sacct) — baseline ctl4(19915)/ctl5(19949) + v-rpb25(19931)/
  v-rpb10(19942,19971,19981)/v-rpbA25(19947)/v-rpb25-r2(19958)/v-rpbA25-r2(19962)/v-rpb50(19980,19998). CORRECT.
- **P1 (CCA) comparison: ALL on ondem-3** (sacct) — v3-cca(19767), v4-cca-lb(19769/r2 19788/r3 19792),
  v5-cca-lb-w90(19794/r2 19796), and the stock baseline (v0-stock family). CORRECT (same-node vs stock).
⇒ The v-srpf-full error was ISOLATED to P3 (launched via eval-on-pool onto an unpredictable node; the
companions' A/Bs used campaign.sh `-w` node-pinning → genuinely same-node). Node attribution now sound across
ALL 4 papers, all verified against sacct ground truth. No companion corrections needed.

### 2026-07-16 (cont.): fresh hostile-PC review of the COMPANIONS (P1/P2/P4) — secondary-number fixes
Applied the P3-re-review thoroughness to the 3 companions (their last review predated this session). Reviewer
(artifact-backed) found NO reject-level errors; sacct-verified all node claims clean; core numbers reproduce.
Fixed the real secondary-number issues it surfaced:
- **P4:** "all variants lossless (Δ≤0.002)" was FACTUALLY FALSE (v-rpb25-r2 hit-rate +0.02–0.035 vs baseline =
  scrape noise) → corrected (3 places) to "lossless BY CONSTRUCTION (only reslices computation), hit-rate within
  scrape noise Δ≤~0.04". Clarified 7037-of-7163 turns.
- **P1:** REGENERATED Figure 2 from committed v0-stock/server.log (stale n=17,384/21.7% → n=35,771, sat≥0.90
  20.6%/idle≤0.05 12.8%/39 at 1.0; redrew all 10 bars, verified on baseline). "pool capped ~0.54" wrong → mean
  ~0.38 (peak still 1.0). Stale "n=5" λ5 → n=7. §5 NCCL lesson reframed as diagnosed+fixed (trace not preserved),
  not fully-traced.
- **P2:** softened "sharp, corpus-general" → "general in form" + noted only 2 of 4 real corpora carry reuse.
⇒ Both hostile reviews (P3 + companions) now fully addressed. Commit 1ca13dba6. ★All 4 papers artifact-backed
reviewed, node-verified via sacct, secondary numbers corrected. Body of work integrity-audited end to end.

### 2026-07-16: Direction 5 (reuse-aware prefetch / transfer-compute overlap) — FAST-SCREENED NEGATIVE
Per program ("always a next direction; don't sit holding"), opened a program-listed ambitious target:
reuse-aware PREFETCH / transfer-compute overlap to hide the L2→L1 load-back of reused conversation prefixes
under load. Free fast-screen from the engine's own Prometheus load-back histogram (analysis/load_back_stats.py,
runs/v0-stock metrics_r*.txt):
- **L2→L1 load-back is NON-BINDING and SCALES:** mean 1.54/1.46/1.35/1.31 ms at λ=3/5/7/10; ≥99.4% <10 ms;
  ALL <30 ms — even as load-back volume grows 3.7× (25K→92.6K load-backs, 0.37→1.36 B tokens). <0.2% of the
  queueing-dominated TTFT (17–24 s); never on the critical path.
- ⇒ **NEGATIVE**: reuse-aware prefetch/overlap has no goodput headroom (nothing to hide). By extension this
  bounds the whole MOVEMENT-HIDING family the program lists — transfer/compute overlap, layout/paging (movement
  cost), conversation co-residency (reload cost): movement is already cheap. Confirms P3's transfer-non-binding
  row, now with cross-rate load-back metrics + the family-level bound. Firmed P3 §3 map row.
- Engine-verified the path: init_load_back→load_back→ongoing_load_back, async via HiCacheController; the
  reused-prefix load is on the request's pre-prefill critical path but each load is <30 ms so it doesn't matter.
- Reuse-/prefix-aware ROUTING = out of scope (single-node TP=8 eval; routing is a multi-replica lever).
This is the rigorous negative behind my "movement axis exhausted" claim, now backed by hard load-back data.

### 2026-07-16: Direction 6 (chunk-level PREEMPTION) — completes P4's axis (complementary primitive)
Per program ("always a next direction"), opened the untested complementary chunk-level primitive: PREEMPTION
(pause a big cold doc's chunked prefill for a waiting short turn, resume later) vs the deployed NON-preemptive
SRPF. Feasibility: TRACTABLE — sglang's stash_chunked_request → maybe_cache_unfinished_req(chunked=True) already
caches partial-prefill KV, so pause/resume is lossless by re-matching the cached partial prefix.
Fast-screen (free, hol_sim.py): preemptive SRPF tail 4031ms vs non-preemptive 6205ms @λ3 (−35%) — BUT both pass
the 8s SLO, so NO goodput change sub-knee; sim λ3-only-reliable (can't assess λ7). Analysis: at the
goodput-SETTING rate the p99 tail is the big cold docs' OWN prefill (compute-bound, P3), which preemption FURTHER
delays — the IDENTICAL failure mode as RPB reservation (both disadvantage the tail-setting large doc; RPB
GPU-confirmed to hurt λ5), plus preemption pays re-prefill/starvation costs reservation avoids.
⇒ NEITHER chunk-level primitive (reservation, preemption) beats whole-request non-preemptive SRPF's goodput.
Decision: did NOT build a full preemptible-prefill GPU test — it's textbook-SRPT-adjacent (disqualified even if
positive, per P4 related-work) AND predicted-negative for the same reason as the GPU-confirmed RPB (building a
2nd confirmation of one insight = incremental; program: "prize contribution over version count"). Instead
COMPLETED P4's axis-closure analytically for BOTH primitives (commit e98fc99e1), honestly scoped (reservation
GPU-confirmed; preemption argued via sim + shared failure mode + costs; direct GPU test = future work).
★Two directions opened this session (D5 reuse-prefetch, D6 preemption) — both fast-screened NEGATIVE, each
firming an existing paper's map (D5→P3 movement-family bound; D6→P4 both-primitives closure). Design space
further bounded; no non-textbook lossless positive found (consistent with the bounded-impossibility thesis).

### 2026-07-16: P5 (5th paper) — kv-movement-nonlever (movement-axis deep-dive negative)
Per program ("series of deep papers, growing list"), turned D5's load-back finding into a full companion paper —
the movement/transfer axis was the one P3 map-axis lacking a deep-dive (caching→P2, admission→P1,
chunk-sched→P4, but transfer only a P3 row). NEW paper `submissions/kv-movement-nonlever/paper.html`:
- CLAIM: the L2→L1 movement tier is NOT a goodput lever. Load-back volume is large + load-GROWING (0.37→1.36B
  tokens λ3→10) but each op <30ms (≥99.4% <10ms; mean DROPS 1.54→1.31ms under load) ⇒ ≈10M tok/s HBM↔DRAM
  (page_first_direct+direct IO); <0.2% of the queueing-dominated TTFT; on the pre-prefill critical path
  (init_load_back→loading_check, code-verified) yet negligible.
- Bounds the whole movement-optimization FAMILY (prefetch, transfer/compute overlap, layout/paging, conversation
  co-residency) — none has goodput headroom (all target a cost that's already ~ms).
- ★SLOW-TIER BOUNDARY criterion (generality, analog of P2's phase boundary): movement matters iff a tier's
  transfer C_move=(reused-prefix-tokens/BW) is a material TTFT fraction — fast 2-tier HBM+DRAM = non-binding;
  the movement lever lives BELOW L2 (L3/disk). Locates where movement optimization would pay.
- Mirrors P2 on the movement axis: retention AND movement both non-levers; lever = scheduling cold prefills.
- Evidence = existing load-back Prometheus histogram (load_back_stats.py); NO new GPU. Measurement (not ablation):
  the histogram bounds achievable gain above by <0.2%, so no impl can beat it (honestly disclosed).
- Integrity pass: reframed disk-tier remarks as PREDICTIONS from my criterion (no external/sibling results).
Registered in INDEX (now 5 papers; every P3 map-axis has a companion). Commits a3370e588/8e702f63d.

### 2026-07-16: re-verified CP (the one compute-side lever) unavailability — confirms P2 §5.1 (no change needed)
Took the program's "always a next direction" seriously by re-examining my highest-upside assumption with fresh
eyes + rigorous code-check: is context-parallel (CP) prefill genuinely unavailable for the Qwen3.5 hybrid model,
or did I dismiss the biggest potential win (curve shift past goodput 4.0) too shallowly? VERIFIED unavailable:
- `qwen3_5.py` forward has ZERO CP wiring (grep context_parallel/cp_size/cp_rank/prefill_cp = empty).
- CP paths are DeepSeek-family-gated: `is_deepseek_dsa()` (server_args.py:3718 "DeepSeek 3.2/GLM 5") for the
  DSA CP auto-derivation, and `use_mla_backend()` (3844) for the MLA CP path. Qwen3.5 = fa3 + GatedDeltaNet
  (15 linear-attn refs), neither DSA nor MLA → no CP path.
- Mamba/GDN CP exists only as a Megatron *debug* reference, not the live path.
⇒ CP is code-verifiably unavailable for this model/version (confirms P2 §5.1, which was already accurate + cites
is_deepseek_dsa). Full hybrid-model prefill CP (ring attention + sequence-parallel GDN chunked state scan for a
122B model) is a major model-parallelism project = honest future work (P2 §5.1), not a tractable in-scope
mechanism; a buggy partial attempt would violate losslessness. The compute-side lever is genuinely out of reach;
the bounded-impossibility (goodput = cold-prefill-compute wall, no in-engine lossless lever) is airtight.
No paper change (P2 §5.1 accurate). This firms the impossibility's most important axis by re-verification.

### 2026-07-16: fresh ~14-angle brainstorm (took program's push seriously) — bound confirmed + P4 completed to 3 policies
Re-examined the space from scratch/creatively for a genuinely-new lever: decode-preemption-for-prefill (bounded:
99% prefill at saturation already, retraction wasteful), multiple-concurrent-big-doc-prefill (bounded: budget
compute-fixed, splitting delays all tail docs), speculative-decode-for-slots (off-scope/lossy, model-level),
lossless L2-KV compression (no headroom: L2 non-binding per mirage), first-token/prefill-tail overlap (needs
full prefill = lossy), cross-conversation doc-sharing (97% singletons per P2), multi-concurrent chunked_req,
non-textbook scheduling signals (all subsumed by SRPF's remaining-work signal or textbook EDF-neg). EVERY angle
bounds to the compute wall / an already-mapped non-lever / out-of-scope-lossy. Bound is airtight.
★Genuine outcome: "concurrent big-doc prefill" is the 4th natural chunk-budget-allocation policy → added to P4's
chunk-scheduling closure (now covers reservation[GPU]+preemption[argued]+concurrent-split[argued], all bounded
by the same tail-setting-big-doc-delay failure mode). Completes the chunk-scheduling axis over ALL budget
policies. Commit follows. No new substantial direction; no GPU experiment adds non-redundant value.

### 2026-07-16: ★FIRMED the flagship's same-node anchor to a BALANCED 4-vs-4 (p=0.05 → 0.014)
The one residual weakness in P3's central claim was that the confound-free *strictly same-node* A/B (node 0-3,
the one node holding both arms) was only **marginal**: SRPF 3/3 vs stock 0/3, Fisher p=0.05 — so the decisive
significance leaned on the *cross-node-pooled* p=0.0006, in mild tension with my own same-node-A/B mandate. With
GPU free (sibling released 0-3) I ran ONE paired SRPF + ONE paired stock full-contract sweep on node 0-3
(jobs 20057/20130; node = `sacct -j <id> -o NodeList`, NOT mtime). Both plain-policy, all contract flags frozen
(`schedule_policy` srpf vs fcfs, `enable_rpb_chunking=False`, mem-frac 0.85, page 64, tp 8, chunk 6144), 7037
completions each → clean, lossless (scheduling reorder only).
- **SRPF firming λ5 = 5.55 s PASS** (4th SRPF λ5 on 0-3); goodput=4.22, hit 0.69 (cache untouched).
- **Stock firming λ5 = 23.25 s FAIL** (4th stock λ5 on 0-3).
- ⇒ **strictly same-node, BALANCED 4-vs-4: SRPF 4/4 vs stock 0/4, Fisher one-tailed p=0.014** — the
  confound-free anchor now *carries* the claim by itself (no cross-node pooling needed).
- Strengthened (real, sacct-verified): λ5 pooled **7/7 vs 0/8, p=0.0002** (was 6/6 vs 0/7, p=0.0006);
  both rates pooled **13/14 vs 4/17, p=0.0001**.
- **HONEST λ3 update:** the SRPF firming's λ3 drew **9.14 s (a FAIL)** — a metastable coin-flip draw — so SRPF
  λ3 is now **6/7** (was 6/6), p=0.12. This does NOT touch goodput (which is λ5-set) and *reinforces* the
  paper's own framing: "λ3 is variance-dominated, not the decisive rate; report λ5."
Integrated: `analysis/goodput_stats.py` (source of truth) + P3 (§5, table caption, limitations, repro) + P1 + P2;
every stale 6/6·0/7·p=0.0006·3/3-vs-0/3 replaced. INDEX registry entry. Commit 90bbdd8f1, pushed. SRPF firming
logged to W&B (v-srpf-samenode-1); stock replicate logs when job 20130's summary.json completes.
**Rigor/completeness firming — NOT a new lossless direction (none remains; the design space stays airtight-bounded).**

### 2026-07-16: fresh codebase inventory (Explore) + NEW trace-negative → P3 9th axis (KV-memory management)
Charter demands I keep hunting for a new direction, not guard a finished body. Ran a fresh, thorough Explore
inventory of ALL stock sglang KV/scheduler primitives and cross-referenced each against my 5-paper bound:
- **Most confirm the bound:** routing-key / in-batch-prefix-dedup / session-radix-cache = prefix-sharing
  primitives, all inert on my 97%-singleton corpus (P2); dynamic-chunking = PP>1-only (I run PP=1), inert;
  async transfer-overlap = the movement-hiding P5 already bounds; cost-aware/gdsf/2q eviction = P2 (LRU=Belady).
- **★INTEGRITY CATCH:** the inventory flagged "WSAC/PGAC" admission gates as stock — but the code comment
  (`scheduler.py:975`) reveals they are a **sibling's** (`base_free`) additions to the shared `extern/sglang`
  fork, NOT stock. Testing them would violate my independence mandate → **DROPPED**. Verified my own clone
  (what my eval runs via PYTHONPATH) is CLEAN of WSAC/PGAC, and that **SRPF is my own** implementation
  (commit 307ddc764 on evolve/floyd, textbook algo on the pluggable interface, cited as textbook — not stock,
  not sibling). Provenance of the flagship is solid.
- **★GENUINE NEW FINDING (free, no GPU):** the inventory raised a question I had NOT tested — could *proactive*
  KV-memory management (decode **retraction** / pool-headroom **reservation**) admit big cold prefills faster
  and cut the p99 tail? This is the OPPOSITE of P1's admission-*deferral* negative. Answered from the stock
  trace (`analysis/mem_admission_stats.py`, n=2 over v0-stock + v-stock-samenode-1, different nodes/days):
  **0 decode retractions and 0 memory-blocked prefill admissions** (#new-seq≥1 at all 34k/36k prefill steps),
  while the KV pool reaches near-full (usage p99 0.98–0.99). Pressure is absorbed **losslessly** by evicting
  clean cached prefixes (LRU=Belady=0 avoidable), never by retracting decode or stalling admission → **no
  memory-wait on the critical path to reclaim; the tail is compute + head-of-line.** ⇒ the proactive-KV-memory-
  management sub-axis is a trace-backed **non-lever**. Added as P3's **9th map axis** (bounded here for the
  first time), intro axis count eight→nine, contributions bullet updated. Commit 1462a6cbc, pushed.
**Net:** the fresh inventory + trace diagnosis independently RE-CONFIRM the design-space bound (every stock/own
primitive maps onto an existing non-lever) and add one genuinely-new trace-negative axis. No un-mapped novel
lossless positive lever exists on this frozen eval. Continuing to hunt (prior-art scan next).

### 2026-07-16: attacked the caching-under-load crack → P2 fortified (live mirage), no lever
Charter demands I keep probing, not re-declare bounded. This cycle I interrogated a specific crack I had NOT
closed rigorously: P2's caching-mirage was established **offline** (trace replay at full capacity, LRU=Belady),
but the **live** hit rate *declines under load*. Since the request mix is identical at every rate (same 1553
convs / 7037 reqs replayed), any λ3→λ10 hit decline is load-induced, not a mix artifact — potentially the
charter's "gap between resident and about-to-be-reused KV widens under concurrency," i.e. a possible reuse-aware
eviction/admission LEVER.
- **Quantified (`analysis/hit_vs_load.py`):** the decline is real, small, monotonic, and CONSISTENT across every
  full sweep and BOTH scheduling policies: −1.86pp (v0-stock 0.678→0.659), −1.85 (stock-samenode), −1.87
  (srpf-full 0.675→0.656), −3.30 (srpf-samenode 0.705→0.672).
- **Verdict = NOT a lever (bounded):** the decline is (a) policy-invariant — stock FCFS and SRPF show the same
  slope (so not scheduling-related), and (b) eviction-policy-invariant by P2 §3.1 (LRU=LFU=Belady; the reuse
  structure is binary — singleton or short-stack-distance — so recency already tracks future reuse, no policy
  can keep more about-to-be-reused KV). It reflects the **frozen KV budget** shared with the growing
  running-request working set as concurrency rises — capacity, not a suboptimal cache decision. A reuse-aware
  policy can't recover it (the space is taken by un-evictable RUNNING KV, not mis-evicted reuse-prefixes).
- **Integrity:** the per-rate metrics_r*.txt are end-of-phase idle snapshots (running-reqs=0), so I could NOT
  cleanly isolate the in-phase concurrency mechanism → I did NOT state an unverified attribution; I stated only
  what's verified (policy-invariance + capacity framing).
- **Output:** fortified P2 with a verified "the mirage holds live, under load" paragraph (§3.1) — closing the #1
  hostile-PC attack on an offline caching analysis (does it hold in the live system?). Commit 2c75f81b9, pushed.
**Net:** another genuine crack attacked and found bounded; a real live-validity fortification of P2 (not gilding —
it closes a distinct offline-vs-live gap). No un-mapped lossless lever. Design space stays bounded.

### 2026-07-16: GPU eviction A/B — scan-resistant SLRU CRATERS (completes the caching-under-load thread)
Last cycle I quantified a load-induced hit decline and argued it's not a lever. This cycle I TESTED the
sharpest lever hypothesis on GPU: the workload is 97% singletons = a massive scan; LRU is classically
scan-susceptible; so under load-shrunk effective capacity a SCAN-RESISTANT policy (SLRU: protect hit_count≥2,
evict singletons first) MIGHT protect about-to-be-reused KV that LRU evicts → recover the decline → a lever.
- **GPU A/B (job 20180, node 0-3, FCFS, differs only in eviction; vs same-node stock LRU v-stock-samenode-1):**
  SLRU **CRATERS** — hit −18.7/−24.1/−25.5/−26.8 pp (λ3/5/7/10: 0.485/0.421/0.402/0.386 vs LRU
  0.673/0.662/0.657/0.654), **goodput 0** (p99 24–41 s, fails every rate). SLRU's own load-decline (−9.9pp)
  is also steeper than LRU's (−1.85pp) → it degrades WORSE under load. Provenance-checked (SLRU stock in base
  a334877e5), verified active on the UnifiedRadixCache path (not dead), lossless (eviction ≠ outputs).
- **Mechanism (generalizes, the real contribution):** in multiturn serving a turn-0 doc has hit_count=1 when
  its OWN conversation's next turn reuses it — **indistinguishable from a never-reused singleton by access
  count**. SLRU (and any frequency signal) puts both in probationary and evicts them first → discards
  about-to-be-reused turn-0 KV BEFORE its first reuse. **Recency (LRU) is uniquely right** (a recently-seen
  turn-0 doc is precisely the one about to be reused). Design rule: for multiturn KV reuse, evict by recency,
  not frequency.
- **Value:** GPU-confirms the P2 §3.1 LRU=Belady PROOF from the opposite direction — the "smarter" scan-resistant
  policy is strictly & severely worse — my first COMPLETED live alternative-eviction sweep. Upgraded P2 §3.3 +
  abstract (from "partial LFU run" → completed same-node A/B) and P3's residency/eviction map row. This is a
  genuine new GPU result strengthening the caching-mirage claim, not gilding. W&B v-evict-slru; commit 5227d49e2.
  GPU economy: cancelled the redundant LRU control (v-stock-samenode-1 already IS the same-node LRU baseline).
**Net:** hypothesis tested on GPU → decisive NEGATIVE (SLRU craters) with a generalizing mechanism. Design space
stays bounded; eviction axis now GPU-confirmed live (not just offline). Keep probing.

### 2026-07-16: completed LFU sweep → 3-policy GPU eviction characterization (recency>frequency is GENERAL + graded)
Followed the SLRU crater to test the GENERALITY of the recency>frequency insight: does LFU (soft graded
frequency, priority (hit_count,last_access)) also degrade, or stay ≈LRU (as offline predicted)? Completed the
LFU full sweep (job 20207, node 1-1).
- **LFU DEGRADES** — hit −10.5/−19.9/−23.2/−24.6pp (λ3/5/7/10: 0.568/0.463/0.426/0.408 vs LRU
  0.673/0.662/0.657/0.654), goodput 0. Less severe than SLRU (−18.7..−26.8pp) but SAME direction and deepening.
- **Complete 3-policy GPU eviction characterization:** LRU (recency) optimal; LFU (graded freq) −10..−25pp;
  SLRU (hard-segment freq) −19..−27pp. Severity scales by how hard the policy protects reused-once nodes:
  **SLRU > LFU > LRU=0. Access frequency is the wrong signal, monotonically.** Both frequency policies goodput 0.
- **Mechanism (general):** in multiturn serving a turn-0 doc is hit_count=1 at its first reuse = indistinguishable
  from a never-reused singleton by access count → any frequency policy evicts it before reuse (SLRU hard, LFU soft)
  → recency (LRU) uniquely right. Design rule: for multiturn KV reuse, evict by recency not frequency.
- **Integrity refinement:** this sharpens P2's OFFLINE "LFU≈LRU" — that held for the cross-pass avoidable-recompute
  metric but MISSED the live concurrency effect (under load-shrunk capacity a frequency policy protects stale
  hit≥2 docs — a conv's doc after its single reuse, never needed again — crowding out fresh turn-0 KV). Added the
  "on avoidable-recompute" qualifier to the abstract; §3.3 now presents the live 3-policy sweep. (LFU cross-node
  vs the 0-3 LRU baseline, but −20..−25pp under load dwarfs ~3pp node variance.)
- **Integrated:** P2 §3.3 (partial→completed 3-policy) + abstract qualifier + P3 residency/eviction map row.
  W&B v-evict-lfu; commit ffdf14e45. Lossless, on-contract, provenance-clean (LFU stock in base).
**Net:** the recency>frequency insight is now GPU-confirmed GENERAL across access-count policies (LFU + SLRU,
graded) — a stronger, more defensible eviction result. Design space stays bounded; caching axis fully
GPU-characterized (3 policies).

### 2026-07-16: GPU-corroborated the LAST analysis-only map axis (prefill/decode split) via mixed_chunk
8 of 9 map axes were GPU/measurement-confirmed; prefill/decode-split was the lone analysis-only axis
(pd_schedule.py: 99% prefill at saturation + prefill-first code). Tested the mechanism that shifts toward
decode: --enable-mixed-chunk (mixes decode running_bs INTO prefill batches; stock base a334877e5, live at
scheduler.py:2918, not auto-disabled with fa3/no-dllm — verified active enable_mixed_chunk=True).
- **Result (job 20222, node 1-1):** mixed_chunk DEGRADES λ3 — median TTFT 2017ms vs stock 577-1056ms (~2-3.5×),
  throughput 2.18 vs 2.87-3.02 req/s (−28%) — then **CRASHES** (8× "pool memory leak detected" KV-accounting
  exceptions, during λ3, 3838/7037 completed; same crash class as the RPB rpb10 run). goodput 0.
- **Handling:** per my RPB-crash precedent, DISCARDED as a clean data point (crashed mid-sweep, no valid full
  curve, hit=0 is a crash artifact — NOT W&B-logged as a curve point). But the pre-crash median degradation is
  robust (median over 3838 reqs) → a genuine directional observation.
- **Interpretation:** shifting toward decode (mixed_chunk) is both HARMFUL (p50 2-3.5×, tput −28%) and UNSTABLE
  (pool-leak crash) for this workload → empirically corroborates the analytically-backed "prefill-first is
  TTFT-optimal" row. Mechanism: at saturation prefill already fills the budget; injecting decode steals prefill
  compute → slower prefill → worse TTFT (+ triggers a KV-pool accounting bug). Added a GPU-corroboration clause
  to P3's prefill/decode-split row (kept modest + honest about the crash). Commit 553818ad2.
**Net:** all 9 map axes now GPU/measurement-confirmed or analysis+GPU-corroborated. Design space fully
empirically characterized; every KV/serving axis a non-lever, the one lever (SRPF) textbook. Keep probing.

### 2026-07-16: fresh hostile-PC review of the materially-changed flagship (verified vs artifacts) — CLEAN
After the recent GPU additions (SLRU/LFU 3-policy eviction, mixed_chunk, memory-mgmt 9th axis), P3 had changed
substantially since its last adversarial review, so I ran a fresh hostile-PC (SOSP/OSDI) review of P3 + P2 §3.3,
instructing it to VERIFY every quantitative claim against the raw run artifacts (re-running goodput_stats.py,
mem_admission_stats.py, reading curve.csv/bench_*.json). ★VERDICT: **NO reject-level issues** — all claims match
the data: mem-admission stats byte-accurate (0 retractions/0 mem-blocked ×4 sweeps), SRPF p=0.014/0.0002 exact,
eviction deltas (LFU −10.5..−24.6pp, SLRU −18.7..−26.8pp) accurate, mixed_chunk magnitudes (~2-3.5×, −28%)
accurate, cross-paper consistent (nine axes, no stale p=0.05/0.0006/6/6). Fixed the only 2 minor items it flagged:
(1) P2 §3.3 LFU endpoint −25→−24.6pp (exact); (2) P3 made the mixed_chunk partial-trace explicit (robust median
over 3838 completed reqs; crashed run = directional corroboration, not a clean point). Commit f5d30cac5.
**Net:** the 5-paper body + 9-axis fully-empirical map is adversarially-verified integrity-clean and
submission-ready. No new lossless lever exists (space exhaustively bounded); no thin/salami 6th paper warranted
(an eviction paper would salami-slice P2). Keep probing; supervisor decides retirement.

### 2026-07-16: ★write_back backup policy raises hit+throughput but NOT goodput — integrity correction + sharpened thesis
Fast-screened the last distinct untested stock mechanism (L1→L2 backup policy, --hicache-write-policy) and
found it genuinely worth a run — it tests whether the multiturn-eager-retention principle extends from eviction
to backup. Result was a GENUINE SURPRISE that required an integrity correction to P2.
- **GPU sweep (v-writeback, node 1-1) + same-node control (v-wt-ctl-n11, node 1-1, stock write_through, FCFS,
  differ ONLY in write policy):** write_back raises hit **+4.9/+6.0/+6.5/+6.6pp** (λ3/5/7/10; 0.70→0.74 … 0.66→0.73)
  and peak throughput **+7.4%** (λ10 4.74→5.09 req/s). Mechanism: write_back (threshold=2) does NOT back up the
  97% singletons to L2 → less L2 pollution → reused prefixes retained → higher hit → less prefill → higher tput.
  Lossless (backup TIMING, cache serves exact KV). **BUT goodput@SLO UNMOVED — p99 fails every rate (25-43s),
  exactly like write_through.**
- **Rigor:** the cross-node number (+7pp/+9.7% vs 0-3 baselines) included ~1pp node inflation; the same-node A/B
  (both node 1-1) gives the clean +4.9-6.6pp / +7.4%. Ran the same-node control specifically to avoid overclaiming.
- **INTEGRITY CORRECTION to P2:** §3.1's "measured hit rate sits at the online optimum" was too strong — it's
  optimal for the EVICTION order (holding backup fixed); the backup policy is a separate lever that DOES raise
  hit. Corrected §3.1 + added §3.4 "the one cache knob that raises hit — and still misses goodput."
- **This STRENGTHENS the headline thesis:** even the one knob that provably improves hit (+6pp) AND the throughput
  ceiling (+7.4%) leaves goodput untouched → goodput is DECOUPLED from hit-rate, set by the cold-doc SLO tail.
  Sharpest possible statement of the caching mirage (you CAN improve hit; it's a mirage for goodput).
- **write_back is a STOCK CONFIG** (charter explicitly excludes config flags, naming "write_back") → reported as
  EVIDENCE, not a claimed contribution. Integrated P2 §3.1+§3.4, P3 residency row. W&B: v-writeback,
  v-wt-ctl-n11 (both config). Commits 3ca47e6fb→4f5ae12d6.
**Net:** a genuine new GPU finding that corrected an overstated claim (integrity) AND sharpened the central
thesis. Fast-screen-then-run discipline paid off (I nearly skipped it as "predictable crater"; it was the opposite).

### 2026-07-16 (cont.): SRPF+write_back composition — analytically bounded + GPU-deferred (bad nodes/congestion)
Attempted the SRPF+write_back composition (does combining the scheduling lever + write_back's raised throughput
ceiling push goodput past λ5?). Two launches hit BAD NODES during heavy cluster congestion: job 20264 (node 1-0)
host DRAM-OOM at init (SIGKILL/-9); job 20267 (node -2) dead CUDA fabric (Error 802 "no accelerator" — node -2
is a known-bad-fabric node). Cluster was 16 alloc / 2 idle (both idle nodes bad), siblings running many jobs.
★DECISION: deferred the GPU run — it is ANALYTICALLY BOUNDED and already subsumed by this cycle's compute-ceiling
correction: write_back raises the throughput ceiling to only ~5.1 req/s, but λ7 offers 7, so λ7 stays overloaded
(5.1 < 7) → p99 fails → goodput stays ~4.0 (λ5). i.e. raising the throughput ceiling (write_back) cannot move
goodput to the next grid rate because even the raised ceiling is below it — exactly what P3's corrected
compute-ceiling row now states ("goodput sits below even the raised ceiling; raising the throughput ceiling does
not move it"). So the composition needs no separate GPU confirmation; fighting bad nodes for a confirmatory run
during congestion isn't warranted. ★OPS: node -2 = dead CUDA fabric (avoid); node 1-0 = DRAM-OOM under
congestion (transient, needs MemAvail headroom); pick a certified/known-good free node, fail-fast on bad ones.
**Net cycle deliverable = the compute-ceiling INTEGRITY correction (necessitated by write_back): the ~4.7 ceiling
is a prefill-work ceiling, raisable losslessly, but goodput is tail-bound below it.** Body current + synced.

### 2026-07-16 (cont.): ★INTEGRITY — scoped the chunk-scheduling negative to RESERVATION (self-audit)
Self-audit of P4's scope: I GPU-tested chunk-level budget RESERVATION (fixed + adaptive RPB) and found it bounded,
but several claims generalized beyond it — "this granularity does not yield a goodput lever" (§6), "close the axis"
(§5), P3 intro "finer granularity backfires." That OVERCLAIMS: reservation is only ONE chunk-budget primitive. A
distinct family — queue-pressure-adaptive chunk SIZING (shrink the in-flight big-doc chunk to interleave waiting
turns WITHOUT holding budget in reserve, so no reserve-waste) — acts at the same granularity and I did NOT evaluate
it; my reserve-waste argument does not apply to it. Fixed (integrity, my own recognition that reservation ≠ all
chunk-scheduling): scoped P4 §5/§6 to the "reservation sub-axis"; added a P4 §7 paragraph explicitly stating the
negative does NOT cover non-reservation chunk sizing (open question; any win must come from tail relief below the
raw-throughput ceiling, since the knee is a compute-saturation wall); scoped P3 intro to "finer-granularity
reservation backfires; chunk sizing left open." Commit 7bbaead6a.
★DECISION not to pursue chunk-sizing myself: a sibling's queue-pressure-chunking result leaked into my recalled
memory index, so building/testing that direction now would violate independence (chasing a known-successful
sibling direction + risking duplication). The clean stance = honestly scope my negative to what I independently
tested + leave chunk-sizing as future work. This makes P4's negative airtight-honest rather than overclaimed.
**Net cycle: two integrity self-corrections (compute-ceiling from write_back; chunk-scheduling scope) — the body
is materially more honest; both from probing my OWN claims. No new lossless lever pursued (independence-clean).**

### 2026-07-16 (cont.): consistency check of the write_back/integrity edits + elevated the decoupling evidence to P3 abstract
The write_back finding + 2 integrity corrections (compute-ceiling, chunk-scheduling scope) were material edits made
AFTER my last hostile-PC review, so I ran a FOCUSED cross-paper consistency check: (1) write_back figures coherent
— P2 §3.4 +4.9–6.6pp / P3 +5–7pp, both +7.4% / 4.74→5.09; (2) chunk-scheduling scoped to "reservation" everywhere,
no residual broad "this granularity does not yield / all chunk-level" overclaim; (3) no stale/contradictory numbers.
PASSED — body coherent after the recent edits. Then elevated the strongest single piece of evidence for P3's thesis
(a KV-cache config, write_back, that improves hit +5–7pp and the throughput ceiling +7% yet leaves goodput
unchanged → goodput decoupled from hit, set by the tail) from the §3 map rows into the P3 ABSTRACT, where a PC
reads first (framed as evidence — write_back is a stock config, not a contribution). Commit 6225b6b88.
**Net: body verified coherent + the flagship's central-thesis evidence sharpened at first read. No new lossless
lever (bounded); chunk-sizing remains an independence-clean deferral (sibling QPAC leaked). Research substantively
complete; careful maintenance, still probing each cycle.**

## ★ CAMPAIGN STATE (2026-07-16, CURRENT — supersedes the stale "3 papers" summary above) — COMPLETE on the frozen eval
**Deliverable: 5 formal papers + a fully-empirical 9-axis bounded-impossibility map** (INDEX.md). P1 admission-neg
(+TP-determinism lesson); P2 caching-mirage (+phase-boundary criterion +3-policy GPU eviction sweep +write_back
decoupling); P3 capstone map + 4-method head-of-line diagnosis (+mechanism figure) + parameter-free lever-selection;
P4 chunk-scheduling-neg (RPB, scoped to *reservation*); P5 movement-nonlever (+slow-tier boundary). **Headline:**
SRPF lifts goodput 0/≤3 → 4.0 req/s (λ5); same-node 4/4 vs 0/4 Fisher p=0.014, λ5 pooled 7/7 vs 0/8 p=0.0002;
SLO-robust [6,41]s. **Thesis:** goodput@SLO is decoupled from hit-rate AND the losslessly-raisable throughput
ceiling — it is set by the cold-doc SLO tail; the one lever is prefill scheduling (textbook SRPF). All
GPU/trace-confirmed, adversarial-review-clean, reproducible (22/22 scripts), cross-paper-consistent (numbers +
prose-scope), figured, registry+overview current, synced.

**EXHAUSTED — do NOT re-tread:** every KV-cache/serving axis is a GPU/measurement-confirmed non-lever for goodput
(caching/eviction, admission, KV-memory-mgmt, movement, Mamba-pool, chunk-reservation, prefill/decode split); the
sole lever (whole-request SRPF) is textbook; no novel lossless goodput lever exists under this frozen eval (~20
cycles, exhaustive incl. repeated fresh brainstorms).

**BLOCKED / OPEN (need external unblock — NOT pursuable now):** (1) **chunk-SIZING** (queue-pressure-adaptive, no
reserve-waste) — the one adjacent untested primitive; **independence-deferred** (a sibling's result on it leaked
into my recalled memory → pursuing = non-independent + duplicative); honest future work, flagged in P4 §7. (2)
**CP-for-hybrid prefill** — the only compute-side lever; architecturally UNAVAILABLE for Qwen3.5 in sglang v0.31
(code-verified). (3) **cross-workload/model generality** — untestable (eval frozen); addressed analytically via
phase-boundary + slow-tier criteria. (4) **prior-art discovery** — WebSearch org-policy-blocked.

**What would unblock further NOVEL work:** a new eval/workload/model (reopens levers + tests generality), CP wiring
for hybrids, or unblocked web. Absent those, the frozen-eval design space is solved. **Supervisor decides
redirect-vs-retire; I hold in disciplined watch-mode — re-probe each cycle, act on any genuinely-new +
independence-clean direction, no gild/fabricate.**

### 2026-07-16 (cont.): fresh hostile-PC review of the materially-changed flagship → hard ACCEPT (artifact-verified)
P3 had changed materially since the last dedicated hostile-PC review (~7 cycles prior): write_back abstract
sentence + residency/compute-row content, the head-of-line mechanism figure, the compute-ceiling correction, the
chunk-scheduling scope corrections. Ran a FRESH adversarial (SOSP/OSDI) review of the CURRENT P3, instructed to
verify every quantitative claim against raw artifacts. ★VERDICT: **ACCEPT** — zero claim mismatches. Verified:
write_back +4.8..+6.6pp hit / +7.4% throughput (4.74→5.09) on a confirmed SAME-NODE pair (both node 1-1 per
sacct, jobs 20232/20242), goodput unmoved (fails all rates); compute-ceiling row internally consistent
(4.0<4.72<5.09); SRPF same-node 4/4 vs 0/4 p=0.014 + λ5 pooled 7/7 vs 0/8 p=0.0002 EXACT; HOL figure accurate;
chunk-scope correctly scoped to reservation; supporting claims (9.6× queue, ~13× decode stall, phase-boundary
step-function, ~5s arithmetic bound) all match. Biggest weakness (SRPF=textbook) EMPHATICALLY ADDRESSED (map/
diagnosis paper; SRPF a confirmatory probe). 3 minor nits, all not-reject-level + already handled (paper states
node 1-1; C_eff-circularity acknowledged + lever-selection non-circular; companions non-load-bearing). Reviewer:
"exemplary systems work — counter-intuitive finding, airtight evidence, honest reporting, generalizable framework."
⇒ the flagship is top-venue-ready after all recent changes; no edits warranted. Genuine rigor confirmation, not gilding.

### 2026-07-16 (cont.): fresh hostile-PC review of P2 (most-changed companion) → hard ACCEPT (artifact-verified)
Parallel to the P3 review: P2 had substantial un-fresh-reviewed new content (NEW §3.4 write_back, rewritten §3.3
3-policy eviction sweep, §3.1 live-under-load fortification). Ran a fresh adversarial review verified against
artifacts. ★VERDICT: **ACCEPT** — all claims match: SLRU −18.7..−26.8pp / LFU −10.5..−24.6pp (ranges "−19..−27"/
"−10..−25" correctly rounded), write_back +4.8..+6.6pp / +7.4% (4.74→5.09) same-node, offline LRU=LFU=Belady=0
avoidable (doc_reuse.py), 4-pass 55.1M/54.0M/23.1M, phase-boundary step-function on constructed + 4 real corpora.
No contradictions (abstract "caching can't move goodput" ↔ §3.4 write_back raises hit not goodput ↔ §3.1
eviction-optimal-hit, all consistent). write_back honestly framed as stock-config evidence. Biggest weakness
(generalization on 2 non-trivial corpora) explicitly + honestly scoped (§3.5 + §7). Only <1pp rounding, within
tolerance. Reviewer: "rigorous bounded negative, proven 3 ways, honest execution, no over-claiming." No edits warranted.
⇒ Both materially-changed papers (P3 + P2) now fresh-adversarial-reviewed → BOTH ACCEPT, all claims artifact-verified.
P1/P4/P5 changed less (verified via consistency audit + P3 cross-checks). Body confirmed top-venue-ready. Genuine rigor.

### 2026-07-16 (cont.): cross-session integrity — propagated the write_back ceiling-correction to P1
Genuine watch-mode probe: does the write_back finding (a LOSSLESS backup policy raises the throughput ceiling
+7%) contradict any PRE-write_back claim in P1/P4/P5? Found one: P1 §2.4 said "the wall that NO POLICY crosses is
that [~4.7] ceiling" — but write_back crosses it (4.74→5.09). Same issue I corrected in P3's compute-ceiling row,
not yet propagated to P1. Fixed: scoped to "no SCHEDULING policy crosses it" (scheduling reorders work, doesn't
reduce it) + noted write_back raises the ceiling losslessly but goodput stays below it (SLO-tail-bound, doesn't
change the admission conclusion). P1 §2.3 ceiling band (~4.2–5) already roughly consistent (write_back 5.09 ≈ band
top). P4/P5 verified clean of the overclaim. ⇒ the write_back ceiling-implication is now fully propagated across
ALL papers (P1/P2/P3 corrected; P4/P5 clean). Commit 1d68fad9d. Genuine cross-session integrity fix, not gilding —
validates that careful probing in watch-mode still catches real inconsistencies.

---
## Cross-session consistency sweep — write_back propagation completed (07-16, cycle post-P1-fix)
Triggered by last cycle's find (P1 §2.4 stale "no policy crosses the ceiling"). A full grep-sweep across
all 5 papers for ceiling/cap/flat-control framing caught two more residual inconsistencies where the
write_back finding (peak req/s +7%, prefill-side, lossless) had not been fully propagated:
- **P3 line 110** (eval-description "honest controls"): "peak tok/s, peak req/s are decode-bound and ~flat
  by construction" — internally inconsistent with P3's own corrected compute-ceiling row. Fixed → peak
  throughput is a prefill-work ceiling not moved by scheduling; only prefill-work reduction (write_back +7%)
  raises it; goodput sits below it regardless. (commit a4bfdcf28)
- **P3 line 386** (g_ceil glossary def): "the work-conservation cap no policy beats" — write_back lowers E[W],
  raising g_ceil. Fixed → "no *scheduling* policy beats (fixed E[W]); reducing E[W] itself raises it ~7%."
  (commit 43bb6d424)
Re-sweep confirms ZERO remaining unscoped ceiling/cap claims across P1–P5; write_back numbers consistent
(4.7/4.72 band, 5.09 raised, +7.4%). P3 well-formed. LESSON (reinforced): a finding that touches a
cross-cutting concept (here: the throughput ceiling) must be grepped across ALL papers — number-focused
reviews and even a fresh full-paper review of the *changed* paper missed these two eval-description/glossary
lines because they carry no headline number. Eval frozen (07-12), WARNINGS clean, web blocked, no new lever.
