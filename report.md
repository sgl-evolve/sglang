# Researcher: floyd

**Branch:** `evolve/floyd`
**Clone:** `workspace/sgl/v0.31/research/researchers/floyd/`
**Base commit:** `a334877e5` (stock sglang)

> **⚠ CURRENT THESIS (read the "★★ PIVOTAL CORRECTION (head-of-line)" section near the end first).**
> This log is chronological. The early "Thesis"/"Mechanism" prose below (admission control, "retraction
> cliff") was my *original* direction and is **superseded**: CCA admission is a rigorous **NEGATIVE**, and
> the λ3 goodput coin-flip is **head-of-line blocking** (small turns queued behind a few large cold docs'
> back-to-back chunked prefill), not a retraction cascade (stock has **0 retractions**). Both papers (v2):
> caching & admission are non-levers; **prefill scheduling (shortest-prefill-first) is the goodput lever**.

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
2. **The raw-throughput ceiling is ~4.7, not ~4.2.** SRPF reaches 4.72 req/s at λ10 (vs stock's 4.22 at λ10
   — SRPF also lifts *peak* throughput ~+12%, a bonus off the goodput metric). The earlier "~4.2 ceiling"
   was a sub-knee estimate; the full sweep measures it. C_eff back-out updated 15k→~17k; band ~4.2–4.7.

The same-node n=3 A/B (node 0-3: srpf λ3 {5.9,6.0,7.4}/λ5 {5.85,6.58,7.46} vs stock control 7.6/22.7) still
provides the *rigorous* λ3,5 comparison + Fisher significance; v-srpf-full (node 1-2) provides the
*contract-complete curve*. Both cited; goodput headline = 4.0.

**Integrated across all 3 papers** (P3 abstract/fig/table/§5/§6.1, P1 §2.3/§4, P2 §4/fig): goodput 4.0,
raw ceiling ~4.7 measured, goodput SLO-tail-bound below it. Figure regenerated data-driven from
`runs/v-srpf-full/curve.csv` (`analysis/gen_curve_svg.py`). W&B logged (v-srpf-full, tag=mechanism).
All papers re-verified well-formed. Committed + pushed to evolve/floyd.
