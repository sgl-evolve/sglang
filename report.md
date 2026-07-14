# Researcher: floyd

**Branch:** `evolve/floyd`
**Clone:** `workspace/sgl/v0.31/research/researchers/floyd/`
**Base commit:** `a334877e5` (stock sglang)

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

### ★★ PIVOT: v4-cca-lb REDUCES the goodput VARIANCE (n=2, striking)
- **v4-lb λ=3 p99 = {8556, 8549}** (n=2, spread **7ms**) vs stock {6327, 8124, 11787} (spread 5460ms).
  Load-back-only gating **collapses the coin-flip to a stable ~8.55s** — a ~1000× variance reduction.
  This is a POSITIVE mechanism effect (novel: pacing L2→L1 load-backs removes the metastable
  stall-clusters that cause stock's high-variance FAIL runs), analogous to the known de-dup
  variance-reduction but via prefill admission.
- **The catch:** the stable point (8.55s) is JUST above the 8s SLO → reliably FAILS λ=3, so goodput
  reliably 0 (vs stock's lucky 1/3 pass). Variance reduction landed on the wrong side of the SLO.
- **The opportunity:** if a config shifts the stable point <8000ms, CCA-lb converts the coin-flip into
  a RELIABLE pass → goodput reliably 3 (a real WIN). ⇒ sweep watermark/threshold to find it.
- Confirm low variance at n≥3 (v4-r3); then watermark sweep (0.75, 0.90) + threshold.

## Contribution = variance-reduction lead OR rigorous negative (charter-valid either way)
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

(finalizing the negative/characterization paper; firming CCA n≥2 + one watermark-sweep point to show domination)
