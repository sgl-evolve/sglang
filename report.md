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

## Formal Submissions

(none yet — awaiting complete same-node CCA curve to judge whether pool-control → p99/goodput win)
