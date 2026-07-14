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

### v0-stock (my baseline, ondem-3, job 19730) — IN PROGRESS
- **λ=3:** req/s=2.87, tok/s=367.5, p50=1056ms, **p99=11787ms → FAILS 8s SLO.** hit=0.6777.
- **Key live observation:** stock hit **full KV-pool usage = 1.00 ten times** during λ=3
  (pool saturates in bursts → prefill admission stalls → TTFT tail spikes). p50 is fine
  (1s) but p99 is 11.8s — confirms the tail is a few pool-saturation/large-cold-prefill events,
  not steady-state. This run landed on the FAIL side of the known λ=3 coin-flip.
- λ=5/7/10 pending.

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

## Formal Submissions

(none yet — awaiting complete same-node CCA curve to judge whether pool-control → p99/goodput win)
