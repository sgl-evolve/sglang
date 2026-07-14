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

## DIRECTION 3 (open): context-parallel prefill for the cold-doc tail

The p99 floor = biggest cold docs' own prefill time is the one thing irreducible under serving POLICY.
The only compute-side lever is parallelizing a single doc's prefill. My prior "CP blocked by fixed tp"
note was WRONG: `--enable-prefill-cp` + `--attention-context-parallel-size` are legal (NOT in FORBIDDEN),
run CP over the 8 TP ranks (attn_cp_size = tp_size//dp_size = 8), fa3 backend HAS a CP-extend path, and
model_runner explicitly supports "MHA-arch prefill CP (Qwen3/Qwen2 MoE)". Open risk = hybrid MAMBA layers
under CP. Plan: GPU smoke-test `--enable-prefill-cp --attention-context-parallel-size 8 --cp-strategy
zigzag`; if it loads + is lossless + cuts big-doc TTFT → NOVEL mechanism = ADAPTIVE CP (invoke only for
heavy-tail cold docs, since CP comms overhead hurts small prefills — ties to the bimodal-cost finding).
If incompatible → bounded finding (the last lever unavailable; wall stands).
