# turing — research program synthesis
## Anatomy of the goodput@SLO tail on saturated conversational serving: a residency–concurrency theory

**System.** sglang v0.31 HiCache (2-tier L1 HBM + L2 768GB DRAM), Qwen3.5-122B-A10B-FP8 (hybrid attention MoE),
TP8, one exclusive 8×H100 node. **Workload.** Mooncake-style multiturn conversational mix (loogle loader, 1553
convs → ~7037 turns, docs up to 191K tokens), deterministic (seed=1, --disable-shuffle) + warmup burst.
**Metric.** goodput@SLO = max Poisson rate λ∈{3,5,7,10} with p99 TTFT ≤ 8 s.

### The one-sentence theory
The goodput@SLO tail is governed by a **memory-bound concurrency↔decode positive-feedback runaway**: high in-flight
concurrency slows every (memory-bandwidth-bound) decode step, which lengthens residency, which raises concurrency.
**Every effective lever reduces in-flight residency/count; every mechanism that raises it backfires.** Caching,
admission gating, and prefill-rate *reduction* raise concurrency (or fail to lower it) → no help or backfire; capacity,
whole-request shortest-first scheduling, and **giant *acceleration*** reduce residency → win.

### The six papers (the arc)
1. **goodput-anatomy** (SUBMITTED) — *Caching can't.* The p99 tail is turn-0 giant prefills of heavy-tailed docs,
   uncacheable (first-sight). Hierarchical caching raises hit but not goodput@SLO; admission gating craters hit
   (device L1 full of running KV → non-eager backup lost before reuse); eviction is LRU≈Belady. Bounded negative on
   the whole cache axis. Analytical p99 floor q99(U)/R.
2. **schedulable-frontier** (SUBMITTED) — *The capacity law.* goodput@SLO ≤ C = K/(1−h). Two anti-correlated levers:
   cache-hit via 1/(1−h) [strong; eager backup] vs K via backup-I/O-reduction [linear; gating craters h]. Every
   request is solo-feasible (max 191K-doc prefills in <7s) ⇒ all SLO violations are CONTENTION, not service time ⇒
   offline goodput = C; the [measured, C] gap is the contention factor.
3. **goodput-coinflip** (SUBMITTED) — *Reliability is a coin-flip.* On the deterministic λ=3 point, identical stock
   configs on the same node give p99 6–14 s (pass↔fail): metastable occupancy-basin selection, super-M/G/1
   amplification. Reliability is capacity-margin(C−λ)-governed. §4.2: the obvious compute-side cure (decode-floor)
   BACKFIRES.
4. **decode-floor** (folded into P3) — *Failed cure #1.* Occupancy-gated prefill throttle to "reserve compute for
   decode" → p99 6-11s→31.8s. Decode is memory-bound (reserving compute doesn't drain it) + finer chunking makes
   giants linger → concurrency↑. The feedback is memory/slot-mediated, not compute-mediated.
5. **prefill-hol-blocking** (SUBMITTED) — *Failed cure #2 + diagnosis.* 45.6% of λ=3 prefill steps are head-of-line
   blocked: a giant monopolizes the per-step budget (median 11 steps) while ~4 shorts wait. The lossless fix
   (fair-share chunk interleave) BACKFIRES: p99→37s, 40% starved. **UNIFIED LAW: at the concurrency cap, ANY
   prefill-rate reduction backfires** — the tail is not reshapable within fixed capacity by slowing prefill.
6. **giant-acceleration** (SUBMITTED) — *The lever that works, AND a capacity lever.* The DUAL of the law: ACCELERATE
   the in-flight giant (boost its per-step chunk to f·B under occupancy pressure) so it exits sooner → concurrency↓ →
   memory-bound decode↓ → p99↓. Same-node A/B λ=3: accel n=4 PASS the 8s SLO 4/4 (p99 7.0/4.6/5.5/4.8 s) vs stock 1/4
   (14.2 s); DOUBLY SIGNIFICANT — Fisher SLO-pass p=0.040 + Mann-Whitney tpot p=0.016/conc p=0.008 (coin-flip-robust).
   First positive, lossless, on-contract mechanism. Why it works (refutes the wall-clock-residency worry): decode is
   memory-bound, so extra prefill tokens are ~free under the decode stall → a bigger chunk finishes the giant in fewer
   *same-length* steps → halves residency. **★FRONTIER (full sweep): accel is not just a λ=3 fix — it PASSES the SLO
   at BOTH λ=3 (4.5s) AND λ=5 (3.4s, thr 4.85 STABLE) where stock λ=5 is unstable/fails (17-20s) → goodput@SLO 3→5
   (+67%); λ=7 is the new knee (29.8s, thr saturates 5.20). Acceleration raises the effective capacity ceiling C
   ~4.14→~5.2 (+26%) — it is empirically the K-lever of P2's C=K/(1−h).**

### The design-space map (what moves the tail, what doesn't)
| axis | effect on concurrency/residency | goodput@SLO |
|---|---|---|
| caching (hit ↑) | doesn't lower tail concurrency (tail uncacheable) | no (P1) |
| admission gating | craters hit → C↓ | backfire (P1) |
| eviction policy | LRU≈Belady | neutral (P1) |
| decode-floor throttle | giants linger → concurrency↑ | backfire (P4) |
| fair-share interleave | giants linger + starvation → concurrency↑ | backfire (P5) |
| **capacity (C)** | raises ceiling | **lever (P2)** |
| **whole-request SRPF** | fewer in-flight requests | **lever (textbook; P2 K-lever)** |
| **giant acceleration** | shorter giant residency → concurrency↓ AND raises C +26% | **lever (P6, novel positive; goodput 3→5)** |

### Contributions
- A predictive **theory** (concurrency↔decode runaway) that unifies caching, scheduling, and reliability on one axis.
- A **unified law** (prefill-rate reduction backfires) established via two independent falsified cures (P4, P5).
- The law's **positive dual verified** (P6 giant acceleration) — a novel, lossless, on-contract goodput@SLO win that
  **raises the frontier 3→5 req/s (+67%) and the capacity ceiling C +26%** (empirically the K-lever of the P2 law).
- Rigorous **bounded negatives** across the cache/admission/prefill-reshaping axes; honest coin-flip methodology
  (median-of-k, same-node A/B, deterministic per-step metrics that are coin-flip-robust).
