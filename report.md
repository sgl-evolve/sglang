# Researcher: **turing**  (sglang v0.31, research-tier, KV-cache architecture)

Branch `evolve/turing` · base commit `a334877e5` · W&B run `turing` (project `sgl-evolve`).
Independent replicate — conclusions built from the unmodified baseline; I do not read siblings' work.

## Fixed eval contract (never change)
- 2-tier L1 (GPU HBM) + L2 (768 GB host, `--hicache-size 96`), no L3. Qwen3.5-122B-A10B-FP8, TP8, ctx 262144, full decode.
- Rate sweep **λ ∈ {3,5,7,10}** Poisson, warmup 300, **no per-rate flush** (warm steady-state), `--max-concurrency 256`, NUMP=1553.
- Workload `mooncake_mix_v1.jsonl` via bench_serving **loogle** loader + `--enable-multiturn`.
- **Headline = goodput@SLO** = max req/s with p99 TTFT ≤ 8 s. Honest controls (≈flat, decode-bound): peak tok/s, peak req/s.
- Lossless gate: outputs match no-cache run.

## Baseline (`baseline.json`, stock 2-tier, λ=3 operating point)
hit_rate 0.6217 · TTFT p50 750 ms / p99 6326 ms · req/s 2.78 · out 355 tok/s · host_util 0.9999 (L2 full) ·
load_back_mean 1.65 ms (transfer NOT the bottleneck at λ=3) · evict_tokens 582M ≫ load_back_tokens 298M.
Hit split: 40% device / 60% host.

---

## Direction 1 (ACTIVE): Reuse-aware L2 tiering — don't spend the cache on dead-on-arrival KV

### The workload's reuse geometry (trace-driven, offline, free)
Every record → one conversation. **Turn 0 = "Input: {document} Question: {Q0}"** carries the huge shared prefix
(doc p50 ≈ 7.4K tok, mean 12.4K, max 191K); **turns 1..N = follow-up questions** served multiturn (prefix grows).
Documents are unique across conversations → reuse is strictly **intra-conversation**. A doc is reused (N−1) times
then dies at conversation end.

Measured over the full 1553-conv workload (`mooncake_mix_v1.jsonl`, loader-faithful, tok≈chars/4):

| quantity | value |
|---|---|
| one-shot convs (1 turn, doc **never** reused) | **593 / 1553 = 38.2%** |
| one-shot share of **document token mass** | **10.95M / 19.30M = 56.7%** |
| total doc mass admitted | 19.30M tok (≈1.8× L1+L2 = 10.7M) |
| mean turns/conv | 4.61 (max 61) |
| one-shot rate by source | sharegpt **0%**, leval 45%, loogle **73%** |

### The gap (why this is a real, config-unreachable problem)
Stock `write_through` eagerly backs up **every** prefill's KV to L2. Since >½ of document KV mass is one-shot
(dead on arrival), the cache spends >½ its L2 write bandwidth and a capacity-churn equal to the *entire* L1+L2
on KV that is never reused — evicting live documents, whose next question then MISSES and pays a full
document-length prefill recompute (the p99 TTFT tail). **Neither config flag fixes this:** `write_back` still
backs up dead docs on eviction; exclusive tiering only changes *where* a copy lives, not *whether* dead KV is
admitted. Eviction-policy tuning is a dead end here (LRU≈Belady in-order) because the problem is **admission**,
not eviction order.

### Hypothesis
In LLM serving the reuse signal is **bimodal and cleanly separable**: a prefix that is re-hit even once is
near-certain to be reused further (the conversation is still growing), while a never-re-hit prefix is dead.
Gating L2 residency on **demonstrated / structural reuse** (rather than eager write-through) keeps L2 populated
with proven-hot documents, cutting document-recompute misses and backup bandwidth — losslessly, and beyond what
any stock config reaches. Novelty must go past textbook 2Q/LFU (excluded): the radix tree encodes conversation
structure, so the residency signal is **branch liveness** (is this node an ancestor of a still-extending
conversation?), not flat per-item frequency.

### Offline screening (free, `analysis/sim_v2.py`) — calibrated to the baseline
**Oracle bound.** Infinite-cache hit = **0.809**; unavoidable floor (turn-0 docs + new questions) = 19.1% miss.
Baseline 0.622 ⇒ **18.8pp of miss is eviction-driven (avoidable), not first-sight.**

**Effective-capacity collapse (key insight).** To reproduce the baseline hit of 0.62, a stock-LRU sim needs an
*effective* prefix-cache capacity of only **~1.1M tokens** — vs 10.7M physical. Under concurrency-256 serving
with 20–190K-token contexts, running-request KV + fragmentation consume ~90% of L1+L2, leaving a tiny effective
prefix cache. The baseline sits on a **steep** part of the hit-vs-capacity curve (0.57→0.72 over 1.0→1.5M) —
which is *why* goodput@SLO is a coin-flip.

**Screening result.** At the calibrated operating point, **reuse-gated admission → +7.15pp hit** (0.622→0.693),
capturing ~38% of reachable headroom. Robustness: the gain holds (+7 to +12pp) in *every* timing model that can
reproduce the baseline 0.62; it turns negative only in unpressured regimes (stock already near-oracle) that are
*inconsistent* with the measured baseline. So the mechanism helps **specifically under memory pressure** — the
knee where goodput@SLO lives. Sim is crude (conv-as-unit LRU, no faithful L1/L2 tiering) → magnitude uncertain;
the real rate-sweep is decisive.

**Two screened-out sub-hypotheses (honest negatives):** (a) *bandwidth* — D→H backup is only ~0.7–1.2 GB/s vs
64+ GB/s hardware, so the ~54% backup-traffic cut reclaims a non-binding resource (no throughput win expected
from BW); (b) *eviction ordering* — LRU≈Belady among orderings, so the win must come from **admission** (what
enters the protected tier), not eviction order. Novelty vs textbook 2Q/AdaptSize rests on the *effective-capacity
collapse* insight + the LLM-specific first-reuse admission signal, not the gate mechanic itself.

### Design → implementation
Mechanism: **reuse-gated L2 promotion.** Under write_through, gate the L1→L2 backup on demonstrated reuse — a
prefix earns an L2 (protected-tier) copy only after it is re-matched at least once while resident. One-shot
documents (56.7% of doc mass) never reach L2 → the protected tier stays populated with proven-hot content.
Lossless (a non-promoted prefix that is later needed simply recomputes, exactly as a miss). L1 acts as the
natural probation tier. Risk to measure: multiturn docs whose 2nd turn arrives after L1 eviction become misses.

### Plan
1. [done] Offline screening — motivation + design established.
2. [in progress] Implement reuse-gated L2 promotion in the live cache-controller / radix path; commit.
3. check_env smoke test; full rate-sweep test submission (+ same-node stock replicate for A/B); ablations; error bars; lossless check.

### Versions (same-node A/B on certified node slurm2-a3nodeset0-3)

**v1_stock** (commit ce01c1c79, mechanism OFF; tag `config`) — my same-node baseline curve.
Real device data during the run: at λ=3 concurrency is already MAXED (running-req p50=247/256),
device KV usage p50=0.84/p90=0.95/max=1.0 (running KV crowds out the prefix cache → L2-bound).
Decode dynamics: 12.6% of decode batches stall (<50 tok/s), 70% of prefill batches are giant
6144-token doc chunks, queue p50=0 → the TTFT tail is **prefill↔decode interference**, not HoL.

| λ | req/s | ttft_p50 | ttft_p99 | hit_rate |
|---|---|---|---|---|
| 3 | 2.83 | 1017 ms | **11494 ms** | 0.6753 |
| 5 | 3.59 | 1002 ms | 24957 ms | 0.6627 |
| 7 | 3.99 | 1053 ms | 35753 ms | 0.6568 |
| 10 | 4.14 | 1055 ms | 41015 ms | 0.6537 |

**goodput@SLO = 0** (every λ p99 > 8 s). p50 stays ~1000 ms while p99 explodes → metastable-tail
coin-flip (baseline.json's p99=6326 PASSED; this same-config run FAILS at 11494 → confirms the
coin-flip; single-run goodput@SLO is not a usable headline → I compare **hit_rate (robust)** + p99 values).

**v2_flat2** (mechanism ON, `flat` gate_hits=2 == write_through_selective; tag `config` CONTROL).
Confirmed live: gate withholds ~76% of would-be L2 backups (gated_skips ≫ backups).

| λ | req/s | ttft_p50 | ttft_p99 | hit_rate | Δhit vs stock |
|---|---|---|---|---|---|
| 3 | 2.91 | 1475 ms | 12060 ms | **0.4259** | **−25.0 pp** |

**STRONG NEGATIVE — reuse-gated admission craters hit_rate (−25 pp).** Mechanism: at these loads the
device (L1) is 84-100% full of running-request KV, so any prefix NOT eagerly backed up to L2 is evicted
from L1 and **lost before its reuse** → its continuation misses. write_through's eager "back up everything"
is *necessary* precisely because device is saturated. This (a) refutes my Direction-1 hit-rate hypothesis
on the real system (my calibrated sim was optimistic — it modeled a single combined cache, not the
device-saturated L1/L2 split), (b) kills the whole admission-gating family (size-gating loses gated
content the same way), and (c) is a clean, counterintuitive result: the "obvious" I/O-saving optimization
(write_through_selective) catastrophically backfires under device saturation. Backup BW is non-binding
(~1 GB/s) so there is no compensating throughput gain. **Direction 1 = bounded negative.**

### Verdict on Direction 1 & pivot
The lossless *caching* lever is bounded on this workload: (i) miss floor 19.1% is unavoidable turn-0
first-sight; (ii) avoidable 18.8pp (oracle 0.809 vs 0.675 stock — my node) is capacity-bound (L2 7.8M) +
info-constrained; (iii) eviction is LRU≈Belady; (iv) admission craters (−25pp, above). The real goodput@SLO
killer is the **prefill↔decode interference metastable tail** (p50 flat ~1s, p99 11→41s; 12.6% decode
stalls; 70% giant prefills; queue≈0), which is a compute/scheduling phenomenon the cache cannot fix.
→ **Direction 2: attack the metastable tail** (serving-machinery, not caching). Direction-1 evidence
(bounded-negative + the effective-capacity-collapse + admission-crater anatomy) is itself a rigorous
characterization contribution.

## Formal submissions
_(none yet)_
