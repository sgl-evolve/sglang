# Paper 4 (candidate NOVEL MECHANISM): occupancy-feedback damping for reliable goodput@SLO

**Status:** design (CPU); implement + test when a certified node frees (after v6_flat, ~21:30, or fresh acquire).
**Motivation:** Paper 3 (goodput-coinflip) VALIDATED that the λ=3 coin-flip is metastable occupancy-basin
selection sustained by a feedback loop:
> high occupancy → more concurrent decode steals prefill compute → prefill AND decode slow (tpot 392→468ms
> measured) → requests linger → occupancy stays high (166 vs 124 measured).
The bad basin is worse for BOTH prefill TTFT and decode. So *breaking the feedback* could improve reliability
AND mean — this is NOT the usual prefill-vs-decode tradeoff, because the runaway hurts both.

## Hypothesis
A controller that caps prefill's per-step compute share when in-flight occupancy exceeds a threshold lets decode
drain, frees slots, and pulls the system back to the low-occupancy basin → collapses the p99 coin-flip variance
(and may lower mean p99), losslessly, without changing capacity C=K/(1−h).

## Why this is novel (not textbook, not config, not sibling)
- NOT SRPF/SJF (no reorder by size), NOT LRU/2Q (not eviction), NOT admission (floyd) or retention (valiant).
- NOT a stock flag (no chunk-size knob is occupancy-adaptive; chunked-prefill uses a *static* budget).
- It is a **feedback controller on occupancy** — a control-theory primitive applied to LLM-serving *reliability*,
  targeting the metastability I measured. Closest prior art = Sarathi-Serve (static chunked prefill to reduce
  interference); ours is *dynamic/occupancy-adaptive* and aimed at variance, not mean throughput.

## Mechanism (engine hook, env-gated `SGLANG_TURING_DECODE_FLOOR`)
In the scheduler's batch-composition step (where chunked prefill token budget is set):
1. Read current in-flight decode count D and running-KV occupancy ρ_occ (device KV util, already instrumented).
2. If ρ_occ > θ_hi (e.g. 0.90): reduce the prefill token budget this step to B·(1 − g), reserving compute for
   decode (a "decode floor"), so decode drains and slots free. g = damping gain (e.g. 0.5).
3. If ρ_occ < θ_lo (e.g. 0.70): restore full prefill budget B (don't throttle when there's headroom — avoids the
   deferral-backfire that hurt cold prefills for floyd).
Hysteresis (θ_lo<θ_hi) prevents oscillation. Only activates in the HIGH-occupancy (bad-basin) regime → giant
turn-0 prefills at low occupancy are NOT delayed (preserves the TTFT tail; addresses the known deferral risk).

## Predicted outcomes (falsifiable)
- PRIMARY: p99 TTFT variance at λ=3 collapses (bad basin unreachable). Test: n≥3 runs, compare p99 spread to
  stock's ~5.3s. WIN = spread < ~1s (like wb's margin effect but via feedback, not capacity).
- SECONDARY: mean p99 at λ=3 drops (feedback broken → low-occupancy basin). 
- GUARD (must not regress): hit_rate invariant (lossless); peak throughput C unchanged (not de-saturation —
  the controller only reshapes the batch at high occupancy, doesn't cap admission/rate).
- RISK / likely-negative modes: (a) throttling prefill delays giants → p99 UP (deferral backfire, floyd);
  mitigated by θ_hi gating (only when occupancy already high) + hysteresis. (b) decode is memory-bound so
  "reserving compute" doesn't speed it → no effect (then it's a characterized negative: the feedback is not
  compute-mediated but memory/slot-mediated → informs the mechanism). Either result is publishable.

## Test protocol (on-contract; lossless)
- Implement in python/sglang/srt/managers/scheduler.py batch-composition (find the chunked-prefill budget site).
- Env-gate; default OFF (contract frozen). Verify lossless (outputs match no-cache) + resolved_args unchanged.
- Eval: full sweep (eval.sh, unchanged) with SGLANG_TURING_DECODE_FLOOR=1; focus λ=3 for the coin-flip; n≥3.
- Compare to stock n≥3 (same node): p99 spread, mean p99, hit (invariant), C (λ=10 unchanged).
- Sweep θ_hi ∈ {0.85,0.90,0.95}, g ∈ {0.3,0.5} if the first run is promising.

## Relationship to the 3 papers
- If WIN → Paper 4 = the first POSITIVE mechanism; turns Paper 3's characterization into a cure (reliability
  mechanism orthogonal to the frontier: damps variance without moving C). Paper 3 §5 already flags this follow-on.
- If NEGATIVE → folds into Paper 3 as "the feedback is memory/slot-mediated, not compute-mediated; decode-floor
  cannot damp it" — a characterized bound on reliability mechanisms.
- Either way lossless, on-contract, novel, motivated by validated data. NEXT ACTION after Paper 2 firming.
