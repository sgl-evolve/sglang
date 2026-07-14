# Paper 2 OUTLINE — "The Reserved Short-Prefill Lane" (working title)
# slug: prefill-hol-lane · author: kleinrock · status: DRAFT-IN-PROGRESS (NOT yet registered — awaiting evidence)
# Register in INDEX.md ONLY once evidence (screen 19836 + full sweep) clears the bar OR bounds a clean negative.

## Thesis (framing decided by the empirical signal — DO NOT register until known)
- POSITIVE branch: prefill inter-request head-of-line (HOL) blocking behind heavy-tailed cold documents is the
  goodput@SLO bottleneck; a small reserved short-prefill lane collapses the victim TTFT tail LOSSLESSLY and
  WITHOUT the starvation shortest-first (SRPF) causes → shifts the throughput–latency curve / raises goodput.
- NEGATIVE branch (if screen null): even fair prefill budget-sharing cannot beat the cold mega-doc prefill
  FLOOR (the p99 is the mega-docs' OWN prefill, not their victims) → a NEW bounded negative, distinct from
  Paper 1 (which bounded caching/scheduling on the coin-flip METRIC; this bounds a prefill-scheduling MECHANISM).

## §1 Introduction (framing pending data)
- Contribution bullets: (i) diagnosis — quantify inter-prefill HOL on the real workload; (ii) mechanism —
  reserved short-prefill lane (non-preemptive, lossless, opt-in); (iii) evidence vs BOTH fcfs and srpf baselines.

## §2 Motivation (trace-driven — FILL with HOL instrumentation data)
- Workload heavy tail (trace_stats.json): doc-tok p50 ~16.7K, p99 ~65.6K, MAX 192.5K; many small/zero
  follow-on turns. [have this]
- CODE MECHANISM [have this, code-verified]: engine serves ONE chunked_req at a time; PrefillAdder.add_chunked_req
  takes min(rem_chunk_tokens, rem_total) = full 6144 for a mega-doc → rem_chunk_tokens→0 → add_one_req loop
  NO_TOKEN for all waiting reqs → HOL for ~doc/6144 iters (192K ≈ 32 iters ≈ 8s).
- FILL: per-request scatter (input_len vs TTFT) under reserve0 showing short reqs' TTFT inflated when behind a
  mega-doc (needs env-gated per-req dump — build if screen positive).

## §3 Design — the reserved short-prefill lane [FINALIZED]
- Algorithm: each prefill iteration, if a chunked (large) req is in flight AND waiting_queue non-empty AND the
  chunked req would otherwise consume > (budget - R) tokens, cap its chunk at (budget - R), reserving R tokens.
  The waiting loop then admits short reqs (input <= R) as NON-chunked prefills in the SAME iteration.
- Invariant preserved: only ONE chunked_req at a time — the reserve admits SHORT (whole-seq) reqs only; a long
  waiting req is NOT allowed to start a 2nd chunked prefill (guard: has_chunked_req → OTHER). [bug caught+fixed]
- Losslessness: only the per-iteration token BUDGET SPLIT changes; every token of every req is still computed,
  same order within a req, same outputs. Opt-in (R=0 => byte-identical stock).
- Novelty vs closest art [from report.md prior-art note]: Sarathi/chunked-prefill target prefill→DECODE stalls
  (mix decode into prefill batch), NOT inter-PREFILL HOL; FastServe preempts (mine is non-preemptive);
  SRPF/SJF reorder+STARVE (mine fair-shares, no starvation); vs classic fair-queuing the novelty is the
  DIAGNOSIS + lossless no-starvation collapse of the victim tail (must be carried by effect size + attribution).

## §4 Implementation [FINALIZED]
- Files: python/sglang/srt/managers/schedule_policy.py (PrefillAdder.__init__ + add_chunked_req cap + add_one_req
  guard), scheduler.py (pass short_lane_reserve), server_args.py (--prefill-short-lane-reserve, default 0).
- ~30 LOC, opt-in. Branch evolve/kleinrock-shortlane @ d106782c8. Hooks the LIVE prefill path (get_new_batch_prefill).

## §5 Evaluation [FILL — screen 19836 then full sweep]
- Same-node A/B reserve0 vs reserveR (fast screen λ5 K3; then full rate sweep λ{3,5,7,10}); goodput@SLO;
  p99/p50 TTFT distribution (Paper-1 methodology: report the distribution, not one coin-flip draw); RESERVE
  ablation {1024,2048,4096}; LOSSLESS verification (outputs vs reserve0); baselines = fcfs AND srpf.
- Honest controls: peak tok/s (decode-bound, ~flat); attribution (per-req input_len vs TTFT).

## §6 Related work [FINALIZED — see report.md prior-art note] Sarathi-Serve, FastServe, Mooncake, SRPF/SJF, fair-queuing/DRR.
## §7 Limitations — reserve slows mega-docs slightly (budget−R); if p99 is mega-doc-own-prefill-bound, no help (=negative branch).
## §8 Reproducibility — commit d106782c8, runs/v4-shortlane-ab (screen) + full-sweep version; tools/{shortlane_ab_eval.sh,ab_analyze.py}.
