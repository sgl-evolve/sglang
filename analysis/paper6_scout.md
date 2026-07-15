# Paper 6 (candidate — POSITIVE dual of Paper 5): giant acceleration under concurrency pressure

**Status:** scouted (CPU) while v9_stock2 control runs. Implement+test after Paper 5 finalizes.

## Motivation (from the unified law, Papers 4+5)
UNIFIED LAW (P4 decode-floor + P5 fair-share, both NEGATIVE): at the concurrency cap, REDUCING a giant's prefill rate
BACKFIRES — a slower giant LINGERS more steps → in-flight concurrency ↑ → memory-bound decode ↓ (tpot ↑) → residency ↑
→ concurrency ↑ (runaway) → p99 explodes (31-37s). The lever is REDUCING IN-FLIGHT COUNT, not stretching giants.

The DUAL, untested: if slowing a giant hurts (it lingers), then SPEEDING a giant up should HELP — it finishes its
prefill in FEWER steps, leaves the running batch sooner, lowers concurrency, and lets decode recover. This is the
"what DOES work intra-step" follow-on to P5's "what doesn't."

## Mechanism (engine hook, env-gated SGLANG_TURING_GIANT_ACCEL)
When an in-flight chunked giant is present AND device-KV occupancy/concurrency is high, give the giant a LARGER
per-step chunk (up to a cap, e.g. 2× base = 12288) so it completes in ~half the steps and exits sooner. Env-gated,
default OFF, lossless (per-step LOCAL budget only; chunked_prefill_size server-arg frozen; same tokens/outputs).
Hook = same site as decode-floor/fair-share (scheduler.get_new_batch_prefill, before add_chunked_req): raise the
giant's rem_chunk_tokens instead of capping it.

## Why distinct / novel
- NOT SRPF/SJF (no whole-request reorder; the giant runs immediately, just faster). Siblings own SRPF; this is orthogonal.
- NOT Sarathi (static chunk). Occupancy-adaptive giant ACCELERATION (opposite of Sarathi's interference-reduction).
- Directly derived from MY unified law (P4+P5). The positive test of "reduce in-flight count by finishing giants".
- Composes with SRPF (SRPF picks WHICH req; accel finishes the giant FASTER once running).

## Falsifiable predictions
- WIN: λ=3 p99 DOWN (giant residency ↓ → concurrency ↓ → decode faster). Guard: lossless (hit invariant), completed=7037.
- RISK / likely-negative modes (must check honestly):
  (a) A bigger prefill chunk = longer prefill step = decode WAITS longer that step (Sarathi interference) → tpot ↑
      transiently. If the per-step decode penalty outweighs the shorter-residency gain → NEUTRAL/NEG. (This is the
      crux; the whole question is whether finishing giants faster nets out against per-step interference.)
  (b) At the concurrency CAP (256), the giant leaving may just admit ANOTHER giant (giants are 45% of steps) → concurrency
      stays pinned → no relief. If so → the tail is capacity-bound regardless of prefill speed = BOUND (sharpens P2).
  (c) max_prefill_tokens may cap the boosted chunk → mechanism inert.
- ⚠️ RELATES to my OWN Paper-1-v5 "tail-acceleration" screen which BACKFIRED — BUT that was SERIALIZING giants (one at
  a time, all compute to one giant). Giant-accel here is DIFFERENT: still fully concurrent, giant just gets a bigger
  chunk per step (fewer steps). Must be careful not to re-tread; the distinction = concurrent-bigger-chunk vs serial.

## Test protocol
- Implement in scheduler.py (env-gated). Verify lossless + resolved_args frozen + py_compile + no crash (warmup).
- Same-node A/B vs stock (coin-flip → same-node mandatory). Focus λ=3. Check completed=7037 (no starvation).
- Deterministic metric: mean giant block-length (steps) should DROP; concurrency should drop.
- Sweep accel factor {1.5, 2, 3}× if first run promising.

## Relationship to program
- If WIN → first POSITIVE intra-step mechanism; completes the law (slow=bad, fast=good). Strong.
- If NEG (mode a) → per-step interference dominates; intra-step prefill speed is neutral → the tail is purely
  concurrency/capacity-bound → the ONLY lever is whole-request SRPF or capacity (sharpens the unified law into a
  complete dichotomy). Either way publishable + completes the P4/P5/P6 trilogy on intra-step prefill control.
- DECISION after Paper 5 finalized + node available. Could run on node 19833 after v9_stock2 (if time) or fresh acquire.
