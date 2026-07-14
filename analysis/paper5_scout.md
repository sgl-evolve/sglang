# Paper 5 (candidate NOVEL MECHANISM): giant-prefill head-of-line blocking → fair-share chunk interleaving

**Status:** scouted on CPU (code-confirmed) while v7 compute-bound 2026-07-14. NEXT direction after Paper 4 resolves.

## The structural finding (code-confirmed, extern/sglang)
The p99 TTFT tail (Papers 1/2: turn-0 giant prefills, docs up to 191K tok, uncacheable first-sight) is amplified by
**head-of-line (HOL) blocking**: an in-flight chunked giant monopolizes the entire per-step prefill budget.

Evidence (scheduler.py `_get_new_batch_prefill_raw`):
- L2857-2859: `if self.chunked_req is not None: self.chunked_req = adder.add_chunked_req(self.chunked_req)` — the
  in-progress chunked giant is added FIRST, BEFORE the waiting queue.
- schedule_policy.py `add_chunked_req` (L728-765): consumes `new_len = min(cand_extend, rem_chunk_tokens)`. For a
  191K giant at `chunked_prefill_size`=8192, new_len=8192 = the WHOLE budget; `_update_prefill_budget` drops
  rem_chunk_tokens→~0; returns req (truncated, unfinished).
- Then waiting-queue drain (L2889+) calls `add_one_req`; schedule_policy.py L872 `if self.rem_chunk_tokens <= 0:
  return AddReqResult.OTHER` — every waiting SHORT req is rejected that step.
- Net: a 191K giant occupies the prefill pipeline for ceil(191K/8192)=~24 CONSECUTIVE steps; shorts arriving in that
  window cannot START prefill → their TTFT += ~24 step-times. At ~200-400ms/step that's ~5-9s — exactly the measured
  p99 tail band (6-11s). So HOL-behind-giants is a plausible primary p99 driver, structurally.

## Mechanism (novel, lossless, on-contract)
**Fair-share chunk interleaving**: cap the in-flight giant's per-step share so a waiting short can also start prefill
in the SAME step. E.g. when `waiting_queue` non-empty AND chunked_req is a giant, give the giant only
`chunked_prefill_size * f` (f∈(0,1), e.g. 0.5) this step, leaving budget for ≥1 short's (small) prefill.
- The giant STILL progresses (just at fraction f/step → ~2× its own steps), but shorts stop waiting behind it.
- Lossless: only reshapes per-step batch composition; same tokens, same outputs, no drop/reorder/evict.
- On-contract: no eval/budget/model change; env-gated default OFF.

## Why novel (not textbook, not sibling, not Paper 4)
- NOT SRPF/SJF: those reorder the WAITING queue (which whole req runs next). This interleaves at the CHUNK level so an
  ALREADY-IN-FLIGHT giant doesn't monopolize the step. Orthogonal to (composes with) queue reordering.
- NOT admission (WSAC/floyd) or retention (valiant): does not defer/gate cold starts; the giant runs immediately,
  just shares the step.
- NOT Paper 4 (occupancy-feedback damping caps prefill when occupancy HIGH to let DECODE drain). Paper 5 splits the
  prefill budget across CONCURRENT PREFILLS to cut SHORT-req HOL latency. Different signal (queue depth vs occupancy),
  different target (structural tail vs basin variance). Could compose.
- Existing `enable_dynamic_chunking`/`predict_next_chunk_size` (L2832) sizes the giant's chunk by history_len but
  does NOT reserve budget for waiting shorts — no fair-share primitive exists.

## Predicted outcomes (falsifiable)
- PRIMARY: λ=3 (and maybe λ=5) p99 TTFT DOWN (shorts freed from behind giants). WIN = p99 drop with SLO pass.
- TRADEOFF risk: the giant's OWN TTFT rises (it prefills slower). Net p99 = win iff #shorts-behind-giant × saved >
  giant's added delay. Likely win when queue depth behind a giant is >1 (measured concurrency 124-166 → many behind).
- GUARD: lossless (hit + outputs invariant); C@λ10 unchanged (no admission change).
- NEGATIVE mode: if the tail is the giant's OWN prefill time (not shorts behind it), interleaving can't help and may
  hurt (giant slower) → characterized bound: "the p99 tail is intrinsic giant-prefill latency, not HOL contention."
  Either result publishable + sharpens Paper 1/2's cold-doc-floor claim.

## Implementation hook (located)
scheduler.py L2857-2859, right before `add_chunked_req`: if `SGLANG_TURING_FAIR_PREFILL` and waiting_queue non-empty
and chunked_req present, pass a reduced budget to add_chunked_req (needs a per-call rem_chunk_tokens override, or set
a temporary `adder.rem_chunk_tokens` cap for the chunked add, then restore for the waiting drain). Env-gate default
OFF; verify resolved_args frozen; verify lossless (outputs match).

## Diagnostic FIRST (before building): confirm HOL is real in OUR traces
Cheap on-node instrumentation run: log, per step, (chunked_req present?, chunked remaining tokens, #waiting rejected
for rem_chunk_tokens<=0). If waiting-rejected-behind-giant is frequent at λ=3, HOL is the driver → build the fix.
If rare (giants finish before shorts queue), the tail is intrinsic → Paper 5 becomes the bound. Do this diagnostic
as the first v-run of the direction.
