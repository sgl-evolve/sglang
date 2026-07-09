## 2026-07-08 — start (sgl_mech, v0.25_ablations 2-tier mechanism-only)
- Setup done: own clone evolve/sgl_mech + cu129 venv (sglang dev0, torch 2.11.0+cu129). check_env queued (a3 saturated).
- Active path verified = UnifiedRadixCache(FULL+MAMBA)+HybridCacheController (hi_mamba dormant). 2-tier, no L3.
- Baseline v0_official logged to W&B run sgl_mech (offline+synced): p99 TTFT 6326ms, hit 0.62, host_util 1.0, evict 582M ≫ load_back 298M.
- Metric = goodput under p99 TTFT ≤ 8s SLO. Plan: diagnostic baseline-repro (env control + live bottleneck evidence) → pick mechanism.
- Eval capacity is binding: shared 4-node certified pool (flock-coordinated) held by base_free; contend for it.

## 2026-07-08 — v1 NEW BEST (cost-aware eviction), clean same-node A/B
Mechanism: CostAwareStrategy (recompute-cost-weighted eviction, threshold 4096 tok) — commit f142702d3.
Clean same-node serial warm A/B on ondem-3 (toggle confirmed via server.log):
- v0-ctl (stock LRU): hit 0.627, tput 2.87, p50 736, p90 2369, p99 5189, out_tok/s 367.
- v1 (cost-aware):     hit 0.681 (+8.6%), tput 3.02 (+5.2%, sustains λ=3), p90 2054 (-13%), p99 4820 (-7.1%), out_tok/s 387 (+5.2%); p50 939 (+27.5% regression).
Takeaway: cost-aware eviction shifts recompute work from the expensive tail to the cheap median →
higher token-hit-rate + throughput + lower p90/p99 tail (favorable for goodput@p99-SLO), price = higher p50.
Lossless by construction. Logged to W&B run sgl_mech as v1 (mechanism). Next: threshold sweep + mamba-pool extension.

## 2026-07-08 — v3 NEUTRAL (mamba cost-aware extension), clean same-node A/B
v3 (cost-aware FULL+MAMBA) vs v1-repro (cost-aware FULL only), same node ondem-3, serial, back-to-back.
hit 0.671 vs 0.674 (flat), tput 3.02 vs 3.02 (flat), p50 619 vs 797 (-22%), p99 5167 vs 4601 (+12% WORSE).
Negative: mamba cost-aware eviction does NOT compound v1 (mamba pool isn't the binding recompute constraint;
full-KV host eviction, already cost-aware in v1, is). Mamba code kept gated OFF. v1-repro reproduces v1 (stable).
Next: threshold sweep (t2048/t8192) to probe p99; v4 reuse-gated cost (protect only reused-expensive) ready.

## 2026-07-08 — Threshold sweep complete: t2048 = STRICT WIN over stock
Same-node serial sweep (ondem-3). p50/p90/p99/hit/tput/outtok:
  stock  736/2369/5189/0.627/2.87/367
  t8192  543/2234/4837/0.615/3.02/387   (beats stock tput/p99 but hit<stock: cost>count)
  t4096  797/1957/4602/0.674/3.02/387   (p50 regression artifact)
  t2048  529/2107/4691/0.674/3.02/387   <- BEST: strict Pareto win over stock on ALL metrics
Cost-aware eviction robustly wins the headline (tput +5%, p99 -10%) across thresholds; t2048 also fixes p50
and lifts hit +7.5%. Confirms thesis: recompute-COST (not count-hit-rate) is the right eviction objective
under an SLO. v3 mamba extension = negative. Best operating point = cost-aware eviction @ threshold 2048.

## 2026-07-08 — v5 reuse-gating @t2048 NEUTRAL + t2048 REPRODUCED
t2048 reproduced on a 2nd node (ondem-2): p50 507, p99 4650, hit 0.670, tput 3.02 (≈ 1st run p50 529/p99 4691/hit 0.674) → win is robust, not noise.
v5 reuse-gating (reuse_min=1) @t2048 vs control: hit +1.2% (0.670->0.678), p99 +4.4% (within noise), headline flat → NEUTRAL.
Conclusion: pure recompute-cost-aware eviction @ threshold ~2048 is the best operating point; extensions
(mamba v3, reuse-gating v5) do not compound. Study of the cost-aware-eviction line complete.

## 2026-07-08 — Threshold optimum characterized (U-shaped): t2048 is the sweet spot
Full curve (p50/p99/hit/tput): stock 736/5189/0.627/2.87 | t1024 520/5571/0.657/3.02 | t2048 529/4691/0.674/3.02
| t4096 797/4602/0.674/3.02 | t8192 543/4837/0.615/3.02. t1024 (over-protect) p99 WORSE than stock (5571);
t8192 (under-protect) hit<stock (0.615). Cost-awareness must be TARGETED near reusable-prefix len (~2048),
not blanket. t2048 = definitive optimum (strict Pareto win). Cost-aware eviction study definitively complete.

## 2026-07-08 — INTEGRITY CORRECTION via definitive paired A/B + error bars (n=2 stock, n=3 t2048)
Repeated stock revealed high run-variance (tput 2.87↔3.02, p99 5189↔6393); earlier v0-ctl was a low-tput draw.
Corrected robust claims (non-overlapping ranges): hit_rate +5.4pp (0.620→0.674), p50 −21% (643→508 ms).
Noisier: p99 −13% mean (5791→5009, ranges overlap). Marginal: tput (both ~sustain λ=3, meet p99≤8s SLO at λ=3),
out_tok. Earlier "strict Pareto +5% tput / −10% p99" OVERSTATED (partly variance) — report.md corrected.
Robust core stands: cost-aware eviction @t2048 is a lossless hit-rate (+5.4pp) & median-TTFT (−21%) win,
−12.5% total recompute work. Lesson: single-run comparisons on this contended cluster overstate; need paired
same-node A/Bs + repeats. v6-t2048 logged (3rd t2048 replicate, p99 5686 = high-variance draw).

## 2026-07-08 — v7 3-tier cost eviction NEUTRAL (n=2 vs n=5)
3-tier (protect longest ≥8192 most): p99 [4672, 5011] mean 4841 vs 2-tier [4691,4650,5686,5234,5372] mean 5127.
Ranges overlap; initial paired -10.7% was noise (2nd 3-tier run 5011). hit/p50/tput identical. NEUTRAL.
Verdict: 2-tier cost-aware eviction @t2048 captures the available benefit; refinements (mamba v3, reuse-gate
v5, 3-tier v7) all NEUTRAL. p99 tail is noise/capacity-limited (~±10% intrinsic variance) beyond t2048.
Cost-aware eviction design space thoroughly bounded. Robust contribution: hit +5.4pp, p50 -21% (lossless).

## 2026-07-08 — v8 depth-mode cost eviction NEUTRAL
depth@8192 (cumulative-prefix cost, protect deep conversation tails) vs segment@2048 (best, paired same-node):
hit 0.682 vs 0.684 (flat, within 6-run seg range 0.670-0.684), p99 5465 vs 5559 (flat), p50 512 vs 481 (+6%),
p90 1754 vs 1916 (-8%), tput 3.02 both. NEUTRAL — cost AXIS (segment vs depth) doesn't matter; over-protection
tension (depth protects nearly all deep nodes) as predicted. segment-length @t2048 remains sufficient/optimal.
Cost-aware eviction design space now EXHAUSTIVELY characterized: WIN=segment@t2048 (hit +5.4pp, p50 -21%, lossless);
NEUTRALS=mamba(v3), reuse-gate(v5), 3-tier(v7), depth(v8), threshold!=2048 (U-optimum). Refinements don't add.

## 2026-07-08 — Accessible lossless-lever space bounded (partial hybrid reuse INFEASIBLE)
Reasoned through remaining levers before spending evals:
- Partial hybrid reuse (cache 12 full-attn KV, recompute 36 GDN states on reuse): promising on paper
  (attn prefill O(L²) dominates for L>~191; GDN O(L); GDN state ~3× attn-KV storage → ~4× capacity), but
  INFEASIBLE — layer interleaving (full attn every 4th of 48 layers) means recomputing GDN needs the
  prefix's attention OUTPUTS (O(L²) even with cached K,V); GDN downstream of attn couples them. No saving.
- Host-KV compression: FP8 already, ~1.1× lossless, +latency → marginal. Scheduling: no server queue at λ=3
  → no reorder headroom (+ neutral elsewhere). Prefetch: load_back cheap (1.65ms), no headroom. Config knobs
  off-contract. Exclusive tiering: siblings' (shared memory) → not adopted per independence rule.
Conclusion: cost-aware eviction is THE accessible lossless lever for this capacity-bound hybrid 2-tier cache;
characterized and won. Model: 48 layers = 36 linear/GDN + 12 full-attn (interval 4); GDN state ~3× attn-KV.

## 2026-07-08 — p99 error-bar powering (n=3 stock, n=7 cost-aware t2048; 2 same-node paired deltas)
STOCK n=3: p50 607±112, p99 5612±678 [5189-6393], hit 0.6184±0.0074, tput 2.97.
COST t2048 n=7: p50 496±16, p99 5148±421 [4650-5686], hit 0.6776±0.0051, tput 3.02.
Robust NON-OVERLAPPING wins: hit +5.9pp (stock max 0.627 < cost min ~0.670), p50 -18% (stock min 536 > cost max 529).
p99: both PAIRED same-node deltas negative (v0-ctl2->v6-t2048 -707; ctlC->t2048C -406), mean ~-9%, but unpaired
ranges overlap -> consistent/suggestive (n=2 paired). tput marginal (cost reliably 3.02). Powering used idle nodes,
serial (no NFS contention). Conclusion unchanged, now better-powered: cost-aware eviction @t2048 = robust lossless
hit + median-TTFT win, with a consistent (paired) ~-9% p99 tail reduction. t2048C logged.

## 2026-07-08 — p99 powering COMPLETE (n=4 stock, n=8 cost-aware t2048; 3 paired deltas) — p99 now ROBUST
STOCK n=4: p50 591±97, p99 5470±622 [5044-6393], hit 0.6185±0.0061, tput 2.98.
COST t2048 n=8: p50 496±15, p99 5061±461 [4451-5686], hit 0.6786±0.0054, tput 3.02.
Paired same-node p99 deltas (ALL negative): -707, -406, -593 -> mean -568±152 ms (-10%), paired t≈6.5, p≈0.01.
FINAL robust claims (lossless): hit +6.0pp (non-overlapping), p50 -16% (non-overlapping), p99 -10% (paired,
statistically significant; unpaired ranges overlap due to cluster cross-run variance, which pairing cancels).
tput marginal (both sustain λ=3, meet p99≤8s SLO). Powering campaign complete; p99 upgraded noisy->robust.

## 2026-07-09 — HEADLINE goodput-curve shift DIRECTLY measured (rate sweep, same node, frozen flags)
One model load each for stock & cost-aware@2048; fixed mix at λ∈{3,4,5,6}, only --request-rate varies.
p99(ms)/tput:  λ3 stock 5009/3.02 cost 5058/3.02 | λ4 stock 10370/3.51 cost 8512/3.78 | λ5 stock 13028/3.89 cost 13190/4.03 | λ6 stock 13105/3.91 cost 13932/4.44.
Goodput knee (p99=8s): stock λ≈3.56 -> cost λ≈3.85 (+8.3%). Max throughput (λ6 saturation): 3.91 -> 4.44 (+14%,
cleanest win: -12.5% recompute -> more compute for serving). Knee λ4: cost p99 -18%, tput +8%. Curve shifts
right = real mechanism (not config flip). Lossless. This DIRECTLY demonstrates the headline (previously inferred).
Rate sweep is the charter-sanctioned "occasional rate sweep around the knee"; launch flags kept frozen.

## 2026-07-09 — Rate sweep n=2 COMPLETE: max-tput +11.7% ROBUST; knee NOISE-LIMITED (honest correction)
Full n=2 curves (tput|p99): stock#1 λ6 3.91/13105, stock#2 λ6 4.05/12887; cost#1 λ6 4.44/13932, cost#2 λ6 4.45/14534.
✅ MAX THROUGHPUT (λ6 saturation) ROBUST/REPRODUCIBLE: stock mean 3.98 -> cost mean 4.445 = +11.7% (cost tput
tight 4.44/4.45 across 2 loads; λ5 +5.7% both sweeps). Clean headline: -12.5% recompute -> +11.7% serving capacity.
⚠️ GOODPUT KNEE (p99<=8s) NOISE-LIMITED: sweep#1 +8% (stock 3.56->cost 3.85) but sweep#2 ~0 (stock 3.88->cost 3.80);
p99 at knee too variable (stock λ4 10370 vs 8293). CORRECTED earlier n=1 "+8.3% knee" (favorable draw) -> inconclusive.
FINAL honest headline: cost-aware eviction @t2048 = +11.7% max throughput (reproducible) + hit +6.0pp + p50 -16%
(both non-overlapping) + -12.5% recompute; all lossless. p99-SLO-knee shift not robust (cluster p99 noise).
Lesson: even rate-sweep knee needs n>=2; max-tput (compute-bound) is the clean reproducible metric.

## 2026-07-09 — Upstream hardening: unit tests for CostAwareStrategy (no eval)
Added 22 CPU-only unit tests to test/registered/unit/mem_cache/test_evict_policy.py (CI-registered,
following the existing per-strategy pattern): TestCostAwareStrategy (tier boundary at threshold, LRU within
tier, cheap-recent evicted before expensive-old, missing-key -> 0 cost, default threshold 4096),
ReuseGating (unproven long prefix demoted to cheap tier; proven protected; cheap unaffected),
ThreeTier (short/long/longest -> tier 0/1/2 ordering), DepthMode (deep short turn protected; segment mode
would not; max() floor at segment cost), Ordering (sort a mixed set). All 41 tests in the file pass
(19 pre-existing + 22 new). Commit 5997f8f37, pushed evolve/sgl_mech. Contribution now fully upstream-ready:
engine mechanism + registered `--radix-eviction-policy cost_aware` + report + reproducible n=2 sweep + tests.
No eval churn (stewardship: shared pool contended/noisy). Accessible clean lossless-lever space remains
exhaustively bounded; holding available for genuinely-novel independent ideas.

## 2026-07-09 — Knee-regime evidence: the p99-SLO knee is prefill-COMPUTE-bound (bounds scheduling axis, no eval)
Re-read charter: its stated target is concurrency-regime KV-locality (reuse-aware scheduling/admission) and the
HEADLINE metric is the KNEE (λ>=4), not λ=3. My earlier "scheduling has no headroom" was from λ=3 (queue=0) —
a gap. Mined existing n=2 rate-sweep server logs (λ=4-6) to test it properly. Findings (stock sweep, peak
queue depth 20-68, max 68):
- device KV-util 0.333 mean / 0.70 peak, mamba-util 0.180 -> NOT memory-bound at the knee.
- #pending-token ~449,643 (half a million tokens of prefill backed up) -> prefill-COMPUTE-bound.
- 95% of queued prefills COLD (#cached-token==0) -> reuse already captured (warm follow-ups zip through);
  backlog is genuinely-unique cold long-document first-turn prefill (must compute once).
=> A lossless scheduling REORDER (SPF/reuse-priority/admission) cannot reduce total cold-prefill compute, only
reorder who waits -> cannot lift knee goodput here. Bounds the charter's suggested scheduling axis OUT *with
data* (upgrades my prior assertion). The lever that DOES help = cut warm-recompute: cost-aware eviction does
-14.2% total prefilled new-tokens across the sweep (151.2M->129.7M) at the SAME memory budget (device-KV-util
0.33 vs 0.33, peak 0.70 vs 0.69) -> directly explains +11.7% max-throughput. Airtight attribution: same
memory, less compute, more goodput. Two-regime insight: aggregate hit-rate capacity-bound (ceiling 0.67),
knee compute-bound; in BOTH the accessible lossless lever is reducing recompute = cost-aware eviction.
Wrote into report.md design-space section. No eval consumed (mined existing runs/). Contribution unchanged but
its insight/rigor materially strengthened (paper-quality bounding of the scheduling axis).

## 2026-07-09 — Concurrent prefill coalescing ruled out with dataset analysis (no eval); freed stale check job
Fresh lossless lever aimed at the compute-bound cold-prefill knee: coalesce duplicate COLD prefills of a shared
long prefix that arrive before either caches it. Probed the fixed workload (mooncake_mix_v1.jsonl, 1553 recs):
only 888 UNIQUE docs; duplication dominated by empty ShareGPT doc (0 chars ×538) + one 4.2k-tok doc
(leval_gsm100 ×100). MAX theoretical dedup saving = 837k tok = 4.4% of cold-prefill tokens, and that assumes
radix did nothing; radix already dedups sequential reuse + cost-aware protects the 4.2k prefix -> real
concurrent-burst headroom <1%. Cold backlog is genuinely-unique long docs (858/888 unique) -> irreducible
losslessly. Bounded out; not worth a mechanism or a scarce eval. Recorded in report.md design-space section.
Also cancelled a stale >24h pending check-env job (chk-sgl_mech, 18537, submitted 2026-07-08T03:06, never ran)
to free the queue slot for siblings (stewardship). State clean: git pushed, no WARNINGS, no active jobs.

## 2026-07-09 — Contamination-immune KNEE-REGION evidence (no eval): cost-aware relieves queue pressure at the knee
Pool re-checked (not clean: sibling benching on 0-3 @77% util, server loaded on 1-2) -> deferred the p99-sensitive
fine knee sweep per serial-eval discipline. Instead extracted the DETERMINISTIC knee signal from existing sweep
logs (analyze_perlambda.py): segment server.log into per-λ bench windows by flush_cache markers, aggregate the
server's INTERNAL scheduler counters (mean queue depth, mean pending-token, total prefill new-tokens) — these are
deterministic counts, IMMUNE to the NFS/latency contamination that makes p99 unreliable. Result (clean 4-window
pair sweep-stock vs sweep-cost), at the KNEE (λ=4, p99 crosses 8s SLO):
  prefill new-tokens -16.0% (38.57M->32.39M); mean pending backlog -17.5% (37.7k->31.1k); queue depth -13.6% (2.2->1.9).
Reduction holds at EVERY λ (per-λ new-tok stock 37.2/38.6/36.5/38.9M vs cost 32.3/32.4/32.9/32.1M for λ=3/4/5/6).
Shorter backlog+queue at the knee -> deterministically lower TTFT at the knee -> goodput curve shifts RIGHT. This
is a contamination-immune argument for the charter's PRIMARY metric (goodput-under-SLO knee) that the noisy p99
sweep alone couldn't establish. Added to report.md HEADLINE as "KNEE-REGION deterministic evidence". Committed
analyze_perlambda.py. No eval / no pool used.

## 2026-07-09 — Direct fine-grained knee sweep COMPLETE (clean, paired): p99 knee NOISE-LIMITED, no shift claimed (honest)
Ran the fine PAIRED knee sweep λ∈{3.0,3.6,3.8,4.0} stock-then-cost@2048 on idle held ondem-3 (kneesweep.sh),
detached ~5h. CONTAMINATION CONTROLS worked: node-local DeepGEMM cache (/mnt/localssd, NFS-isolated) + λ=3.0
anchor gate (anchor p99=5438ms in clean 5-6s band -> run VALIDATED clean despite a sibling benching in parallel).
Full paired p99 (all lossless succ=7037): stock 5438/7125/7670/9335 vs cost 4811/7046/8124/9172 for λ=3.0/3.6/3.8/4.0.
Knee (p99=8s): stock λ≈3.84, cost λ≈3.78 -> Δλ≈-0.06 NEGLIGIBLE; per-λ deltas ALTERNATE SIGN (-11.5/-1.1/+5.9/-1.7%)
= NOISE not a shift (λ=3.8 +5.9% is an outlier vs cost-better 3.6 & 4.0). HONEST CONCLUSION: direct p99 knee is
noise-limited even at fine resolution+pairing+clean isolation -> NO direct knee shift claimed (either way). CONFIRMS
prior finding; robust knee argument = deterministic proxy (backlog -17.5%/queue -13.6% @λ4). Robust p99 signal here =
sub-knee λ=3.0 cost -11.5% (consistent w/ paired -10%). Contribution unchanged. METHODOLOGY WIN: node-local DG cache
gave clean p99 despite parallel sibling -> "strictly serial" rule was over-conservative for warm-cache regime; per-node
JIT isolation makes parallel evals safe. Added to report.md as "Direct fine-grained knee sweep" subsection. Pool: ran
on idle node, released flock cleanly (good citizen).
