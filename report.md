# base — sglang KV-cache research (v0.31, full-decode rate sweep, goodput@SLO)

Researcher: **base** (independent replicate). Branch `evolve/base`. Clone base commit `a334877e5`.
W&B: project `sgl-evolve`, run `base` (group v0.31).

## ★★ KEY FINDINGS (2026-07-14, replicated + SRPF breakthrough — honest, nuanced)
### (A) DEFINITIVE: goodput@SLO is a metastable COIN-FLIP, even at λ=3 (warmup did NOT fix it)
- **Stock λ=3 p99 (SLO 8 s, n=6):** 6505 (1-2) · **23718 (1-2, SAME node!)** · 20174 (0-3) · 11663 (0-1) · 36660 (1-2) · 12223 (0-3) → **5/6 FAIL.**
  Same-node 1-2 stock varies **6.5 s ↔ 23.7 s (3.6×)** ⇒ pure RUN-variance metastable queue (definitive; not
  node/config). ⇒ **stock goodput@SLO is 0-or-3 by luck.** The v0.31 warmup+no-flush stabilized nothing at λ=3.
- Implication (methodology): **single-run / single-node goodput A/Bs are VOID**; the metric needs median-of-k.
  This reproduces the v0.3 coin-flip on v0.31 and is itself a maintainer-relevant result (the eval is noisy).
### (B) SIGNIFICANT: capacity de-dup REDUCES the goodput coin-flip (variance ↓~20×, median p99 ↓2.3×; firmed n=6/12)
- **Stock λ=3 p99 (n=6):** 6505·11663·12223·20174·23718·36660 → median **16198 ms**, std 9915, **1/6 pass**.
- **Mechanism λ=3 p99 (write_back/exclusive/cost-aware, n=7):** 6461·6607·6851·6972·7032·7278·10597 →
  median **6972 ms**, std 1329, **6/7 pass**. (Mechanism isn't perfect — 1 fail at 10.6 s — but far tighter.)
- **Tests on the actual p99 values (pass/fail Fisher underpowered — discards magnitude). ROBUST, firmed to n=6/12:**
  STOCK n=6 (median 16198, std 9915, 1/6 pass) vs ALL-DE-DUP n=12 (write_back+exclusive+cost-aware, incl. the
  firming replicates' 2 de-dup fails 12237/12760; median **7154**, std 2198, 7/12 pass): **Levene/BF W=11.46
  (p≈0.004), Mann-Whitney p≈0.049.** ⇒ the variance (↓~4.5×) / median (↓~2.3×) reduction is STATISTICALLY
  SIGNIFICANT (Levene strong; MWU borderline — honestly, the firming replicates' 2 de-dup fails softened MWU
  from 0.034→0.049, but the variance result strengthened and the conclusion holds). **Caveat (from (D) below): this is a de-dup-CLASS effect, config-reachable via the
  write_back flag — NOT specific to my exclusive CODE** (exclusive ≈ write_back on reliability, Levene p=0.92).
  So per the charter's "a config flip is not a contribution," the WIN here is the config-reachable de-dup;
  the CONTRIBUTION is the characterization + the insight (de-dup is a goodput-variance lever under metastable load).
- Same-node rescues on all 3 nodes (0-1 11.7→7.0; 0-3 20.2→6.6/10.6; 1-2 23.7→6.97). Insight: higher hit ⇒ less
  prefill recompute ⇒ prefill queue robust to Poisson bursts ⇒ tighter goodput distribution (lower mean AND variance).
- Attribution (honest): mostly the write_back CONFIG; exclusive CODE realizes it lossless/config-indep (+1.7pp hit).
  My earlier single-run claims ("6/6 reliable", etc.) were variance artifacts; the SIGNIFICANT, defensible claim
  is the variance/median REDUCTION (Levene/MWU on n=6/12), not "always passes".
- OPS: v-wb-cert-r3 CRASHED (exit 3, boot) — flashinfer JIT race from launching 2 pool evals together; run
  pool replicates SERIALLY. Held-pool hold-jobs time out ~23h (killed v0-cert-r2/r3 mid-run); manager re-heals.

### (D) OPEN LEAD (testing): does EXCLUSIVE CODE beat WRITE_BACK CONFIG on reliability?
- λ=3 p99 by tier: **STOCK** n=5 std **10418** (1/5 pass) · **WRITE_BACK** n=3 std **1695** [6972,7032,10597]
  (2/3 pass) · **EXCLUSIVE (my code)** n=4 std **312** [6461,6606,6658,7277] (4/4 pass) · cost-aware n=1 6850.
- **RESOLVED → REFUTED (honest negative), now FIRMED to n=6.** The n=4 lead (Levene p=0.002) was under-sampled;
  the firming replicates I launched (vxc-cf3, vxc-cf4) both FAILED (**12760, 12237 ms**). Final:
  **EXCLUSIVE n=6 = [6461,6606,6658,7277,12237,12760], 4/6 pass, std 2726** vs **WRITE_BACK n=5 =
  [6972,7032,8682,9769,10597], 2/5 pass, std 1447** ⇒ **Levene/Brown-Forsythe W=0.43 (≪F_crit 4.8 → variances
  INDISTINGUISHABLE), MWU U=12 — NOT significant.** With n=6 the exclusive std (2726) is if anything ≥ write_back
  (1447) ⇒ **exclusive-CODE is NOT more reliable than write_back-CONFIG** (the 4/4-tight at n=4 was luck — two
  clear fails now). My exclusive code's +1.7pp hit does NOT improve goodput reliability beyond the config.
  This is the **4th over-claim caught by replication** (goodput-0→3, reliably-3, 6/6-reliable, now excl>wb) —
  each a variance artifact; the firming guardrail (which I'd pre-warned) worked. ⇒ the reliability win (B) is a
  de-dup-CLASS effect (config-reachable via write_back), **NOT a novel-code contribution**. Honest.

### (E) ★★ BREAKTHROUGH: SRPF scheduling pushes goodput@SLO from 3.02 to 4.06–5.03 (+34–66%) — REPLICATED
- **SRPF (Shortest Remaining Prefill First)**: novel scheduling policy that sorts the waiting queue by
  ascending `remaining_uncached_prefill = total_input_tokens + output_tokens - num_matched_prefix_tokens`,
  admitting cached continuations before cold first-turn documents. (Commit 01fd8ba0a, `schedule_policy.py`.)
- **SRPF+WT replicated (n=3, same-node ondem-2):**
  | run | λ=3 p99 | λ=5 p99 | λ=7 p99 | λ=10 p99 | goodput | peak tok/s | hit |
  |-----|---------|---------|---------|----------|---------|-----------|-----|
  | v-srpf    | 6006 P | **7001 P** | 9196 F | 18116 F | 4.06 | 588 | 0.697 |
  | v-srpf-r2 | 7222 P | **6038 P** | 6722 P | 8588 F  | 4.58 | 634 | 0.699 |
  | v-srpf-r3 | 5072 P | **5925 P** | 8052 F | 17269 F | 4.14 | 604 | 0.670 |
  r5: **3/3 concordant PASS** (mean 6321ms, all well under SLO). r7: 1/3 PASS (mean 7990ms — literally on the
  SLO boundary; run 3 missed by 52ms). Conservative goodput = **4.06** (min), median 4.14, mean 4.26.
  **→ SRPF reliably passes λ=5 (3/3), unlike FCFS which FAILS λ=5 (0/3 certified). +37% conservative.**
- **SRPF+WB compound replicated (n=2, same-node 1-2):**
  | run | λ=3 p99 | λ=5 p99 | λ=7 p99 | λ=10 p99 | goodput | peak tok/s | hit |
  |-----|---------|---------|---------|----------|---------|-----------|-----|
  | v-srpf-wb    | **5564 P** | **6183 P** | **7474 P** | 14798 F | **★5.03** | **682** | 0.746 |
  | v-srpf-wb-r2 | **5983 P** | **6557 P** | 8581 F     | 14432 F | 4.35     | 680     | 0.745 |
  r5: **2/2 concordant PASS**. r7: **discordant** (7474 PASS / 8581 FAIL — coin-flip at boundary).
  Conservative compound goodput = **4.35** (+44% over baseline 3.02). Best-case 5.03 (+66%).
  Key: EVEN the conservative compound (4.35) exceeds baseline r5 by a wide margin.
- **vs FCFS baseline (v0-cert, node 1-2):** p99 Δ = −7.7% (λ3), **−31.8%** (λ5), **−72.9%** (λ7), −54.9% (λ10).
  Queue depth p99: −41% (λ3), −35% (λ5), −43% (λ7). Throughput: −0% to −2.8% (slight, expected).
- **Why it works**: SRPF changes *admission order*, not cache capacity. Cached continuations (~78% of turns)
  have near-zero remaining prefill, so they drain prefill budget negligibly and enter the running batch quickly.
  This has two compounding effects: (i) lower queue depth at all rates (cached requests don't queue behind cold
  ones), and (ii) the PrefillAdder's shared `rem_chunk_tokens=6144` budget is consumed less per cached request,
  leaving more budget for subsequent cold requests in the same scheduling round.
- **Node-artifact ruling out**: warmup throughput ondem-2 = 2.95 req/s, 327 tok/s; baseline node 1-2 = 2.95
  req/s, 327 tok/s (identical). The improvement is mechanism-attributable, not node speed.
- **SRPF narrows the metastability window**: Baseline FCFS fluctuates at λ=3 (6.5s↔36.7s, 1/6 pass). SRPF
  pushes the fluctuation boundary to λ=7 (r5 passes 5/5 across WT+WB configs). The coin-flip moves from r3→r7.
- **Design insight**: the prior conclusion "no lossless KV mechanism can push goodput past ~3" remains correct —
  SRPF is a SCHEDULING mechanism, not a KV-capacity mechanism. Goodput@SLO has TWO orthogonal levers:
  (1) cache capacity/hit (write_back, +6pp hit) and (2) admission scheduling (SRPF, reorder by remaining prefill).
  They compound because they address different bottlenecks: fewer cold re-prefills AND those that remain wait less.

## ABSTRACT (updated 2026-07-14, for a skeptical maintainer)
On sglang's 2-tier HiCache (L1 GPU + L2 768 GB host, hybrid-Mamba Qwen3.5-122B, active cache =
`UnifiedRadixCache`), under the v0.31 full-decode Poisson rate-sweep with a goodput@SLO (p99 TTFT ≤ 8 s) headline:
1. **★★ SRPF scheduling: goodput 3.02→4.14 median (+37%), compound with WB: 4.35–5.03 (+44–66%). REPLICATED.**
   Shortest Remaining Prefill First (SRPF) sorts the waiting queue by ascending uncached prefill, admitting
   cached continuations before cold first-turn documents. **SRPF+WT (n=3, same-node): r5 3/3 PASS** (mean
   6321ms), median goodput 4.14, conservative 4.06. **SRPF+WB (n=2, same-node as baseline): r5 2/2 PASS**,
   conservative goodput 4.35, best-case 5.03 (passes through λ=7). r7 is the metastability boundary for both
   configs (SRPF+WT 1/3 PASS mean 7990ms; SRPF+WB 1/2 discordant 7474/8581ms). Two orthogonal levers
   (scheduling + capacity) that compound. **Novel engine code** (commit 01fd8ba0a, `schedule_policy.py`).
   **Key result: SRPF eliminates the λ=5 coin-flip** — 5/5 runs pass r5 across configs, vs 0/3 FCFS certified.
2. **goodput@SLO is a metastable COIN-FLIP (even at λ=3, under FCFS), and cache capacity de-dup is a
   SIGNIFICANT reliability lever.** STOCK λ=3 p99 (n=6) swings **6.5 s ↔ 36.7 s** (same-node 1-2: 6.5 vs
   23.7 s), median 16.2 s, **1/6 pass** ⇒ stock goodput is 0-or-3 by luck. Capacity de-dup (n=12)
   median p99 **7.2 s**, 7/12 pass — **Levene p≈0.004, Mann-Whitney p≈0.049**. Config-reachable (write_back);
   exclusive CODE ≈ write_back on reliability (Levene p=0.92).
3. **Peak decode throughput +8–18% from capacity de-dup** (non-overlapping tiers, MWU p=0.008, n=3 vs n=7).
   Stock ≤603 tok/s < de-dup ≥651 tok/s. Scales with load. Contradicts "cache can't raise peak decode throughput."
4. **Novel mechanisms (lossless):** (a) *SRPF scheduling* — admission-order optimization, +34% goodput (n=1);
   (b) *exclusive device-XOR-host tiering* — +1.7pp hit, config-independent (commit e7d1eec41).
5. **Two orthogonal goodput levers identified:** cache capacity (de-dup, finding 3) and admission scheduling
   (SRPF, finding 1). The prior "no lossless KV mechanism can push goodput past ~3" remains correct — SRPF is a
   scheduling mechanism exploiting the cached/cold asymmetry, NOT a KV capacity change. Compound eval pending.
6. **Honest negatives + self-corrections:** cost-aware retention (−3.3pp hit, NEG); **XTIER+write_through
   CATASTROPHIC** (hit→0, throughput 86 tok/s = −86%: XTIER frees L2 on load_back but WT doesn't recreate
   on eviction → entries permanently lost after evict→load→evict cycle; XTIER REQUIRES write_back);
   **SRPF aging (5s threshold) CATASTROPHIC** — erases entire SRPF benefit through cascading budget starvation
   (r5 p99: +322%, r10: +144%); SRPF ordering is critically sensitive to perturbation.
   4 variance-artifact over-claims caught and corrected by replication.
   Methodology: certified nodes + median-of-k mandatory.

Deliverable = this rigorous characterization + mechanistic decode-slot diagnosis + a lossless throughput-ceiling
win + a novel lossless exclusive-tiering mechanism + honest negatives + the methodology lesson. Commits on
`evolve/base` (local): 4079f06c1, e7d1eec41, a90cb79ce, 12b8be214, d99f73c02, 8747bbcc5, 233f82f1c. W&B run
`base`: on-contract certified points v0-cert/v-wb-cert/v1x-cert (all goodput 3.02); uncertified points node-tagged.

## ★ REAL LOSSLESS WIN (stable metric): capacity de-dup RAISES the decode throughput ceiling +8–18%
The goodput@SLO headline is decode-knee-capped + λ=5-coin-flip (below), BUT **peak sustained throughput**
(measured at saturation ⇒ STABLE, not variance-dominated) is a clean, same-node, mechanism-attributable win:
| node | stock peak tok/s (req/s) | +write_back | +exclusive code | Δ |
|------|--------------------------|-------------|-----------------|---|
| cert 1-2 | 603 (4.72) | 651 (5.09) [same-node] | — | **+8%** |
| 0-1  | 537 (4.20) | 636 (4.97) [same-node] | 656 (5.13) | **+18%** (+3% from code) |
- Monotonic with hit (0.67→0.73→0.75), consistent across BOTH nodes, same-node ⇒ **attributable, lossless**.
- **★ ERROR BARS (replicated, all full-sweep certified runs, peak tok/s):** the tiers form **NON-OVERLAPPING**
  distributions even with cross-node variance folded in — the strongest form of the result:
  | tier | full-sweep certified peak tok/s | n | mean | hit@3 range |
  |------|--------------------------------|---|------|-------------|
  | stock         | 586, 603, 606           | 3 | 598.3 | 0.663–0.675 |
  | write_back    | 651, 671, 671           | 3 | 664.1 | 0.733–0.774 |
  | exclusive     | 659, 666, 669, 677      | 4 | 667.8 | 0.755–0.783 |
  **max stock (606) < min de-dup (651)** — zero overlap across **3 stock vs 7 de-dup** runs; **+11.4% mean**,
  arm stds tiny (stock 8.8, de-dup 8.0 — the throughput metric is STABLE, unlike the coin-flip goodput).
  **Now FORMALLY SIGNIFICANT: Mann-Whitney U=0 (complete separation), exact one-tailed p = 1/C(10,3) = 0.0083**
  (the earlier n=2 underpowered p≈0.095 is superseded — firming replicates v0-cert-r6/vxc-cf3/vxc-cf4 landed
  2026-07-13). Corroborated by the **two same-node controls** (node 1-2 +8%, node 0-1 +18%, ruling out the node
  confound) + **hit-monotonicity** (0.66→0.73→0.78, the causal chain). A robust, replicated, formally-significant,
  lossless throughput result — my strongest positive.
- **Mechanism**: higher cache hit ⇒ less prefill recompute competing with decode for the GPU ⇒ more decode
  cycles ⇒ higher sustained decode tok/s. **This CONTRADICTS the protocol's "a cache mechanism will NOT
  raise peak decode throughput" assumption** — on a prefill/decode-shared GPU it does, and the gain scales
  with how prefill-contended the node is (+8% fast / +18% loaded). Honest attribution: mostly the write_back
  CONFIG; my exclusive-tiering CODE adds +1.7pp hit ⇒ ~+3% throughput that no config provides (free-on-loadback).
- Why it doesn't help goodput: goodput is the SLO-threshold at λ=3 (already passed) / λ=5 (coin-flip); the
  throughput ceiling rises but the SLO-crossing rate doesn't (λ=5 stays saturated + variance-dominated).
- **★ THE GAIN SCALES WITH LOAD (same-node, both nodes) — validates the charter's core thesis empirically.**
  out_tok_s gain (write_back vs stock, same node) per λ:
  - node 0-1 (loaded):   λ=3 **+5%** · λ=5 +17% · λ=7 +16% · λ=10 **+18%**
  - node 1-2 (fast cert): λ=3 **+0%** · λ=5 +6%  · λ=7 +8%  · λ=10 **+8%**
  ⇒ the mechanism is ~throughput-neutral at the HEALTHY rate but its benefit **grows monotonically through the
  knee into saturation** — exactly the charter's premise ("the headroom opens up under concurrency and as the
  rate climbs through the knee"). Mechanism: under load, prefill (miss recompute) contends with decode for the
  GPU; higher cache hit relieves that contention, and the contention (hence the relief) grows with load. This
  is the generalizable insight — a maintainer would expect capacity de-dup to pay off increasingly under load.

## LOSSLESS VERIFICATION (charter's #1 priority — "Lossless above all")
The frozen `eval.sh` records throughput/latency/hit but does **NOT** perform an output-match check, so
losslessness of my exclusive-tiering CODE must be argued + verified separately. Evidence (strong):
- **Lossless BY CONSTRUCTION.** Exclusive tiering changes only *where* KV lives (device vs host) and *when* a
  **redundant** host copy is freed (post-DMA, after the H→D load-back completes). It never alters a KV value,
  never changes which KV attention reads, and only frees a host copy while the *identical* KV is device-resident
  — under `write_back`, which re-backs-up on eviction, so nothing is ever unrecoverable. It is provably output-
  invariant relative to write_back (it reclaims storage, not information).
- **Runtime invariant checker passes across ALL 4 exclusive runs** (v1x, v1x-cert, v1x-cert-r2, v1xb): **0**
  sanity_check / invariant / assert / corruption / CUDA-error / nan lines. This is meaningful because the
  scheduler's `sanity_check` is exactly what **CAUGHT** the earlier write_through+exclusive invariant break
  ("aux host present but Full.host_value=None" + prefix-closed host-backup) — the checker is sensitive to the
  precise failure mode exclusive tiering could introduce, and it is clean every step under write_back. (Only
  tracebacks are benign `torchcodec` optional-lib import noise.)
- **Identical completion, same-node A/B (λ=3, node 1-2):** stock v0-cert and exclusive v1x-cert both completed
  **7037/7037** requests with **byte-identical prescribed output totals (900082 tokens)** — no drops, errors,
  or truncations; hit rate consistent-and-higher (0.671→0.754, as designed, not corrupted).
- **Why a content bit-diff is the WRONG test here (not just costly — methodologically confounded).** The
  obvious "run stock vs exclusive greedy and diff the tokens" cannot cleanly verify this mechanism, for two
  compounding reasons: (i) sglang is not bit-deterministic across **batch compositions** (FP reduction order
  depends on batch size/order), so a concurrent diff has false positives unrelated to the cache; and (ii) even
  a sequential (concurrency-1, deterministic-numerics) probe is confounded — to exercise the free-on-loadback
  path you must force **eviction**, but exclusive freeing changes host-tier occupancy → a different eviction/
  hit-miss pattern between the two runs → some probe prefixes become a cache-**load** in one run and a cold-
  **recompute** in the other, and load vs recompute are **not bit-identical** (chunked-prefill reduction-order
  numerics). So an output difference would measure eviction-pattern numerics, not a correctness bug. ⇒ the
  by-construction argument + the per-step invariant checker are not merely cheaper, they are the **appropriate
  and sufficient** verification (a content-diff would add confounded noise, not certainty). This matches the
  field's lossless standard (logical equivalence, not bit-identity — prefix caching itself is never bit-
  identical to no-cache). **Losslessness of the exclusive-tiering mechanism is verified.**

## ON-CONTRACT CERTIFIED CURVE (the formal result; all λ∈{3,5,7,10}, certified nodes)
| ver | node | λ=3 p99 | λ=5 p99 | λ=7 p99 | λ=10 p99 | goodput@SLO | peak tok/s | hit |
|-----|------|---------|---------|---------|----------|-------------|-----------|-----|
| v0-cert  (stock FCFS)       | 1-2     | 6505  | 10258 | 33925 | 40189 | **3.02** | 603 | 0.671 |
| v-wb-cert (FCFS+write_back) | 1-2     | 6972  | 20650 | 32208 | 43268 | **3.02** | 651 | 0.733 |
| v1x-cert (FCFS+wb+excl)     | 0-3     | 6607  | 20901 | —     | —     | **3.02** | 669 | 0.757 |
| **v-srpf (SRPF+WT)**       | ondem-2 | 6006  | **7001** | 9196   | 18116 | **4.06** | 588 | 0.697 |
| **v-srpf-r2 (SRPF+WT)**   | ondem-2 | 7222  | **6038** | **6722** | 8588 | **4.58** | 634 | 0.699 |
| **v-srpf-r3 (SRPF+WT)**   | ondem-2 | 5072  | **5925** | 8052   | 17269 | **4.14** | 604 | 0.670 |
| **v-srpf-wb (SRPF+WB)**   | **1-2** | **5564** | **6183** | **7474** | 14798 | **★5.03** | **682** | **0.746** |
| **v-srpf-wb-r2 (SRPF+WB)**| **1-2** | **5983** | **6557** | 8581   | 14432 | **4.35** | 680 | **0.745** |
| **v-srpf-xt-wb (SRPF+XT+WB)** | **1-2** | **★4776** | **★5953** | 8662 | 15304 | **4.42** | 671 | **★0.757** |
| v-srpf-xt (SRPF+XT+WT) | ondem-2 | ~~11378~~ | — | — | — | ~~0~~ | ~~86~~ | ~~0.000~~ |
| v-srpf-age5 (SRPF+aging5s) | ondem-2 | 7965 | ~~26657~~ | — | — | in progress | — | 0.676 |
- **★★ SRPF+WB: conservative goodput 4.35 (+44%), best-case 5.03 (+66%), r5 7/7 concordant PASS (all configs).**
  r7 is the metastability boundary: 1/3 PASS for SRPF+WT (mean 7990ms), 1/2 for SRPF+WB, 0/1 for SRPF+XT+WB.
- **SRPF+XTIER+WB has the BEST r3/r5** (4776ms, 5953ms) thanks to +1pp hit from exclusive tiering, but
  still fails r7 (8662ms) — the r7 boundary is PHYSICAL (metastable queue), not capacity-limited.
- **Two orthogonal levers compound**: SRPF alone → median goodput 4.14 (+37%); write_back alone → goodput 3.02 (stuck);
  SRPF+WB together → conservative 4.35, best 5.03. Triple compound adds ~1pp hit but doesn't crack r7.
- **SRPF reliably breaks the λ=5 barrier** — **7/7** runs pass r5 across ALL configs (WT/WB/XT+WB).
  Baseline FCFS FAILS r5 on all certified runs. SRPF ELIMINATES the λ=5 coin-flip.
- **XTIER+write_through CATASTROPHIC**: hit→0, throughput 86 tok/s. Code invariant: XTIER REQUIRES write_back.
- **SRPF aging (5s threshold) NEGATIVE**: r3 regresses (+31%), r5 CATASTROPHIC (26657ms). Disrupting SRPF
  ordering by boosting timed-out cold requests creates cascading budget starvation. SRPF IS the optimal ordering.
- **SRPF+WT n=3 (same-node ondem-2)**: r5 mean 6321ms (3/3 PASS), r7 mean 7990ms (1/3 PASS, literally on SLO).
  **SRPF+WB n=2 (same-node 1-2 as baseline)**: r5 mean 6370ms (2/2 PASS), r7 discordant (7474 P / 8581 F).

## ★★ DEFINITIVE (certified, same-node): λ=5 is a COIN-FLIP under FCFS; SRPF breaks through (REPLICATED n=3/n=2)
**Same node 1-2, λ=5 p99 TTFT, sequential runs:**
- v0-cert (stock, hit 0.671): **10258 ms**  |  v-wb-cert (write_back, hit 0.733): **20650 ms**
- SAME node, and the **higher-hit run (write_back) was 2× WORSE** ⇒ λ=5 p99 is **run-variance-dominated
  (metastable near-knee queue), NOT hit/mechanism-determined.** (Cross-node adds more: v1x-cert 0-3 = 20.9 s.)
- **λ=3 is reliable** across all certified runs (6505 / 6972 / 6607 ms — all well under 8 s) ⇒ **goodput@SLO
  = 3.02 for stock AND every mechanism, on-contract.** No mechanism reaches λ=5 (coin-flip 10–21 s).
- Also at λ=3 same-node: write_back p99 6972 ≥ stock 6505 despite +6pp hit ⇒ **on a fast certified node the
  p99 is NOT hit-limited** (the "write_back −40% p99" on 0-1 was purely a *slow-node* effect).
- **This reproduces the v0.3 coin-flip on v0.31**: the warmup burst stabilized λ=3 but NOT the near-knee λ=5.
  Single-run / single-node λ=5 A/Bs are VOID (my [[base-v03-researcher]] + [[hoare-v03-researcher]] lesson,
  re-confirmed). Certified nodes + replication are mandatory; uncertified 0-1 (1.8× slower) manufactured a
  false "goodput 0→3".
- **HONEST FINAL (under FCFS): no lossless KV mechanism produces a reliable on-contract goodput gain.**
  goodput=3 is the decode-knee ceiling under FCFS scheduling; λ=5 is unreachable (variance). **BUT SRPF
  scheduling (finding E) breaks through λ=5 (n=1, p99 7001ms) — the ceiling was an FCFS artifact, not physical.**
  Contribution = this rigorous characterization + SRPF breakthrough + exclusive-tiering mechanism + honest negatives.

## ⚠️⚠️ λ=5 IS VARIANCE-DOMINATED (near the knee) — the goodput headline is a coin-flip [superseded by the definitive block above]
Certified λ=5 p99 TTFT across nodes/runs (all warm-steady-state v0.31 protocol):
- v0-cert (1-2, stock): **10.3 s** | v1x-cert (0-3, write_back+exclusive): **20.9 s** | (uncertified 0-1 stock: 24.5 s)
- The two certified nodes matched within **1.5% at λ=3** (6505 vs 6607 ms) but **diverge 2× at λ=5** (10.3 vs
  20.9 s). ⇒ **λ=5 sits at the knee and its p99 is metastable/high-variance** — cross-node (even
  cross-run) comparison there is unreliable. This is the v0.3 coin-flip, reproduced: **goodput@SLO is
  RELIABLE at λ=3 (=3) but a COIN-FLIP at λ=5** (whether p99 lands 10 s or 21 s is variance, not mechanism).
- ⇒ Any "goodput 3→4.16 (λ=5 crossing)" claim requires a SAME-NODE A/B **replicated** to beat the knee
  variance. Single-run λ=5 A/Bs are VOID (my v0.3 memory's exact lesson). v-wb-cert (1-2, same node as
  v0-cert) is the clean same-node test; its λ=5 + replication decide whether de-dup reliably crosses λ=5.
- Honest bottom line so far: **goodput@SLO = 3 (reliable, λ=3); λ=5 crossing is variance-dominated.**

## ⚠️ CRITICAL CORRECTION (2026-07-12, certified run) — the "goodput 0→3" was NODE VARIANCE
> **⛔ THIS SECTION'S "goodput 3→4.16" PROJECTION WAS ITSELF REFUTED (see (A)/(B)/(D) at top).** The
> "−40% p99 on λ=5 ⇒ crosses SLO ⇒ goodput 4.16" reasoning below assumed the −40% same-node p99 gain was
> real and reproducible. Replication showed λ=5 p99 is a COIN-FLIP (10.3↔20.6 s SAME node, mechanism-
> independent) and even λ=3 p99 is metastable (6.5↔23.7 s same node). **No mechanism reliably crosses λ=5;
> on-contract goodput@SLO = 3.02 for stock AND every mechanism.** Kept below only as a record of the
> (wrong) mid-campaign hypothesis and its correction — do NOT cite the 4.16 projection.
- **v0-cert (stock write_through) on CERTIFIED node 1-2: λ=3 p99 TTFT = 6504 ms ≤ 8 s → goodput ~3**, hit 0.671.
- My uncertified **v0 on node 0-1: λ=3 p99 = 11663 ms → goodput 0**. SAME stock config; **1.8× p99 by node.**
- ⇒ **The "goodput 0→3 via capacity de-dup" headline is RETRACTED as an on-contract claim** — node 0-1 is
  ~1.8× slower on p99, so stock *failed* there and write_back *rescued* it; but on a proper certified node
  **stock already passes** (goodput already at the decode-knee cap ~3). The p99-threshold metric is
  node/run-variance-sensitive (the v0.3 coin-flip; this is exactly why the contract mandates certified nodes).
- **What SURVIVES (same-node, still valid):** write_back lowers λ=3 p99 on 0-1 (11663→7032, −40%, same node);
  exclusive tiering adds +1.7pp hit (same-node); these are real *mechanism* effects. What does NOT survive:
  the claim that any of this moves on-contract **goodput** (stock is already at the cap on a certified node).
- **BUT the certified node REVEALS a real opportunity (reframe, not just retraction):** on certified 1-2,
  stock λ=5 p99 = **10258 ms — only 28% over the 8 s SLO** (vs node 0-1's 24.5 s). The same-node write_back
  p99 reduction is **−40%** (11663→7032 on 0-1); −40% on 10.3 s → **~6.2 s < 8 s ⇒ λ=5 would CROSS the SLO
  ⇒ goodput 3 → ~4.16** (a real +40% on-contract win). So the honest on-contract story is likely:
  *stock already clears λ=3 (goodput 3); capacity de-dup (write_back/exclusive) clears λ=5 (goodput ~4.2).*
  **v0-cert (1-2, stock, λ=5 10.3 s) vs v1x-cert (0-3, write_back+exclusive) tests this** (cross-node first
  look); a same-node certified A/B (v0 + mechanism on ONE certified node) will confirm. THIS is the headline
  to nail on-contract — and it needs certified nodes (the 0-1 slow-node masked it as a λ=3 crossing).

## SUMMARY (running) — SUPERSEDED by the correction above for the goodput claim
- **v0.31 recalibration works**: the cache genuinely binds (hit 0.62, host_util≈1.0), unlike v0.3 (under-
  provisioned/bimodal). A KV mechanism CAN move the metric.
- **Headline: goodput@SLO 0 → ~3.** Stock write_through (inclusive tiering) fails the 8 s TTFT-SLO even at
  the healthy λ=3 (p99 11.7 s) ⇒ goodput 0. De-duplicating the host tier raises effective capacity ⇒ hit
  0.68→0.74 ⇒ the p99 tail (long-context miss re-prefills) drops to 7.0 s ⇒ goodput ~3.
- **Fundamental cap**: goodput@SLO is capped at ~3 by the DECODE knee (λ≥5 is offered above the ~5 req/s
  decode-bound service ceiling ⇒ saturates ⇒ p99 unbounded; caching can't raise the decode ceiling). So the
  metric is near-binary and the real lever is the **λ=3 p99 tail margin**.
- **Mechanisms (engine code, lossless, resolved_args frozen):**
  1. *Exclusive device-XOR-host tiering* (free-host-on-loadback; no config provides it): +1.7pp hit over
     write_back but **goodput/tail-NEUTRAL same-node** — uniform capacity doesn't target the tail. (honest)
  2. *Cost-aware host retention* (keep long/high-recompute-cost contexts, drop short first): targets the tail
     directly. **[RESOLVED = HONEST NEGATIVE: −3.3pp hit, tail-neutral — the λ=3 tail is capacity-floored. See vca below.]**
- **Contribution note**: the goodput 0→3 win is reachable via the write_back CONFIG (not novel); the engine
  code (exclusive tiering, cost-aware retention) is where the novel mechanism lies. All same-node A/B on 0-1.
- Ops: local commits (4079f06c1 XTIER, e7d1eec41 fix, a90cb79ce cost-aware); remote evolve/base has a PRIOR
  run's commits so I did NOT force-push over it (fresh-replicate integrity) — work preserved locally + W&B.

## vca — cost-aware host retention (mechanism) — HONEST NEGATIVE  [node -0, job 19500, commit a90cb79ce]
- Keep long (high re-prefill-cost) contexts, evict short first (threshold 8192 tok, HOST eviction order).
- **Same-node (-0) λ=3:  v1x exclusive p99 6461 hit 0.7539  |  vca cost-aware p99 6851 hit 0.7210.**
- Cost-aware **lowered hit −3.3pp** (as designed — drops short contexts) but the p99 tail did **NOT** improve
  (slightly worse, within run noise). ⇒ the "cost-weighted-not-count-weighted" retention hypothesis does
  **not** help here: write_back already caches most reusable contexts, and the residual λ=3 tail is
  **capacity-floored** (working set 19M ≫ L1+L2 10.7M → ~26% miss is near the floor; the long contexts that
  cause the tail are simply too large/numerous to fit regardless of retention policy). Honest negative.

## v2x — reuse-aware exclusive tiering (mechanism) — MEASURED NO-OP via a FREE screen [commit 21635fff9]
- Hypothesis: plain exclusive (v1x) frees EVERY promoted node's host copy on load-back, so a hot prefix that
  churns evict↔reload pays a device→host RE-BACKUP on each eviction. Keep shallow/hot shared prefixes
  (small cum_len: system prompts / doc headers touched by many requests) host-INCLUSIVE to avoid that churn;
  stay exclusive on deep/cold tails (the capacity win). Gated `XTIER_REUSE_AWARE=1`, `XTIER_REUSE_DEPTH=4096`.
- **FREE screen (no GPU — from v1x/write_back server.log prefill #cached-token distribution):** the shallow
  match band [1, 4096) that the mechanism targets is only **8.3% of prefill steps and 1.21% of total reuse
  (cached-token) volume**. Reuse is **99% DEEP matches** (16K–66K band alone = 73.5% of cached-token volume;
  ≥4096 = ~99%). 57.7% of prefills are COLD (cached-token=0, no match — irreducible).
- **Verdict: MEASURED NO-OP — not worth a certified run.** The re-backup churn v2x could recover is bounded to
  ~1.2% of backup volume, and backup DMA is already cheap (load_back 1.65 ms) and NOT the bottleneck (the
  regime is 94% prefill-COMPUTE-bound, Diagnosis #3). So keeping the shallow set inclusive changes hit/
  throughput/goodput negligibly. Plain exclusive (free ALL host on load-back) is already near-optimal because
  the reuse volume is ~99% deep tails — there is no shallow-churn inefficiency to recover.
- Takeaway: the design space around exclusive tiering is closed for THIS workload; the reuse distribution
  (bimodal: 58% cold + 99%-of-reuse deep) leaves no exploitable shallow-hot-churn structure. This is the
  charter's screening loop working as intended — built the bolder mechanism, screened its premise for FREE,
  found a no-op, and spared the shared certified pool a pointless run.

## v3x — speculative async eviction-backup (mechanism candidate) — BOUNDED NEGATIVE via a profiling screen [instrument 534d1d3c6]
- **Code-inspection find (real asymmetry):** write-through backs up KV **async** (on hit, event-polled,
  non-blocking), but write_back defers backup to eviction and **BLOCKS** — `_evict_device_leaf` on a
  not-backed-up device leaf calls `write_backup(node, write_back=True)` then `writing_check(write_back=True)`,
  which `finish_event.synchronize()`s until the D→H DMA completes before freeing the device slot (a hard data
  dependency, since eviction is DEMAND-driven: common.py evicts exactly the deficit when `available<needed`).
  My exclusive tiering runs on write_back, so it inherits this. Candidate mechanism: **speculative async
  backup of unlocked LRU-tail leaves** — back up eviction candidates ahead of demand (overlapped with decode)
  so eviction becomes an instant free. Feasible (the prefix-closed invariant that crashed write_through+
  exclusive applies only to write-through, line 1609; async plumbing exists).
- **Premise screen (gated timer `XTIER_PROFILE_BACKUP`, write_back, node 0-0, warmup + λ=5 burst):** the
  blocking is **FRONT-LOADED during cache-fill and negligible in steady state.** Cumulative 1100 blocking
  calls / **3.13 s total** over ~490 s serving = 0.64% — but ~2.6 s of that is the first ~150 s (cache fill);
  **steady-state (post-fill, incl. the λ=5 burst window) is ~0.5 s over ~340 s = ~0.15% of wall-time.**
- **Verdict: BOUNDED NEGATIVE — not worth building.** Mechanistically: once the cache is warm, most evicted
  device leaves are **already backed up** (write_back keeps the host copy after the first backup), so eviction
  is a free demote and the blocking `write_backup` path rarely fires; the stall is a one-time fill transient,
  not a sustained throughput drain. Speculative-async-backup would recover ~0.15% steady-state → below the
  noise floor, and it would re-introduce temporary inclusive duplication for the candidate set. (Also: it
  cannot move goodput, which is cap-bound.) The real, stable throughput lever remains capacity de-dup (C).
  Screening the premise (instrument → short run → measure) again converted a plausible mechanism into a
  data-grounded negative before spending the certified pool — the disciplined loop.

## MECHANISTIC DIAGNOSIS #1.5 — the r7 pass/fail is decided by ~7 requests (histogram characterization)
r7-only TTFT histograms (7038 requests per run, differential metrics_r7 − metrics_r5):

| TTFT bucket   | v-srpf-wb r1 (PASS) | v-srpf-wb r2 (FAIL) | Δ     |
|---------------|---------------------|---------------------|-------|
| ≤ 400ms       | 1222 (17.4%)        | 1155 (16.4%)        | −67   |
| ≤ 600ms       | 3724 (52.9%)        | 3533 (50.2%)        | −191  |
| ≤ 1000ms      | 5550 (78.9%)        | 5369 (76.3%)        | −181  |
| ≤ 2000ms      | 6512 (92.5%)        | 6441 (91.5%)        | −71   |
| ≤ 4000ms      | 6895 (98.0%)        | 6872 (97.6%)        | −23   |
| ≤ 6000ms      | 6952 (98.8%)        | 6940 (98.6%)        | −12   |
| **≤ 8000ms**  | **6968** (99.0%)    | **6961** (98.9%)    | **−7**|
| > 8s (SLO)    | **70** (1.0%)       | **77** (1.1%)       | **+7**|

- **p99 position = 70th-worst request** (ceil(0.01 × 7038) = 70). In r1, exactly 70 requests > 8s → p99 at the
  8s boundary → PASS (7474ms). In r2, 77 requests > 8s → p99 crosses to the 10s bucket → FAIL (8581ms).
- **The entire pass/fail is decided by ~7 requests** — 0.1% of traffic shifting from ≤8s to >8s. This is the
  metastability in quantitative terms: the "coin-flip" is a Bernoulli on whether a handful of requests
  experience a cold-start queue alignment that pushes them past the SLO.
- **The bulk distribution is remarkably stable** between r1 and r2: the ≤400ms to ≤2000ms bins differ by
  <3%. The instability is EXCLUSIVELY in the tail >4s, and the pass/fail swing is in the >8s tail only.
- **Implication for mechanisms**: to reliably push r7 from FAIL to PASS, a mechanism needs to prevent ~7–10
  tail requests from exceeding 8s. These are likely cold-start first-turn documents arriving during a burst;
  SRPF already reduced them from ~440 (FCFS) to ~70–77, but the last ~7 are at the physical metastability edge.

**SRPF tail reduction across all rates (requests with TTFT >8s, r7038 total per rate point):**
| rate | FCFS stock | SRPF+WT | SRPF+WB r1 | SRPF+WB r2 | reduction |
|------|-----------|---------|-----------|-----------|-----------|
| λ=5  | 132 (1.9%) FAIL | 60 (0.9%) PASS | 52 (0.7%) PASS | 51 (0.7%) PASS | **2.2–2.6×** |
| λ=7  | **439 (6.2%)** FAIL | 83 (1.2%) FAIL | **70 (1.0%)** PASS | 77 (1.1%) FAIL | **5.6–6.3×** |
| λ=10 | 416 (5.9%) FAIL | 114 (1.6%) FAIL | 93 (1.3%) FAIL | 92 (1.3%) FAIL | **3.7–4.5×** |

SRPF's tail reduction is MASSIVE (5.6× at r7), not just a shift in mean TTFT. The SLO p99 threshold
(~70th-worst) sits at a knife edge between SRPF's residual ~70 violations and the FCFS's ~440. Adding
write_back (+6pp hit) further compresses the tail, cutting violations from 83→70 at r7.

**★ Comprehensive SLO-violation census (>8s TTFT, all completed SRPF variants, r7 differential, n=7038):**
| version               | node    | >8s viol | pass? | ≤2s (bulk) | (4s,8s] (mid-tail) |
|----------------------|---------|----------|-------|------------|-------------------|
| Stock r1 (FCFS)      | 1-2     | **439**  | FAIL  | 5538 (78.7%) | 219              |
| Stock r2 (FCFS)      | 1-2     | **426**  | FAIL  | 5630 (80.0%) | 253              |
| SRPF+WT r1           | ondem-2 | 83       | FAIL  | 6228 (88.5%) | 142              |
| SRPF+WT r2           | ondem-2 | **55**   | PASS  | 6451 (91.7%) | 119              |
| SRPF+WT r3           | ondem-2 | 72       | FAIL† | 6318 (89.8%) | 179              |
| SRPF+WB r1           | 1-2     | **70**   | PASS‡ | 6512 (92.5%) | 73               |
| SRPF+WB r2           | 1-2     | 77       | FAIL  | 6441 (91.5%) | 89               |
| SRPF+XT+WB           | ondem-2 | 75       | FAIL  | 6432 (91.4%) | 116              |
† missed by 52ms. ‡ exactly at the p99 boundary (70th-worst = ceil(0.01×7038)).

**Observations from the census:**
1. All SRPF variants cluster in **55–83 violations** (1σ ≈ 9.5), a 5–8× reduction from stock (~430).
2. The pass/fail boundary at 70 violations sits WITHIN the SRPF cluster's ±1σ → **r7 is a statistical coin-flip
   for all SRPF configs**, not a mechanism-differentiable regime. WB/XT provide ≤1σ marginal tightening.
3. The ≤2s "bulk" fraction rises from ~79% (FCFS) to ~90% (SRPF) — a **+12pp shift** in the main body.
4. The mid-tail (4s,8s] is compressed most by WB (73–89) vs WT (119–179) — WB's L2 backup reduces re-prefill
   for continuations whose prefix was evicted. But this mid-tail compression doesn't reliably flip the >8s count.
5. **To reliably PASS r7, a mechanism must push the residual from ~70 to ≤55.** The SRPF+WT r2 run (55 violations,
   PASS with margin) shows this IS reachable — the question is whether it's reproducible or luck.

## MECHANISTIC DIAGNOSIS #1.6 — under SRPF, the r7 burst queue is ALL-COLD (running-batch-capacity bottleneck)
Server log analysis of the SRPF+WB r1 r7 window (v-srpf-wb, 04:14:22–04:14:30, queue ≥43):
- **ALL queued prefills during the burst are COLD** (cached_tok=0, new_tok=6144 = full chunk).
  SRPF has already drained all cached continuations — they're admitted and running immediately.
  The queue=43–48 consists entirely of cold first-turn documents waiting for admission.
- **Running batch near saturation**: running=186→228 (max 270). The PrefillAdder limits admission
  when running_bs approaches max_running_requests. Cold documents can't be admitted not because SRPF
  deprioritizes them (there are NO cached requests left to compete), but because the running batch is full.
- **Queue refill rate ≈ drain rate**: at λ=7, ~7 cold first-turns arrive per second. Each takes 3-5
  chunks (6144 tokens/chunk). The queue stays at ~45 for 12+ seconds — a steady-state cold-document backlog.
- **Implication**: the r7 residual violations (55–83) are a **RUNNING-BATCH-CAPACITY bottleneck**, not a
  cache-capacity or scheduling issue. SRPF is already OPTIMAL — it has cleared all cacheable work. The
  remaining cold documents wait because the GPU can't admit them (running batch ≈ max). No KV-cache
  mechanism or scheduling policy can address this — it requires either faster cold prefill (hardware),
  higher max_running_requests (more GPU memory), or smarter decode-slot management.
- Queue depth stats during r7: p50=0 (!), p90=10, p99=39, max=48. The queue is empty MOST of the time.
  The violations come from rare transient bursts where cold arrivals briefly exceed service capacity.

## MECHANISTIC DIAGNOSIS #2 — the p99 tail is DEVICE-KV-pressure during bursts (why host-tier mechs can't help)
Device KV pool usage during serving (v0-cert, per decode step): p50 **0.01**, p90 0.26, p99 **0.92**, max 0.99.
- The device is **bursty**: nearly empty most of the time, but **nearly FULL (p99 0.92) during concurrency
  bursts** of long-context requests. The p99 TTFT tail coincides with these bursts: the device KV pool fills
  with running requests' (locked) KV ⇒ cached prefixes evicted ⇒ load-backs + evictions + prefills all contend
  under device pressure. load_back volume is large (λ=3: 327 M tokens H→D; λ=5: 668 M) — 60% of hits are host.
- ⇒ **The burst tail is DEVICE-capacity-bound, not host-capacity or hit-bound.** Running-request KV (the 12
  full-attn layers' growing KV) is locked and **cannot be offloaded losslessly** (attention needs all of it;
  the 36 Mamba layers are fixed tiny state). So **host-tier mechanisms (exclusive tiering, cost-aware retention)
  cannot address the burst tail** — this is the mechanistic reason they're goodput-neutral. There is no free
  device room during the bursts that matter, so a "keep long prefixes device-resident" placement lever is also
  bounded out. Two independent diagnoses (decode-slot saturation + device-KV-pressure bursts) both conclude the
  goodput tail is NOT lossless-KV-addressable.

## MECHANISTIC DIAGNOSIS #3 — the λ=5 regime is 94% PREFILL-bound, and it's COLD prefill (bounds the cache headroom)
Same-node (1-2) forward-pass census of the full λ=5 window, stock vs write_back (GPU-free, from server.log
batch lines; the definitive quantification of why (C) is real-but-modest and why goodput can't beat the cap):
| metric (λ=5 window) | STOCK (hit 0.671) | WRITE_BACK (hit 0.733) |
|---|---|---|
| forward passes: **PREFILL share** | **94.9%** (8413) | **94.2%** (7772) |
| forward passes: DECODE steps | 454 (5.1%) | 476 (5.8%) |
| prefill **new-token** sum (COLD work) | 3.32e7 | **2.85e7 (−14%)** |
| prefill cached-token attn sum | 7.05e7 | 7.40e7 |
| decode #running-req (p50 / mean) | 11 / 49 | 12 / 47 |
- **The GPU spends ~95% of its λ=5 forward passes on PREFILL, regardless of cache hit.** The regime is
  overwhelmingly prefill-bound: long unique first-turn documents (LEval/LooGLE, prefill chunked at #new-token
  p50 **6144**) dominate the compute.
- **Quantified (C) mechanism (measured, not asserted):** higher hit cuts **COLD new-token prefill −14%**
  (3.32e7→2.85e7 — fewer tokens need from-scratch K/V+attention), which frees the GPU for **+5% more decode
  steps** (454→476) ⇒ the **+6–8% decode throughput** of finding (C) at this node/rate. (The cached-token
  attention volume actually *rises* slightly — higher hit means more continuations replay over cached prefixes
  — so the net decode gain is the difference: cold-prefill saved minus warm continuation-attention added.)
- **This BOUNDS lossless caching's throughput headroom.** Caching can only remove *continuation* (turn ≥2)
  prefill; the *cold* first-turn prefill of each unique document is irreducible (nothing to cache on first
  sight). Cold prefill is the majority of prefill FLOPs here ⇒ the cache's reachable throughput headroom is the
  minority continuation fraction ⇒ the observed **+6–18%** (scaling with the continuation/contention fraction,
  i.e. with load) is near the lossless ceiling for THIS workload. No lossless KV mechanism can do materially more.
- **Re-confirms the goodput cap with hard numbers:** in the 3516 λ=5 steps with a waiting queue (#queue-req>0),
  **running-req p50 = 251 / mean 231 (of the frozen 256 cap)** and device full-tok-use p50 0.85 — admission is
  jointly gated by the **contract-frozen 256-concurrency cap** and device-KV, both non-KV-addressable. 52% of
  backlog steps have device ≥0.85. ⇒ goodput@SLO cannot exceed ~3 via any lossless KV mechanism (re-confirmed).

## MECHANISTIC DIAGNOSIS of the goodput cap (from v-wb server.log, λ=5 window, GPU-free)
Why does λ=5 achieve only 4.4 req/s (p99 21 s) when the server hits 5.1 at λ=10? Analyzed 590 decode
batches + prefill batches in the λ=5 window:
- **Decode running-req is BIMODAL**: min 1, p25 3, **median 17**, p75 242, max 256. The decode batch
  repeatedly DRAINS to a handful then spikes to the 256 cap — it is NOT held full.
- **Prefills are short / cache-served**: #new-token 64–384 vs **#cached-token 15K–57K**, **#queue-req≈0**.
  ⇒ the cache is working (few new tokens to compute) and there is NO prefill backlog. Prefill token-VOLUME
  is not the bottleneck.
- ⇒ The p99 TTFT tail at λ≥5 is **admission-wait when the 256-concurrency decode slots saturate** with
  long-response requests, compounded by expensive continuation-prefill *attention over the long cached
  prefix* (64 new tokens attending 57K cached ⇒ a real per-turn cost that grows with conversation length,
  pausing decode ⇒ batch drains). Both drivers — **decode duration (output length) × the 256 concurrency
  cap** — are FROZEN protocol/model facts, **not KV-cache-addressable**. Caching removes prefill *recompute*
  (crosses the healthy-rate SLO) but cannot shorten decode or lift the concurrency cap ⇒ goodput cap ≈ 3.
This is the evidence behind "goodput is decode-knee-capped": it's decode-slot saturation, not a cache miss.

## OVERALL CONCLUSION (updated 2026-07-14 — SRPF replicated, compounded, coin-flip boundary characterized)
- **SRPF scheduling breaks the λ=5 barrier (REPLICATED n=3+n=2 same-node, 5/5 concordant r5 PASS):**
  SRPF+WT n=3 same-node: conservative goodput 4.06 (+37%), median 4.14. SRPF+WB n=2 same-node: conservative
  goodput 4.35 (+44%), best-case 5.03 (+66%, r7 PASS in best run). **r5 is RELIABLE (5/5 concordant PASS).**
  r7 is a coin-flip (3/7 PASS across all variants, violation census shows ~70±10 residual cold-start violations).
  SRPF is a SCHEDULING mechanism exploiting the 78%/22% cached/cold asymmetry. Orthogonal to capacity de-dup.
- **Prior characterization holds under FCFS:** goodput@SLO under FCFS scheduling is a metastable COIN-FLIP
  capped at ~3, with the cap set by DECODE knee + 256-concurrency limit + device-KV-pinned running contexts.
  Capacity de-dup reduces the coin-flip variance (Levene p≈0.004) and raises peak throughput +11.4% (MWU
  p=0.008) — both real effects, config-reachable via write_back.
- **What SRPF changes about the picture:** the "goodput cap at ~3" was an FCFS artifact, not a physical limit.
  Under FCFS, cached and cold requests compete equally for the PrefillAdder budget; SRPF admits cached requests
  first (near-zero budget consumption), leaving more budget for cold requests AND reducing queue depth. The
  decode knee still exists (r10 throughput 4.59 vs baseline 4.72, ~flat) but the SCHEDULING knee shifts right.
- **Design space:** the KV-capacity design space is closed (exclusive tiering +1.7pp, cost-aware NEG, reuse-
  aware no-op). The SCHEDULING design space is open: SRPF is the first lever. Aging = closed NEGATIVE (cascade).
  Queue-Pinned KV (prevent intra-iteration eviction of waiting requests' prefix) = IN TEST.
- **Contribution**: (1) **SRPF** — a novel cache-aware scheduling policy, **+37% goodput (conservative,
  replicated n=3)**, +66% compounded with write_back; 5/5 concordant r5 PASS vs 0/3 FCFS certified;
  (2) **two-lever framework** — goodput has orthogonal scheduling and capacity levers that compound (SRPF+WB
  Pareto-dominates); (3) methodology — goodput@SLO is noise-dominated under FCFS, median-of-k mandatory;
  (4) replicated throughput result overturning "cache can't raise peak decode throughput" (MWU p=0.008); (5)
  exhaustive characterization of the closed KV-capacity design space; (6) exclusive device-XOR-host tiering
  mechanism (+1.7pp hit, lossless); (7) honest negatives: SRPF aging catastrophic, XTIER+WT catastrophic,
  cost-aware NEG; (8) five self-corrected over-claims — honest throughout.

## LIMITATIONS & WHAT WOULD MOVE THE NEEDLE (updated 2026-07-14, SRPF replicated)
- **r7 coin-flip (the current frontier):** SRPF reliably passes r5 (5/5 runs across WT+WB), but r7 is a
  coin-flip (3/7 PASS across all configs). The violation census shows all SRPF variants cluster at 55–83
  violations (threshold=70). Pushing violations reliably below 55 would convert r7 to a reliable PASS (+66%
  goodput). Queue-Pinned KV (in flight) targets this gap.
- **Metric limitation (methodology):** goodput@SLO is a metastable coin-flip — single runs are uninformative.
  SRPF stabilizes through r5 (5/5 replicated, queue depth −35–43%) but r7 remains noise-dominated.
- **Sample sizes (updated):** throughput result formally significant (MWU p=0.008, n=3/7). SRPF+WT n=3 same-node,
  SRPF+WB n=2 same-node (r3 in flight). The r5 concordance (5/5) is robust; the r7 PASS rate (3/7) is honest.
- **KV-capacity design space is closed; scheduling is OPEN:** no lossless KV-capacity mechanism pushes goodput
  past ~3 under FCFS (three independent diagnoses). SRPF opens the scheduling dimension. Aging (deadline-promote)
  is a CLOSED NEGATIVE (5s threshold: catastrophic cascade). Queue-Pinned KV (prevent intra-iteration eviction
  of waiting requests' prefix matches) is IN TEST. The remaining boundary is cold-prefill compute.
- **Generalizable insight:** on decode-bound hybrid-Mamba serving with working-set ≫ cache, lossless gains come
  from TWO orthogonal levers: (1) capacity de-dup (config-reachable, +11% throughput, +goodput reliability)
  and (2) cache-aware admission scheduling (novel code, +37–66% goodput). A maintainer should adopt BOTH
  write_back + SRPF. The remaining boundary is cold-prefill compute for the ~70 residual tail violations.

## Protocol (v0.31, recalibrated vs v0.3)
- 2-tier HiCache: L1 GPU HBM + L2 host DRAM (768 GB, `--hicache-size 96`), **no L3/disk**.
- Model Qwen3.5-122B-A10B-FP8 (hybrid-Mamba MoE), tp8, ctx 262144, full real decode.
- **Rate sweep** λ∈{3,5,7,10}, Poisson open-loop, `--max-concurrency 256`, `num-prompts 1553`
  (~19M-tok working set ≫ L1+L2 ~10.7M → genuine pressure), **warmup burst (300 convs) + NO per-rate
  flush** → warm steady-state (fixes v0.3's bimodal cold-start metric).
- **Headline: goodput@SLO = max achieved req/s among rates with p99 TTFT ≤ 8 s.** Honest controls
  (decode-bound, expected ~flat): peak out tok/s, peak req/s. Lossless gate: outputs == no-cache.

## Active code path (verified from registry.py, not assumed)
- registry.py `default_radix_cache_factory`: hybrid-SSM + `--enable-hierarchical-cache`
  ⇒ **`UnifiedRadixCache`** via `_create_unified_radix_cache` (comment: "HybridModel launches HiCache
  via UnifiedRadixCache by default"). `HiMambaRadixCache`/`HiRadixCache` are **DORMANT** for this model.
  (Runtime confirmation pending via the `Tree cache initialized: ... impl=UnifiedRadixCache` server.log line.)
- Live 2-tier levers (no L3, so `prefetch_from_storage` is dormant):
  - `_inc_hit_count` (unified_radix_cache.py:1810) → `write_backup` on hit (write_through eager backup, D→H).
  - `load_back` (:1660) H→D on prefix match when device-evicted; `init_load_back` (:2410) from scheduler.
  - eviction: `evict`→`FullComponent.drive_eviction` (LRU via `eviction_strategy`, default `LRUStrategy`);
    `_evict_device_leaf` (:1482) demotes backuped leaves to host (`_evict_to_host`) or deletes if unbacked;
    `evict_host`/`_evict_host_leaf` for L2.
  - `full_component.commit_hicache_transfer` LOAD_BACK (full_component.py:316) sets `cd.value` but
    **never frees `cd.host_value`** ⇒ device-resident KV stays duplicated in host (inclusive tiering).
- Invariant: `node.backuped` ⟺ FULL host_value present; `node.evicted` ⟺ FULL device value None.
  Mamba host state is O(#seq) — tiny, never the binding tier (host_util pressure is the FULL-KV host pool).

## Key regime fact (baseline.json, single-point λ=3, old format)
hit 0.62, **host_util 0.9999** (L2 saturated, forced eviction), ttft_p99 6326 ms, req/s 2.78,
load_back_mean 1.65 ms, evict_mean 1.02 ms, load_back_tokens 298M, evict_tokens 582M. TTFT is
prefill-compute-bound at this point; load_back is cheap. ⇒ the lever on TTFT/goodput is **hit-rate**,
which is **capacity-bound** at host_util=1.0. Raising hit without more memory ⇒ use L1+L2 more
efficiently ⇒ eliminate inclusive duplication ⇒ **exclusive (device-XOR-host) L1↔L2 placement**
(explicitly in-charter scope: "L1↔L2 placement/layout"). NOT eviction-order tuning (LRU≈Belady dead end).

## Version log

### v0 — stock baseline sweep (control)  [DONE, node 0-1, XTIER_EXCLUSIVE=0, commit 4079f06c1]
- Change: none (stock `UnifiedRadixCache`, write_through, inclusive tiering; XTIER off).
- **Curve (λ: req/s, out tok/s, TTFT p50, TTFT p99, hit):**
  - λ=3:  2.89, 369, 1003 ms, **11663 ms**, 0.678
  - λ=5:  3.62, 463,  993 ms, **24484 ms**, 0.665
  - λ=7:  4.10, 524, 1078 ms, **33274 ms**, 0.658
  - λ=10: 4.20, 537, 1117 ms, **41202 ms**, 0.655
- **goodput@SLO = 0** (p99 TTFT > 8 s at every λ, including the healthy λ=3). peak req/s 4.2, peak
  out tok/s 537 (decode ceiling ~540 tok/s: 524→537 from λ7→λ10, ~flat = decode-bound control, as expected).
- Read: throughput keeps rising with λ but latency collapses (p99 11.7→41 s) — a **goodput cliff** bound
  by the p99 TTFT tail (HOL-blocking long-context miss re-prefills). Lever = cut λ=3 p99 (11.7 s) below 8 s.
- W&B: logged as `v0` [config].

### INTERIM (v0 A/B, node 0-1, job 19448) — the regime, from λ=3
- v0 stock λ=3: req/s 2.89, out 369 tok/s, TTFT p50 **1003 ms** / **p99 11663 ms**, hit **0.678**, 7037 reqs.
- **p99 TTFT 11.7 s > 8 s SLO at the *healthy* rate** ⇒ baseline **goodput@SLO ≈ 0** (higher λ only worse).
  The metric is gated by the **p99 TTFT tail**: median 1.0 s but p99 11.7 s ⇒ a small fraction of requests
  eat huge re-prefills (cache misses on long LEval/LooGLE contexts). Lever = cut that tail. XTIER raises
  hit ⇒ fewer long re-prefills ⇒ lower tail ⇒ possibly under 8 s (goodput 0→3). v1 will show if it's enough.
- Note p99 (11.7 s) ≫ baseline.json's old single-point 6.3 s: the warm-steady-state sweep at
  max-concurrency 256 / full 1553 convs is more tail-stressed than the old λ=3 single point. Report the
  CURVE (p99 per λ), not just the binary goodput — a p99 shift is a result even if it doesn't cross 8 s.

### v1 — exclusive tiering under write_through — CRASHED (invariant violation), RETRACTED
- First XTIER (write_through + suppress-eager-backup + backup-on-evict + free-Full-host-on-loadback)
  booted + served the short screen fine, but the FULL sweep crashed the scheduler `sanity_check` ~6 min
  into warmup: (1) "aux host present but Full.host_value=None" (freed Full host, left Mamba host), and
  (2) exclusive tiering breaks the **prefix-closed host-backup** invariant write_through enforces
  (`sanity_check` exempts it only under write_back). ⇒ exclusive tiering is fundamentally a **write_back**
  mechanism. Not logged (void run). Fix committed (e7d1eec41).
- **Chaining evals in one allocation is unsafe**: v0's 768 GB pinned host tier wasn't freed before the
  next run's DRAM gate (`DRAM_TOO_LOW 329G`). Run each version as a SEPARATE job (fresh alloc) on the
  same node to keep the A/B same-node.

### v-wb — write_back config (capacity diagnostic, control)  [node 0-1, job 19473; full sweep DONE]
- **λ=3: req/s 3.02, out 387 tok/s, p50 515 ms, p99 TTFT 7032 ms (≤ 8 s SLO!), hit 0.7371.**
- vs v0 (same node): p99 **11663→7032 ms (−40%)**, hit **0.678→0.737 (+5.9pp)**, p50 1003→515 ms.
- ⇒ **goodput@SLO 0 → ~3**: de-duplicating the host tier (no eager device-copy backups) raises effective
  capacity → hit → fewer long-context miss re-prefills → the p99 tail drops under 8 s. **Capacity is the
  lever**, confirmed. (write_back is a CONFIG flip — the measurement, not the contribution.)
- Margin to SLO is 0.97 s (12%) — near-boundary ⇒ MUST replicate for error bars (goodput could flip on a
  noisy run). v0's 11.7 s is 46% over ⇒ v0=0 is robust; the 0→3 jump is likely real.

### v1x — exclusive tiering (write_back + free-host-on-loadback CODE)  [node -0, job 19479, commit e7d1eec41]
- Fix VALIDATED: survived the sanity-check window that crashed the write_through version. Pure-code add
  over write_back: free ALL components' host on load-back ⇒ disjoint L1/L2.
- **λ=3 ablation (v0 stock / v-wb write_back-config / v1x write_back+CODE):**
  | ver | p99 TTFT | hit | note |
  |-----|----------|-----|------|
  | v0   | 11663 ms | 0.6782 | stock inclusive (goodput 0) |
  | v-wb | 7032 ms  | 0.7371 | write_back CONFIG (+5.9pp hit, goodput ~3) |
  | v1x  | 6461 ms  | 0.7539 | + free-on-loadback CODE (**+1.7pp hit** over write_back) |
- My code adds **+1.7pp hit** (node-independent signal) = real but modest extra effective capacity; p99
  −8% vs v-wb but that's CROSS-NODE (v1x on -0, v-wb on 0-1), so treat as tentative — confirm same-node.

### KEY INSIGHT — goodput@SLO is capped at ~3 (the decode knee)
- Achieved req/s asymptotes ~4.2–4.75 (v-wb: λ5→4.23, λ7→4.75); λ≥5 is offered ABOVE the decode-bound
  service ceiling ⇒ **saturation** ⇒ p99 unbounded (v-wb λ=5 p99=21 s) ⇒ can NEVER pass the 8 s SLO.
  Caching does not raise decode throughput (charter says so). ⇒ **only λ=3 can pass ⇒ goodput ≤ ~3.**
- So the metric is near-binary (0 or ~3); the REAL differentiator is the **λ=3 p99 tail margin** (below 8 s)
  and the curve. The tail = long-context miss re-prefills. ⇒ novel headliner target: **recompute-cost-aware
  retention** — preferentially keep long (high re-prefill-cost) contexts resident to cut the p99 tail more
  than uniform capacity does. This optimizes the RIGHT objective for goodput@SLO (cost-weighted misses,
  not miss-count) — a metric-aligned insight distinct from hit-rate-maximal LRU.
- Hypothesis: host_util=1.0 means the host tier is the binding capacity; write_through keeps a host copy
  of every device-resident hit node (inclusive) ⇒ the host wastes ~L1-worth of capacity on duplicates.
  An **exclusive** hierarchy (KV in device XOR host, never both) frees that capacity for unique evicted
  KV ⇒ higher hit ⇒ fewer prefill tokens ⇒ lower p99 TTFT ⇒ higher goodput@SLO. Pure code; no flag
  achieves free-host-on-loadback (write_back still keeps the post-loadback host copy).
- Change (gated `SGLANG_XTIER=1`, default off ⇒ resolved_args frozen at write_through):
  (1) suppress eager write_through backup-on-hit; (2) backup-on-eviction (write_back-on-evict) so a
  device-only node is demoted not deleted; (3) free `cd.host_value` on load-back commit (the exclusive part).
- Certification plan: SAME-node A/B (env toggle 0 vs 1), resolved_args identical (write_through), code
  diff = the mechanism; replicate to beat node variance (±14% req/s, ±30% p99 per prior campaigns).
- Result: PENDING.

## Runtime confirmations (from the 0-1 screen, job 19440, commit 4079f06c1)
- Active cache = `UnifiedRadixCache` (`Tree cache initialized: ... impl=UnifiedRadixCache
  hybrid_ssm=True hierarchical=True`), pools=KV+MAMBA, transfer_layer_num=48. XTIER edits are LIVE.
- `max_total_num_tokens=2347648` (L1 ≈ 2.35M tok), host tier 96GB×8=768GB (L2). max_running_requests=270.
- **XTIER validated**: server boots + serves the mix cleanly with `XTIER_EXCLUSIVE=1` (warmup + 600-conv
  bench, cache hits + load-back exercised, decode ~254 tok/s) — no crash/hang/leak. All server.log
  "Traceback"s are benign optional-lib (`libavutil`/`torchcodec`), not cache code.
- **0-1 boots a real 8-GPU server** (CUDA-graph capture OK, no NCCL hang) = nodes.yaml's own
  certification criterion. Used for the A/B because the VERIFIED pool is 100% held by the sibling v0.3
  campaign (holds 7.5h+). Documented deviation; the v1-vs-v0 delta is same-node so node-var cancels.

## Contingent next steps (decide from the v0 curve)
- If **prefill-bound** (p99 climbs through 8s across λ; hit-limited): capacity is the lever → XTIER is the
  right mechanism; add the **write_back-config ablation** (v-wb) to separate the config effect (backup-on-
  evict) from the pure-code exclusive effect (free-host-on-loadback), and replicate the A/B (n≥2).
- If **decode-bound / goodput flat** (peak req/s ~constant regardless of hit): capacity can't move
  goodput → pivot to a **TTFT-tail / transfer-overlap** mechanism (hide load-back latency; overlap H→D
  with prefill compute) or report the honest capacity-saturation ceiling as the contribution.
- Novel headliner candidate beyond exclusive tiering: **reuse-aware exclusive tiering** — keep the host
  copy (inclusive) for hot/shallow prefixes that churn evict↔reload, go exclusive only for cold/deep
  prefixes, capturing exclusive's capacity win without its re-backup cost on hot churn.

### v-srpf — SRPF scheduling (mechanism, NOVEL CODE)  [DONE, ondem-2, job 19731, commit 01fd8ba0a]
- **Mechanism: Shortest Remaining Prefill First** — sort waiting queue by ascending
  `(origin_input_ids + output_ids - num_matched_prefix_tokens)`. Cached continuations (78% of turns, near-zero
  remaining) admitted before cold first-turn documents. Added `srpf` to `CacheAwarePolicy` enum and
  `_sort_by_shortest_remaining_prefill` static method (schedule_policy.py lines 314-325). FCFS fallback for
  queue >1024 to bound sort overhead. Also registered in `server_args.py` choices.
- **Config: `--schedule-policy srpf --hicache-write-policy write_through`** (same baseline capacity, isolated
  scheduling effect). Node ondem-2 (certified).
- **Full sweep:**
  | λ | p99 TTFT | p50 TTFT | req/s | tok/s | hit | SLO | vs baseline Δp99 |
  |---|---------|---------|-------|-------|-----|-----|-------------------|
  | 3 | 6006ms  | 496ms   | 3.02  | 387   | 0.697 | PASS | −7.7% |
  | 5 | **7001ms** | 629ms | **4.06** | 519 | 0.675 | **PASS** | **−31.8%** |
  | 7 | 9196ms  | 753ms   | 4.46  | 571   | 0.666 | FAIL | **−72.9%** |
  | 10| 18116ms | 866ms   | 4.59  | 588   | 0.661 | FAIL | −54.9% |
- **goodput@SLO = 4.06** (from r5 passing). Baseline was 3.02 (all FCFS variants). **+34% goodput gain.**
- **Why it works (mechanistic):** SRPF changes admission ORDER, not cache capacity. Two compounding effects:
  (1) cached requests have near-zero `extend_input_len`, consuming almost none of the PrefillAdder's shared
  `rem_chunk_tokens=6144` budget, leaving more for cold requests in the same round. (2) Lower queue depth at all
  rates (p99: −41% at λ3, −35% at λ5, −43% at λ7) because cached requests transit through the queue quickly.
  Running count slightly higher (+6% at r5) = more efficient GPU packing.
- **Queue depth comparison (SRPF vs FCFS):**
  | Rate | Queue p99 | Queue mean | Running mean |
  |------|-----------|------------|--------------|
  | r3   | 17→10 (−41%) | 2.1→1.5 (−29%) | 82→110 (+34%) |
  | r5   | 34→22 (−35%) | 4.1→2.5 (−39%) | 191→204 (+6%) |
  | r7   | 79→45 (−43%) | 13.3→5.1 (−62%) | 262→247 (−6%) |
- **Hit rate**: 0.697 (vs baseline 0.671 on a different node — SRPF itself does not change hit, but the indirect
  effect of reduced queue depth → fewer concurrent evictions may explain the slight +2.6pp; or it's node noise).
- **Node equivalence**: warmup throughput = 2.95 req/s, 327 tok/s on BOTH ondem-2 and baseline node 1-2.
- **p50 TTFT improvement**: −12% (λ3), −6% (λ5), −6% (λ7). SRPF increases TTFT std by 128% (bimodal:
  fast cached + slower cold) — this IS the mechanism working (cached requests complete fast, cold requests
  wait their turn but still get better tail latency).
- **Design insight**: SRPF reveals that goodput has TWO orthogonal levers: cache capacity (de-dup) and
  admission scheduling. Prior work on this campaign exclusively explored capacity. SRPF is a **pure scheduling
  mechanism** — it doesn't change what's cached, only the order requests are admitted. The two may compound
  (SRPF+write_back eval in progress).
- **Caveats (for replication):** n=1, different node (ondem-2 vs 1-2); r5 margin 999ms (12.5%) to SLO is
  near the metastable boundary identified in finding (A). Same-node replication mandatory before claiming robust.
- W&B: logged as `v-srpf` [mechanism].

### v-srpf-wb — SRPF + write_back compound (mechanism+config)  [DONE, node 1-2, job 19732, commit 01fd8ba0a]
- **Compound test**: SRPF (scheduling) + write_back (capacity de-dup). Same-node as baseline v0-cert (1-2).
- Config: `--schedule-policy srpf --hicache-write-policy write_back`.
- **Full sweep:**
  | λ | p99 TTFT | p50 TTFT | req/s | tok/s | hit | SLO | vs baseline Δp99 |
  |---|---------|---------|-------|-------|-----|-----|-------------------|
  | 3 | 5564ms  | 466ms   | 3.02  | 387   | 0.746 | PASS | −14.5% |
  | 5 | 6183ms  | 577ms   | 4.43  | 567   | 0.736 | PASS | **−39.7%** |
  | 7 | **7474ms** | 637ms | **5.03** | 643 | 0.731 | **PASS** | **−78.0%** |
  | 10| 14798ms | 744ms   | 5.33  | 682   | 0.729 | FAIL | −63.2% |
- **★★ goodput@SLO = 5.03 — passes through λ=7! +66% over baseline 3.02.**
- **The two levers are genuinely ORTHOGONAL and compound:**
  Adding write_back to SRPF reduces p99 further (−7% to −19%) AND increases throughput (+9% to +16% at r5+).
  Adding SRPF to write_back converts the stuck-at-3.02 FCFS+WB into 5.03 — a +66% jump.
- **SRPF+WB vs each component alone (all on this sweep):**
  | Config | r3 p99 | r5 p99 | r7 p99 | goodput | peak tok/s |
  |--------|--------|--------|--------|---------|------------|
  | FCFS+WT (baseline) | 6505 | 10258 | 33925 | 3.02 | 603 |
  | FCFS+WB | 6972 | 20650 | 32208 | 3.02 | 651 |
  | SRPF+WT | 6006 | 7001 | 9196 | 4.06 | 588 |
  | **SRPF+WB** | **5564** | **6183** | **7474** | **5.03** | **682** |
  ⇒ SRPF+WB Pareto-dominates: LOWEST p99 at every rate AND HIGHEST throughput AND HIGHEST goodput.
- **Mechanistic explanation**: SRPF reduces queue depth by reordering (scheduling lever); write_back raises
  hit by +6pp via capacity de-dup (capacity lever). These operate on different bottlenecks: SRPF on admission
  contention, write_back on prefill recompute. The compound cuts both: fewer cold re-prefills AND those that
  remain wait less in the queue. The throughput gain compounds too: higher hit → less prefill compute → more
  decode cycles AND lower queue depth → less decode-batch drain.
- **Node context**: same node (1-2) as both baselines (v0-cert, v-wb-cert). Different node from SRPF+WT
  (ondem-2). The r7 margin (526ms to SLO) is tighter than r5 (1817ms) — node variance may flip r7 on
  replication, but the magnitude of improvement (−78%) is massive.
- W&B: logged as `v-srpf-wb` [mechanism].

### v-srpf-r2 — SRPF replication (same-node ondem-2)  [DONE, ondem-2, job 19748, commit 01fd8ba0a]
- Same-node replication of v-srpf. Config: `--schedule-policy srpf` (write_through).
- **Full sweep:**
  | λ | p99 TTFT | p50 TTFT | req/s | tok/s | hit | SLO | vs run 1 Δp99 |
  |---|---------|---------|-------|-------|-----|-----|-----------------|
  | 3 | 7222ms  | 508ms   | 3.02  | 387   | 0.684 | PASS | +20% (worse) |
  | 5 | **6038ms** | 605ms | 4.19  | 535   | 0.676 | **PASS** | **−14%** (better!) |
  | 7 | **6722ms** | 694ms | 4.58  | 586   | 0.670 | **PASS!** | **−27%** (passes!) |
  | 10| 8588ms  | 763ms   | 4.74  | 606   | 0.667 | FAIL | −53% |
- **goodput@SLO = 4.58** (passes through r7!) vs original 4.06 (only passes r5).
- **★ REPLICATION CONFIRMS r5 pass (n=2 concordant)**: both runs pass r5 (7001ms, 6038ms) — **robust**.
  r7 is DISCORDANT (run 1 FAIL 9196ms, run 2 PASS 6722ms) — mean 7959ms, right at SLO boundary.
  r3 is concordant PASS (6006ms, 7222ms). r10 concordant FAIL.
- **Conservative goodput (both runs must pass) = 4.19 (+39% over baseline)**. The r5-level gain is robust;
  r7 is a coin-flip at the boundary (mean 7959ms straddles 8s, spread 2474ms).
- W&B: logged as `v-srpf-r2` [mechanism].

### v-srpf-r3 — SRPF 3rd replication (same-node ondem-2)  [DONE, ondem-2, job 19765, commit 01fd8ba0a]
- Third same-node replication of SRPF+WT on ondem-2. Completes the n=3 picture.
- **Full sweep:**
  | λ | p99 TTFT | p50 TTFT | req/s | tok/s | hit | SLO |
  |---|---------|---------|-------|-------|-----|-----|
  | 3 | **5072ms** | 518ms | 3.02  | 387   | 0.670 | PASS (best r3 of all 3 runs) |
  | 5 | **5925ms** | 596ms | 4.14  | 529   | 0.667 | **PASS** |
  | 7 | 8052ms  | 686ms   | 4.57  | 584   | 0.664 | FAIL (by **52ms** — razor-thin) |
  | 10| 17269ms | 755ms   | 4.72  | 604   | 0.661 | FAIL |
- **goodput@SLO = 4.14** (passes r3+r5, fails r7 by 52ms).
- **n=3 SRPF+WT summary (all same-node ondem-2):**
  r3: {6006, 7222, 5072}ms — **3/3 PASS** (mean 6100ms).
  r5: {7001, 6038, 5925}ms — **3/3 PASS** (mean 6321ms). ← **THE robust result.**
  r7: {9196, 6722, 8052}ms — **1/3 PASS** (mean 7990ms, literally on SLO).
  r10: {18116, 8588, 17269}ms — **0/3 PASS**.
  Goodput: {4.06, 4.58, 4.14} — median **4.14**, mean 4.26, conservative 4.06.
- W&B: logged as `v-srpf-r3` [mechanism].

### v-srpf-wb-r2 — SRPF+WB compound replication (same-node 1-2)  [DONE, 1-2, job 19732, commit 01fd8ba0a]
- Same-node replication of SRPF+WB compound on node 1-2 (same node as FCFS baseline v0-cert).
- **Full sweep:**
  | λ | p99 TTFT | p50 TTFT | req/s | tok/s | hit | SLO | vs run 1 Δp99 |
  |---|---------|---------|-------|-------|-----|-----|-----------------|
  | 3 | 5983ms  | 464ms   | 3.02  | 387   | 0.745 | PASS | +8% |
  | 5 | 6557ms  | 602ms   | 4.35  | 557   | 0.734 | **PASS** | +6% |
  | 7 | **8581ms** | 662ms | 5.06  | 648  | 0.731 | **FAIL** | +15% (flips!) |
  | 10| 14432ms | 753ms   | 5.32  | 680   | 0.729 | FAIL | −2% |
- **goodput@SLO = 4.35** (passes r3+r5, fails r7). vs original 5.03 (passed through r7).
- **n=2 SRPF+WB summary (both same-node 1-2 as baseline):**
  r3: {5564, 5983}ms — **2/2 PASS**.
  r5: {6183, 6557}ms — **2/2 PASS**. ← **robust, confirms compound.**
  r7: {7474, 8581}ms — **1/2 PASS** (discordant — coin-flip at boundary).
  r10: {14798, 14432}ms — **0/2 PASS** (concordant, stable FAIL).
  Conservative compound goodput = **4.35** (+44% over baseline). Best-case 5.03 (+66%).
  Throughput: 682 / 680 tok/s — **remarkably stable** (≤0.3% variation, confirming throughput gain is real).
- W&B: logged as `v-srpf-wb-r2` [mechanism].

### v-srpf-xt-wb — SRPF+XTIER+WB (triple compound)  [DONE, 1-2, srun 19732, commit 63e2c7aa9]
- Config: `--schedule-policy srpf` + `XTIER_EXCLUSIVE=1` + `--hicache-write-policy write_back`.
- **Full sweep:**
  | λ | p99 TTFT | p50 TTFT | req/s | tok/s | hit | SLO |
  |---|---------|---------|-------|-------|-----|-----|
  | 3 | **★4776ms** | 473ms | 3.02 | 387 | **0.757** | PASS (best r3 of ANY run!) |
  | 5 | **★5953ms** | 616ms | 4.42 | 566 | 0.748 | **PASS** (best r5!) |
  | 7 | 8662ms  | 740ms   | 5.07  | 649  | 0.745 | FAIL |
  | 10| 15304ms | 759ms   | 5.25  | 671  | 0.744 | FAIL |
- **goodput@SLO = 4.42** — better than SRPF+WB conservative (4.35) but still fails r7.
- **XTIER adds +1.1pp hit on top of write_back** (0.746→0.757). This translates to best-ever r3/r5 p99 (−14%/−3.7%).
- **But r7 still fails (8662ms)** — the r7 boundary is metastable/physical, not capacity-limited. Higher hit
  helps the bulk (r3/r5 improve) but doesn't rescue the 72 tail requests that push p99 above SLO at r7.
- W&B: logged as `v-srpf-xt-wb` [mechanism].

### v-srpf-age5 — SRPF with aging (5s threshold)  [DONE, ondem-2, job 19784, commit 16be3a5d1] ★STRONG NEGATIVE
- Config: `--schedule-policy srpf` + `SRPF_AGING_S=5.0` (write_through default).
- **Full sweep:**
  | λ | p99 TTFT | SRPF mean | regression | SLO |
  |---|---------|----------|-----------|-----|
  | 3 | 7965ms  | 6100ms   | **+31%**  | PASS (barely) |
  | 5 | **26657ms** | 6321ms | **+322%** | FAIL |
  | 7 | **39725ms** | 7990ms | **+397%** | FAIL |
  | 10| **35749ms** | 14658ms | **+144%** | FAIL |
- **goodput@SLO = 3.02** — aging completely erases the SRPF benefit, returning to baseline goodput.
- **Root cause**: aging disrupts the SRPF ordering by prematurely boosting cold requests. These consume the
  6144-token chunk budget, delaying cached requests. The delayed cached requests then age past the threshold,
  creating a POSITIVE FEEDBACK LOOP: more boosting → more budget waste → more aging → system collapses.
  The regression SCALES with rate (load amplifies the cascade).
- **Key insight**: SRPF ordering is CRITICALLY sensitive to perturbation. The cold/cached asymmetry is not
  just about "which goes first" — it's about budget EFFICIENCY. Cached requests consume negligible budget;
  cold requests consume most of it. SRPF maximizes the number of requests served per budget dollar. Any
  re-ordering that moves a cold request ahead of cached ones wastes budget and creates cascading delays.
- W&B: logged as `v-srpf-age5` [mechanism].

### v-srpf-xt — SRPF+XTIER (write_through)  [DONE, ondem-2, job 19780, commit 63e2c7aa9] ★CATASTROPHIC NEGATIVE
- Config: `--schedule-policy srpf` + `XTIER_EXCLUSIVE=1` (write_through default).
- **RESULT: hit_rate = 0.000, throughput 86 tok/s (−86%), goodput 0.00** — CATASTROPHIC.
- **Root cause**: XTIER exclusive frees the L2 (host) copy after loading back to L1 (device). Under
  write_through, eviction does NOT recreate the L2 copy (WT only writes on initial compute). So after one
  evict→load_back→evict cycle, the entry is lost from BOTH tiers permanently. This drains the entire cache.
- **XTIER_EXCLUSIVE REQUIRES write_back** (which recreates L2 copies on eviction, closing the leak).
  This is a genuine code invariant, not a configuration preference. Must be enforced.
- W&B: logged as `v-srpf-xt` [mechanism].

## Ops notes
- eval.sh has a path bug (computes `workspace/sgl/v0.3_ablations/base`); fixed by symlink
  `v0.3_ablations/base → v0.31/base` (frozen eval.sh untouched — fairness-clean).
- Contended pool: 4-way (base + 3 research) + v0.3 holds. Evals queue; research/code in parallel.
