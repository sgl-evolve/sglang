# §5.5 empirical required-k integration (apply when v0-medk-bign K=10 lands → n=16 ondem-3 stock draws)

GOAL: upgrade §5.5 required-k from MODEL-ONLY to model + EMPIRICAL (real same-node data). Reviewers of a
measurement paper specifically ask "is your required-k divergence a model artifact or seen in real data?"
Answer it with the n=16 ondem-3 stock pool via tools/required_k_empirical.py (bootstrap, deterministic).

## Data source
runs/v0-medk (K=5) + runs/v0-medk-bign (K=10) + runs/v0-stock λ=3 → pooled ondem-3 stock cold λ=3 p99.
Preview on n=6 (before K=10): median 11.4s, σ/m=1.94, pass 2/6, EMPIRICAL required-k=23 (verdict goodput=0).
=> already confirms "required-k huge near knee" on real data; n=16 firms the number.

## Insertion point
paper.html §5.5, immediately AFTER the paragraph ending "...not the tail deadline." (the "unifies the
methodology" para, ~line 679), BEFORE the "Scope/honesty:" clause. So the flow is:
  model required-k table -> unifies-methodology para -> [NEW empirical confirmation para+table] -> scope/honesty.

## New block (FILL X/Y/Z/W from `python3 tools/required_k_empirical.py` after K=10):
<p><b>The measured data confirm the divergence.</b> The model's required-k rests on its own assumptions, so
we verify it directly against measurement: we pool all <b>n=NN same-node stock cold-λ=3 draws</b> (ondem-3,
median MED s, σ/m = SM, PASS/NN passing) and bootstrap the required median-of-k straight from the empirical
p99 distribution (<code>tools/required_k_empirical.py</code>, deterministic). At this operating point
(median MED s, MARGIN s from the SLO) the empirical required-k is <b>REQK</b> — i.e. even REQK back-to-back
same-node replicates would be needed for a 95%-reproducible goodput verdict. This is real-data confirmation,
not a model extrapolation, that median-of-k does not rescue goodput@SLO near the knee: it is exactly why the
same-node K=5 experiments (§3.1) — five replicates apiece — could not resolve the configurations.</p>

Optionally add a one-row addendum to the model table OR a tiny 2-col table (source | required-k):
  model @ σ/m≈1.9 (+1.2s): 7 ;  MODEL @ σ/m≈X (this op pt): __ ;  EMPIRICAL n=16 @ σ/m=X: REQK
(only if the model vs empirical σ/m are comparable — if empirical median is ABOVE SLO like n=6, compare to
the model's +σ rows; note the empirical op-pt σ/m and required-k together so the reader can place it.)

## Also update
- abstract / §1 contribution 6 (methodology): change "even median-of-k cannot resolve" to cite the EMPIRICAL
  required-k=REQK (n=16) as the hard evidence (currently leans on the 2-node median-of-5 disagreement).
- §7 (node×run grid limitation): note n=16 single-node draws now anchor the required-k empirically.
- §8 reproducibility: add v0-medk-bign (K=10 ondem-3) + tools/required_k_empirical.py.
- report.md + memory: log the empirical required-k number.
- W&B: log as a note on kleinrock run (variance study), NOT a new curve version (on-contract stock λ=3).

## Integrity
- All draws stock (git diff a334877e5 python/ empty, verified at launch commit 702678216).
- Report the EXACT bootstrap output; if empirical required-k differs from model, say so honestly (both are
  "large near the knee" — the qualitative claim; exact k is pool-dependent).
- n=16 bootstrap draws with replacement from 16 values → required-k is an estimate; state n and method.
