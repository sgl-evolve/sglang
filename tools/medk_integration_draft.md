# medk integration draft (apply to paper.html when K=5 completes)

Same-node ondem-3 stock+flush λ=3 draws (medk, job 19516, commit ce281d9ff = stock engine):
- rep1: 13.39 s  FAIL
- rep2:  6.61 s  PASS
- rep3: 25.76 s  FAIL  (!! huge)
- rep4:  6.47 s  PASS
- rep5:  TBD
So far 2 PASS / 2 FAIL at fixed node -> goodput {0,3,0,3} literal coin toss. Spread 6.47-25.76 = 4.0x.
p50 dead-flat ~0.54s across ALL draws (cache serves body identically); only p99 swings. Perfect demo.
Plus prior ondem-3 same-config draw: v0-stock 14.06 s FAIL (n=6 total same-node once K=5 done).

KEY: rep1 (13.39, fail) and rep2 (6.61, pass) STRADDLE the 8 s SLO on the SAME node, SAME config
→ same-node goodput@SLO flips {0,3}. Node confound RESOLVED: the coin-flip is run-to-run cold-start
metastability at a FIXED node, not node heterogeneity. (2.02x spread at fixed node.)

## Edit 1 — §3.1: after the cross-node table + paragraph, ADD a same-node subsection.

New block (insert after the coin-flip scatter figure, before §3.2):

<h3>3.1b Same-node replication: the flip is run-variance, not node heterogeneity</h3>
<p>The cross-node evidence above cannot separate run-to-run metastability from node heterogeneity
(§7). We resolve this directly: <b>K=5 replicates of the identical stock configuration, back-to-back on
one node</b> (ondem-3, the same node as v0-stock), each an independent cold-cache λ=3 draw (the load
generator flushes at the start of every run, so every replicate reproduces the eval's λ=3 point).
Because λ≥5 reliably fails the SLO (§5.1), goodput@SLO is decided entirely by λ=3, so the spread of
these same-node λ=3 p99 values <i>is</i> the goodput coin-flip with node held constant.</p>
<table>
<tr><th>same-node draw (ondem-3, stock)</th><th>λ=3 p99 TTFT</th><th>vs 8 s SLO</th><th>goodput</th></tr>
<tr><td>v0-stock (earlier)</td><td>14.06 s</td><td>fail</td><td>0</td></tr>
<tr><td>medk rep1</td><td>13.39 s</td><td>fail</td><td>0</td></tr>
<tr><td>medk rep2</td><td>6.61 s</td><td><b>pass</b></td><td><b>3</b></td></tr>
<tr><td>medk rep3</td><td>TBD</td><td>TBD</td><td>TBD</td></tr>
<tr><td>medk rep4</td><td>TBD</td><td>TBD</td><td>TBD</td></tr>
<tr><td>medk rep5</td><td>TBD</td><td>TBD</td><td>TBD</td></tr>
</table>
<p><b>On a single fixed node, the identical stock configuration produces λ=3 p99 values that straddle
the SLO</b> (rep1 13.39 s fails, rep2 6.61 s passes — a 2.02x spread), so goodput@SLO flips {0, 3}
with node, workload, and configuration all held constant. This <b>isolates run-to-run cold-start
metastability as the coin-flip's cause</b> and removes the node-heterogeneity confound from the central
claim: the metric is variance-dominated even at a fixed node. [FILL: median = X.X s; N of 5 pass; so the
same-node goodput is unresolved/{0 or 3} depending on the draw.] Node heterogeneity (§7) is an
<i>additional</i>, independent variance source layered on top.</p>

## Edit 2 — §7 bullet-1 (node confound): future work -> RESULT.
Replace "The clean experiment (future work, compute-permitting) is same-node, median-of-k ... void."
with: we RAN same-node median-of-k (§3.1b): K=5 stock replicates on ondem-3 straddle the SLO (rep1
13.39 s fail, rep2 6.61 s pass), directly demonstrating the coin-flip at a fixed node. So the coin-flip
is confirmed run-variance-driven; node heterogeneity is an additional layer. The remaining honest limit:
K=5 on one node characterizes run-variance for that node; a full node x run grid would map both axes.

## Edit 3 — §5.3 re-point: change "(§7.)" ref for warm median-of-k to also cite §3.1b (now exists).

## Edit 4 — subtitle: drop "same-node median-of-k underway" -> "n=3 cross-node + K=5 same-node replicates + node-controlled no-flush diagnostic" (remove DRAFT if fully done).

## Edit 5 — abstract: add one clause that same-node K=5 confirms run-variance (flip at fixed node).

## W&B: medk draws are on-contract stock λ=3 points. Log median as a note on kleinrock run (variance study), NOT a new curve version.
