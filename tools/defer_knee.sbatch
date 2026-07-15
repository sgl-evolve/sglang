#!/bin/bash
# kleinrock: DOES DEFERRAL COLLAPSE THE GOODPUT@SLO COIN-FLIP AT THE KNEE? — the P1<->P2 bridge.
# Paper 1 (prefill-slo-tail): goodput@SLO is a cold-start COIN-FLIP at the knee (lambda=3), variance-dominated
# (stock lambda3 p99 {10.8,20.1,34.2}s, sigma/m huge). Paper 2 (prefill-hol-defer): the tail is HOL, cleared by
# DEFERRAL (SRPF). OPEN QUESTION unifying both: at the knee, does deferral merely SHIFT the median under the SLO,
# or does it COLLAPSE the metastable run-to-run variance (making goodput@SLO RELIABLE, not just passing)? n=3 is
# too few to tell (SRPF lambda3 {6.2,6.7,8.1}s — 2/3 pass, one GRAZED the SLO). This is a same-node K=5 test:
# fcfs (stock) vs srpf (deferral REFERENCE, NOT claimed as a mechanism — a characterization probe, like P1's use
# of stable metrics) at lambda=3. Outcome either way is a real result: collapse => "deferral makes the metric
# reliable at the knee" (strong systems claim, bridges P1+P2); shift-but-still-coin-flip => "the knee coin-flip is
# scheduling-robust" (a genuine boundary strengthening P1). Contract-legal (--schedule-policy not forbidden).
#SBATCH --job-name=kleinrock-deferknee
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset0-2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/deferknee_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SRPF_SPECS="3:5" SRPF_POLICIES="srpf" SRPF_OUT="runs/v9-defer-knee" PORT=30051
echo ">>> [deferknee] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SRPF_SPECS] policies=[fcfs $SRPF_POLICIES] out=$SRPF_OUT"
bash "$MAIN/tools/srpf_ab_eval.sh"
echo ">>> [deferknee] EXIT $(date -u)"
