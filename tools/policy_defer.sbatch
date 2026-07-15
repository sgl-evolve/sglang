#!/bin/bash
# kleinrock: DEFERRAL-SPECIFICITY test for Paper 1 §5.6 — is it deferral-by-remaining-prefill (SRPF) that fixes
# the goodput tail, or would ANY queue reordering help? Same-node A/B at λ=5, K=3: fcfs (baseline) vs LOF
# (longest-output-first — a real reordering that does NOT reorder by prefill size, so it should NOT defer the
# heavy docs) vs srpf (deferral). Prediction: LOF ≈ fcfs (fails), only srpf passes → the lever is deferral
# SPECIFICALLY, not reordering in general. Stock policies used as characterization/controls (not claimed as
# contributions), via the srpf-worktree engine (has all policies + srpf). Fair: idle node, 0 pending.
#SBATCH --job-name=kleinrock-poldefer
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset0-0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/poldefer_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SRPF_SPECS="5:3" SRPF_POLICIES="lof srpf" SRPF_OUT="runs/v7-policy-defer" PORT=30047
echo ">>> [poldefer] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SRPF_SPECS] policies=[fcfs $SRPF_POLICIES] out=$SRPF_OUT"
bash "$MAIN/tools/srpf_ab_eval.sh"
echo ">>> [poldefer] EXIT $(date -u)"
