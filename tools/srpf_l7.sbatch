#!/bin/bash
# kleinrock: SRPF goodput-ceiling test at λ=7 (severe saturation) — completes the across-λ characterization for
# Paper 1 §5.1/§5.6. SRPF moves the tail at λ3 (knee) and λ5 (moderate saturation); does deferral still help at
# λ=7 (offered 7 ≫ compute ceiling ~4.2 req/s, so the backlog grows unboundedly), or does severe saturation
# push even short reqs past the SLO (bounding SRPF's goodput ceiling to λ≤5)? Same-node fcfs vs srpf, K=3.
# Stock+srpf via the srpf-worktree engine; characterization (SRPF is known, not claimed as a contribution).
#SBATCH --job-name=kleinrock-srpfl7
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset0-0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/srpfl7_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SRPF_SPECS="7:3" SRPF_POLICIES="srpf" SRPF_OUT="runs/v8-srpf-l7" PORT=30049
echo ">>> [srpfl7] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SRPF_SPECS] policies=[fcfs $SRPF_POLICIES] out=$SRPF_OUT"
bash "$MAIN/tools/srpf_ab_eval.sh"
echo ">>> [srpfl7] EXIT $(date -u)"
