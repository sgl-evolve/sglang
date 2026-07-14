#!/bin/bash
# kleinrock: DURABLE exclusive sbatch for the PHASE-MAP diagnostic (low-lambda sweep to complete the empirical
# phase diagram). Durable => immune to hold timeouts. Pinned to 1-2 (idle certified); all 4 certified nodes are
# idle with no pending siblings, so this uses genuinely-spare capacity. Autonomous; I check periodically.
#SBATCH --job-name=kleinrock-phasemap
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset1-2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/phasemap_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export PHASE_LAMBDAS="1.5 2.5" PHASE_K=3 PORT=30031
echo ">>> [phasemap] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID lambdas=[$PHASE_LAMBDAS] K=$PHASE_K"
bash "$MAIN/tools/phasemap_eval.sh"
echo ">>> [phasemap] EXIT $(date -u)"
