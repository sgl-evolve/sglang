#!/bin/bash
# kleinrock: DURABLE exclusive sbatch to firm the empirical required-k (ondem-3 n=6 -> n=16).
# Durable (own exclusive allocation) => immune to the hold-reclamation that killed the --overlap attempt.
# Pinned to ondem-3 so K=10 pools with the existing K=5 medk + v0-stock draws on the SAME node.
# On-contract: medk_bign_eval.sh uses byte-identical eval flags (stock write_through) and is node-guarded.
#SBATCH --job-name=kleinrock-reqk16
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodesetondem-3
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/reqk16_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export MEDK_K=10 PORT=30011
echo ">>> [reqk16] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID K=$MEDK_K"
bash "$MAIN/tools/medk_bign_eval.sh"
echo ">>> [reqk16] DONE $(date -u)"
