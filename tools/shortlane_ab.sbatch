#!/bin/bash
# kleinrock: DURABLE exclusive sbatch for the SHORT-LANE fast screen (Paper 2). Idle node (8 idle a3 + 0
# pending => fair spare capacity, parallel with the SRPF diagnostic on ondem-2). Isolated JIT cache avoids the
# flashinfer race. ~4h wall (2 arms x [warmup + lambda5 K3]); 6h walltime margin.
#SBATCH --job-name=kleinrock-shortlane
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset1-1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/shortlane_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SL_SPECS="5:3" SL_RESERVE=2048 PORT=30041
echo ">>> [shortlane] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SL_SPECS] reserve=$SL_RESERVE"
bash "$MAIN/tools/shortlane_ab_eval.sh"
echo ">>> [shortlane] EXIT $(date -u)"
