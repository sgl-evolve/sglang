#!/bin/bash
# kleinrock: DURABLE exclusive sbatch for the SESSION-VARIANCE experiment (N server launches x K draws in
# ONE allocation, controlling the cross-allocation node-state confound). Durable => immune to hold timeouts.
# Pinned to 0-3 (idle certified node); leaves ondem-3 free for siblings. Autonomous: writes sessvar.csv;
# I check periodically (not babysit) and integrate when done.
#SBATCH --job-name=kleinrock-sessvar
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset0-3
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=09:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/sessvar_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SESS_N=3 SESS_K=3 PORT=30021
echo ">>> [sessvar] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID N=$SESS_N K=$SESS_K"
bash "$MAIN/tools/session_var_eval.sh"
echo ">>> [sessvar] EXIT $(date -u)"
