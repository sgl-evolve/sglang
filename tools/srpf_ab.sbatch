#!/bin/bash
# kleinrock: DURABLE exclusive sbatch for the SRPF A/B (fcfs default vs SRPF scheduler, same node, median-of-k).
# Durable => immune to hold timeouts. Pinned to an idle a3 node; 9 idle nodes + 0 pending in the queue, so this
# uses genuinely-spare capacity and starves no sibling. ~10h wall (2 server launches x [warmup + lambda5 K5 +
# lambda3 K3]); 14h walltime for JIT/startup margin. Autonomous; I check periodically.
#SBATCH --job-name=kleinrock-srpfab
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodesetondem-2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=14:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/srpfab_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SRPF_SPECS="5:5 3:3" PORT=30037
echo ">>> [srpfab] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SRPF_SPECS]"
bash "$MAIN/tools/srpf_ab_eval.sh"
echo ">>> [srpfab] EXIT $(date -u)"
