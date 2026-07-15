#!/bin/bash
# kleinrock: SHORT-LANE A/B at lambda=3 (moderate load / coin-flip knee) — the decisive test of whether the
# reserved short-prefill lane helps the p99 goodput tail when the system is NOT saturated (unlike lambda=5, where
# it backfired because the tail is mega-doc-own-prefill and the reserve slows mega-docs). Same-node reserve0 vs
# reserve2048, K=3, with per-request attribution dump (SL_PERREQ=1). Separate output dir + JIT cache (parallel
# with the lambda=5 job on another node — avoids the shared flashinfer cache race). Fair: idle node, 0 pending.
#SBATCH --job-name=kleinrock-sl-l3
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset0-1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/sl_l3_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SL_SPECS="3:3" SL_RESERVE=2048 SL_OUT="runs/v5-shortlane-l3" SL_CACHE=".cache-shortlane-l3" SL_PERREQ=1 PORT=30043
echo ">>> [sl-l3] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SL_SPECS] reserve=$SL_RESERVE out=$SL_OUT"
bash "$MAIN/tools/shortlane_ab_eval.sh"
echo ">>> [sl-l3] EXIT $(date -u)"
