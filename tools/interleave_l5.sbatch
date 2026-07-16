#!/bin/bash
# kleinrock: chunk-interleave A/B (l5) — fcfs vs srpf vs interleave at lambda=5, K=3, same node.
# Decisive question: does chunk-interleaved deferral clear the short-req HOL tail BETTER than SRPF, and does it
# PASS goodput@SLO at lambda=7 where SRPF fails (9.6s)? Interleave parks the in-flight chunked mega-doc to let
# short reqs prefill (bounds HOL ~1 chunk). Sim predicts interleave short-req p99 << srpf. Engine on branch
# evolve/kleinrock-interleave (built off srpf, so one binary runs all 3 arms). fcfs/srpf byte-identical to stock.
#SBATCH --job-name=kleinrock-ill5
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset0-1
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/ill5_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SRPF_SPECS="5:3" SRPF_OUT="runs/v11-interleave-l5" IL_CACHE=".cache-il5" PORT=30072
echo ">>> [ill5] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SRPF_SPECS] out=$SRPF_OUT"
bash "$MAIN/tools/interleave_ab_eval.sh"
echo ">>> [ill5] EXIT $(date -u)"
