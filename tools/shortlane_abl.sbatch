#!/bin/bash
# kleinrock: SHORT-LANE RESERVE ABLATION (λ=5, same node) — answers the obvious reviewer question on Paper 1
# §5.6: is the fair-reserve failure fundamental, or specific to the thin (2048) regime? Sweeps reserve ∈
# {0, 1024, 4096} at λ=5 (K=3 each). Hypothesis test: monotonic (more reserve→worse, or →better approaching SRPF)
# vs U-shaped (thin hurts, large helps). If large reserve crosses over to HELPING, that is a novel bounded-
# deferral result (SRPF-like benefit with a guaranteed per-iter floor for the heavy doc); if all fail, the
# negative control is fundamental. Same-node (all arms one node); separate out dir + JIT cache. Fair: idle node.
#SBATCH --job-name=kleinrock-sl-abl
#SBATCH --partition=a3
#SBATCH --nodelist=slurm2-a3nodeset0-0
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/sl_abl_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SL_SPECS="5:3" SL_RESERVES="1024 4096" SL_OUT="runs/v6-shortlane-abl" SL_CACHE=".cache-shortlane-abl" SL_PERREQ=1 PORT=30045
echo ">>> [sl-abl] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SL_SPECS] reserves=[$SL_RESERVES] out=$SL_OUT"
bash "$MAIN/tools/shortlane_ab_eval.sh"
echo ">>> [sl-abl] EXIT $(date -u)"
