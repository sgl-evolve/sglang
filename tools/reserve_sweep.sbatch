#!/bin/bash
# kleinrock: CLEAN SINGLE-NODE reserve-size sweep — rigor upgrade for Paper 2 (prefill-hol-defer) §4.2.
# Current §4.2 patches together two runs with different stock baselines (reserve2048 from v4/stock 24.2s;
# reserve0/1024/4096 from v6/stock 23.3s), forcing a normalized-to-own-run-stock footnote. This runs ALL reserve
# sizes {0,1024,2048,3072,4096} on ONE node with ONE stock baseline, same binary (short-lane worktree engine),
# K=3 at lambda=5 -> a single honest same-node MONOTONE ablation. Adds 3072 (the cliff-locator between 2048's
# +75% and 4096's catastrophe) to pin where the collapse onsets. reserve0 = byte-identical stock (lossless).
# NOTE: reserve4096 OOM-kills the server mid-arm (congestion backlog) -> that arm yields n=1 (reps 2-3 hit the
# dead server); the negative's n>=3 rests on 0/1024/2048/3072. Unpinned (fair-share queue) + exclude the
# GPU-less node; isolated .cache-reserve avoids a flashinfer-JIT race with the concurrent defer-knee job.
#SBATCH --job-name=kleinrock-ressweep
#SBATCH --partition=a3
#SBATCH --exclude=slurm2-a3nodeset-2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --time=10:00:00
#SBATCH --output=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock/runs/ressweep_sbatch_%j.out
set -uo pipefail
MAIN=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
export SL_SPECS="5:3" SL_RESERVES="1024 2048 3072 4096" SL_OUT="runs/v10-reserve-sweep" SL_CACHE=".cache-reserve" PORT=30061
echo ">>> [ressweep] $(date -u) node=$(hostname -s) jobid=$SLURM_JOB_ID specs=[$SL_SPECS] reserves=[0 $SL_RESERVES] out=$SL_OUT"
bash "$MAIN/tools/shortlane_ab_eval.sh"
echo ">>> [ressweep] EXIT $(date -u)"
