#!/usr/bin/env bash
set -uo pipefail
POOL=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/_pool
K=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.31/research/researchers/kleinrock
NODE=slurm2-a3nodesetondem-3
JID=$(cat "$POOL/held/$NODE" 2>/dev/null)
[ -z "$JID" ] && { echo "no hold job for $NODE"; exit 1; }
exec 200>"$POOL/locks/$NODE.lock"
flock -w 120 200 || { echo "FLOCK_BUSY $NODE"; exit 2; }
echo "[medk_wb-pool] flock acquired on $NODE (hold $JID) $(date -u +%H:%M:%S)"
srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 bash "$K/tools/medk_wb_eval.sh"
rc=$?
flock -u 200; exec 200>&-
echo "[medk_wb-pool] DONE rc=$rc $(date -u +%H:%M:%S)"
