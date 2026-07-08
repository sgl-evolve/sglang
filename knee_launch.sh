#!/usr/bin/env bash
# Grab a free held certified node (flock, like eval-on-pool) and run knee_node.sh into it.
# Env (XTIER_*) propagates via srun --export=ALL. Retries on transient node-busy.
# Usage: bash knee_launch.sh <label> <lambdas_csv> [extra launch args...]
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free
_H="$SGL_HOME"; _W="$SGL_WORKSPACE"; set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export SGL_HOME="$_H"; export SGL_WORKSPACE="$_W"; export HF_TOKEN="$HF_API_KEY"
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_free_dg
POOL=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/_pool
NODE_SH=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free/knee_node.sh
LABEL="${1:?label}"; LAMS="${2:?lambdas}"; shift 2
for attempt in $(seq 1 300); do
  for f in "$POOL"/held/*; do
    node=$(basename "$f"); jid=$(cat "$f" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[knee] running knee-$LABEL on $node (job $jid) attempt $attempt"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 \
        bash "$NODE_SH" base_free "$LABEL" "$LAMS" "$@"
      rc=$?; flock -u 200; exec 200>&-
      [ $rc -eq 0 ] && { echo "[knee] DONE $LABEL"; exit 0; }
      echo "[knee] rc=$rc (transient?) retry in 45s"; sleep 45; break
    fi
    exec 200>&-
  done
  sleep 30
done
echo "[knee] gave up"; exit 1
