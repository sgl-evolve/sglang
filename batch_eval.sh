#!/usr/bin/env bash
# batch_eval.sh — run a sequence of evals, one at a time, waiting for pool nodes.
# Each line in the EVALS array = "version_name ENV_VAR1=val ENV_VAR2=val ..."
set -uo pipefail
SGL_HOME=/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/base_mech
EVAL_SCRIPT="$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
WS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_mech/researchers/base_mech
NAME=base_mech
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_mech_dg

EVALS=(
  # Exclusive + eviction strategy variants
  "v6-costaware BM_EXCL=1 BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=2048"
  "v7-gdsf BM_EXCL=1 BM_EVICT_STRATEGY=gdsf"
  "v8-lfu-excl BM_EXCL=1 BM_EVICT_STRATEGY=lfu"
  "v9-fifo-excl BM_EXCL=1 BM_EVICT_STRATEGY=fifo"
  "v10-mru-excl BM_EXCL=1 BM_EVICT_STRATEGY=mru"
  "v11-slru-excl BM_EXCL=1 BM_EVICT_STRATEGY=slru"
  "v12-freqdecay-excl BM_EXCL=1 BM_EVICT_STRATEGY=freq_decay"
  "v13-sizeweight-excl BM_EXCL=1 BM_EVICT_STRATEGY=size_weighted"
  "v14-depthaware-excl BM_EXCL=1 BM_EVICT_STRATEGY=depth_aware"
  # No-exclusive variants (pure strategy changes on baseline)
  "v15-costaware-noexcl BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=2048"
  "v16-gdsf-noexcl BM_EVICT_STRATEGY=gdsf"
  "v17-lfu-noexcl BM_EVICT_STRATEGY=lfu"
  # Selective host eviction
  "v18-selhost BM_SELECTIVE_HOST=1"
  "v19-selhost-excl BM_EXCL=1 BM_SELECTIVE_HOST=1"
  # Exclusive with different cost thresholds
  "v20-costaware-t1024 BM_EXCL=1 BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=1024"
  "v21-costaware-t4096 BM_EXCL=1 BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=4096"
  # Pure exclusive replicate (error bar)
  "v22-excl-rep1 BM_EXCL=1"
  "v23-excl-rep2 BM_EXCL=1"
  # Baseline replicate (error bar)
  "v24-baseline-rep1"
  "v25-baseline-rep2"
)

for spec in "${EVALS[@]}"; do
  read -r ver envvars <<< "$spec"
  if [ -f "$WS/runs/$ver/summary.json" ]; then
    echo "=== SKIP $ver (already has summary.json) ==="
    continue
  fi
  echo "=== START $ver (envs: $envvars) ==="
  # Export env vars for this eval
  for ev in $envvars; do
    export "$ev"
  done
  bash "$EVAL_SCRIPT" "$NAME" "$ver"
  rc=$?
  # Unset env vars
  for ev in $envvars; do
    unset "${ev%%=*}"
  done
  if [ $rc -eq 0 ]; then
    echo "=== DONE $ver (exit $rc) ==="
  else
    echo "=== FAILED $ver (exit $rc) ==="
  fi
done
echo "=== ALL EVALS COMPLETE ==="
