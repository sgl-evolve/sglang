#!/usr/bin/env bash
# batch_eval3.sh — backup evals if batch1/2 don't produce enough versions
set -uo pipefail
SGL_HOME=/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/base_mech
EVAL_SCRIPT="$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
WS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_mech/researchers/base_mech
NAME=base_mech
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_mech_dg

EVALS=(
  # More error bar replicates (these are cheap and valuable)
  "v44-excl-rep3 BM_EXCL=1"
  "v45-baseline-rep3"
  # Write-through threshold sweep (more granular)
  "v46-wt2 BM_WT_THRESHOLD=2"
  "v47-wt5 BM_WT_THRESHOLD=5"
  # Admission with non-exclusive (pure admission effect)
  "v48-admit256-noexcl BM_ADMIT_MIN_TOKENS=256"
  "v49-admit1024 BM_EXCL=1 BM_ADMIT_MIN_TOKENS=1024"
  # Adaptive exclusivity at 70% (more aggressive)
  "v50-adaptive-excl-70 BM_ADAPTIVE_EXCL=0.70"
  # SJF + cost-aware eviction
  "v51-sjf-costaware BM_SJF=1 BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=2048"
  # Selective dev + selective host (dual selective, no exclusive)
  "v52-dual-sel BM_SELECTIVE_DEV=1 BM_SELECTIVE_HOST=1"
  # Cost-aware with higher threshold
  "v53-cost-t8192 BM_EXCL=1 BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=8192"
)

for spec in "${EVALS[@]}"; do
  read -r ver envvars <<< "$spec"
  if [ -f "$WS/runs/$ver/summary.json" ]; then
    echo "=== SKIP $ver (already has summary.json) ==="
    continue
  fi
  echo "=== START $ver (envs: $envvars) ==="
  for ev in $envvars; do export "$ev"; done
  bash "$EVAL_SCRIPT" "$NAME" "$ver"
  rc=$?
  for ev in $envvars; do unset "${ev%%=*}"; done
  echo "=== $([ $rc -eq 0 ] && echo DONE || echo FAILED) $ver (exit $rc) ==="
done
echo "=== ALL BATCH3 EVALS COMPLETE ==="
