#!/usr/bin/env bash
# batch_eval2.sh — additional mechanism evals (run AFTER batch_eval.sh finishes)
set -uo pipefail
SGL_HOME=/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/base_mech
EVAL_SCRIPT="$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
WS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_mech/researchers/base_mech
NAME=base_mech
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_mech_dg

EVALS=(
  # SJF scheduling variants
  "v26-sjf BM_SJF=1"
  "v27-sjf-excl BM_SJF=1 BM_EXCL=1"
  # Warm-first + exclusive (combination)
  "v28-warmfirst-excl BM_WARMFIRST=1 BM_EXCL=1"
  # Selective host + cost-aware
  "v29-selhost-costaware BM_SELECTIVE_HOST=1 BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=2048"
  # Pure exclusive with cost-aware on both tiers
  "v30-excl-costaware-selhost BM_EXCL=1 BM_EVICT_STRATEGY=cost_aware BM_COST_THRESHOLD=2048 BM_SELECTIVE_HOST=1"
  # Adaptive exclusivity variants (bugfix: now uses pool.size correctly)
  "v31-adaptive-excl-90 BM_ADAPTIVE_EXCL=0.90"
  "v32-adaptive-excl-95 BM_ADAPTIVE_EXCL=0.95"
  "v33-adaptive-excl-80 BM_ADAPTIVE_EXCL=0.80"
  # Admission control: min tokens for backup
  "v34-admit256 BM_EXCL=1 BM_ADMIT_MIN_TOKENS=256"
  "v35-admit512 BM_EXCL=1 BM_ADMIT_MIN_TOKENS=512"
  # Write-through threshold override (back up after N hits, not 1)
  "v36-wt3 BM_WT_THRESHOLD=3"
  "v37-wt3-excl BM_EXCL=1 BM_WT_THRESHOLD=3"
  # Hybrid host eviction (separate strategy for host tier)
  "v38-hostcost BM_HOST_EVICT_STRATEGY=cost_aware BM_HOST_COST_THRESHOLD=2048"
  "v39-hostcost-excl BM_EXCL=1 BM_HOST_EVICT_STRATEGY=cost_aware BM_HOST_COST_THRESHOLD=2048"
  # Combined: excl + host-cost-aware + selective
  "v40-excl-hostcost-sel BM_EXCL=1 BM_HOST_EVICT_STRATEGY=cost_aware BM_HOST_COST_THRESHOLD=2048 BM_SELECTIVE_HOST=1"
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
echo "=== ALL BATCH2 EVALS COMPLETE ==="
