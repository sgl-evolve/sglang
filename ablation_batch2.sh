#!/usr/bin/env bash
# Batch ablation 2 — post-batch experiments testing new mechanisms.
# Usage: nohup bash ablation_batch2.sh > ablation_batch2.log 2>&1 &
set -uo pipefail
EVAL_SCRIPT="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
NAME="sgl_mech"

run_eval() {
    local ver="$1"; shift
    echo "========== $(date) ========== STARTING: $ver =========="
    env | grep -E "SGLANG_(EVICTION|ENABLE_COST|SCHEDULE|WRITE_ADMISSION|LOADBACK|WRITE_THROUGH)" || echo "(no override envs)"
    bash "$EVAL_SCRIPT" "$NAME" "$ver"
    local rc=$?
    echo "========== $(date) ========== FINISHED: $ver (rc=$rc) =========="
    return $rc
}

clean_env() {
    unset SGLANG_EVICTION_POLICY_OVERRIDE SGLANG_ENABLE_COST_AWARE_EVICTION \
          SGLANG_COST_AWARE_EVICT_THRESHOLD SGLANG_SCHEDULE_OVERRIDE \
          SGLANG_WRITE_ADMISSION_MIN_COST SGLANG_CONTINUOUS_COST_ALPHA \
          SGLANG_COST_FREQ_THRESHOLD SGLANG_FREQ_DECAY_RATE \
          SGLANG_SIZE_AWARE_THRESHOLD SGLANG_LOADBACK_MIN_COST \
          SGLANG_LOADBACK_VALUE_GATE SGLANG_WRITE_THROUGH_THRESHOLD 2>/dev/null || true
}

# --- v21: CostAware + value-gated load-back ---
# Hypothesis: skip loading nodes from L2 when they'd displace more valuable L1 nodes
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_LOADBACK_VALUE_GATE=1
run_eval v21-valuegate || true

# --- v22: Write-through threshold=2 (admission control for L2) ---
# Hypothesis: only back up re-hit nodes to L2, improving L2 quality
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v22-wt2 || true

# --- v23: Write-through threshold=3 (stricter L2 admission) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_WRITE_THROUGH_THRESHOLD=3
run_eval v23-wt3 || true

# --- v24: CostAware + WT threshold=2 (best eviction + better L2) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v24-cost-wt2 || true

# --- v25: CostAware + value gate + WT2 (full stack) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_LOADBACK_VALUE_GATE=1
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v25-fullstack || true

# --- v26: stock LRU + WT2 (isolate WT2 effect from eviction) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v26-lru-wt2 || true

clean_env
echo "========== $(date) ========== BATCH 2 COMPLETE =========="
