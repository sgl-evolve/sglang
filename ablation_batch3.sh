#!/usr/bin/env bash
# Batch ablation 3 — new strategies + tuned combinations.
# Usage: nohup bash ablation_batch3.sh > ablation_batch3.log 2>&1 &
set -uo pipefail
EVAL_SCRIPT="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_mech/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
NAME="sgl_mech"

run_eval() {
    local ver="$1"; shift
    echo "========== $(date) ========== STARTING: $ver =========="
    env | grep -E "SGLANG_(EVICTION|ENABLE_COST|SCHEDULE|WRITE_ADMISSION|LOADBACK|WRITE_THROUGH|CONTINUOUS_COST|FREQ_BOOST|COST_AWARE)" || echo "(no override envs)"
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
          SGLANG_LOADBACK_VALUE_GATE SGLANG_WRITE_THROUGH_THRESHOLD \
          SGLANG_FREQ_BOOST_WEIGHT SGLANG_COST_AWARE_REUSE_MIN \
          SGLANG_COST_AWARE_COST_MODE SGLANG_COST_AWARE_EVICT_THRESHOLD2 2>/dev/null || true
}

# --- v27: BackupAwareCost — 4-tier (cost × backup status) eviction ---
# Hypothesis: prefer evicting L2-backed nodes (safe/recoverable) over unbacked
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=backup_aware_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
run_eval v27-backupcost || true

# --- v28: RecencyBoostedCost — CostAware tiers + frequency bonus ---
# Hypothesis: within tiers, frequently re-hit nodes deserve extra protection
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=recency_boosted_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_FREQ_BOOST_WEIGHT=5.0
run_eval v28-freqcost || true

# --- v29: ContinuousCost with high alpha (50) ---
# Hypothesis: stronger cost weighting makes continuous cost competitive with CostAware
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost
export SGLANG_CONTINUOUS_COST_ALPHA=50.0
run_eval v29-contcost-a50 || true

# --- v30: ContinuousCost with very high alpha (200) ---
# Hypothesis: very strong cost preference — does it degrade to size-aware?
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost
export SGLANG_CONTINUOUS_COST_ALPHA=200.0
run_eval v30-contcost-a200 || true

# --- v31: CostAware + reuse_min=1 (protect only re-hit expensive nodes) ---
# Hypothesis: one-shot long prefixes waste protection; only protect re-hit ones
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_COST_AWARE_REUSE_MIN=1
run_eval v31-reuse1 || true

# --- v32: CostAware threshold=4096 (protect only very expensive prefixes) ---
# Hypothesis: higher threshold focuses protection on the most expensive nodes
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=4096
run_eval v32-t4096 || true

# --- v33: BackupAwareCost + WT2 (amplifies backup-status differentiator) ---
# Hypothesis: WT2 creates more unbacked nodes, making backup-aware more impactful
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=backup_aware_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v33-backupcost-wt2 || true

# --- v34: CostAware 3-tier (threshold=1024, threshold2=4096) ---
# Hypothesis: 3-tier segmentation (short/medium/long) is more granular than 2-tier
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=1024
export SGLANG_COST_AWARE_EVICT_THRESHOLD2=4096
run_eval v34-3tier || true

# --- v35: RecencyBoostedCost with stronger freq weight (20) ---
# Hypothesis: stronger frequency boost creates more differentiation within tiers
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=recency_boosted_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_FREQ_BOOST_WEIGHT=20.0
run_eval v35-freqcost-w20 || true

# --- v0-ctl4: 4th control (stock LRU) for variance estimation ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
run_eval v0-ctl4 || true

clean_env
echo "========== $(date) ========== BATCH 3 COMPLETE =========="
