#!/usr/bin/env bash
# Combined ablation batch — all remaining experiments.
# SKIPS versions that already have summary.json.
# Usage: nohup bash ablation_all.sh > ablation_all.log 2>&1 &
set -uo pipefail
EVAL_SCRIPT="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech/fast_acquire.sh"
NAME="sgl_mech"
WS="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_mech/researchers/sgl_mech"

run_eval() {
    local ver="$1"; shift
    local sf="$WS/runs/$ver/summary.json"
    if [ -f "$sf" ]; then
        echo "========== $(date) ========== SKIPPING: $ver (already has summary.json) =========="
        return 0
    fi
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

# ============================================================
# BATCH 1 remaining (v10-v19 + ctl3)
# ============================================================

# --- v10: stock LFU eviction ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_EVICTION_POLICY_OVERRIDE=lfu
run_eval v10-lfu || true

# --- v11: stock SLRU eviction ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_EVICTION_POLICY_OVERRIDE=slru
run_eval v11-slru || true

# --- v12: cost-aware + frequency hybrid ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_EVICTION_POLICY_OVERRIDE=cost_freq
export SGLANG_COST_FREQ_THRESHOLD=2048
run_eval v12-costfreq || true

# --- v13: continuous cost (log-weighted, alpha=1.0) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost
export SGLANG_CONTINUOUS_COST_ALPHA=1.0
run_eval v13-contcost || true

# --- v14: write-admission control (cost-aware + defer cheap backup) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_WRITE_ADMISSION_MIN_COST=2048
run_eval v14-writeadmit || true

# --- v15: SJF scheduling (cost-aware eviction + shortest-prefill-first) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_SCHEDULE_OVERRIDE=sjf
run_eval v15-sjf || true

# --- v16: warm-first scheduling (cost-aware + prioritize cache-hit requests) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_SCHEDULE_OVERRIDE=warmfirst
run_eval v16-warmfirst || true

# --- v17: frequency-decay eviction ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_EVICTION_POLICY_OVERRIDE=freq_decay
export SGLANG_FREQ_DECAY_RATE=0.999
run_eval v17-freqdecay || true

# --- v18: size-aware LRU (evict large nodes first) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_EVICTION_POLICY_OVERRIDE=size_aware_lru
export SGLANG_SIZE_AWARE_THRESHOLD=2048
run_eval v18-sizelru || true

# --- v19: cost-aware load_back skip (recompute cheap, load expensive) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_LOADBACK_MIN_COST=512
run_eval v19-loadback || true

# --- v0-ctl3: 3rd control run ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
run_eval v0-ctl3 || true

# ============================================================
# BATCH 2 (v21-v26)
# ============================================================

# --- v21: value-gated load_back (only load if more valuable than cheapest L1 leaf) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_LOADBACK_VALUE_GATE=1
run_eval v21-valuegate || true

# --- v22: write-through threshold=2 (only backup on 2nd hit) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v22-wt2 || true

# --- v23: write-through threshold=3 ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_WRITE_THROUGH_THRESHOLD=3
run_eval v23-wt3 || true

# --- v24: stock LRU + value-gated load_back (isolate value-gate from CostAware) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_LOADBACK_VALUE_GATE=1
run_eval v24-lru-valuegate || true

# --- v25: full stack (CostAware + WT2 + value-gate) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_WRITE_THROUGH_THRESHOLD=2
export SGLANG_LOADBACK_VALUE_GATE=1
run_eval v25-fullstack || true

# --- v26: stock LRU + WT2 (isolate WT2 effect from CostAware) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v26-lru-wt2 || true

# ============================================================
# BATCH 3 (v27-v35 + ctl4)
# ============================================================

# --- v27: BackupAwareCost (4-tier: cost × backup status) ---
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=backup_aware_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
run_eval v27-backupcost || true

# --- v28: RecencyBoostedCost (CostAware tiers + frequency bonus) ---
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=recency_boosted_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_FREQ_BOOST_WEIGHT=5.0
run_eval v28-freqcost || true

# --- v29: ContinuousCost alpha=50 ---
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost
export SGLANG_CONTINUOUS_COST_ALPHA=50.0
run_eval v29-contcost-a50 || true

# --- v30: ContinuousCost alpha=200 ---
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=continuous_cost
export SGLANG_CONTINUOUS_COST_ALPHA=200.0
run_eval v30-contcost-a200 || true

# --- v31: CostAware + reuse_min=1 ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_COST_AWARE_REUSE_MIN=1
run_eval v31-reuse1 || true

# --- v32: CostAware threshold=4096 ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=4096
run_eval v32-t4096 || true

# --- v33: BackupAwareCost + WT2 ---
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=backup_aware_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_WRITE_THROUGH_THRESHOLD=2
run_eval v33-backupcost-wt2 || true

# --- v34: CostAware 3-tier (threshold=1024, threshold2=4096) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=1
export SGLANG_COST_AWARE_EVICT_THRESHOLD=1024
export SGLANG_COST_AWARE_EVICT_THRESHOLD2=4096
run_eval v34-3tier || true

# --- v35: RecencyBoostedCost freq_weight=20 ---
clean_env
export SGLANG_EVICTION_POLICY_OVERRIDE=recency_boosted_cost
export SGLANG_COST_AWARE_EVICT_THRESHOLD=2048
export SGLANG_FREQ_BOOST_WEIGHT=20.0
run_eval v35-freqcost-w20 || true

# --- v0-ctl4: 5th control (for variance) ---
clean_env
export SGLANG_ENABLE_COST_AWARE_EVICTION=0
run_eval v0-ctl4 || true

clean_env
echo "========== $(date) ========== ALL ABLATIONS COMPLETE =========="
