#!/usr/bin/env bash
# batch_eval.sh — run a sequence of eval experiments unattended.
# Each experiment runs sequentially on the held pool (one at a time).
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free

# Preserve cell env vars across .env sourcing
_SGL_HOME="$SGL_HOME"; _SGL_WS="$SGL_WORKSPACE"
set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export SGL_HOME="$_SGL_HOME"; export SGL_WORKSPACE="$_SGL_WS"; export HF_TOKEN="$HF_API_KEY"
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_free_dg

EVALPOOL="$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
LOGDIR="$PWD/eval-logs"
mkdir -p "$LOGDIR"

run_one() {
  local ver="$1" envs="$2" extras="$3"
  # Check if already completed
  if [[ -f "$PWD/runs/$ver/summary.json" ]]; then
    echo "===== [$ver] already done — skip ====="
    return 0
  fi
  echo "===== [$ver] starting $(date) ====="
  # Clear env vars from previous run first
  unset SGLANG_XTIER_LAZY SGLANG_XTIER_WM_FRAC SGLANG_XTIER_BATCH SGLANG_XTIER_PERIOD SGLANG_XTIER_ADAPTIVE SGLANG_XTIER_BACKUP_SELECT SGLANG_XTIER_REUSE_GATE SGLANG_COSTEVICT_THRESHOLD SGLANG_WSAC_MAX_COLD SGLANG_WSAC_COLD_RATIO SGLANG_PGAC_THRESHOLD SGLANG_PGAC_COLD_RATIO 2>/dev/null || true
  # Export env vars for this experiment
  if [[ -n "$envs" ]]; then
    eval "export $envs"
  fi
  # Parse extras into array
  local -a extra_arr=()
  if [[ -n "$extras" ]]; then
    read -ra extra_arr <<< "$extras"
  fi
  local logfile="$LOGDIR/eval-$ver.log"
  for attempt in $(seq 1 30); do
    echo "[$ver] attempt $attempt $(date +%H:%M:%S)"
    bash "$EVALPOOL" base_free "$ver" "${extra_arr[@]}" > "$logfile" 2>&1
    rc=$?
    case $rc in
      0) echo "[$ver] SUCCESS $(date)"; return 0 ;;
      2|3|6) echo "[$ver] transient (rc=$rc) — retry in 45s"; sleep 45 ;;
      5) echo "[$ver] REFUSED — skip"; return 5 ;;
      8) echo "[$ver] bench failed — skip"; return 8 ;;
      *) echo "[$ver] rc=$rc — retry in 45s"; sleep 45 ;;
    esac
  done
  echo "[$ver] gave up after 30 attempts"
  return 1
}

# ===== EXPERIMENT QUEUE =====
# Format: run_one "VERSION" "ENV_EXPORTS" "EXTRA_CLI_ARGS"

# --- Config experiments (stock flags, tagged "config") ---
run_one "v6-cfg-lpm" "" "--schedule-policy lpm"
run_one "v7-cfg-wb" "" "--hicache-write-policy write_back"
run_one "v8-cfg-wb-lpm" "" "--hicache-write-policy write_back --schedule-policy lpm"

# --- Novel eviction policies (new code, tagged "mechanism") ---
run_one "v9-cost-evict" "" "--radix-eviction-policy cost_aware"
run_one "v10-lfu-evict" "" "--radix-eviction-policy lfu"
run_one "v11-slru-evict" "" "--radix-eviction-policy slru"
run_one "v12-gdsf-evict" "" "--radix-eviction-policy gdsf"
run_one "v13-2q-evict" "" "--radix-eviction-policy 2q"
run_one "v14-sizelru-evict" "" "--radix-eviction-policy size_lru"

# --- Novel scheduling (new code, tagged "mechanism") ---
run_one "v15-srpf" "" "--schedule-policy srpf"

# --- XTIER + eviction combos ---
run_one "v16-xtier-cost" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_aware"
run_one "v17-xtier-lfu" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy lfu"
run_one "v18-xtier-2q" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy 2q"

# --- XTIER + scheduling combos ---
run_one "v19-xtier-lpm" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--schedule-policy lpm"
run_one "v20-xtier-srpf" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--schedule-policy srpf"

# --- XTIER knob variants ---
run_one "v21-xtier-wm05" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.05" ""
run_one "v22-xtier-batch64" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_BATCH=64" ""
run_one "v23-xtier-period1" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_PERIOD=1" ""

# --- Triple combos ---
run_one "v24-xtier-cost-lpm" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_aware --schedule-policy lpm"
run_one "v25-xtier-cost-srpf" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# --- Replicates for error bars ---
run_one "v26-baseline-rep" "" ""
run_one "v27-xtier-rep3" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" ""

# --- Cost-aware eviction threshold sweep ---
run_one "v28-cost-t4096" "SGLANG_COSTEVICT_THRESHOLD=4096" "--radix-eviction-policy cost_aware"
run_one "v29-cost-t1024" "SGLANG_COSTEVICT_THRESHOLD=1024" "--radix-eviction-policy cost_aware"

# --- Pressure-gated admission (PGAC) ---
run_one "v30-pgac99" "SGLANG_PGAC_THRESHOLD=0.99" ""
run_one "v31-pgac95" "SGLANG_PGAC_THRESHOLD=0.95" ""

# --- XTIER cost-aware backup selection ---
run_one "v32-xtier-costly" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_BACKUP_SELECT=costly" ""

# --- XTIER reuse-gated backup ---
run_one "v33-xtier-reuse1" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_REUSE_GATE=1" ""

# --- write_back + novel eviction combos ---
run_one "v34-wb-cost" "" "--hicache-write-policy write_back --radix-eviction-policy cost_aware"
run_one "v35-wb-lfu" "" "--hicache-write-policy write_back --radix-eviction-policy lfu"

# --- XTIER with cost threshold variants ---
run_one "v36-xtier-cost-t4096" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_COSTEVICT_THRESHOLD=4096" "--radix-eviction-policy cost_aware"
run_one "v37-xtier-cost-t1024" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_COSTEVICT_THRESHOLD=1024" "--radix-eviction-policy cost_aware"

# --- PGAC + XTIER ---
run_one "v38-xtier-pgac99" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_PGAC_THRESHOLD=0.99" ""

# --- write_back + XTIER (write_back makes XTIER a no-op but test to verify) ---
run_one "v39-wb-srpf" "" "--hicache-write-policy write_back --schedule-policy srpf"

# --- XTIER WM_FRAC sweep ---
run_one "v40-xtier-wm15" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.15" ""
run_one "v41-xtier-wm20" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.20" ""

# --- Cost eviction + SRPF ---
run_one "v42-cost-srpf" "" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# --- Frequency-decay eviction (LFUDA) ---
run_one "v43-lfuda" "" "--radix-eviction-policy lfuda"
run_one "v44-xtier-lfuda" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy lfuda"

# --- Cost+Frequency combined eviction ---
run_one "v45-cost-freq" "" "--radix-eviction-policy cost_freq"
run_one "v46-xtier-cost-freq" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_freq"

# --- Adaptive XTIER (wm_frac auto-adjusts based on unbacked eviction rate) ---
run_one "v47-xtier-adaptive" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_ADAPTIVE=1" ""

# --- DFS-weight scheduling (built-in) ---
run_one "v48-cfg-dfs" "" "--schedule-policy dfs-weight"
run_one "v49-xtier-dfs" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--schedule-policy dfs-weight"

echo "===== BATCH COMPLETE $(date) ====="
