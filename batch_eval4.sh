#!/usr/bin/env bash
# batch_eval4.sh — additional experiments beyond v49.
# Same structure as batch_eval3.sh. Submits ONE job at a time.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free

_SGL_HOME="$SGL_HOME"; _SGL_WS="$SGL_WORKSPACE"
set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export SGL_HOME="$_SGL_HOME"; export SGL_WORKSPACE="$_SGL_WS"; export HF_TOKEN="$HF_API_KEY"
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_free_dg

EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
CERT_SCRIPT="$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/certified-nodes.sh"
LOGDIR="$PWD/eval-logs"
mkdir -p "$LOGDIR"

run_one() {
  local ver="$1" envs="$2" extras="$3"
  if [[ -f "$PWD/runs/$ver/summary.json" ]]; then
    echo "===== [$ver] already done — skip ====="
    return 0
  fi
  echo "===== [$ver] starting $(date) ====="

  local env_prefix="export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_free_dg"
  env_prefix="$env_prefix; unset SGLANG_XTIER_LAZY SGLANG_XTIER_WM_FRAC SGLANG_XTIER_BATCH SGLANG_XTIER_PERIOD SGLANG_XTIER_ADAPTIVE SGLANG_XTIER_BACKUP_SELECT SGLANG_XTIER_REUSE_GATE SGLANG_COSTEVICT_THRESHOLD SGLANG_PGAC_THRESHOLD SGLANG_PGAC_COLD_RATIO SGLANG_LFUDA_HALFLIFE 2>/dev/null || true"
  if [[ -n "$envs" ]]; then
    env_prefix="$env_prefix; export $envs"
  fi

  local logfile="$LOGDIR/eval4-$ver.out"
  local excl
  excl=$(bash "$CERT_SCRIPT" --exclude 2>/dev/null || true)
  local ex_args=()
  [[ -n "$excl" ]] && ex_args=(--exclude="$excl")

  for attempt in $(seq 1 10); do
    echo "[$ver] attempt $attempt $(date +%H:%M:%S)"

    local jid
    jid=$(sbatch --parsable -p a3 -N1 --exclusive --gres=gpu:8 "${ex_args[@]}" -t 6:00:00 \
          -J "eval-$ver" -o "$logfile" \
          --wrap "$env_prefix; bash $EVAL base_free $ver $extras") 2>/dev/null

    if [[ -z "$jid" ]]; then
      echo "[$ver] sbatch failed — retry in 60s"
      sleep 60
      continue
    fi
    echo "[$ver] submitted job $jid"

    local waited=0
    while (( waited < 14400 )); do
      sleep 30
      waited=$((waited + 30))
      local state
      state=$(squeue -h -j "$jid" -o "%t" 2>/dev/null || true)
      if [[ -z "$state" ]]; then
        break
      fi
      if (( waited % 300 == 0 )); then
        local prog
        prog=$(grep -oP '\d+/7037' "runs/$ver/mix.txt" 2>/dev/null | tail -1 || echo "?")
        echo "[$ver] running ($((waited/60))m, progress=$prog)"
      fi
    done

    if [[ -f "$PWD/runs/$ver/summary.json" ]]; then
      echo "[$ver] SUCCESS $(date)"
      return 0
    fi
    echo "[$ver] job $jid finished without summary.json — retry"
    sleep 30
  done
  echo "[$ver] gave up after 10 attempts"
  return 1
}

# ===== EXPERIMENT QUEUE v50+ =====
# Three-way best combo (XTIER+CostAware+SRPF) with different WM_FRAC
run_one "v50-xtier-cost-srpf-wm05" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.05" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# write_back + CostAware + SRPF (three-way without novel XTIER — compare to v25)
run_one "v51-wb-cost-srpf" "" "--hicache-write-policy write_back --radix-eviction-policy cost_aware --schedule-policy srpf"

# XTIER + 2Q + SRPF (2Q was only non-LRU that matched LRU — test as three-way)
run_one "v52-xtier-2q-srpf" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy 2q --schedule-policy srpf"

# XTIER + CostAware + REUSE_GATE (test if filtering one-shot docs helps)
run_one "v53-xtier-cost-reuse1" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_REUSE_GATE=1" "--radix-eviction-policy cost_aware"

# Four-way: XTIER + CostAware + SRPF + REUSE_GATE
run_one "v54-xtier-cost-srpf-reuse1" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_XTIER_REUSE_GATE=1" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# Three-way with higher WM_FRAC
run_one "v55-xtier-cost-srpf-wm15" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.15" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# Baseline replicate 2 (for cross-validation of node variance)
run_one "v56-baseline-rep2" "" ""

# XTIER best three-way replicate (for statistical confidence)
run_one "v57-xtier-cost-srpf-rep2" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# write_back + 2Q (compare stock write_back + 2Q to inclusive baseline)
run_one "v58-wb-2q" "" "--hicache-write-policy write_back --radix-eviction-policy 2q"

# XTIER + CostAware(t=1024) + SRPF (threshold sweep in three-way combo)
run_one "v59-xtier-cost-srpf-t1024" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_COSTEVICT_THRESHOLD=1024" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# XTIER + CostAware(t=4096) + SRPF (threshold sweep in three-way combo)
run_one "v60-xtier-cost-srpf-t4096" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_COSTEVICT_THRESHOLD=4096" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# SRPF with raised queue threshold (1024 vs default 128) — test if SRPF works better on longer queues
run_one "v61-xtier-cost-srpf-qlim" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# XTIER + 2Q eviction (2Q matched LRU in v13; test with XTIER for comparison to XTIER+CostAware)
run_one "v62-xtier-2q" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy 2q"

echo "===== BATCH4 COMPLETE $(date) ====="
