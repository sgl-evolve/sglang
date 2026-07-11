#!/usr/bin/env bash
# batch_eval5.sh — experiments v63+. Focused on:
#   - Alternative eviction policies in the three-way combo (CostFreq, LFUDA)
#   - Clean ablation table (isolate each mechanism's marginal)
#   - Replicates for statistical confidence
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

  local logfile="$LOGDIR/eval5-$ver.out"
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

# ===== EXPERIMENT QUEUE v63+ =====

# --- Alternative eviction in three-way combo ---
# XTIER + CostFreq + SRPF (cost * (1+hit_count) — combines recompute cost AND frequency)
run_one "v63-xtier-costfreq-srpf" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_freq --schedule-policy srpf"

# XTIER + LFUDA + SRPF (frequency with dynamic aging — prevents stale popular nodes)
run_one "v64-xtier-lfuda-srpf" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_LFUDA_HALFLIFE=500" "--radix-eviction-policy lfuda --schedule-policy srpf"

# --- Clean ablation: strip each mechanism from the three-way ---
# XTIER-only (LRU eviction, no SRPF) — isolate XTIER's solo contribution
run_one "v65-xtier-only" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" ""

# XTIER + SRPF (no CostAware) — what does CostAware add on top?
run_one "v66-xtier-srpf" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--schedule-policy srpf"

# wb + CostAware (no SRPF) — already have v34 (hit 0.736/p99 4388), this is the match
# SKIP v67 — v34-wb-cost already covers this (write_back + cost_aware, no SRPF)

# CostAware + SRPF (no XTIER, no wb — inclusive tiering) — does the combo work without exclusive?
run_one "v67-cost-srpf-inclusive" "" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# SRPF-only (no XTIER, no CostAware, inclusive tiering)
run_one "v68-srpf-inclusive" "" "--schedule-policy srpf"

# --- Replicates for statistical confidence ---
# Baseline replicate 3
run_one "v69-baseline-rep3" "" ""

# Three-way best replicate 3 (v25/v57 are rep1/rep2)
run_one "v70-xtier-cost-srpf-rep3" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# wb+SRPF replicate 2 (v39 was rep1 — compare to three-way)
run_one "v71-wb-srpf-rep2" "" "--hicache-write-policy write_back --schedule-policy srpf"

# --- Novel mechanism: WM_FRAC micro-sweep ---
# WM_FRAC 0.03 (even less backup — pushing the boundary)
run_one "v72-xtier-cost-srpf-wm03" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.03" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# WM_FRAC 0.02 (extreme minimal backup)
run_one "v73-xtier-cost-srpf-wm02" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.02" "--radix-eviction-policy cost_aware --schedule-policy srpf"

# --- LFUDA halflife sweep ---
# LFUDA halflife=200 (faster aging — more aggressive frequency decay)
run_one "v74-xtier-lfuda200-srpf" "SGLANG_XTIER_LAZY=1 SGLANG_XTIER_WM_FRAC=0.1 SGLANG_LFUDA_HALFLIFE=200" "--radix-eviction-policy lfuda --schedule-policy srpf"

echo "===== BATCH5 COMPLETE $(date) ====="
