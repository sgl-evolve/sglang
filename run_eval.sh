#!/usr/bin/env bash
# Robust eval launcher for the SHARED 4-cell pool: retries on transient node collisions
# (DRAM_TOO_LOW / NCCL fail / server-died = a sibling occupies the flock-free "held" node).
# Stops on real outcomes (exit 0 success; 5 refused; 8 bench failed).
# Usage: bash run_eval.sh <version> [extra eval args...]
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_free/researchers/base_free
_SGL_HOME="$SGL_HOME"; _SGL_WS="$SGL_WORKSPACE"
set -a; source /home/junyanch_google_com/autoresearch/.env; set +a
export SGL_HOME="$_SGL_HOME"; export SGL_WORKSPACE="$_SGL_WS"; export HF_TOKEN="$HF_API_KEY"
export SGLANG_DG_CACHE_DIR=/mnt/localssd/base_free_dg
# WSAC knobs passed through the environment (default off unless caller exported them)
EVALPOOL="$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
VER="${1:?version}"; shift || true
for attempt in $(seq 1 20); do
  echo "[run_eval] attempt $attempt for $VER $(date +%H:%M:%S)"
  bash "$EVALPOOL" base_free "$VER" "$@"
  rc=$?
  echo "[run_eval] eval-on-pool rc=$rc"
  case $rc in
    0) echo "[run_eval] SUCCESS $VER"; exit 0 ;;
    2|3|6) echo "[run_eval] transient (node busy/DRAM/NCCL/died) — retry in 45s"; sleep 45 ;;
    5) echo "[run_eval] REFUSED arg — stop"; exit 5 ;;
    8) echo "[run_eval] bench failed — stop"; exit 8 ;;
    *) echo "[run_eval] rc=$rc — retry in 45s"; sleep 45 ;;
  esac
done
echo "[run_eval] gave up after 20 attempts"; exit 1
