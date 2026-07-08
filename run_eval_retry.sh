#!/usr/bin/env bash
# Auto-retry eval-on-pool until a bench actually starts (handles flock-loss to
# sibling cell + DRAM-gate failures on just-freed nodes). Stops once the eval
# reaches the MIX bench or completes. Usage: run_eval_retry.sh <version> [extra args...]
set -uo pipefail
SGL_HOME="/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/base_mech"
WORK="/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/base_mech/researchers/base_mech"
cd "$SGL_HOME/researcher"
VER="${1:?usage: run_eval_retry.sh <version> [args...]}"; shift || true
LOG="$WORK/eval-$VER.log"
for attempt in $(seq 1 200); do
  echo "[retry] attempt $attempt for $VER at $(date -u +%H:%M:%SZ)" >> "$WORK/retry-$VER.log"
  bash .claude/skills/submit-gpu-job/scripts/eval-on-pool.sh base_mech "$VER" "$@" > "$LOG" 2>&1
  # success signals: bench started or eval done
  if grep -qE "EVAL_DONE|Serving Benchmark Result|MIX \(ShareGPT" "$LOG" 2>/dev/null; then
    echo "[retry] $VER reached bench/done on attempt $attempt" >> "$WORK/retry-$VER.log"
    break
  fi
  # transient failures -> wait then retry (DRAM settling / flock lost)
  tail -3 "$LOG" >> "$WORK/retry-$VER.log" 2>/dev/null
  sleep 40
done
