#!/usr/bin/env bash
# retry_eval.sh <version> [policy args...]
# Loops eval-on-pool until a run actually STARTS on a DRAM-healthy node (base_free cycles evals directly
# on the shared held nodes, bypassing the pool flock, so we keep landing on its low-DRAM nodes and DRAM-fail
# fast). A fast-fail (DRAM_TOO_LOW / NODE_NCCL_FAIL / SERVER_DIED) returns in <180s; a real eval blocks ~1.5h.
# We detect success by: the run's summary.json appears, OR the call blocks past START_GRACE (means it's serving).
set -uo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=/home/junyanch_google_com/autoresearch
SGL_HOME="$(cd "$SELF/../../../../../../programs/sgl/v0.25_ablations/sgl_free" && pwd 2>/dev/null)" || true
[ -d "$SGL_HOME/researcher" ] || SGL_HOME=/home/junyanch_google_com/autoresearch/programs/sgl/v0.25_ablations/sgl_free
EVALPOOL="$SGL_HOME/researcher/.claude/skills/submit-gpu-job/scripts/eval-on-pool.sh"
WS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free
VER="${1:?usage: retry_eval.sh <version> [args...]}"; shift
OUT="$WS/runs/$VER"; SUMMARY="$OUT/summary.json"
START_GRACE=${START_GRACE:-240}   # if a call runs longer than this, assume it's really serving
MAXTRIES=${MAXTRIES:-200}

echo "[retry] version=$VER args=[$*]  (grace ${START_GRACE}s)"
for try in $(seq 1 "$MAXTRIES"); do
  [ -f "$SUMMARY" ] && { echo "[retry] summary already exists -> done"; break; }
  rm -f "$OUT"/_started 2>/dev/null
  log="/tmp/sgl_free_retry_${VER}_try${try}.log"
  echo "[retry] try $try $(date -u +%H:%M:%S) -> $log"
  bash "$EVALPOOL" sgl_free "$VER" "$@" > "$log" 2>&1 &
  pid=$!
  # watch: if it dies fast (DRAM/NCCL fail) retry; if it survives the grace window it's serving -> wait it out
  alive_at_grace=0
  for s in $(seq 1 $((START_GRACE/10))); do
    sleep 10
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
  done
  if kill -0 "$pid" 2>/dev/null; then
    echo "[retry] try $try survived grace -> serving; waiting for completion"
    wait "$pid"; rc=$?
    if [ -f "$SUMMARY" ]; then echo "[retry] SUCCESS try $try (rc=$rc) summary written"; break; fi
    echo "[retry] try $try ended (rc=$rc) but no summary; tail:"; tail -3 "$log"
  else
    wait "$pid" 2>/dev/null; rc=$?
    echo "[retry] try $try fast-fail (rc=$rc): $(grep -oE 'DRAM_TOO_LOW[^\n]*|NODE_NCCL_FAIL[^\n]*|SERVER_DIED[^\n]*|all held nodes busy' "$log" | head -1)"
  fi
  sleep 45
done
[ -f "$SUMMARY" ] && echo "[retry] DONE: $SUMMARY" || echo "[retry] gave up after $MAXTRIES tries"
