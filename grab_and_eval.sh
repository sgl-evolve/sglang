#!/usr/bin/env bash
# grab_and_eval.sh — reliable node acquisition for base_mech under heavy cross-cell
# contention. Replicates eval-on-pool's flock loop BUT, after grabbing a held node's
# flock, HOLDS it while waiting for that node's DRAM to settle (>=1.3T) — instead of
# bailing on the transient low-DRAM window right after a sibling's eval frees the node.
# Then srun's the frozen eval.sh. Usage: grab_and_eval.sh <version> [eval args...]
set -uo pipefail
ROOT="/home/junyanch_google_com/autoresearch"
SGL_HOME="$ROOT/programs/sgl/v0.25_ablations/base_mech"
EVAL="$SGL_HOME/researcher/.claude/skills/evaluation-sop/scripts/eval.sh"
POOL="${SGL_POOL_DIR:-$ROOT/workspace/sgl/v0.25_ablations/_pool}"
NAME=base_mech
VER="${1:?usage: grab_and_eval.sh <version> [eval args...]}"; shift || true

held_nodes(){ compgen -G "$POOL/held/*" >/dev/null 2>&1 && for f in "$POOL/held"/*; do basename "$f"; done; }

while :; do
  any_alive=0
  for node in $(held_nodes); do
    jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    any_alive=1
    exec 200>"$POOL/locks/$node.lock"
    if flock -n 200; then
      echo "[grab] got flock on $node (hold $jid) at $(date -u +%H:%M:%SZ); checking DRAM..."
      dram_ok=0
      for i in $(seq 1 8); do   # up to ~2min holding the flock while sibling teardown frees memory
        dram=$(srun --jobid="$jid" --overlap -N1 -w "$node" \
                 awk '/MemAvailable/{printf "%d",$2/1048576}' /proc/meminfo 2>/dev/null)
        echo "[grab]  $node DRAM=${dram:-?}G (need>=1300) try $i"
        if [ "${dram:-0}" -ge 1300 ]; then dram_ok=1; break; fi
        sleep 15
      done
      if [ "$dram_ok" -eq 1 ]; then
        echo "[grab] DRAM settled on $node — launching eval.sh $VER (BM_WARMFIRST=${BM_WARMFIRST:-unset} BM_WARM_THRESHOLD=${BM_WARM_THRESHOLD:-default})"
        EXTRA_STR=""; [ "$#" -gt 0 ] && EXTRA_STR=$(printf "%q " "$@")
        ENVPFX=""
        [ -n "${BM_WARMFIRST:-}" ] && ENVPFX="$ENVPFX export BM_WARMFIRST=$(printf %q "$BM_WARMFIRST");"
        [ -n "${BM_WARM_THRESHOLD:-}" ] && ENVPFX="$ENVPFX export BM_WARM_THRESHOLD=$(printf %q "$BM_WARM_THRESHOLD");"
        [ -n "${BM_AGE_LIMIT_S:-}" ] && ENVPFX="$ENVPFX export BM_AGE_LIMIT_S=$(printf %q "$BM_AGE_LIMIT_S");"
        [ -n "${BM_EXCL:-}" ] && ENVPFX="$ENVPFX export BM_EXCL=$(printf %q "$BM_EXCL");"
        [ -n "${BM_EXCL_KEEP_HITS:-}" ] && ENVPFX="$ENVPFX export BM_EXCL_KEEP_HITS=$(printf %q "$BM_EXCL_KEEP_HITS");"
        srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 \
          bash -c "$ENVPFX exec bash '$EVAL' '$NAME' '$VER' $EXTRA_STR"
        rc=$?
        flock -u 200; exec 200>&-
        echo "[grab] eval.sh exited rc=$rc on $node"
        exit $rc
      fi
      echo "[grab] DRAM never settled on $node — releasing flock, trying another"
      flock -u 200
    fi
    exec 200>&-
  done
  [ "$any_alive" -eq 0 ] && { echo "[grab] no live held nodes; sleeping"; }
  sleep 20
done
