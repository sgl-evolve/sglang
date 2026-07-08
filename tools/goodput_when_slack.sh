#!/usr/bin/env bash
# goodput_when_slack.sh — capture the DEFINITIVE node-controlled, full-protocol, both-knees-bracketed
# goodput A/B, using only capacity nobody else needs. Fires when the shared pool has genuine slack
# (>=SLACK_MIN nodes simultaneously free) OR when a node has been idle for >=IDLE_STREAK consecutive polls
# (persistently-idle = genuinely-wasted capacity, not an actively-cycled sibling buffer) — so it never
# starves active siblings. Grabs ONE such node, holds its flock across BOTH legs (baseline fcfs then
# exclusive), sweeps full-protocol rates bracketing both knees.
# Idempotent (skips if done), time-budgeted (won't linger), DRAM-preflighted (sweep bails on occupied nodes).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
ONNODE="$WS/tools/sweep_onnode.sh"
RATES="${RATES:-3 4 4.5}"        # baseline knee bracketed 3-4; exclusive knee bracketed 4-4.5
NP="${NP:-1553}"                 # FULL protocol
BASE=defgood_base; EXC=defgood_exc
SLACK_MIN="${SLACK_MIN:-2}"      # fire if >=2 free nodes (pool has slack)
IDLE_STREAK="${IDLE_STREAK:-3}"  # OR fire if a node is grabbable for >=3 consecutive polls (persistently
                                 # idle = genuinely-wasted capacity, not an actively-cycled sibling buffer)
MAX_POLLS="${MAX_POLLS:-144}"    # ~12h at 5-min cadence, then give up (don't linger)
held_nodes(){ for f in "$POOL"/held/*; do [ -e "$f" ] && basename "$f"; done; }
is_free(){ local lk="$POOL/locks/$1.lock" r=1; if exec 202>"$lk" && flock -n 202; then r=0; flock -u 202; fi; exec 202>&- 2>/dev/null; return $r; }
declare -A STREAK   # per-node consecutive-idle poll count

echo "[gws] start $(date -u +%FT%TZ); fire if >=$SLACK_MIN free OR a node idle >=$IDLE_STREAK polls; rates=[$RATES] np=$NP"
for poll in $(seq 1 "$MAX_POLLS"); do
  [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && { echo "[gws] already done"; exit 0; }
  # update per-node idle streaks + collect currently-free nodes (ordered: highest streak first)
  slack=0; persistent=""; freelist=""
  for node in $(held_nodes); do
    if is_free "$node"; then slack=$((slack+1)); STREAK[$node]=$(( ${STREAK[$node]:-0} + 1 ));
      freelist+="$node "; [ "${STREAK[$node]}" -ge "$IDLE_STREAK" ] && persistent+="$node ";
    else STREAK[$node]=0; fi
  done
  echo "[gws] poll $poll/$MAX_POLLS $(date -u +%H:%M:%S): $slack free [$freelist] persistent-idle:[${persistent:-none}]"
  fire=0; order="$freelist"
  if [ "${slack:-0}" -ge "$SLACK_MIN" ]; then fire=1; fi
  if [ -n "$persistent" ]; then fire=1; order="$persistent $freelist"; fi   # prefer persistently-idle node
  if [ "$fire" = 1 ]; then
    for node in $order; do
      jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
      squeue -h -j "$jid" >/dev/null 2>&1 || continue
      exec 200>"$POOL/locks/$node.lock"
      if flock -n 200; then
        echo "[gws] GRABBED $node (job $jid) $(date -u +%H:%M:%S) — running baseline then exclusive"
        unset SGLANG_HICACHE_EXCLUSIVE
        RATES="$RATES" NPROMPTS="$NP" srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$ONNODE" "$BASE"
        rcb=$?
        if [ ! -f "$WS/runs/sweep_$BASE/curve.csv" ]; then
          echo "[gws] baseline leg failed on $node (rc=$rcb; likely DRAM) — release + keep polling"
          flock -u 200; exec 200>&-; break
        fi
        export SGLANG_HICACHE_EXCLUSIVE=1
        RATES="$RATES" NPROMPTS="$NP" srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$ONNODE" "$EXC"
        flock -u 200; exec 200>&-
        [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && { echo "[gws] SUCCESS both legs on $node"; break; }
        echo "[gws] exclusive leg failed on $node — will retry"; break
      fi
      exec 200>&- 2>/dev/null
    done
    [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && break
  fi
  sleep 300
done
echo "[gws] end $(date -u +%FT%TZ)"
if [ -f "$WS/runs/sweep_$EXC/curve.csv" ]; then
  echo "=== DEFINITIVE SAME-NODE GOODPUT CURVES (full protocol, np=$NP) ==="
  echo "baseline:";  cat "$WS/runs/sweep_$BASE/curve.csv"
  echo "exclusive:"; cat "$WS/runs/sweep_$EXC/curve.csv"
else
  echo "[gws] no slack window within budget — definitive run NOT taken (existing 3 evidence pieces stand)."
fi
