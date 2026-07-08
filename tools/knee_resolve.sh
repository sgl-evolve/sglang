#!/usr/bin/env bash
# knee_resolve.sh — resolve the p99<=8s SLO goodput KNEE that the coarse defgood grid (3/4/4.5) left
# co-bracketed in (3,4) for BOTH baseline and exclusive. Same contention-gated, same-node, full-protocol
# A/B as goodput_when_slack.sh, but a FINE rate grid straddling the 8s crossing: 3.25/3.5/3.75. Holds one
# idle pool node's flock across both legs (baseline fcfs then exclusive), so the knee shift is node-controlled.
# Fires on >=SLACK_MIN free nodes OR a node idle >=IDLE_STREAK consecutive polls (never starves siblings).
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
POOL="$ROOT/workspace/sgl/v0.25_ablations/_pool"
WS="$ROOT/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free"
ONNODE="$WS/tools/sweep_onnode.sh"
RATES="${RATES:-3.25 3.5 3.75}"   # fine grid across the (3,4) 8s crossing
NP="${NP:-1553}"                  # FULL protocol
BASE=knee_base; EXC=knee_exc
SLACK_MIN="${SLACK_MIN:-2}"
IDLE_STREAK="${IDLE_STREAK:-3}"
MAX_POLLS="${MAX_POLLS:-144}"
held_nodes(){ for f in "$POOL"/held/*; do [ -e "$f" ] && basename "$f"; done; }
is_free(){ local lk="$POOL/locks/$1.lock" r=1; if exec 202>"$lk" && flock -n 202; then r=0; flock -u 202; fi; exec 202>&- 2>/dev/null; return $r; }
declare -A STREAK

echo "[knee] start $(date -u +%FT%TZ); fire if >=$SLACK_MIN free OR node idle >=$IDLE_STREAK polls; rates=[$RATES] np=$NP"
for poll in $(seq 1 "$MAX_POLLS"); do
  [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && { echo "[knee] already done"; exit 0; }
  slack=0; persistent=""; freelist=""
  for node in $(held_nodes); do
    if is_free "$node"; then slack=$((slack+1)); STREAK[$node]=$(( ${STREAK[$node]:-0} + 1 ));
      freelist+="$node "; [ "${STREAK[$node]}" -ge "$IDLE_STREAK" ] && persistent+="$node ";
    else STREAK[$node]=0; fi
  done
  echo "[knee] poll $poll/$MAX_POLLS $(date -u +%H:%M:%S): $slack free [$freelist] persistent-idle:[${persistent:-none}]"
  fire=0; order="$freelist"
  [ "${slack:-0}" -ge "$SLACK_MIN" ] && fire=1
  [ -n "$persistent" ] && { fire=1; order="$persistent $freelist"; }
  if [ "$fire" = 1 ]; then
    for node in $order; do
      jid=$(cat "$POOL/held/$node" 2>/dev/null) || continue
      squeue -h -j "$jid" >/dev/null 2>&1 || continue
      exec 200>"$POOL/locks/$node.lock"
      if flock -n 200; then
        echo "[knee] GRABBED $node (job $jid) $(date -u +%H:%M:%S) — baseline then exclusive"
        unset SGLANG_HICACHE_EXCLUSIVE
        RATES="$RATES" NPROMPTS="$NP" srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$ONNODE" "$BASE"
        if [ ! -f "$WS/runs/sweep_$BASE/curve.csv" ]; then
          echo "[knee] baseline leg failed on $node — release + keep polling"; flock -u 200; exec 200>&-; break
        fi
        export SGLANG_HICACHE_EXCLUSIVE=1
        RATES="$RATES" NPROMPTS="$NP" srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$ONNODE" "$EXC"
        flock -u 200; exec 200>&-
        [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && { echo "[knee] SUCCESS both legs on $node"; break; }
        echo "[knee] exclusive leg failed on $node — will retry"; break
      fi
      exec 200>&- 2>/dev/null
    done
    [ -f "$WS/runs/sweep_$EXC/curve.csv" ] && break
  fi
  sleep 300
done
echo "[knee] end $(date -u +%FT%TZ)"
if [ -f "$WS/runs/sweep_$EXC/curve.csv" ]; then
  echo "=== KNEE-RESOLVER SAME-NODE CURVES (full protocol, np=$NP, fine grid) ==="
  echo "baseline:";  cat "$WS/runs/sweep_$BASE/curve.csv"
  echo "exclusive:"; cat "$WS/runs/sweep_$EXC/curve.csv"
else
  echo "[knee] no slack window within budget — knee-resolver NOT taken (defgood + rate-3 win stand)."
fi
