#!/usr/bin/env bash
# run_eval_robust.sh <version> [extra eval args...]
# Held-pool eval with AUTO-RETRY on init-hang / collision. The cluster intermittently hangs during
# distributed init (FlashInfer/c10d 600s rendezvous timeout) under heavy concurrent load, and the
# shared held pool can be collided by a flock-bypassing neighbour. This wrapper:
#   1. picks a clean held node (disk>=1800G, dram>=1400G, no foreign sglang server) behind the flock
#   2. starts eval.sh into it (unique PORT)
#   3. monitors: once ">>> MIX" appears (serving), the hard part is past -> just wait for completion.
#      Before that, if server.log mtime freezes >6min (init hang) or a foreign server appears
#      (collision), it kills the attempt and retries on the next clean node.
#   nohup bash run_eval_robust.sh <version> [args...] > eval-<version>.log 2>&1 &
set -uo pipefail
NAME=onyx-7q2
ROOT=/home/junyanch_google_com/autoresearch
RESEARCHER=$ROOT/programs/sgl/researcher
EVAL=$RESEARCHER/.claude/skills/evaluation-sop/scripts/eval.sh
RT=$ROOT/programs/sgl/manager/.runtime
WORK=$ROOT/workspace/sgl/researchers/$NAME
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"; }
export PORT="${PORT:-30729}"
VER="${1:?usage: run_eval_robust.sh <version> [extra args...]}"; shift || true
SLOG="$WORK/runs/$VER/server.log"
BADNODES="slurm2-a3nodeset0-2"   # known-faulty (persistent FlashInfer-init hang)

held_nodes(){ compgen -G "$RT/held/*" >/dev/null 2>&1 && for f in "$RT/held"/*; do basename "$f"; done; }
foreign(){ srun --jobid="$1" --overlap -N1 -w "$2" pgrep -c -f sglang.launch_server 2>/dev/null; }

attempt=0
while :; do
  attempt=$((attempt+1))
  # ---- pick a clean node ----
  node=""; jid=""
  for n in $(held_nodes); do
    case " $BADNODES " in *" $n "*) continue;; esac
    j=$(cat "$RT/held/$n" 2>/dev/null) || continue
    squeue -h -j "$j" >/dev/null 2>&1 || continue
    d=$(srun --jobid="$j" --overlap -N1 -w "$n" df -BG /mnt/localssd 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4);print $4}')
    [ "${d:-0}" -lt 1800 ] && continue
    m=$(srun --jobid="$j" --overlap -N1 -w "$n" free -g 2>/dev/null | awk '/Mem:/{print $7}')
    [ "${m:-0}" -lt 1400 ] && continue
    [ "$(foreign "$j" "$n")" -gt 0 ] 2>/dev/null && continue
    node="$n"; jid="$j"; break
  done
  [ -z "$node" ] && { echo "[robust] attempt $attempt: no clean node, waiting..."; sleep 45; continue; }

  exec 200>"$RT/locks/$node.lock"
  flock -n 200 || { exec 200>&-; sleep 5; continue; }
  [ "$(foreign "$jid" "$node")" -gt 0 ] 2>/dev/null && { echo "[robust] $node foreign appeared"; flock -u 200; exec 200>&-; continue; }

  echo "[robust] attempt $attempt: eval $VER on $node (job $jid); port=$PORT extra=[$*]"
  rm -rf "$WORK/runs/$VER" 2>/dev/null
  srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 --export=ALL,PORT="$PORT" \
       bash "$EVAL" "$NAME" "$VER" "$@" &
  SR=$!

  # ---- monitor init ----
  serving=0
  while kill -0 $SR 2>/dev/null; do
    if grep -q ">>> MIX" "$WORK/eval-$VER.log" 2>/dev/null; then serving=1; break; fi
    # init-hang: server.log exists but frozen >6min
    if [ -f "$SLOG" ]; then
      age=$(( $(date +%s) - $(stat -c %Y "$SLOG" 2>/dev/null || echo 0) ))
      if [ "$age" -gt 480 ]; then echo "[robust] init hang on $node (log frozen ${age}s) -> retry"; break; fi
    fi
    # collision: a 2nd server on my node
    [ "$(foreign "$jid" "$node")" -gt 1 ] 2>/dev/null && { echo "[robust] collision on $node -> retry"; break; }
    sleep 30
  done

  if [ "$serving" -eq 1 ]; then
    echo "[robust] $VER serving on $node — waiting for completion"
    wait $SR; rc=$?
    flock -u 200; exec 200>&-
    echo "[robust] $VER finished rc=$rc on $node"; exit $rc
  fi

  # failed init -> kill this attempt, clean, retry
  kill $SR 2>/dev/null
  srun --jobid="$jid" --overlap -N1 -w "$node" pkill -9 -f sglang.launch_server 2>/dev/null
  flock -u 200; exec 200>&-
  sleep 10
done
