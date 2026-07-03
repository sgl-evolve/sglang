#!/usr/bin/env bash
# run_eval.sh — like eval-on-pool.sh but pre-checks /mnt/localssd free (>=1800G) so we skip
# held nodes whose L3 dir is full (e.g. node0-0 with 614G). Locks the first good held node via
# the same flock the pool uses, srun's the frozen eval.sh into it. Run in background + tail.
#   nohup bash run_eval.sh <version> [extra eval args...] > eval-<version>.log 2>&1 &
set -uo pipefail
NAME=onyx-7q2
ROOT=/home/junyanch_google_com/autoresearch
RESEARCHER=$ROOT/programs/sgl/researcher
EVAL=$RESEARCHER/.claude/skills/evaluation-sop/scripts/eval.sh
RT=$ROOT/programs/sgl/manager/.runtime
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"; }
VER="${1:?usage: run_eval.sh <version> [extra args...]}"; shift || true
# Unique port so we never collide with a foreign server on the shared held node's :30000.
export PORT="${PORT:-30729}"

held_nodes(){ compgen -G "$RT/held/*" >/dev/null 2>&1 && for f in "$RT/held"/*; do basename "$f"; done; }

while :; do
  any=0
  for node in $(held_nodes); do
    jid=$(cat "$RT/held/$node" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    any=1
    # disk pre-check (must be >=1800G to hold the L3 tier)
    free=$(srun --jobid="$jid" --overlap -N1 -w "$node" df -BG /mnt/localssd 2>/dev/null | tail -1 | awk '{gsub(/G/,"",$4);print $4}')
    [ "${free:-0}" -lt 1800 ] && { echo "[run_eval] skip $node: only ${free}G ssd free"; continue; }
    # DRAM pre-check (need ~1.3TB free for 768G pinned host tier + weights; skip busy/low nodes)
    dram=$(srun --jobid="$jid" --overlap -N1 -w "$node" free -g 2>/dev/null | awk '/Mem:/{print $7}')
    [ "${dram:-0}" -lt 1400 ] && { echo "[run_eval] skip $node: only ${dram}G dram avail"; continue; }
    # Foreign-server check: the held pool is shared via --overlap and coordination
    # is imperfect; a second 122B server on the node collides (port + 2x768G -> OOM).
    # Skip any node that already has a foreign sglang server, even if the flock is free.
    foreign=$(srun --jobid="$jid" --overlap -N1 -w "$node" pgrep -c -f sglang.launch_server 2>/dev/null)
    [ "${foreign:-0}" -gt 0 ] && { echo "[run_eval] skip $node: ${foreign} foreign sglang server(s) present"; continue; }
    exec 200>"$RT/locks/$node.lock"
    if flock -n 200; then
      # re-check foreign server after acquiring the lock (race window)
      foreign=$(srun --jobid="$jid" --overlap -N1 -w "$node" pgrep -c -f sglang.launch_server 2>/dev/null)
      if [ "${foreign:-0}" -gt 0 ]; then echo "[run_eval] skip $node: foreign server appeared"; flock -u 200; exec 200>&-; continue; fi
      echo "[run_eval] eval $VER on held node $node (job $jid, ${free}G ssd, ${dram}G dram); port=$PORT extra=[$*]"
      srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 --export=ALL,PORT="$PORT" bash "$EVAL" "$NAME" "$VER" "$@"
      rc=$?; flock -u 200; exec 200>&-; exit $rc
    fi
    exec 200>&-
  done
  [ "$any" -eq 0 ] && { echo "[run_eval] no live held nodes"; exit 1; }
  echo "[run_eval] all good held nodes busy — waiting..."; sleep 30
done
