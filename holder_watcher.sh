#!/usr/bin/env bash
# holder_watcher.sh — my GUARANTEED, race-free eval path. My exclusive 8h a3 hold job (JOBID in
# .holdjob) is pending for a node. The shared held pool is unwinnable (foreign researchers cycle their
# pinned nodes with ~0 gap; my collision-safe waiter loses every micro-race), so this watcher makes the
# exclusive holder productive: when it starts, disk-check its node; if eval-capable (>=1800G, healthy),
# run my queue there SEQUENTIALLY (one 122B eval at a time, ~2h each, ~4 fit in 8h), autologging each.
# If it landed on a small-disk / unusable node, RELEASE it (never squat a node I can't eval on).
# Pops from the SAME experiment_queue.txt under the SAME lock as the waiters -> no double-runs.
set -uo pipefail
NAME=onyx-7q2
ROOT=/home/junyanch_google_com/autoresearch
EVAL=$ROOT/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
WORK=$ROOT/workspace/sgl/researchers/$NAME
Q=$WORK/experiment_queue.txt
PORT=30761
cd "$WORK"
[ -f "$ROOT/.env" ] && { set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"; export WANDB_API_KEY; }
JID=$(cat "$WORK/.holdjob" 2>/dev/null)
[ -z "$JID" ] && { echo "[holder] no .holdjob -> exit"; exit 0; }

# SINGLE-INSTANCE: never allow two holder_watchers (a duplicate double-launched v11 onto the same node
# and corrupted the hold). A 2nd instance exits immediately.
exec 210>/tmp/onyx-7q2-holder.lock
flock -n 210 || { echo "[holder] another holder_watcher holds the lock -> exit"; exit 0; }

qpop(){ exec 201>"$Q.lock"; flock 201; local l; l=$(grep -vE '^[[:space:]]*$' "$Q" 2>/dev/null|head -1); [ -n "$l" ] && { grep -vFx "$l" "$Q">"$Q.tmp" 2>/dev/null; mv "$Q.tmp" "$Q"; }; flock -u 201; exec 201>&-; printf '%s' "$l"; }
qpush_front(){ exec 201>"$Q.lock"; flock 201; { printf '%s\n' "$1"; cat "$Q" 2>/dev/null; } > "$Q.tmp" && mv "$Q.tmp" "$Q"; flock -u 201; exec 201>&-; }

echo "[holder] watching hold job $JID for a node..."
while :; do
  st=$(squeue -h -j "$JID" -o "%T" 2>/dev/null)
  [ -z "$st" ] && { echo "[holder] job $JID gone (done/cancelled) -> exit"; exit 0; }
  [ "$st" != "RUNNING" ] && { sleep 60; continue; }
  NODE=$(squeue -h -j "$JID" -o "%N" 2>/dev/null); [ -z "$NODE" ] && { sleep 10; continue; }
  echo "[holder] job $JID RUNNING on $NODE"
  # Clean MY OWN leftovers first (a freed pool node often still holds prior L3 caches; I may only
  # delete my own dir, never foreign data or foreign processes). Then measure free disk.
  srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c 'rm -rf /mnt/localssd/onyx-7q2 2>/dev/null; true' 2>/dev/null
  # disk/dram health of my exclusive node
  info=$(srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c "df -BG /mnt/localssd 2>/dev/null|tail -1|awk '{gsub(/G/,\"\",\$4);print \$4}'; free -g 2>/dev/null|awk '/Mem:/{print \$7}'" 2>/dev/null)
  d=$(printf '%s\n' "$info"|sed -n 1p); m=$(printf '%s\n' "$info"|sed -n 2p)
  if [ "${d:-0}" -lt 1800 ] || [ "${m:-0}" -lt 1400 ]; then
    echo "[holder] $NODE not eval-capable (disk=${d}G dram=${m}G) -> record bad + release (scancel $JID)"
    echo "$NODE" >> bad_nodes.txt; scancel "$JID"; exit 0
  fi
  # GPU preflight: a freed node can carry a zombie holding GPU memory (slurm didn't reap it) -> every
  # eval would NCCL/OOM (rc=6). Refuse + release if any GPU already has >2GB used (not a clean 8-GPU node).
  gpu=$(srun --jobid="$JID" --overlap -N1 -w "$NODE" bash -c "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|sort -rn|head -1" 2>/dev/null)
  if [ "${gpu:-0}" -gt 2000 ]; then
    echo "[holder] $NODE GPU-wedged (max mem.used=${gpu}MB, zombie holding GPU) -> record bad + release"
    echo "$NODE" >> bad_nodes.txt; scancel "$JID"; exit 0
  fi
  echo "[holder] $NODE eval-capable (disk=${d}G dram=${m}G gpu_used=${gpu:-0}MB) -> running queue sequentially"
  break
done

# Run the queue sequentially in my exclusive node until it drains or the hold job ends.
# ALWAYS release the node (scancel) when done hill-climbing — never squat it (per submit-gpu-job skill).
while :; do
  squeue -h -j "$JID" -o "%T" 2>/dev/null | grep -q RUNNING || { echo "[holder] hold job ended -> exit"; exit 0; }
  exp=$(qpop); [ -z "$exp" ] && { echo "[holder] queue drained -> releasing hold (scancel $JID)"; scancel "$JID" 2>/dev/null; exit 0; }
  IFS='|' read -r lbl env flags tag <<<"$exp"
  echo "[holder] === running $lbl (env=[$env] flags=[$flags]) on $NODE ==="
  rm -rf "$WORK/runs/$lbl" 2>/dev/null
  ( unset SGLANG_HICACHE_FILE_READ_THREADS SGLANG_HICACHE_FILE_WRITE_THREADS SGLANG_HICACHE_PREFETCH_TIMEOUT_BASE SGLANG_HICACHE_PREFETCH_TIMEOUT_PER_KI SGLANG_HICACHE_PREFETCH_TIMEOUT_MAX
    [ "$env" != "-" ] && export $env
    export PORT
    srun --jobid="$JID" --overlap -N1 -w "$NODE" --gres=gpu:8 --export=ALL,PORT="$PORT" \
      bash "$EVAL" "$NAME" "$lbl" --enforce-disable-flashinfer-allreduce-fusion $flags > "eval-$lbl.log" 2>&1 )
  rc=$?
  echo "[holder] $lbl finished rc=$rc"
  if [ -f "runs/$lbl/summary.json" ]; then
    # SUCCESS -> autolog + advance to next experiment
    bash finish_eval.sh "$lbl" "$tag" >> "logs_finish_$lbl.txt" 2>&1; touch "runs/$lbl/.logged"; echo "[holder] $lbl autologged"
  else
    # FAILURE (rc=6 NCCL/OOM node-wedge, or other) -> DO NOT drain the queue. Re-queue this experiment
    # at the front, record the node as bad, release it, and exit. A fresh hold (excluding bad nodes)
    # retries $lbl on a healthy node. Only successful evals advance the queue.
    echo "[holder] $lbl FAILED (rc=$rc, no summary) -> re-queue front + record bad node $NODE + release"
    qpush_front "$exp"; echo "$NODE" >> bad_nodes.txt; scancel "$JID"; exit 0
  fi
done
