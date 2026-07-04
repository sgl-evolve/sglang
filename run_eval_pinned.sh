#!/usr/bin/env bash
# Disk-aware, lock-respecting eval launcher over the manager's held certified pool.
# flock-FIRST (grab the node the instant it frees), THEN disk-check; skip held nodes with < 1.8T free.
# On a transient load failure (hang/timeout during warmup), RETRY on the same held node up to
# MAXTRY times WHILE HOLDING THE FLOCK (so we don't lose the node to contention); if still failing,
# release and move to another node. Exits 0 only on a clean eval (exit 0) or a real MIX failure.
# Usage: run_eval_pinned.sh <name> <version> [eval args...]
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a; export HF_TOKEN="$HF_API_KEY"
RT=/home/junyanch_google_com/autoresearch/programs/sgl/manager/.runtime
EVAL=/home/junyanch_google_com/autoresearch/programs/sgl/researcher/.claude/skills/evaluation-sop/scripts/eval.sh
NAME="${1:?name}"; VER="${2:?version}"; shift 2
MINGB="${MINGB:-1800}"; MAXTRY=2   # best_effort skips L3 writes -> safe with a low MINGB (disk stable under our flock)
WS="$ROOT/workspace/sgl/researchers/$NAME"; MUTEX="$WS/.mutex-$VER"
done_already(){ [ -f "$WS/runs/$VER/mix.txt" ] && grep -aqE "Benchmark duration" "$WS/runs/$VER/mix.txt" 2>/dev/null; }
while :; do
  # cross-path coordination: if another path (hold_watch) already produced/owns this version, stop.
  done_already && { echo "[pinned] $VER already complete; exiting"; exit 0; }
  any=0
  for f in "$RT"/held/*; do
    [ -e "$f" ] || continue
    node=$(basename "$f"); jid=$(cat "$f" 2>/dev/null) || continue
    squeue -h -j "$jid" >/dev/null 2>&1 || continue
    any=1
    exec 200>"$RT/locks/$node.lock"
    # BLOCKING flock with a bounded wait (not flock -n): peers use blocking flock and queue FIFO,
    # so a non-blocking poller loses every release window to an already-queued waiter. -w lets us
    # fairly QUEUE for up to FLOCKWAIT sec per node, then move on to try the next node.
    if flock -w "${FLOCKWAIT:-30}" 200; then
      # ONE srun probes BOTH free disk AND foreign-server residency. The manager flock only
      # excludes flock-respecting peers; foreign pinned launchers srun --overlap WITHOUT the
      # flock, so a node can read flock-free while a foreign 8-GPU server is resident. Launching
      # a 2nd server there = OOM/SIGKILL collision. So skip any node with a live sglang server.
      probe=$(timeout 20 srun --jobid="$jid" --overlap -N1 -w "$node" -t 0:01:00 bash -c \
                'echo "DISK=$(df -BG --output=avail /mnt/localssd 2>/dev/null | tail -1 | tr -dc 0-9)"; echo "SRV=$(pgrep -cf "[s]glang.launch_server" 2>/dev/null || echo 0)"' 2>/dev/null)
      freeG=$(printf '%s\n' "$probe" | sed -n 's/^DISK=//p')
      nsrv=$(printf '%s\n' "$probe" | sed -n 's/^SRV=//p')
      if [ -n "${nsrv:-}" ] && [ "$nsrv" != "0" ]; then
        echo "[pinned] $node: foreign sglang server resident (nsrv=$nsrv) — release+skip (collision guard)"
        flock -u 200; exec 200>&-
      elif [ -n "$freeG" ] && [ "$freeG" -ge "$MINGB" ]; then
        # atomic mutex (shared with hold_watch) so we never double-run into the same run dir
        if ! mkdir "$MUTEX" 2>/dev/null; then
          done_already && { echo "[pinned] $VER complete elsewhere; exiting"; flock -u 200; exec 200>&-; exit 0; }
          echo "[pinned] $VER already running elsewhere (mutex) — defer; release node"; flock -u 200; exec 200>&-; exit 0
        fi
        try=1
        while [ "$try" -le "$MAXTRY" ]; do
          echo "[pinned] eval $VER on $node (job $jid) free=${freeG}G, attempt $try/$MAXTRY"
          srun --jobid="$jid" --overlap -N1 -w "$node" --gres=gpu:8 bash "$EVAL" "$NAME" "$VER" "$@"
          rc=$?
          # rc 0 = success; rc 8 = MIX_BENCH_FAILED (real, don't retry); else = load/timeout (retry)
          if [ "$rc" -eq 0 ] || [ "$rc" -eq 8 ]; then
            rmdir "$MUTEX" 2>/dev/null; flock -u 200; exec 200>&-; echo "[pinned] eval $VER exit $rc (final)"; exit $rc
          fi
          echo "[pinned] eval $VER attempt $try failed (rc=$rc); retrying on same node"
          try=$((try+1)); sleep 5
        done
        rmdir "$MUTEX" 2>/dev/null
        echo "[pinned] $node: $MAXTRY load failures; releasing, trying another node"
        flock -u 200; exec 200>&-
      else
        echo "[pinned] $node: free=${freeG:-?}G < ${MINGB}G, release+skip"
        flock -u 200; exec 200>&-
      fi
    else
      exec 200>&-
    fi
  done
  [ "$any" -eq 0 ] && { echo "[pinned] no live held nodes"; exit 3; }
  sleep 15
done
