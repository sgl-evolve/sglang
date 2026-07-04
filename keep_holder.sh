#!/usr/bin/env bash
# keep_holder.sh — minimal autonomy insurance for a long capacity block. Keeps EXACTLY the
# holder_watcher alive (NO pool node_waiters — the pool is foreign-jammed, so racing it is useless
# churn/bad-citizenship). Detached (setsid), single-instance (flock). Relaunches holder_watcher only
# if it died AND the hold job still exists AND the queue is non-empty. Exits when queue drains or the
# hold is gone (work done / released). This makes v19 run on node-recovery even if pings stop.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/onyx-7q2
exec 208>/tmp/onyx-7q2-keepholder.lock
flock -n 208 || { echo "keep_holder already running -> exit"; exit 0; }
echo "[keep] started $(date '+%F %T')"
while :; do
  # done? queue drained -> stop insurance.
  [ -s experiment_queue.txt ] || { echo "[keep] queue empty -> exit $(date '+%F %T')"; exit 0; }
  jid=$(cat .holdjob 2>/dev/null)
  # hold gone (cancelled/finished and not resubmitted) -> nothing to watch; stop.
  if [ -z "$jid" ] || ! squeue -h -j "$jid" >/dev/null 2>&1; then
    echo "[keep] hold $jid gone -> exit $(date '+%F %T')"; exit 0
  fi
  # holder_watcher alive? (its own flock is the source of truth)
  if ( flock -n 210 true ) 210>/tmp/onyx-7q2-holder.lock 2>/dev/null; then
    # lock acquired => NO holder running => relaunch it
    echo "[keep] holder_watcher dead -> relaunch $(date '+%F %T')"
    setsid bash holder_watcher.sh > holder_watcher.out 2>&1 < /dev/null &
  fi
  sleep 600
done
