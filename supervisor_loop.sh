#!/usr/bin/env bash
# supervisor_loop.sh — self-sustaining "supervisor of supervisors". Runs ensure_waiters.sh every ~10min
# so the 3 collision-safe pool waiters stay alive and armed EVEN IF the interactive loop pauses (a died
# waiter otherwise stays dead and misses the next genuine pool gap). Detached (setsid), single-instance
# (flock), self-logging. This is what makes "never stop" real at the infra level: the eval-capture
# machinery self-heals continuously without needing an active ping.
set -uo pipefail
cd /home/junyanch_google_com/autoresearch/workspace/sgl/researchers/onyx-7q2
exec 209>/tmp/onyx-7q2-suploop.lock
flock -n 209 || { echo "supervisor_loop already running -> exit"; exit 0; }
echo "[suploop] started $(date '+%F %T')"
while :; do
  bash ensure_waiters.sh >> supervisor_loop.out 2>&1 || true
  # exit if the whole experiment queue is drained (all work done)
  if [ ! -s experiment_queue.txt ]; then echo "[suploop] queue empty -> exit $(date '+%F %T')"; exit 0; fi
  sleep 600
done
