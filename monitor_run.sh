#!/usr/bin/env bash
# Watch a running eval's server.log (shared FS) and snapshot the throughput/bottleneck signals.
# Usage: monitor_run.sh <version>   e.g. monitor_run.sh v1-timeout
set -uo pipefail
VER="${1:?version}"
W=/home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-heron-e29
LOG="$W/runs/$VER/server.log"
MIX="$W/runs/$VER/mix.txt"
OUT="$W/monitor-$VER.txt"
: > "$OUT"
echo "[monitor] waiting for $LOG ..." >> "$OUT"
for _ in $(seq 1 720); do [ -f "$LOG" ] && break; sleep 10; done
[ -f "$LOG" ] || { echo "[monitor] server.log never appeared (30min)" >> "$OUT"; exit 0; }
echo "[monitor] server.log found $(date '+%H:%M:%S')" >> "$OUT"
# poll until the bench finishes (mix.txt has the summary) or 3h
for i in $(seq 1 180); do
  ts=$(date '+%H:%M:%S')
  # last decode-batch stat line (running-req, token usage, gen throughput, queue)
  db=$(grep -aE "Decode batch" "$LOG" 2>/dev/null | tail -1)
  pb=$(grep -aE "Prefill batch" "$LOG" 2>/dev/null | tail -1)
  nretr=$(grep -acE "Retract requests" "$LOG" 2>/dev/null)
  ready=$(grep -aE "server ready|SERVER_DIED|SERVER_TIMEOUT" "$LOG" 2>/dev/null | tail -1)
  {
    echo "==== $ts (poll $i) ===="
    [ -n "$ready" ] && echo "  status: $ready"
    echo "  retract_warnings_total: ${nretr:-0}"
    [ -n "$pb" ] && echo "  PREFILL: ${pb##*Prefill batch}"
    [ -n "$db" ] && echo "  DECODE:  ${db##*Decode batch}"
  } >> "$OUT"
  # bench progress line (bench_serving prints running stats)
  grep -aE "Benchmark duration|Mean TTFT|Request throughput|Output token throughput" "$MIX" 2>/dev/null | tail -4 >> "$OUT" 2>/dev/null
  # done?
  grep -aqE "Benchmark duration" "$MIX" 2>/dev/null && { echo "[monitor] bench complete $ts" >> "$OUT"; break; }
  sleep 60
done
echo "[monitor] exiting" >> "$OUT"
