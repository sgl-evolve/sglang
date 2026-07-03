#!/usr/bin/env bash
# mon.sh <version> [N]  — clean live view of an eval's server.log (filters the giant server_args line)
VER="${1:?usage: mon.sh <version> [N]}"; N="${2:-12}"
SRV="/home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-flint-2c/runs/$VER/server.log"
[ -f "$SRV" ] || { echo "no server.log at $SRV"; exit 1; }
echo "=== ready/warmup ==="
grep -aiE "server is fired up|Uvicorn running|ready to roll" "$SRV" | tail -2
grep -aiE "warmup" "$SRV" | tail -1
echo "=== retract/preempt/oom counts ==="
printf "  retract: %s  preempt: %s  OOM: %s\n" \
  "$(grep -aic "retract" "$SRV")" "$(grep -aic "preempt" "$SRV")" "$(grep -aic -E "out of memory|OOM|KV cache pool is full" "$SRV")"
echo "=== last $N scheduler stat lines ==="
grep -aiE "Prefill batch|Decode batch" "$SRV" | grep -av "server_args" | tail -"$N"
echo "=== bench progress (mix.txt) ==="
MIX="/home/junyanch_google_com/autoresearch/workspace/sgl/researchers/kv-flint-2c/runs/$VER/mix.txt"
[ -f "$MIX" ] && tail -3 "$MIX" || echo "  (bench not started)"
