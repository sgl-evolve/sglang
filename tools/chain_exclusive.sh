#!/usr/bin/env bash
# Wait for the s_writeback gate to finish, then immediately grab a node for the exclusive-tiering
# mechanism eval (write_back + SGLANG_HICACHE_EXCLUSIVE=1) so base_free doesn't steal the freed node.
set -uo pipefail
WS=/home/junyanch_google_com/autoresearch/workspace/sgl/v0.25_ablations/sgl_free/researchers/sgl_free
cd "$WS"
echo "[chain] waiting for s_writeback summary..."
for i in $(seq 1 240); do
  [ -f runs/s_writeback/summary.json ] && break
  sleep 30
done
[ -f runs/s_writeback/summary.json ] || { echo "[chain] s_writeback never finished; launching exclusive anyway"; }
echo "[chain] s_writeback done at $(date -u +%H:%M:%S); launching exclusive mechanism eval"
export SGLANG_HICACHE_EXCLUSIVE=1
exec bash tools/retry_eval.sh v1_exclusive --hicache-write-policy write_back
