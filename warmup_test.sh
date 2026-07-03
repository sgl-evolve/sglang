#!/usr/bin/env bash
# Warmup-only test: launch the 122B server with the EXACT frozen eval flags, wait for /health,
# then kill. Purpose: (1) test whether first-run warmup deadlocks, (2) pre-warm $WORK/.cache so
# real evals on certified nodes are fast + avoid the cold-cache concurrent-compile race.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch
set -a; . "$ROOT/.env"; set +a
export HF_TOKEN="$HF_API_KEY" HF_HOME=/rmeng_data/junyanch-data/hf_cache HF_HUB_OFFLINE=1
WORK=$ROOT/workspace/sgl/researchers/kv-heron-eb9
SSD=/mnt/localssd/kv-heron-eb9-warmup; rm -rf "$SSD"; mkdir -p "$SSD"
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$SSD"
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"; cd "$WORK"
export TRITON_CACHE_DIR="$WORK/.cache/triton" CUDA_CACHE_PATH="$WORK/.cache/nv" \
       FLASHINFER_CACHE_DIR="$WORK/.cache/flashinfer" XDG_CACHE_HOME="$WORK/.cache"
mkdir -p "$WORK/.cache"
MODEL=Qwen/Qwen3.5-122B-A10B-FP8
PORT=31577
OUT="$WORK/warmup"; mkdir -p "$OUT"
echo "host=$(hostname); free disk $(df -BG /mnt/localssd|tail -1|awk '{print $4}'); MemAvail $(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)G"
python3 -m sglang.launch_server \
  --model-path "$MODEL" --tp 8 --trust-remote-code \
  --context-length 262144 --chunked-prefill-size 6144 --mem-fraction-static 0.85 \
  --enable-hierarchical-cache --hicache-size 96 \
  --hicache-storage-backend file --hicache-storage-backend-extra-config "{\"file_path\":\"$SSD\",\"max_size\":\"200G\",\"min_free_space\":\"50G\"}" \
  --page-size 64 --hicache-io-backend direct --hicache-mem-layout page_first_direct \
  --hicache-write-policy write_through --hicache-storage-prefetch-policy best_effort \
  --disable-cuda-graph --enforce-disable-flashinfer-allreduce-fusion \
  --enable-metrics --enable-cache-report --port "$PORT" > "$OUT/warmup_server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null; pkill -9 -f "sglang::" 2>/dev/null; rm -rf "$SSD"' EXIT
ok=0
for i in $(seq 1 1000); do   # up to 1000*3=3000s = 50 min (patient: first-run compile is slow)
  if curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then echo "WARMUP_OK ready after ~$((i*3))s"; ok=1; break; fi
  kill -0 $SRV 2>/dev/null || { echo "WARMUP_SERVER_DIED"; break; }
  if [ $((i % 20)) -eq 0 ]; then
    comp=$(pgrep -c -f 'cicc|ptxas|cc1plus' 2>/dev/null || echo 0)
    gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    echo "[t=$((i*3))s] warming; compilers=$comp gpu0=$gu; last: $(grep -vE 'warnings.warn|UserWarning|deprecated|frame #|Multi-thread' "$OUT/warmup_server.log" | tail -1 | cut -c1-90)"
  fi
  sleep 3
done
[ "$ok" -eq 1 ] && echo "WARMUP SUCCESS — cache is warm at $WORK/.cache" || { echo "WARMUP FAILED/timeout — server.log tail:"; grep -vE "warnings.warn|UserWarning|deprecated" "$OUT/warmup_server.log" | tail -20; }
echo "cache sizes:"; du -sh "$WORK/.cache"/* 2>/dev/null
