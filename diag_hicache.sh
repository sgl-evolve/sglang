#!/usr/bin/env bash
# diag_hicache.sh — does the HICACHE path reach READY on a healthy node?
# Distinguishes node-specific capture flakiness (ondem-3/1-2) from a universal
# hicache-path stall (gloo prefetch sync groups / hicache capture). Uses a SMALL
# hicache-size + small local L3 so it doesn't need 768GB DRAM or 1.8TB disk — the
# goal is to exercise the hicache init/gloo/capture code path, not the full budget.
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch; WORK=$ROOT/workspace/sgl/researchers/quill-7m3
set -a; source "$ROOT/.env"; set +a; export HF_TOKEN="${HF_API_KEY:-}"
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"
export HF_HOME=/rmeng_data/junyanch-data/hf_cache
PORT=31013; LOG=$WORK/diag_hicache_server.log
L3=/mnt/localssd/quill-7m3-hcdiag; rm -rf "$L3" 2>/dev/null; mkdir -p "$L3"
echo "hicache diag on $(hostname) @ $(date +%H:%M:%S)"
free -g | head -2
python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-122B-A10B-FP8 --tp 8 \
  --trust-remote-code --context-length 262144 --chunked-prefill-size 6144 \
  --mem-fraction-static 0.85 --page-size 64 --enable-metrics \
  --enforce-disable-flashinfer-allreduce-fusion \
  --enable-hierarchical-cache --hicache-size 8 --hicache-write-policy write_through \
  --hicache-io-backend direct --hicache-mem-layout page_first_direct \
  --hicache-storage-backend file --hicache-storage-prefetch-policy wait_complete \
  --hicache-storage-backend-extra-config "{\"file_path\":\"$L3\",\"max_size\":\"100G\"}" \
  --port $PORT > "$LOG" 2>&1 &
SRV=$!; res="TIMEOUT"
for i in $(seq 1 1000); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { res="HICACHE_READY(~$((i*3/60))min)"; break; }
  kill -0 $SRV 2>/dev/null || { res="SERVER_DIED"; break; }
  if (( i % 40 == 0 )); then echo "[$((i*3/60))min] gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|paste -sd,) comp=$(ps -eo comm|grep -icE 'cicc|ptxas|nvcc|cc1')"; fi
  sleep 3
done
echo "HICACHE_DIAG_RESULT: $res @ $(date +%H:%M:%S)"
grep -iE "Bus error|fired up|Guessing device|allreduce fusion|Traceback|died|prefetch" "$LOG" 2>/dev/null | tail -8
kill $SRV 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null || true
rm -rf "$L3" 2>/dev/null
echo "hicache diag done @ $(date +%H:%M:%S)"
