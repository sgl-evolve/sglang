#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/junyanch_google_com/autoresearch; WORK=$ROOT/workspace/sgl/researchers/quill-7m3
set -a; source "$ROOT/.env"; set +a; export HF_TOKEN="${HF_API_KEY:-}"
source "$WORK/.venv/bin/activate"; export PYTHONPATH="$WORK/python"
export HF_HOME=/rmeng_data/junyanch-data/hf_cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID          # <-- consistent rank->GPU enumeration (lossless)
export NCCL_DEBUG=WARN
PORT=31011; LOG=$WORK/diag_devorder_server.log
echo "devorder diag on $(hostname) @ $(date +%H:%M:%S) CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER"
python3 -m sglang.launch_server --model-path Qwen/Qwen3.5-122B-A10B-FP8 --tp 8 \
  --trust-remote-code --context-length 262144 --chunked-prefill-size 6144 \
  --mem-fraction-static 0.85 --page-size 64 --enable-metrics --port $PORT > "$LOG" 2>&1 &
SRV=$!; res="TIMEOUT"
for i in $(seq 1 900); do
  curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && { res="READY(~$((i*3/60))min)"; break; }
  kill -0 $SRV 2>/dev/null || { res="SERVER_DIED"; break; }
  if (( i % 40 == 0 )); then echo "[$((i*3/60))min] gpu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader|paste -sd,) compilers=$(ps -eo comm|grep -icE 'cicc|ptxas|cc1|nvcc')"; fi
  sleep 3
done
echo "DEVORDER_RESULT: $res @ $(date +%H:%M:%S)"
grep -iE "Bus error|fired up|Guessing device|NCCL WARN|error" "$LOG" 2>/dev/null | tail -8
kill $SRV 2>/dev/null; sleep 3; pkill -9 -f sglang.launch_server 2>/dev/null || true
